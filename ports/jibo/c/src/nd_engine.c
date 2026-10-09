/* nd_engine.c: see nd_engine.h. */
#include "nd_engine.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "nd_prompt.h"

const char *nd_stop_name(nd_stop s) {
  switch (s) {
  case ND_STOP_EOS: return "Eos";
  case ND_STOP_IM_END: return "ImEnd";
  case ND_STOP_MAX_TOKENS: return "MaxTokens";
  case ND_STOP_MAX_SEQ_LEN: return "MaxSeqLen";
  case ND_STOP_CANCELLED: return "Cancelled";
  }
  return "?";
}

void nd_gen_opts_default(nd_gen_opts *o) {
  memset(o, 0, sizeof *o);
  o->max_new_tokens = ND_DEFAULT_MAX_NEW_TOKENS;
  o->kv = ND_KV_F32;
}

void nd_result_free(nd_result *r) {
  free(r->text);
  free(r->tokens);
  memset(r, 0, sizeof *r);
}

static double now_s(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

int nd_engine_from_bytes(uint8_t *raw, size_t len, size_t depth, nd_engine **out, nd_err *e) {
  nd_engine *g = calloc(1, sizeof *g);
  size_t i, blen = 0;
  const uint8_t *blob = NULL;
  int rc;
  *out = NULL;
  if (!g) {
    free(raw);
    { nd_seterr(e, "out of memory"); return ND_E_NOMEM; }
  }
  g->raw = raw;
  if ((rc = nd_cact_parse(raw, len, &g->cact, e)) != ND_OK) {
    free(raw);
    free(g);
    return rc;
  }
  if ((rc = nd_model_load(&g->cact, depth, &g->model, e)) != ND_OK) goto fail;
  /* tokenizer_index: the first RAW record. */
  for (i = 0; i < g->cact.nrec; i++)
    if (g->cact.recs[i].dtype == ND_DT_RAW) {
      blob = nd_cact_blob(&g->cact, i, &blen);
      break;
    }
  if (!blob) {
    nd_seterr(e, "container carries no tokenizer");
    rc = ND_E_FORMAT;
    goto fail;
  }
  if (nd_tok_load(blob, blen, &g->tok, e) != ND_OK) {
    nd_seterr(e, "embedded tokenizer did not decode");
    rc = ND_E_FORMAT;
    goto fail;
  }
  /* Upstream fixes these in needle/model/tokenizer.py. */
  g->eos_id = 1;
  g->bos_id = 2;
  g->im_end_id = nd_tok_id_of(g->tok, ND_IM_END);
  g->tc_start_id = nd_tok_id_of(g->tok, ND_TOOL_CALL_START);
  g->tc_end_id = nd_tok_id_of(g->tok, ND_TOOL_CALL_END);
  g->tools_end_id = nd_tok_id_of(g->tok, ND_TOOLS_END);
  *out = g;
  return ND_OK;
fail:
  nd_engine_free(g);
  return rc;
}

int nd_engine_load(const char *path, size_t depth, nd_engine **out, nd_err *e) {
  FILE *f = fopen(path, "rb");
  uint8_t *raw;
  long l;
  *out = NULL;
  if (!f) { nd_seterr(e, "%s: cannot open", path); return ND_E_IO; }
  if (fseek(f, 0, SEEK_END) != 0 || (l = ftell(f)) < 0 || fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    { nd_seterr(e, "%s: cannot size", path); return ND_E_IO; }
  }
  raw = malloc(l ? (size_t)l : 1);
  if (!raw) {
    fclose(f);
    { nd_seterr(e, "%s: out of memory", path); return ND_E_NOMEM; }
  }
  if (fread(raw, 1, (size_t)l, f) != (size_t)l) {
    fclose(f);
    free(raw);
    { nd_seterr(e, "%s: short read", path); return ND_E_IO; }
  }
  fclose(f);
  return nd_engine_from_bytes(raw, (size_t)l, depth, out, e);
}

void nd_engine_free(nd_engine *g) {
  if (!g) return;
  free(g->byte_table);
  free(g->mask);
  free(g->prefix_ids);
  nd_cache_free(g->prefix_cache);
  nd_tok_free(g->tok);
  nd_model_free(g->model);
  nd_cact_free(&g->cact);
  free(g->raw);
  free(g);
}

/* alloc_prompt_ids: BOS, then the encoded prompt. */
static int prompt_ids_of(const nd_engine *g, const char *prompt, uint32_t **ids, size_t *n) {
  uint32_t *enc = NULL, *all;
  size_t m = 0, total;
  int rc = nd_tok_encode(g->tok, prompt, strlen(prompt), &enc, &m);
  if (rc != ND_OK) return rc;
  if (!nd_add_ok(m, 1, &total) || !(all = malloc((m + 1) * sizeof *all))) {
    free(enc);
    return ND_E_NOMEM;
  }
  all[0] = g->bos_id;
  if (m) memcpy(all + 1, enc, m * sizeof *all);
  free(enc);
  *ids = all;
  *n = m + 1;
  return ND_OK;
}

int nd_engine_prompt_ids(const nd_engine *g, const char *query, const char *tools_json, const char *system,
                         uint32_t **ids, size_t *n) {
  char *p = nd_build_prompt(query, tools_json, system);
  int rc;
  if (!p) return ND_E_NOMEM;
  rc = prompt_ids_of(g, p, ids, n);
  free(p);
  return rc;
}

/* constrained::byte_table over the tokenizer, and the mask buffer. */
static int ensure_byte_table(nd_engine *g) {
  size_t n = nd_tok_size(g->tok), i, rows = nd_logit_rows(&g->model->cfg);
  const char **pieces;
  size_t *lens;
  uint8_t *types;
  nd_err e;
  int rc = ND_E_NOMEM;
  if (g->byte_table) return ND_OK;
  pieces = nd_calloc(n, sizeof *pieces);
  lens = nd_calloc(n, sizeof *lens);
  types = nd_calloc(n, 1);
  if (pieces && lens && types && (g->mask = nd_calloc(rows, sizeof *g->mask))) {
    for (i = 0; i < n; i++) {
      pieces[i] = nd_tok_piece(g->tok, (uint32_t)i, &lens[i]);
      types[i] = (uint8_t)nd_tok_type(g->tok, (uint32_t)i);
    }
    rc = nd_grammar_byte_table_from_pieces(pieces, lens, types, n, &g->byte_table, &e);
    if (rc == ND_OK) g->n_byte_table = n;
  }
  free(pieces);
  free(lens);
  free(types);
  return rc;
}

/* ---- the tool-prefix cache ---- */

void nd_engine_enable_prefix_cache(nd_engine *g) { g->prefix_on = 1; }

/* Where the reusable prefix ends: just after the first `</tools>`, when something follows. */
static size_t prefix_len(const nd_engine *g, const uint32_t *ids, size_t n) {
  size_t i;
  if (g->tools_end_id < 0) return 0;
  for (i = 0; i < n; i++)
    if (ids[i] == (uint32_t)g->tools_end_id) return i + 1 < n ? i + 1 : 0;
  return 0;
}

/* The cache after ids[0..l): a copy of the stored entry when it matches (*reused = l), else
 * computed and stored (*reused = 0). */
static int prefix_state(nd_engine *g, const uint32_t *ids, size_t l, nd_kv_precision kv, nd_cache **out,
                        size_t *reused) {
  const nd_cfg *cfg = &g->model->cfg;
  float *scratch;
  nd_cache *c;
  uint32_t *keep;
  int rc;
  *out = NULL;
  *reused = 0;
  if (g->prefix_cache && g->prefix_kv == kv && g->prefix_n == l && !memcmp(g->prefix_ids, ids, l * sizeof *ids)) {
    if (!(*out = nd_cache_clone(g->prefix_cache))) return ND_E_NOMEM;
    *reused = l;
    return ND_OK;
  }
  /* Sized to the prefix, so the stored entry and each copy stay small; copies grow as they go. */
  if (!(c = nd_cache_new(cfg, l, kv))) return ND_E_NOMEM;
  if (!(scratch = malloc(nd_logit_rows(cfg) * sizeof *scratch))) {
    nd_cache_free(c);
    return ND_E_NOMEM;
  }
  rc = nd_prefill(g->model, ids, l, c, scratch);
  free(scratch);
  if (rc != ND_OK) {
    nd_cache_free(c);
    return rc;
  }
  /* Store a copy; if that fails, the request still proceeds without storing. */
  keep = malloc(l * sizeof *keep);
  if (keep) {
    nd_cache *snap = nd_cache_clone(c);
    if (snap) {
      memcpy(keep, ids, l * sizeof *ids);
      free(g->prefix_ids);
      nd_cache_free(g->prefix_cache);
      g->prefix_ids = keep;
      g->prefix_n = l;
      g->prefix_kv = kv;
      g->prefix_cache = snap;
    } else {
      free(keep);
    }
  }
  *out = c;
  return ND_OK;
}

/* ---- sampling ---- */

static uint32_t argmax(const float *l, size_t n) {
  size_t i, best = 0;
  float top = -INFINITY;
  for (i = 0; i < n; i++)
    if (l[i] > top) {
      top = l[i];
      best = i;
    }
  return (uint32_t)best;
}

static uint64_t splitmix_next(uint64_t *s) {
  uint64_t z;
  *s += 0x9E3779B97F4A7C15ull;
  z = *s;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

/* Rust's f32::exp in the std engine is the platform libm's expf, as here. */
static uint32_t sample(const float *l, size_t n, float temperature, uint64_t *rng) {
  float inv = 1.0f / temperature, mx = -INFINITY;
  double sum = 0.0, target;
  size_t i;
  for (i = 0; i < n; i++)
    if (l[i] > mx) mx = l[i];
  for (i = 0; i < n; i++) sum += (double)expf((l[i] - mx) * inv);
  target = (double)(splitmix_next(rng) >> 11) / (double)(1ull << 53) * sum;
  for (i = 0; i < n; i++) {
    target -= (double)expf((l[i] - mx) * inv);
    if (target <= 0.0) return (uint32_t)i;
  }
  return (uint32_t)(n - 1);
}

/* ---- streaming ---- */

/* stream_delta: `full` minus trailing U+FFFD, if it extends `emitted`. */
static size_t stream_delta(const char *full, size_t flen, const char *emitted, size_t elen) {
  while (flen >= 3 && (uint8_t)full[flen - 3] == 0xEF && (uint8_t)full[flen - 2] == 0xBF &&
         (uint8_t)full[flen - 1] == 0xBD)
    flen -= 3;
  if (flen <= elen || memcmp(full, emitted, elen) != 0) return 0;
  return flen - elen;
}

int nd_generate(nd_engine *g, const char *query, const char *tools_json, const nd_gen_opts *o,
                nd_on_token on_token, nd_keep_going keep_going, void *user, nd_result *r) {
  const nd_cfg *cfg = &g->model->cfg;
  size_t rows = nd_logit_rows(cfg), max = cfg->max_seq_len, budget, step, nids = 0, elen = 0, split;
  uint32_t *ids = NULL;
  float *logits = NULL;
  nd_cache *cache = NULL;
  char *prompt, *full = NULL, *emitted = NULL;
  size_t flen = 0;
  uint64_t rng = o->seed;
  nd_grammar *gram = NULL;
  int in_tool_call = 0, rc = ND_OK;
  double t0 = now_s(), t1, t2;

  memset(r, 0, sizeof *r);
  r->stop = ND_STOP_MAX_TOKENS;
  if (!(prompt = nd_build_prompt(query, tools_json, o->system))) return ND_E_NOMEM;
  rc = prompt_ids_of(g, prompt, &ids, &nids);
  free(prompt);
  if (rc != ND_OK) return rc;
  r->prompt_tokens = nids;
  r->t_tokenize = now_s() - t0;

  r->prompt_truncated = nids > max;
  if (r->prompt_truncated) nids = max;
  budget = max > nids ? max - nids : 0;
  if (budget > o->max_new_tokens) budget = o->max_new_tokens;

  if (!(r->tokens = malloc((budget ? budget : 1) * sizeof *r->tokens)) || !(logits = malloc(rows * sizeof *logits)) ||
      !(emitted = calloc(1, 1))) {
    rc = ND_E_NOMEM;
    goto done;
  }
  if (o->constrain) {
    nd_tooldefs defs = {NULL, 0};
    nd_err ge;
    if ((rc = ensure_byte_table(g)) != ND_OK) goto done;
    if ((rc = nd_grammar_parse_tools(tools_json, strlen(tools_json), &defs, &ge)) != ND_OK) goto done;
    /* No declared tools: no grammar, as Rust's `(!defs.is_empty()).then(..)`. */
    if (defs.n) {
      rc = nd_grammar_new(defs.tools, defs.n, g->byte_table, g->n_byte_table, &gram, &ge);
      if (rc == ND_OK) nd_grammar_set_unique_arg_keys(gram, 1);
    }
    nd_tooldefs_free(&defs);
    if (rc != ND_OK) goto done;
  }

  /* With the prefix cache, the tool prefix comes from (or goes into) the stored entry and the
   * rest of the prompt is stepped: the same cache and logits as one prefill. */
  t1 = now_s();
  split = g->prefix_on ? prefix_len(g, ids, nids) : 0;
  if (split) {
    size_t i;
    if ((rc = prefix_state(g, ids, split, o->kv, &cache, &r->prefix_reused)) != ND_OK) goto done;
    for (i = split; i < nids; i++)
      if ((rc = nd_decode_step(g->model, cache, ids[i], logits)) != ND_OK) goto done;
  } else {
    if (!(cache = nd_cache_new(cfg, nids + budget, o->kv))) {
      rc = ND_E_NOMEM;
      goto done;
    }
    if (nids && (rc = nd_prefill(g->model, ids, nids, cache, logits)) != ND_OK) goto done;
  }
  r->t_prefill = now_s() - t1;
  t2 = now_s();

  for (step = 0; step < budget; step++) {
    uint32_t next;
    size_t d;
    if (keep_going && !keep_going(user)) {
      r->stop = ND_STOP_CANCELLED;
      break;
    }
    if (in_tool_call && gram) {
      size_t i;
      if ((rc = nd_grammar_logit_mask(gram, g->mask, rows)) != ND_OK) goto done;
      for (i = 0; i < rows; i++) logits[i] += g->mask[i];
    }
    next = o->temperature <= 0.0f ? argmax(logits, rows) : sample(logits, rows, o->temperature, &rng);
    if (next == g->eos_id) {
      r->stop = ND_STOP_EOS;
      break;
    }
    if (g->im_end_id >= 0 && next == (uint32_t)g->im_end_id) {
      r->stop = ND_STOP_IM_END;
      break;
    }
    r->tokens[r->ntokens++] = next;
    free(full);
    full = NULL;
    if ((rc = nd_tok_decode(g->tok, r->tokens, r->ntokens, &full, &flen)) != ND_OK) goto done;
    d = stream_delta(full, flen, emitted, elen);
    if (d) {
      char *grown = realloc(emitted, elen + d + 1);
      if (!grown) {
        rc = ND_E_NOMEM;
        goto done;
      }
      emitted = grown;
      if (on_token) on_token(user, next, full + elen, d);
      memcpy(emitted + elen, full + elen, d);
      elen += d;
      emitted[elen] = 0;
    }
    if (gram) {
      if (g->tc_start_id >= 0 && next == (uint32_t)g->tc_start_id)
        in_tool_call = 1;
      else if (g->tc_end_id >= 0 && next == (uint32_t)g->tc_end_id)
        in_tool_call = 0;
      else if (in_tool_call)
        if ((rc = nd_grammar_update(gram, next)) != ND_OK) goto done;
    }
    if (nd_cache_pos(cache) >= max) {
      r->stop = ND_STOP_MAX_SEQ_LEN;
      break;
    }
    if ((rc = nd_decode_step(g->model, cache, next, logits)) != ND_OK) goto done;
  }
  r->t_decode = now_s() - t2;

  free(full);
  full = NULL;
  if ((rc = nd_tok_decode(g->tok, r->tokens, r->ntokens, &full, &flen)) != ND_OK) goto done;
  /* What was held back (a trailing U+FFFD that never completed) goes out now. */
  if (on_token && r->ntokens && flen > elen && memcmp(full, emitted, elen) == 0)
    on_token(user, r->tokens[r->ntokens - 1], full + elen, flen - elen);
  r->text = full;
  r->text_len = flen;
  full = NULL;
  r->positions = nd_cache_pos(cache);
  if (r->prompt_truncated) r->stop = ND_STOP_MAX_SEQ_LEN;
done:
  free(full);
  free(emitted);
  free(logits);
  free(ids);
  nd_grammar_free(gram);
  nd_cache_free(cache);
  if (rc != ND_OK) nd_result_free(r);
  return rc;
}

int nd_confidence_for(const nd_engine *g, const char *query, const char *tools_json, const char *completion,
                      float *p) {
  const nd_head *h = g->model->confidence;
  char *prompt, *text;
  uint32_t *ids;
  size_t n, plen, clen;
  float logit;
  int rc;
  if (!h) return 0;
  if (!(prompt = nd_build_prompt(query, tools_json, NULL))) return ND_E_NOMEM;
  plen = strlen(prompt);
  clen = strlen(completion);
  if (!(text = malloc(plen + clen + 1))) {
    free(prompt);
    return ND_E_NOMEM;
  }
  memcpy(text, prompt, plen);
  memcpy(text + plen, completion, clen + 1);
  free(prompt);
  rc = prompt_ids_of(g, text, &ids, &n);
  free(text);
  if (rc != ND_OK) return rc;
  if (n > g->model->cfg.max_seq_len) n = g->model->cfg.max_seq_len;
  rc = nd_forward_head(g->model, h, ids, n, &logit);
  free(ids);
  if (rc != ND_OK) return rc;
  *p = 1.0f / (1.0f + expf(-logit));
  return 1;
}

/* ---- text helpers ---- */

/* char::is_whitespace (White_Space): the code point starting at s, or 0. Sets *w to its width. */
static int ws_at(const uint8_t *s, size_t n, size_t *w) {
  uint32_t c;
  if (!n) return 0;
  if (s[0] < 0x80) {
    *w = 1;
    return s[0] == ' ' || (s[0] >= 0x09 && s[0] <= 0x0D);
  }
  if ((s[0] & 0xE0) == 0xC0 && n >= 2) {
    c = ((uint32_t)(s[0] & 0x1F) << 6) | (s[1] & 0x3F);
    *w = 2;
    return c == 0x85 || c == 0xA0;
  }
  if ((s[0] & 0xF0) == 0xE0 && n >= 3) {
    c = ((uint32_t)(s[0] & 0x0F) << 12) | ((uint32_t)(s[1] & 0x3F) << 6) | (s[2] & 0x3F);
    *w = 3;
    return c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F ||
           c == 0x205F || c == 0x3000;
  }
  return 0;
}

const char *nd_trim(const char *str, size_t n, size_t *out_len) {
  const uint8_t *s = (const uint8_t *)str;
  size_t w = 0, b;
  while (n && ws_at(s, n, &w)) s += w, n -= w;
  for (;;) {
    /* Step back over one code point. */
    if (!n) break;
    b = n - 1;
    while (b && (s[b] & 0xC0) == 0x80 && n - b < 4) b--;
    if (!ws_at(s + b, n - b, &w) || b + w != n) break;
    n = b;
  }
  *out_len = n;
  return (const char *)s;
}

const char *nd_extract_between(const char *text, const char *open, const char *close, size_t *len) {
  const char *a = strstr(text, open), *b;
  if (!a) return NULL;
  a += strlen(open);
  b = strstr(a, close);
  return nd_trim(a, b ? (size_t)(b - a) : strlen(a), len);
}
