/* nd_service.c: see nd_service.h. Transcription of runner/src/service.rs. */
#include "nd_service.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "nd_sha256.h"
#include "nd_sysinfo.h"

double nd_now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

double nd_ms(double seconds) { return round(seconds * 1e6) / 1e3; }

/* ms() of Duration::from_millis(n): as_secs_f64 is secs + nanos / 1e9. */
static double ms_of_millis(uint64_t n) {
  double s = (double)(n / 1000) + (double)((n % 1000) * 1000000u) / 1e9;
  return nd_ms(s);
}

static char *xstrdup(const char *s) {
  size_t n = strlen(s);
  char *d = malloc(n + 1);
  if (d) memcpy(d, s, n + 1);
  return d;
}

static char *xprintf(const char *fmt, ...) {
  va_list ap;
  int n;
  char *s;
  va_start(ap, fmt);
  n = vsnprintf(NULL, 0, fmt, ap);
  va_end(ap);
  if (n < 0 || !(s = malloc((size_t)n + 1))) return NULL;
  va_start(ap, fmt);
  vsnprintf(s, (size_t)n + 1, fmt, ap);
  va_end(ap);
  return s;
}

void nd_limits_default(nd_limits *l) {
  memset(l, 0, sizeof *l);
  l->max_total_tokens = 512;
  l->max_new_tokens = 256;
  l->min_new_tokens = 64;
  l->max_query_bytes = 2048;
}

void nd_options_default(nd_options *o) {
  memset(o, 0, sizeof *o);
  o->grounding = ND_GROUND_ENFORCE;
}

int nd_grounding_parse(const char *s, nd_grounding_mode *out) {
  static const char *names[4] = {"off", "report", "enforce", "strict"};
  int i;
  for (i = 0; i < 4; i++)
    if (!strcmp(s, names[i])) {
      *out = (nd_grounding_mode)i;
      return 1;
    }
  return 0;
}

const char *nd_grounding_name(nd_grounding_mode g) {
  switch (g) {
  case ND_GROUND_OFF: return "off";
  case ND_GROUND_REPORT: return "report";
  case ND_GROUND_ENFORCE: return "enforce";
  case ND_GROUND_STRICT: return "strict";
  }
  return "?";
}

const char *nd_stop_wire_name(nd_stop s) {
  switch (s) {
  case ND_STOP_EOS: return "eos";
  case ND_STOP_IM_END: return "im_end";
  case ND_STOP_MAX_TOKENS: return "max_tokens";
  case ND_STOP_MAX_SEQ_LEN: return "max_seq_len";
  case ND_STOP_CANCELLED: return "cancelled";
  }
  return "?";
}

int nd_stop_from_wire(const char *s, size_t len, nd_stop *out) {
  nd_stop i;
  for (i = ND_STOP_EOS; i <= ND_STOP_CANCELLED; i++) {
    const char *n = nd_stop_wire_name(i);
    if (strlen(n) == len && !memcmp(n, s, len)) {
      *out = i;
      return 1;
    }
  }
  return 0;
}

void nd_request_free(nd_request *r) {
  size_t i;
  free(r->id);
  free(r->query);
  for (i = 0; i < r->n_tools; i++) free((char *)r->tools[i].ptr);
  free(r->tools);
  memset(r, 0, sizeof *r);
}

void nd_refusal_obj(nd_obj *o, const char *id, const char *status, const char *detail, size_t detail_len) {
  nd_obj_cstr(o, "request_id", id);
  nd_obj_cstr(o, "status", status);
  nd_obj_str(o, "detail", detail, detail_len);
}

char *nd_refusal(const char *id, const char *status, const char *detail) {
  nd_obj o;
  size_t n;
  nd_obj_init(&o);
  nd_refusal_obj(&o, id, status, detail, strlen(detail));
  return nd_obj_finish(&o, &n);
}

/* ---- parse_line ---- */

static char *dup_n(const char *s, size_t n) {
  char *d = malloc(n + 1);
  if (d) {
    memcpy(d, s, n);
    d[n] = 0;
  }
  return d;
}

nd_line_kind nd_parse_line(const char *line, size_t len, size_t max_line, nd_request *req, char **refusal) {
  nd_json_doc *doc = NULL;
  const nd_json_value *v, *x;
  const nd_json_member **keys = NULL;
  char *msg = NULL, *id = NULL;
  size_t nk, i;
  nd_line_kind kind = ND_LINE_ERROR;
  memset(req, 0, sizeof *req);
  *refusal = NULL;
#define BAD(text) (*refusal = nd_refusal(id ? id : "", "invalid_request", (text)))
  if (len > max_line) {
    *refusal = nd_refusal("", "invalid_request", "request line too long");
    return ND_LINE_ERROR;
  }
  if (nd_serde_parse(line, len, &doc, &msg) != ND_OK) {
    char *d = xprintf("not JSON: %s", msg ? msg : "out of memory");
    *refusal = nd_refusal("", "invalid_request", d ? d : "not JSON");
    free(d);
    free(msg);
    return ND_LINE_ERROR;
  }
  v = nd_json_root(doc);
  if (v->type != ND_JSON_OBJECT) {
    *refusal = nd_refusal("", "invalid_request", "request is not an object");
    goto out;
  }
  x = nd_json_get_cstr(v, "request_id");
  if (x) {
    if (x->type != ND_JSON_STRING || x->u.str.len > 128) {
      *refusal = nd_refusal("", "invalid_request", "request_id must be a string of <= 128 bytes");
      goto out;
    }
    id = dup_n(x->u.str.ptr, x->u.str.len);
  } else {
    id = dup_n("", 0);
  }
  if (!id) goto out;
  x = nd_json_get_cstr(v, "health");
  if (x && x->type == ND_JSON_BOOL && x->u.boolean) {
    kind = ND_LINE_HEALTH;
    goto out;
  }
  /* m.keys(): the map's keys, sorted, duplicates once. */
  if (nd_sorted_members(v, &keys, &nk) != ND_OK) goto out;
  for (i = 0; i < nk; i++) {
    const nd_json_member *m = keys[i];
    static const char *known[4] = {"request_id", "query", "tools", "max_new_tokens"};
    int k, ok = 0;
    for (k = 0; k < 4; k++) ok |= m->key_len == strlen(known[k]) && !memcmp(m->key, known[k], m->key_len);
    if (!ok) {
      nd_json_writer w;
      nd_json_writer_init(&w, 0);
      nd_json_writer_raw(&w, "unknown key ", 12);
      nd_rust_debug_str(&w, m->key, m->key_len);
      if (!w.err) BAD(w.data);
      nd_json_writer_free(&w);
      goto out;
    }
  }
  x = nd_json_get_cstr(v, "query");
  if (!x || x->type != ND_JSON_STRING) {
    BAD("query must be a string");
    goto out;
  }
  if (!(req->query = dup_n(x->u.str.ptr, x->u.str.len))) goto out;
  req->query_len = x->u.str.len;
  x = nd_json_get_cstr(v, "tools");
  if (x && x->type != ND_JSON_NULL) {
    if (x->type != ND_JSON_ARRAY) {
      BAD("tools must be an array of catalogue names");
      goto out;
    }
    for (i = 0; i < x->u.arr.len; i++)
      if (x->u.arr.items[i].type != ND_JSON_STRING) {
        BAD("tools must be an array of catalogue names");
        goto out;
      }
    req->has_tools = 1;
    req->tools = nd_calloc(x->u.arr.len, sizeof *req->tools);
    if (!req->tools) goto out;
    for (i = 0; i < x->u.arr.len; i++) {
      const nd_json_value *t = &x->u.arr.items[i];
      if (!(req->tools[i].ptr = dup_n(t->u.str.ptr, t->u.str.len))) goto out;
      req->tools[i].len = t->u.str.len;
      req->n_tools++;
    }
  }
  x = nd_json_get_cstr(v, "max_new_tokens");
  if (x) {
    uint64_t n;
    if (!nd_json_as_u64(x, &n) || n > (uint64_t)SIZE_MAX) {
      BAD("max_new_tokens must be a non-negative integer");
      goto out;
    }
    req->has_max_new = 1;
    req->max_new = (size_t)n;
  }
  req->id = id;
  id = NULL;
  kind = ND_LINE_REQUEST;
out:
#undef BAD
  if (kind == ND_LINE_ERROR) {
    nd_request_free(req);
    if (!*refusal) *refusal = nd_refusal("", "invalid_request", "out of memory");
  }
  free(id);
  free(keys);
  nd_json_doc_free(doc);
  return kind;
}

/* ---- classify ---- */

void nd_verdict_free(nd_verdict *v) {
  free(v->detail);
  free(v->calls);
  free(v->rejected);
  nd_strlist_free(&v->ungrounded);
  memset(v, 0, sizeof *v);
  v->grounded = -1;
}

static int set_detail(nd_verdict *v, const char *status, const char *d, size_t n) {
  v->status = status;
  free(v->detail);
  v->detail = dup_n(d, n);
  v->detail_len = n;
  return v->detail ? ND_OK : ND_E_NOMEM;
}

static int set_cstr(nd_verdict *v, const char *status, const char *d) { return set_detail(v, status, d, strlen(d)); }

/* ", ".join(list) */
static char *join(const nd_strlist *l, size_t *len) {
  size_t n = 0, i, o = 0;
  char *s;
  for (i = 0; i < l->n; i++) n += l->lens[i] + (i ? 2 : 0);
  if (!(s = malloc(n + 1))) return NULL;
  for (i = 0; i < l->n; i++) {
    if (i) memcpy(s + o, ", ", 2), o += 2;
    memcpy(s + o, l->items[i], l->lens[i]);
    o += l->lens[i];
  }
  s[o] = 0;
  *len = o;
  return s;
}

int nd_classify(const nd_catalogue *cat, const nd_options *opts, const char *query, size_t query_len,
                const nd_strv *allowed, size_t n_allowed, int has_allowed, const char *text, size_t text_len,
                nd_stop stop, int prompt_truncated, size_t budget, nd_verdict *v) {
  nd_outcome oc;
  int rc = ND_OK;
  memset(v, 0, sizeof *v);
  v->status = "candidate";
  v->grounded = -1;
  v->detail = dup_n("", 0);
  if (!v->detail) return ND_E_NOMEM;
  if (prompt_truncated) return set_cstr(v, "truncated", "the engine truncated the prompt");
  switch (stop) {
  case ND_STOP_CANCELLED: return set_cstr(v, "timeout", "deadline reached during generation");
  case ND_STOP_MAX_TOKENS:
  case ND_STOP_MAX_SEQ_LEN: {
    char *d = xprintf("generation budget of %zu tokens ran out", budget);
    if (!d) return ND_E_NOMEM;
    rc = set_cstr(v, "incomplete", d);
    free(d);
    return rc;
  }
  case ND_STOP_EOS:
  case ND_STOP_IM_END: break;
  }
  if ((rc = nd_validate(text, text_len, cat, has_allowed ? allowed : NULL, has_allowed ? n_allowed : 0, &oc)) !=
      ND_OK)
    return rc;
  switch (oc.kind) {
  case ND_OUTCOME_CALLS: {
    nd_json_writer w;
    size_t i;
    if (opts->grounding != ND_GROUND_OFF) {
      nd_mentions m;
      if ((rc = nd_mentions_read(query, query_len, &m)) != ND_OK) break;
      for (i = 0; i < oc.n_calls && rc == ND_OK; i++) {
        const nd_tool *t = nd_catalogue_get(cat, oc.calls[i].name, oc.calls[i].name_len);
        if (t) rc = nd_ungrounded(&oc.calls[i], t, &m, opts->grounding == ND_GROUND_STRICT, &v->ungrounded);
      }
      nd_mentions_free(&m);
      if (rc != ND_OK) break;
      v->grounded = v->ungrounded.n == 0;
    }
    nd_json_writer_init(&w, 0);
    nd_calls_write(&w, oc.calls, oc.n_calls);
    if (w.err) {
      nd_json_writer_free(&w);
      rc = ND_E_NOMEM;
      break;
    }
    if ((opts->grounding == ND_GROUND_ENFORCE || opts->grounding == ND_GROUND_STRICT) && v->grounded == 0) {
      size_t jl;
      char *j = join(&v->ungrounded, &jl), *d;
      d = j ? xprintf("not stated in the request: %s", j) : NULL;
      free(j);
      if (!d) {
        nd_json_writer_free(&w);
        rc = ND_E_NOMEM;
        break;
      }
      v->status = "needs_clarification";
      free(v->detail);
      v->detail = d;
      v->detail_len = strlen(d);
      v->rejected = dup_n(w.data, w.len);
    } else {
      v->calls = dup_n(w.data, w.len);
    }
    nd_json_writer_free(&w);
    if (!v->calls && !v->rejected) rc = ND_E_NOMEM;
    break;
  }
  case ND_OUTCOME_NO_CALL:
    rc = set_cstr(v, "no_call", "the model chose no tool");
    if (rc == ND_OK && !(v->calls = dup_n("[]", 2))) rc = ND_E_NOMEM;
    break;
  case ND_OUTCOME_NO_MARKER: rc = set_cstr(v, "invalid_output", "no <tool_call> in the completion"); break;
  case ND_OUTCOME_UNTERMINATED: rc = set_cstr(v, "invalid_output", "unterminated <tool_call>"); break;
  case ND_OUTCOME_MALFORMED:
  case ND_OUTCOME_INVALID: rc = set_detail(v, "invalid_output", oc.detail, oc.detail_len); break;
  case ND_OUTCOME_UNSUPPORTED: rc = set_detail(v, "unsupported", oc.detail, oc.detail_len); break;
  case ND_OUTCOME_NEEDS_CLARIFICATION: rc = set_detail(v, "needs_clarification", oc.detail, oc.detail_len); break;
  }
  nd_outcome_free(&oc);
  return rc;
}

int nd_gate_confidence(nd_verdict *v, int has_min, float min, int has_p, float p) {
  if (has_min && has_p && !strcmp(v->status, "candidate") && p < min) {
    char m[64], *d;
    nd_fmt_f32_display(min, m, sizeof m);
    /* {p:.3}: exact decimal rounding, as printf. */
    if (!(d = xprintf("confidence %.3f below %s", (double)p, m))) return ND_E_NOMEM;
    v->status = "low_confidence";
    free(v->detail);
    v->detail = d;
    v->detail_len = strlen(d);
    free(v->rejected);
    v->rejected = v->calls;
    v->calls = NULL;
  }
  return ND_OK;
}

/* ---- model hash ---- */

/* Rust's io::Error Display for an OS error. */
static char *os_error(const char *path, int err) { return xprintf("%s: %s (os error %d)", path, strerror(err), err); }

static int is_hex(const char *s, size_t n) {
  size_t i;
  for (i = 0; i < n; i++)
    if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f') || (s[i] >= 'A' && s[i] <= 'F'))) return 0;
  return 1;
}

static void model_hash(const char *path, const uint8_t *bytes, size_t len, int verify, char out[65],
                       const char **source) {
  struct stat st;
  char stamp[64], *side = xprintf("%s.sha256", path);
  int has_stamp = 0;
  if (stat(path, &st) == 0 && st.st_mtim.tv_sec >= 0) {
    /* "{len} {mtime as_nanos}" */
    unsigned long long ns = (unsigned long long)st.st_mtim.tv_sec * 1000000000ull + (unsigned long long)st.st_mtim.tv_nsec;
    snprintf(stamp, sizeof stamp, "%llu %llu", (unsigned long long)st.st_size, ns);
    has_stamp = 1;
  }
  if (!verify && has_stamp && side) {
    FILE *f = fopen(side, "rb");
    if (f) {
      char buf[512];
      size_t n = fread(buf, 1, sizeof buf - 1, f), a, b;
      int whole = feof(f);
      fclose(f);
      buf[n] = 0;
      if (whole && nd_json_utf8_valid((const unsigned char *)buf, n)) {
        const char *sp;
        nd_rust_trim(buf, n, &a, &b);
        buf[b] = 0;
        sp = memchr(buf + a, ' ', b - a);
        if (sp && (size_t)(sp - (buf + a)) == 64 && !strcmp(sp + 1, stamp) && is_hex(buf + a, 64)) {
          size_t i;
          for (i = 0; i < 64; i++) {
            char c = buf[a + i];
            out[i] = (char)(c >= 'A' && c <= 'F' ? c - 'A' + 'a' : c);
          }
          out[64] = 0;
          *source = "cache";
          free(side);
          return;
        }
      }
    }
  }
  nd_sha256_hex(bytes, len, out);
  if (has_stamp && side) {
    /* Best effort: a read-only model directory just means hashing again next time. */
    FILE *f = fopen(side, "wb");
    if (f) {
      fprintf(f, "%s %s\n", out, stamp);
      fclose(f);
    }
  }
  free(side);
  *source = "computed";
}

static int ieq(const char *a, const char *b) {
  for (; *a && *b; a++, b++) {
    char x = *a, y = *b;
    if (x >= 'A' && x <= 'Z') x = (char)(x - 'A' + 'a');
    if (y >= 'A' && y <= 'Z') y = (char)(y - 'A' + 'a');
    if (x != y) return 0;
  }
  return !*a && !*b;
}

nd_service *nd_service_load(const char *path, nd_catalogue *cat, int has_depth, size_t depth, const nd_limits *l,
                            const nd_options *o, char **err) {
  double t = nd_now();
  nd_service *s = calloc(1, sizeof *s);
  FILE *f;
  uint8_t *bytes = NULL;
  long len = 0;
  nd_err e = {""};
  *err = NULL;
  if (!s) {
    nd_catalogue_free(cat);
    *err = xstrdup("out of memory");
    return NULL;
  }
  s->cat = cat;
  s->limits = *l;
  s->opts = *o;
  f = fopen(path, "rb");
  if (!f) {
    *err = os_error(path, errno);
    goto fail;
  }
  if (fseek(f, 0, SEEK_END) != 0 || (len = ftell(f)) < 0 || fseek(f, 0, SEEK_SET) != 0 ||
      !(bytes = malloc(len ? (size_t)len : 1)) || fread(bytes, 1, (size_t)len, f) != (size_t)len) {
    *err = os_error(path, errno ? errno : EIO);
    fclose(f);
    goto fail;
  }
  fclose(f);
  model_hash(path, bytes, (size_t)len, o->verify_model || o->expect_sha256, s->sha256, &s->sha256_source);
  if (o->expect_sha256 && !ieq(o->expect_sha256, s->sha256)) {
    *err = xprintf("%s: sha256 %s, expected %s", path, s->sha256, o->expect_sha256);
    goto fail;
  }
  s->model_bytes = (size_t)len;
  if (has_depth && depth == 0) {
    /* depth 0 means "the container's own" to nd_engine; Rust refuses a rung of 0. */
    *err = xprintf("%s: depth 0 is outside the ladder", path);
    goto fail;
  }
  if (nd_engine_from_bytes(bytes, (size_t)len, has_depth ? depth : 0, &s->engine, &e) != ND_OK) {
    bytes = NULL; /* taken over and freed */
    *err = xprintf("%s: %s", path, e.msg);
    goto fail;
  }
  bytes = NULL;
  if (!o->no_prefix_cache) nd_engine_enable_prefix_cache(s->engine);
  s->depth = s->engine->model->cfg.num_layers;
  if (l->max_total_tokens > s->engine->model->cfg.max_seq_len) {
    *err = xprintf("max_total_tokens %zu exceeds the model's context %zu", l->max_total_tokens,
                   s->engine->model->cfg.max_seq_len);
    goto fail;
  }
  if (l->min_new_tokens > l->max_new_tokens) {
    *err = xstrdup("min_new_tokens exceeds max_new_tokens");
    goto fail;
  }
  s->load_ms = nd_ms(nd_now() - t);
  return s;
fail:
  free(bytes);
  if (!*err) *err = xstrdup("out of memory");
  nd_service_free(s);
  return NULL;
}

void nd_service_free(nd_service *s) {
  if (!s) return;
  nd_engine_free(s->engine);
  nd_catalogue_free(s->cat);
  free(s);
}

const char *nd_service_kv_name(const nd_service *s) { return s->opts.kv_int8 ? "int8" : "f32"; }

void nd_service_health(const nd_service *s, nd_obj *o) {
  const nd_cfg *c = &s->engine->model->cfg;
  nd_obj lim;
  nd_json_writer w;
  size_t i;
  nd_obj_cstr(o, "status", "ok");
  nd_obj_cstr(o, "model_sha256", s->sha256);
  nd_obj_cstr(o, "model_sha256_source", s->sha256_source);
  nd_obj_u64(o, "model_bytes", s->model_bytes);
  nd_obj_f64(o, "load_ms", s->load_ms);
  nd_obj_u64(o, "depth", s->depth);
  nd_obj_u64(o, "d_model", c->d_model);
  nd_obj_u64(o, "vocab", c->vocab_size);
  nd_obj_u64(o, "max_seq_len", c->max_seq_len);
  nd_obj_cstr(o, "kv_precision", nd_service_kv_name(s));
  nd_obj_bool(o, "constrained", s->opts.constrain);
  nd_obj_bool(o, "prefix_cache", s->engine->prefix_on);
  nd_obj_cstr(o, "grounding", nd_grounding_name(s->opts.grounding));
  if (s->opts.has_min_confidence)
    nd_obj_f64(o, "min_confidence", (double)s->opts.min_confidence);
  else
    nd_obj_null(o, "min_confidence");
  nd_obj_bool(o, "confidence_head", s->engine->model->confidence != NULL);
  nd_json_writer_init(&w, 0);
  nd_json_writer_begin_array(&w);
  for (i = 0; i < s->cat->n_tools; i++) nd_json_writer_string(&w, s->cat->tools[i].name, s->cat->tools[i].name_len);
  nd_json_writer_end_array(&w);
  if (w.err)
    o->err = 1;
  else
    nd_obj_raw(o, "tools", w.data, w.len);
  nd_json_writer_free(&w);
  nd_obj_init(&lim);
  nd_obj_u64(&lim, "max_total_tokens", s->limits.max_total_tokens);
  nd_obj_u64(&lim, "max_new_tokens", s->limits.max_new_tokens);
  nd_obj_u64(&lim, "min_new_tokens", s->limits.min_new_tokens);
  nd_obj_u64(&lim, "max_query_bytes", s->limits.max_query_bytes);
  if (s->limits.has_deadline)
    nd_obj_f64(&lim, "deadline_ms", ms_of_millis(s->limits.deadline_ms));
  else
    nd_obj_null(&lim, "deadline_ms");
  nd_obj_child(o, "limits", &lim);
  nd_obj_bool(o, "parallel", 0);
}

static void meta(const nd_service *s, nd_obj *o) {
  nd_obj_cstr(o, "model_sha256", s->sha256);
  nd_obj_u64(o, "depth", s->depth);
  nd_obj_cstr(o, "kv_precision", nd_service_kv_name(s));
  nd_obj_bool(o, "constrained", s->opts.constrain);
}

typedef struct {
  double started, first_token;
  int has_first;
  int has_deadline;
  double deadline;
} gen_ctx;

static void on_token(void *u, uint32_t tok, const char *d, size_t n) {
  gen_ctx *g = u;
  (void)tok, (void)d, (void)n;
  if (!g->has_first) {
    g->first_token = nd_now() - g->started;
    g->has_first = 1;
  }
}

static int keep_going(void *u) {
  gen_ctx *g = u;
  return !g->has_deadline || nd_now() < g->deadline;
}

void nd_service_handle(nd_service *s, const nd_request *r, double received, nd_obj *out) {
  double started = nd_now(), queue = started - received;
  const char *id = r->id ? r->id : "";
  char *tools_json = NULL, *msg = NULL;
  size_t tl = 0, want_new, prompt_tokens, room, budget, need;
  uint32_t *ids = NULL;
  nd_gen_opts go;
  nd_result res;
  nd_verdict v;
  gen_ctx g;
  int has_conf = 0, ran_conf = 0, rc;
  float conf = 0.0f;
  double conf_ms = 0.0;
  memset(&res, 0, sizeof res);
  memset(&v, 0, sizeof v);
  v.grounded = -1;

  if (s->limits.has_deadline && queue >= (double)s->limits.deadline_ms / 1e3) {
    nd_obj t;
    nd_obj_cstr(out, "request_id", id);
    nd_obj_cstr(out, "status", "timeout");
    nd_obj_cstr(out, "detail", "deadline passed while queued");
    nd_obj_init(&t);
    nd_obj_f64(&t, "queue_ms", nd_ms(queue));
    nd_obj_child(out, "timing", &t);
    meta(s, out);
    return;
  }
  if (r->query_len == 0 || r->query_len > s->limits.max_query_bytes) {
    char *d = xprintf("query must be 1..=%zu bytes", s->limits.max_query_bytes);
    nd_refusal_obj(out, id, "invalid_request", d ? d : "", d ? strlen(d) : 0);
    free(d);
    meta(s, out);
    return;
  }
  if (memchr(r->query, 0, r->query_len)) {
    nd_refusal_obj(out, id, "invalid_request", "query contains NUL", 18);
    meta(s, out);
    return;
  }
  if (nd_catalogue_tools_json(s->cat, r->has_tools ? r->tools : NULL, r->has_tools ? r->n_tools : 0,
                              &tools_json, &tl, &msg) != ND_OK) {
    nd_refusal_obj(out, id, "invalid_request", msg ? msg : "out of memory", msg ? strlen(msg) : 13);
    free(msg);
    meta(s, out);
    return;
  }
  want_new = r->has_max_new ? r->max_new : s->limits.max_new_tokens;
  if (want_new == 0 || want_new > s->limits.max_new_tokens) {
    char *d = xprintf("max_new_tokens must be 1..=%zu", s->limits.max_new_tokens);
    nd_refusal_obj(out, id, "invalid_request", d ? d : "", d ? strlen(d) : 0);
    free(d);
    goto done_meta;
  }

  /* Admission on the real prompt length, before any model compute. */
  if (nd_engine_prompt_ids(s->engine, r->query, tools_json, s->opts.system, &ids, &prompt_tokens) != ND_OK) {
    out->err = 1;
    goto done_meta;
  }
  free(ids);
  ids = NULL;
  room = s->limits.max_total_tokens > prompt_tokens ? s->limits.max_total_tokens - prompt_tokens : 0;
  need = s->limits.min_new_tokens < want_new ? s->limits.min_new_tokens : want_new;
  if (room < need) {
    nd_obj t;
    char *d = xprintf("prompt is %zu tokens; %zu allowed with at least %zu to generate", prompt_tokens,
                      s->limits.max_total_tokens, need);
    nd_obj_cstr(out, "request_id", id);
    nd_obj_cstr(out, "status", "truncated");
    nd_obj_cstr(out, "detail", d ? d : "");
    free(d);
    nd_obj_bool(out, "prompt_truncated", 0);
    nd_obj_init(&t);
    nd_obj_u64(&t, "prompt", prompt_tokens);
    nd_obj_child(out, "tokens", &t);
    goto done_meta;
  }
  budget = want_new < room ? want_new : room;

  nd_gen_opts_default(&go);
  go.max_new_tokens = budget;
  go.temperature = 0.0f;
  go.seed = 0;
  go.system = s->opts.system;
  go.constrain = s->opts.constrain;
  go.kv = s->opts.kv_int8 ? ND_KV_INT8 : ND_KV_F32;
  memset(&g, 0, sizeof g);
  g.started = started;
  g.has_deadline = s->limits.has_deadline;
  g.deadline = received + (double)s->limits.deadline_ms / 1e3;
  if (nd_generate(s->engine, r->query, tools_json, &go, on_token, keep_going, &g, &res) != ND_OK) {
    out->err = 1;
    goto done_meta;
  }

  rc = nd_classify(s->cat, &s->opts, r->query, r->query_len, r->tools, r->n_tools, r->has_tools, res.text,
                   res.text_len, res.stop, res.prompt_truncated, budget, &v);
  if (rc != ND_OK) {
    out->err = 1;
    goto done_meta;
  }
  if (s->opts.confidence && !strcmp(v.status, "candidate")) {
    double t = nd_now();
    int c = nd_confidence_for(s->engine, r->query, tools_json, res.text, &conf);
    if (c < 0) {
      out->err = 1;
      goto done_meta;
    }
    has_conf = c == 1;
    ran_conf = 1;
    conf_ms = nd_ms(nd_now() - t);
  }
  if (nd_gate_confidence(&v, s->opts.has_min_confidence, s->opts.min_confidence, has_conf, conf) != ND_OK) {
    out->err = 1;
    goto done_meta;
  }

  {
    nd_obj tok, tim;
    int schema_valid = !strcmp(v.status, "candidate") || !strcmp(v.status, "no_call") || v.grounded >= 0;
    nd_obj_cstr(out, "request_id", id);
    nd_obj_cstr(out, "status", v.status);
    nd_obj_bool(out, "prompt_truncated", res.prompt_truncated);
    nd_obj_cstr(out, "stop_reason", nd_stop_wire_name(res.stop));
    nd_obj_bool(out, "schema_valid", schema_valid);
    if (v.grounded >= 0)
      nd_obj_bool(out, "grounded", v.grounded);
    else
      nd_obj_null(out, "grounded");
    /* An f32 in a serde_json Value is widened to f64. */
    if (has_conf)
      nd_obj_f64(out, "confidence_raw", (double)conf);
    else
      nd_obj_null(out, "confidence_raw");
    nd_obj_init(&tok);
    nd_obj_u64(&tok, "prompt", res.prompt_tokens);
    nd_obj_u64(&tok, "generated", res.ntokens);
    nd_obj_u64(&tok, "budget", budget);
    nd_obj_u64(&tok, "positions", res.positions);
    nd_obj_u64(&tok, "prefix_reused", res.prefix_reused);
    nd_obj_child(out, "tokens", &tok);
    nd_obj_init(&tim);
    nd_obj_f64(&tim, "queue_ms", nd_ms(queue));
    nd_obj_f64(&tim, "wall_ms", nd_ms(nd_now() - started));
    nd_obj_f64(&tim, "tokenize_ms", nd_ms(res.t_tokenize));
    nd_obj_f64(&tim, "prefill_ms", nd_ms(res.t_prefill));
    nd_obj_f64(&tim, "decode_ms", nd_ms(res.t_decode));
    if (g.has_first)
      nd_obj_f64(&tim, "first_token_ms", nd_ms(g.first_token));
    else
      nd_obj_null(&tim, "first_token_ms");
    if (ran_conf)
      nd_obj_f64(&tim, "confidence_ms", conf_ms);
    else
      nd_obj_null(&tim, "confidence_ms");
    nd_obj_child(out, "timing", &tim);
    if (v.detail_len) nd_obj_str(out, "detail", v.detail, v.detail_len);
    if (v.calls) nd_obj_raw(out, "calls", v.calls, strlen(v.calls));
    if (v.rejected) nd_obj_raw(out, "rejected_calls", v.rejected, strlen(v.rejected));
    if (v.ungrounded.n) {
      nd_json_writer w;
      size_t i;
      nd_json_writer_init(&w, 0);
      nd_json_writer_begin_array(&w);
      for (i = 0; i < v.ungrounded.n; i++) nd_json_writer_string(&w, v.ungrounded.items[i], v.ungrounded.lens[i]);
      nd_json_writer_end_array(&w);
      if (w.err)
        out->err = 1;
      else
        nd_obj_raw(out, "ungrounded", w.data, w.len);
      nd_json_writer_free(&w);
    }
    if (s->opts.debug_text) nd_obj_str(out, "text", res.text, res.text_len);
  }
done_meta:
  meta(s, out);
  nd_verdict_free(&v);
  nd_result_free(&res);
  free(tools_json);
}
