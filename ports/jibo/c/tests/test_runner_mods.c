/* test_runner_mods.c: replays a reference corpus produced by the Rust runner modules
 * (catalog.rs, validate.rs, grounding.rs; generator: a throwaway program outside the repo, see
 * the port notes) against nd_catalog / nd_validate / nd_grounding, and reports every mismatch.
 *
 *   test_runner_mods REF.jsonl
 *
 * One JSON object per line, by "t":
 *   str  {s, dbg, lc, trim:[a,b], trim_end}   format!("{:?}"), to_lowercase, trim, trim_end
 *   f64  {s, v: hexbits|null}                 str::parse::<f64>
 *   sj   {text, out | err}                    serde_json::from_str::<Value>, to_string / error
 *   cat  {text, ok: dump | err}               Catalogue::parse (each ok one is appended to the list
 *                                             that later records index with "cat")
 *   tj   {cat, names|null, ok | err}          Catalogue::tools_json
 *   val  {cat, allowed|null, text, sp, kind, detail, calls}   strict_payload + validate
 *   men  {q, numbers, durations, clock, zero, max, min, lower} Mentions::read
 *   ug   {cat, tool, args, q, strict, out}    ungrounded
 * Exit status 0 only with zero mismatches. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nd_catalog.h"
#include "nd_grounding.h"
#include "nd_json.h"
#include "nd_validate.h"

static unsigned long n_checked, n_bad;
static unsigned long per_ok[16], per_bad[16];
static const char *const TYPES[] = {"str", "f64", "sj", "cat", "tj", "val", "men", "ug", NULL};

static void show(const char *s, size_t n) {
  size_t k;
  for (k = 0; k < n && k < 400; k++) {
    unsigned char c = (unsigned char)s[k];
    if (c < 0x20 || c == 0x7f)
      printf("\\x%02x", c);
    else
      putchar(c);
  }
  if (n > 400) printf("...(%lu bytes)", (unsigned long)n);
}

static int mismatch(unsigned long line, const char *what, const char *want, size_t wl,
                    const char *got, size_t gl) {
  n_bad++;
  if (n_bad <= 40) {
    printf("line %lu: %s\n  want: ", line, what);
    if (want) show(want, wl); else printf("(none)");
    printf("\n  got:  ");
    if (got) show(got, gl); else printf("(none)");
    printf("\n");
  }
  return 1;
}

static const nd_json_value *get(const nd_json_value *o, const char *k) {
  return nd_json_get_cstr(o, k);
}
static int is_null(const nd_json_value *v) { return !v || v->type == ND_JSON_NULL; }
static int str_eq(const nd_json_value *v, const char *s, size_t n) {
  return v && v->type == ND_JSON_STRING && v->u.str.len == n &&
         (n == 0 || memcmp(v->u.str.ptr, s, n) == 0);
}
static int check_str(unsigned long line, const char *what, const nd_json_value *want,
                     const char *got, size_t gl) {
  if (str_eq(want, got, gl)) return 0;
  return mismatch(line, what, want && want->type == ND_JSON_STRING ? want->u.str.ptr : NULL,
                  want && want->type == ND_JSON_STRING ? want->u.str.len : 0, got, gl);
}
static uint64_t hexbits(const nd_json_value *v) {
  return v && v->type == ND_JSON_STRING ? strtoull(v->u.str.ptr, NULL, 16) : 0;
}
static int same_double(uint64_t want_bits, double got) {
  double w;
  uint64_t g;
  memcpy(&w, &want_bits, sizeof w);
  memcpy(&g, &got, sizeof g);
  if (isnan(w) && isnan(got)) return 1;
  return g == want_bits;
}
static size_t vlen(const nd_json_value *v) {
  return v && v->type == ND_JSON_ARRAY ? v->u.arr.len : 0;
}
static const nd_json_value *at(const nd_json_value *v, size_t i) { return nd_json_at(v, i); }

/* ─── per record ───────────────────────────────────────────────────────────────────────────── */

static int t_str(unsigned long line, const nd_json_value *r) {
  const nd_json_value *s = get(r, "s");
  nd_json_writer w;
  char *lc = NULL;
  size_t ll = 0, a, b, e;
  int bad = 0;
  nd_json_writer_init(&w, 0);
  nd_rust_debug_str(&w, s->u.str.ptr, s->u.str.len);
  bad |= check_str(line, "debug", get(r, "dbg"), w.data, w.len);
  nd_json_writer_free(&w);
  if (nd_rust_to_lowercase(s->u.str.ptr, s->u.str.len, &lc, &ll)) return mismatch(line, "lc nomem", 0, 0, 0, 0);
  bad |= check_str(line, "to_lowercase", get(r, "lc"), lc, ll);
  free(lc);
  nd_rust_trim(s->u.str.ptr, s->u.str.len, &a, &b);
  nd_rust_trim_end(s->u.str.ptr, s->u.str.len, &e);
  {
    int64_t wa = 0, wb = 0, we = 0;
    nd_json_as_i64(at(get(r, "trim"), 0), &wa);
    nd_json_as_i64(at(get(r, "trim"), 1), &wb);
    nd_json_as_i64(get(r, "trim_end"), &we);
    if ((size_t)wa != a || (size_t)wb != b || (size_t)we != e) {
      char g[64];
      int n = snprintf(g, sizeof g, "%lu %lu %lu", (unsigned long)a, (unsigned long)b, (unsigned long)e);
      bad |= mismatch(line, "trim", s->u.str.ptr, s->u.str.len, g, (size_t)n);
    }
  }
  return bad;
}

static int t_f64(unsigned long line, const nd_json_value *r) {
  const nd_json_value *s = get(r, "s"), *v = get(r, "v");
  double d = 0.0;
  int ok = nd_rust_parse_f64(s->u.str.ptr, s->u.str.len, &d);
  if (is_null(v) ? !ok : (ok && same_double(hexbits(v), d))) return 0;
  {
    char g[64];
    int n = snprintf(g, sizeof g, ok ? "%.17g" : "error", d);
    return mismatch(line, "parse f64", s->u.str.ptr, s->u.str.len, g, (size_t)n);
  }
}

static int t_sj(unsigned long line, const nd_json_value *r) {
  const nd_json_value *text = get(r, "text");
  nd_json_doc *doc = NULL;
  char *err = NULL;
  int rc = nd_serde_parse(text->u.str.ptr, text->u.str.len, &doc, &err), bad = 0;
  if (rc == ND_OK) {
    char *s = NULL;
    size_t sl = 0;
    if (!get(r, "out")) {
      bad = mismatch(line, "serde: accepted what serde_json rejects", get(r, "err")->u.str.ptr,
                     get(r, "err")->u.str.len, text->u.str.ptr, text->u.str.len);
    } else if (nd_json_to_string(nd_json_root(doc), ND_JSON_KEYS_SORTED, &s, &sl) == ND_OK) {
      bad = check_str(line, "serde value", get(r, "out"), s, sl);
    } else {
      bad = mismatch(line, "to_string failed", 0, 0, 0, 0);
    }
    free(s);
    nd_json_doc_free(doc);
  } else {
    if (!get(r, "err"))
      bad = mismatch(line, "serde: rejected what serde_json accepts", text->u.str.ptr,
                     text->u.str.len, err, err ? strlen(err) : 0);
    else
      bad = check_str(line, "serde error", get(r, "err"), err, err ? strlen(err) : 0);
    free(err);
  }
  return bad;
}

static int cmp_dump(unsigned long line, const nd_catalogue *c, const nd_json_value *d) {
  size_t k, j;
  int bad = 0;
  if (vlen(d) != c->n_tools) return mismatch(line, "tool count", 0, 0, 0, 0);
  for (k = 0; k < c->n_tools; k++) {
    const nd_tool *t = &c->tools[k];
    const nd_json_value *td = at(d, k), *ps = get(td, "params"), *rq = get(td, "required");
    bad |= check_str(line, "tool name", get(td, "name"), t->name, t->name_len);
    bad |= check_str(line, "snake", get(td, "snake"), t->snake_name, t->snake_len);
    bad |= check_str(line, "json span", get(td, "json"), t->json, t->json_len);
    if (vlen(rq) != t->n_required) {
      bad |= mismatch(line, "required count", 0, 0, 0, 0);
    } else {
      for (j = 0; j < t->n_required; j++)
        bad |= check_str(line, "required", at(rq, j), t->required[j], t->required_lens[j]);
    }
    if (vlen(ps) != t->n_params) {
      bad |= mismatch(line, "param count", 0, 0, 0, 0);
      continue;
    }
    for (j = 0; j < t->n_params; j++) {
      const nd_param *p = &t->params[j];
      const nd_json_value *pd = at(ps, j), *kind = get(pd, "kind");
      static const char *const kinds[] = {"string", "integer", "number", "boolean"};
      bad |= check_str(line, "param name", get(pd, "name"), p->name, p->name_len);
      bad |= check_str(line, "param kind", kind, kinds[p->kind], strlen(kinds[p->kind]));
      if (p->kind == ND_PARAM_STRING) {
        const nd_json_value *e = get(pd, "enum"), *ml = get(pd, "max_length");
        if (is_null(e) != !p->has_enum || (p->has_enum && vlen(e) != p->n_enum)) {
          bad |= mismatch(line, "enum presence/count", 0, 0, 0, 0);
        } else {
          size_t q;
          for (q = 0; q < p->n_enum; q++)
            bad |= check_str(line, "enum value", at(e, q), p->enum_values[q], p->enum_lens[q]);
        }
        if (is_null(ml) != !p->has_max_length) {
          bad |= mismatch(line, "maxLength presence", 0, 0, 0, 0);
        } else if (p->has_max_length) {
          uint64_t u = 0;
          if (!nd_json_as_u64(ml, &u) || u != (uint64_t)p->max_length)
            bad |= mismatch(line, "maxLength", 0, 0, 0, 0);
        }
      } else if (p->kind == ND_PARAM_INTEGER) {
        const nd_json_value *mn = get(pd, "imin"), *mx = get(pd, "imax");
        int64_t a = 0, b = 0;
        if (is_null(mn) != !p->has_imin || is_null(mx) != !p->has_imax ||
            (p->has_imin && (!nd_json_as_i64(mn, &a) || a != p->imin)) ||
            (p->has_imax && (!nd_json_as_i64(mx, &b) || b != p->imax)))
          bad |= mismatch(line, "integer bounds", 0, 0, 0, 0);
      } else if (p->kind == ND_PARAM_NUMBER) {
        const nd_json_value *mn = get(pd, "fmin"), *mx = get(pd, "fmax");
        if (is_null(mn) != !p->has_fmin || is_null(mx) != !p->has_fmax ||
            (p->has_fmin && !same_double(hexbits(mn), p->fmin)) ||
            (p->has_fmax && !same_double(hexbits(mx), p->fmax)))
          bad |= mismatch(line, "number bounds", 0, 0, 0, 0);
      }
    }
  }
  return bad;
}

typedef struct {
  nd_catalogue **v;
  size_t n, cap;
} catlist;

static int t_cat(unsigned long line, const nd_json_value *r, catlist *cl) {
  const nd_json_value *text = get(r, "text"), *ok = get(r, "ok");
  nd_catalogue *c = NULL;
  char *err = NULL;
  int rc = nd_catalogue_parse(text->u.str.ptr, text->u.str.len, &c, &err), bad = 0;
  if (rc == ND_OK) {
    if (!ok) {
      bad = mismatch(line, "catalogue accepted, Rust refused", get(r, "err")->u.str.ptr,
                     get(r, "err")->u.str.len, "ok", 2);
      nd_catalogue_free(c);
      c = NULL;
    } else {
      bad = cmp_dump(line, c, ok);
    }
  } else {
    if (ok)
      bad = mismatch(line, "catalogue refused, Rust accepted", "ok", 2, err, err ? strlen(err) : 0);
    else
      bad = check_str(line, "catalogue error", get(r, "err"), err, err ? strlen(err) : 0);
  }
  free(err);
  if (ok) {
    /* Keep indices aligned with the reference even when C refused. */
    if (cl->n == cl->cap) {
      size_t nc = cl->cap ? cl->cap * 2 : 64;
      nd_catalogue **q = (nd_catalogue **)realloc(cl->v, nc * sizeof *q);
      if (!q) {
        fprintf(stderr, "out of memory\n");
        exit(2);
      }
      cl->v = q;
      cl->cap = nc;
    }
    cl->v[cl->n++] = c;
  }
  return bad;
}

static nd_strv *names_of(const nd_json_value *a, size_t *n) {
  nd_strv *v;
  size_t k;
  *n = vlen(a);
  v = (nd_strv *)nd_calloc(*n, sizeof *v);
  if (!v) {
    fprintf(stderr, "out of memory\n");
    exit(2);
  }
  for (k = 0; k < *n; k++) {
    v[k].ptr = at(a, k)->u.str.ptr;
    v[k].len = at(a, k)->u.str.len;
  }
  return v;
}

static const nd_catalogue *cat_of(const nd_json_value *r, const catlist *cl) {
  int64_t i = -1;
  nd_json_as_i64(get(r, "cat"), &i);
  if (i < 0 || (size_t)i >= cl->n) return NULL;
  return cl->v[i];
}

static int t_tj(unsigned long line, const nd_json_value *r, const catlist *cl) {
  const nd_catalogue *c = cat_of(r, cl);
  const nd_json_value *names = get(r, "names");
  nd_strv *nv = NULL;
  size_t nn = 0, ol = 0;
  char *out = NULL, *err = NULL;
  int rc, bad;
  if (!c) return mismatch(line, "tools_json: catalogue missing", 0, 0, 0, 0);
  if (!is_null(names)) nv = names_of(names, &nn);
  rc = nd_catalogue_tools_json(c, is_null(names) ? NULL : nv, nn, &out, &ol, &err);
  if (rc == ND_OK)
    bad = get(r, "ok") ? check_str(line, "tools_json", get(r, "ok"), out, ol)
                       : mismatch(line, "tools_json ok, Rust err", get(r, "err")->u.str.ptr,
                                  get(r, "err")->u.str.len, out, ol);
  else
    bad = get(r, "err") ? check_str(line, "tools_json err", get(r, "err"), err, err ? strlen(err) : 0)
                        : mismatch(line, "tools_json err, Rust ok", 0, 0, err, err ? strlen(err) : 0);
  free(out);
  free(err);
  free(nv);
  return bad;
}

static const char *kind_name(nd_outcome_kind k) {
  switch (k) {
    case ND_OUTCOME_CALLS: return "calls";
    case ND_OUTCOME_NO_CALL: return "no_call";
    case ND_OUTCOME_NO_MARKER: return "no_marker";
    case ND_OUTCOME_UNTERMINATED: return "unterminated";
    case ND_OUTCOME_MALFORMED: return "malformed";
    case ND_OUTCOME_UNSUPPORTED: return "unsupported";
    case ND_OUTCOME_NEEDS_CLARIFICATION: return "needs_clarification";
    case ND_OUTCOME_INVALID: return "invalid";
  }
  return "?";
}

static int t_val(unsigned long line, const nd_json_value *r, const catlist *cl) {
  const nd_catalogue *c = cat_of(r, cl);
  const nd_json_value *text = get(r, "text"), *allowed = get(r, "allowed"), *sp = get(r, "sp");
  nd_strv *av = NULL;
  size_t na = 0, s0 = 0, sl = 0;
  nd_outcome o;
  nd_outcome_kind sk;
  int bad = 0, rc;
  if (!c) return mismatch(line, "validate: catalogue missing", 0, 0, 0, 0);
  sk = nd_strict_payload(text->u.str.ptr, text->u.str.len, &s0, &sl);
  if (sp->type == ND_JSON_STRING) {
    bad |= check_str(line, "strict_payload", sp, kind_name(sk), strlen(kind_name(sk)));
  } else {
    int64_t a = -1, b = -1;
    nd_json_as_i64(at(sp, 0), &a);
    nd_json_as_i64(at(sp, 1), &b);
    if (sk != ND_OUTCOME_CALLS || (size_t)a != s0 || (size_t)b != sl)
      bad |= mismatch(line, "strict_payload span", text->u.str.ptr, text->u.str.len,
                      kind_name(sk), strlen(kind_name(sk)));
  }
  if (!is_null(allowed)) av = names_of(allowed, &na);
  rc = nd_validate(text->u.str.ptr, text->u.str.len, c, is_null(allowed) ? NULL : av, na, &o);
  if (rc != ND_OK) {
    bad |= mismatch(line, "validate failed", 0, 0, 0, 0);
  } else {
    const char *kn = kind_name(o.kind);
    bad |= check_str(line, "outcome kind", get(r, "kind"), kn, strlen(kn));
    if (is_null(get(r, "detail")) != (o.detail == NULL))
      bad |= mismatch(line, "detail presence", 0, 0, o.detail, o.detail_len);
    else if (o.detail)
      bad |= check_str(line, "detail", get(r, "detail"), o.detail, o.detail_len);
    if (o.kind == ND_OUTCOME_CALLS) {
      nd_json_writer w;
      nd_json_writer_init(&w, 0);
      nd_calls_write(&w, o.calls, o.n_calls);
      if (w.err)
        bad |= mismatch(line, "calls write failed", 0, 0, 0, 0);
      else
        bad |= check_str(line, "calls", get(r, "calls"), w.data, w.len);
      nd_json_writer_free(&w);
    }
  }
  nd_outcome_free(&o);
  free(av);
  return bad;
}

static int cmp_doubles(unsigned long line, const char *what, const nd_json_value *want,
                       const double *got, size_t n, size_t stride, size_t off) {
  size_t k;
  if (vlen(want) != n) {
    char g[32];
    int l = snprintf(g, sizeof g, "count %lu", (unsigned long)n);
    return mismatch(line, what, "count differs", 13, g, (size_t)l);
  }
  for (k = 0; k < n; k++) {
    const nd_json_value *x = at(want, k);
    if (stride == 2) x = at(x, off);
    if (!same_double(hexbits(x), got[k * stride + off])) {
      char g[48];
      int l = snprintf(g, sizeof g, "[%lu] %.17g", (unsigned long)k, got[k * stride + off]);
      return mismatch(line, what, x->u.str.ptr, x->u.str.len, g, (size_t)l);
    }
  }
  return 0;
}

static int t_men(unsigned long line, const nd_json_value *r) {
  const nd_json_value *q = get(r, "q");
  nd_mentions m;
  int bad = 0, zb, xb, nb;
  if (nd_mentions_read(q->u.str.ptr, q->u.str.len, &m)) return mismatch(line, "mentions failed", 0, 0, 0, 0);
  bad |= cmp_doubles(line, "numbers", get(r, "numbers"), m.numbers, m.n_numbers, 1, 0);
  bad |= cmp_doubles(line, "durations", get(r, "durations"), m.durations_s, m.n_durations, 1, 0);
  bad |= cmp_doubles(line, "clock hour", get(r, "clock"), m.clock, m.n_clock, 2, 0);
  bad |= cmp_doubles(line, "clock minute", get(r, "clock"), m.clock, m.n_clock, 2, 1);
  nd_json_as_bool(get(r, "zero"), &zb);
  nd_json_as_bool(get(r, "max"), &xb);
  nd_json_as_bool(get(r, "min"), &nb);
  if (zb != m.zero_alias || xb != m.max_alias || nb != m.min_alias)
    bad |= mismatch(line, "aliases", q->u.str.ptr, q->u.str.len, "", 0);
  bad |= check_str(line, "lower", get(r, "lower"), m.lower, m.lower_len);
  if (bad) {
    printf("  query: ");
    show(q->u.str.ptr, q->u.str.len);
    printf("\n");
  }
  nd_mentions_free(&m);
  return bad;
}

static int t_ug(unsigned long line, const nd_json_value *r, const catlist *cl) {
  const nd_catalogue *c = cat_of(r, cl);
  const nd_json_value *tn = get(r, "tool"), *args = get(r, "args"), *q = get(r, "q"),
                      *want = get(r, "out");
  const nd_tool *t;
  nd_json_doc *doc = NULL;
  nd_mentions m;
  nd_call call;
  nd_strlist out;
  int strict = 0, bad = 0;
  size_t k;
  if (!c) return mismatch(line, "ungrounded: catalogue missing", 0, 0, 0, 0);
  t = nd_catalogue_get(c, tn->u.str.ptr, tn->u.str.len);
  if (!t) return mismatch(line, "ungrounded: tool missing", 0, 0, 0, 0);
  if (nd_serde_parse(args->u.str.ptr, args->u.str.len, &doc, NULL))
    return mismatch(line, "ungrounded: args do not parse", 0, 0, 0, 0);
  nd_json_as_bool(get(r, "strict"), &strict);
  memset(&out, 0, sizeof out);
  if (nd_mentions_read(q->u.str.ptr, q->u.str.len, &m) ||
      nd_call_from_object(t, nd_json_root(doc), &call) ||
      nd_ungrounded(&call, t, &m, strict, &out)) {
    bad = mismatch(line, "ungrounded failed", 0, 0, 0, 0);
  } else if (vlen(want) != out.n) {
    char g[32];
    int l = snprintf(g, sizeof g, "%lu items", (unsigned long)out.n);
    bad = mismatch(line, "ungrounded count", args->u.str.ptr, args->u.str.len, g, (size_t)l);
    for (k = 0; k < out.n && n_bad <= 40; k++) {
      printf("   got[%lu] ", (unsigned long)k);
      show(out.items[k], out.lens[k]);
      printf("\n");
    }
  } else {
    for (k = 0; k < out.n; k++) bad |= check_str(line, "ungrounded", at(want, k), out.items[k], out.lens[k]);
  }
  if (bad && n_bad <= 40) {
    printf("  query: ");
    show(q->u.str.ptr, q->u.str.len);
    printf("  strict=%d\n", strict);
  }
  nd_call_release(&call);
  nd_strlist_free(&out);
  nd_mentions_free(&m);
  nd_json_doc_free(doc);
  return bad;
}

int main(int argc, char **argv) {
  FILE *f;
  char *buf;
  long sz;
  size_t pos = 0;
  unsigned long line = 0;
  catlist cl = {NULL, 0, 0};
  nd_json_limits lim = nd_json_limits_default();
  size_t k;
  if (argc != 2) {
    fprintf(stderr, "usage: %s REF.jsonl\n", argv[0]);
    return 2;
  }
  f = fopen(argv[1], "rb");
  if (!f || fseek(f, 0, SEEK_END) || (sz = ftell(f)) < 0 || fseek(f, 0, SEEK_SET)) {
    fprintf(stderr, "%s: cannot read\n", argv[1]);
    return 2;
  }
  buf = (char *)malloc((size_t)sz + 1);
  if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
    fprintf(stderr, "%s: cannot read\n", argv[1]);
    return 2;
  }
  fclose(f);
  buf[sz] = '\0';
  lim.max_input = (size_t)1 << 24;
  lim.max_elements = (size_t)1 << 22;
  while (pos < (size_t)sz) {
    char *nl = (char *)memchr(buf + pos, '\n', (size_t)sz - pos);
    size_t len = nl ? (size_t)(nl - (buf + pos)) : (size_t)sz - pos;
    nd_json_doc *doc = NULL;
    nd_err err;
    const nd_json_value *r, *t;
    int bad = 0, ti = -1;
    line++;
    if (len == 0) {
      pos += 1;
      continue;
    }
    if (nd_json_parse(buf + pos, len, &lim, &doc, &err)) {
      fprintf(stderr, "line %lu: reference does not parse: %s\n", line, err.msg);
      return 2;
    }
    pos += len + 1;
    r = nd_json_root(doc);
    t = get(r, "t");
    for (k = 0; TYPES[k]; k++)
      if (str_eq(t, TYPES[k], strlen(TYPES[k]))) ti = (int)k;
    switch (ti) {
      case 0: bad = t_str(line, r); break;
      case 1: bad = t_f64(line, r); break;
      case 2: bad = t_sj(line, r); break;
      case 3: bad = t_cat(line, r, &cl); break;
      case 4: bad = t_tj(line, r, &cl); break;
      case 5: bad = t_val(line, r, &cl); break;
      case 6: bad = t_men(line, r); break;
      case 7: bad = t_ug(line, r, &cl); break;
      default:
        fprintf(stderr, "line %lu: unknown record\n", line);
        return 2;
    }
    n_checked++;
    if (bad)
      per_bad[ti]++;
    else
      per_ok[ti]++;
    nd_json_doc_free(doc);
  }
  for (k = 0; k < cl.n; k++) nd_catalogue_free(cl.v[k]);
  free(cl.v);
  free(buf);
  for (k = 0; TYPES[k]; k++)
    printf("%-4s %7lu ok %5lu mismatched\n", TYPES[k], per_ok[k], per_bad[k]);
  printf("%lu records, %lu mismatches\n", n_checked, n_bad);
  return n_bad ? 1 : 0;
}
