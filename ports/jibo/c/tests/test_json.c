/* test_json.c: tests for nd_json.
 *
 *   test_json                         built-in cases (expected strings taken from serde_json 1.0.149)
 *   test_json --docs CORPUS EXPECTED  every document of a reference corpus: accept/reject, the
 *                                     ordered DOM (number kinds and f64 bits), the sorted
 *                                     (serde_json::Value) and insertion-order serialisations
 *   test_json --floats64 FILE         records (u64 bits, fnv(serde f64 text), fnv(via Value))
 *   test_json --floats32 FILE         records (u32 bits, fnv(serialize_f32 text), fnv(via Value))
 *   test_json --f32all FILE           256 block hashes over all 2^32 f32 bit patterns
 *
 * The reference files come from a throwaway Rust program against serde_json 1.0.149.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nd_json.h"

static long g_fail, g_checks;

#define CHECK(c)                                                         \
  do {                                                                   \
    g_checks++;                                                          \
    if (!(c)) {                                                          \
      g_fail++;                                                          \
      fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #c); \
    }                                                                    \
  } while (0)

static uint64_t fnv(const char *s, size_t n) {
  uint64_t h = 0xcbf29ce484222325u;
  size_t i;
  for (i = 0; i < n; i++) {
    h ^= (unsigned char)s[i];
    h *= 0x100000001b3u;
  }
  return h;
}

static double dbits(uint64_t b) {
  double d;
  memcpy(&d, &b, sizeof d);
  return d;
}
static float fbits(uint32_t b) {
  float f;
  memcpy(&f, &b, sizeof f);
  return f;
}

/* ─── growable string for dumps ─── */
typedef struct {
  char *p;
  size_t n, c;
} sbuf;
static void sb_add(sbuf *s, const char *t, size_t n) {
  if (s->n + n + 1 > s->c) {
    size_t nc = s->c ? s->c : 256;
    while (nc < s->n + n + 1) nc *= 2;
    s->p = (char *)realloc(s->p, nc);
    if (!s->p) {
      fprintf(stderr, "oom\n");
      exit(2);
    }
    s->c = nc;
  }
  memcpy(s->p + s->n, t, n);
  s->n += n;
  s->p[s->n] = 0;
}
static void sb_str(sbuf *s, const char *t) { sb_add(s, t, strlen(t)); }
static void sb_hex(sbuf *s, const char *t, size_t n) {
  static const char hx[] = "0123456789abcdef";
  size_t i;
  for (i = 0; i < n; i++) {
    char b[2];
    b[0] = hx[(unsigned char)t[i] >> 4];
    b[1] = hx[(unsigned char)t[i] & 15];
    sb_add(s, b, 2);
  }
}

static void dump(const nd_json_value *v, sbuf *s) {
  char b[64];
  size_t i;
  switch (v->type) {
    case ND_JSON_NULL: sb_str(s, "n"); break;
    case ND_JSON_BOOL: sb_str(s, v->u.boolean ? "t" : "f"); break;
    case ND_JSON_NUMBER:
      if (v->u.num.kind == ND_JSON_NUM_U64) {
        b[0] = 'U';
        nd_json_write_u64(v->u.num.u, b + 1);
      } else if (v->u.num.kind == ND_JSON_NUM_I64) {
        b[0] = 'I';
        nd_json_write_i64(v->u.num.i, b + 1);
      } else {
        uint64_t bits;
        memcpy(&bits, &v->u.num.f, 8);
        sprintf(b, "F%08lx%08lx", (unsigned long)(bits >> 32), (unsigned long)(bits & 0xffffffffu));
      }
      sb_str(s, b);
      break;
    case ND_JSON_STRING:
      sb_str(s, "s");
      sb_hex(s, v->u.str.ptr, v->u.str.len);
      break;
    case ND_JSON_ARRAY:
      sb_str(s, "[");
      for (i = 0; i < v->u.arr.len; i++) {
        if (i) sb_str(s, ",");
        dump(&v->u.arr.items[i], s);
      }
      sb_str(s, "]");
      break;
    case ND_JSON_OBJECT:
      sb_str(s, "{");
      for (i = 0; i < v->u.obj.len; i++) {
        if (i) sb_str(s, ",");
        sb_hex(s, v->u.obj.members[i].key, v->u.obj.members[i].key_len);
        sb_str(s, ":");
        dump(&v->u.obj.members[i].value, s);
      }
      sb_str(s, "}");
      break;
  }
}

/* Every member: nd_json_get returns the last member with that key; as_* agree with the kind. */
static void check_dom(const nd_json_value *v) {
  size_t i, j;
  if (v->type == ND_JSON_ARRAY) {
    for (i = 0; i < v->u.arr.len; i++) check_dom(&v->u.arr.items[i]);
    CHECK(nd_json_at(v, v->u.arr.len) == NULL);
  } else if (v->type == ND_JSON_OBJECT) {
    for (i = 0; i < v->u.obj.len; i++) {
      const nd_json_member *m = &v->u.obj.members[i];
      const nd_json_value *got = nd_json_get(v, m->key, m->key_len);
      const nd_json_value *want = &m->value;
      for (j = i + 1; j < v->u.obj.len; j++) {
        const nd_json_member *o = &v->u.obj.members[j];
        if (o->key_len == m->key_len && memcmp(o->key, m->key, m->key_len) == 0) want = &o->value;
      }
      if (got != want) {
        g_fail++;
        fprintf(stderr, "nd_json_get did not return the last duplicate\n");
      }
      check_dom(&m->value);
    }
  } else if (v->type == ND_JSON_NUMBER) {
    int64_t i64;
    uint64_t u64;
    double f;
    int hi = nd_json_as_i64(v, &i64), hu = nd_json_as_u64(v, &u64);
    nd_json_as_f64(v, &f);
    if (v->u.num.kind == ND_JSON_NUM_F64 && (hi || hu)) {
      g_fail++;
      fprintf(stderr, "float has integer view\n");
    }
    if (v->u.num.kind == ND_JSON_NUM_U64 && (!hu || hi != (v->u.num.u <= (uint64_t)INT64_MAX))) {
      g_fail++;
      fprintf(stderr, "u64 views wrong\n");
    }
    if (v->u.num.kind == ND_JSON_NUM_I64 && (hu || !hi || f != (double)i64)) {
      g_fail++;
      fprintf(stderr, "i64 views wrong\n");
    }
  }
}

static unsigned char *read_file(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  unsigned char *b;
  long sz;
  if (!f) {
    perror(path);
    exit(2);
  }
  fseek(f, 0, SEEK_END);
  sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  b = (unsigned char *)malloc((size_t)sz + 1);
  if (!b || fread(b, 1, (size_t)sz, f) != (size_t)sz) {
    fprintf(stderr, "read %s failed\n", path);
    exit(2);
  }
  fclose(f);
  b[sz] = 0;
  *n = (size_t)sz;
  return b;
}

static int run_docs(const char *corpus, const char *expected) {
  size_t cn, en, cp = 0, ep = 0, ndocs = 0, nok = 0, nbad = 0, mism = 0, nmsg = 0;
  unsigned char *c = read_file(corpus, &cn), *e = read_file(expected, &en);
  nd_json_limits serde_lim, def = nd_json_limits_default();
  sbuf want = {0, 0, 0}, got = {0, 0, 0};
  serde_lim.max_input = (size_t)1 << 30;
  serde_lim.max_depth = 1000; /* capped to serde_json's 127 */
  serde_lim.max_elements = (size_t)1 << 30;
  while (cp + 4 <= cn) {
    size_t len = (size_t)c[cp] | (size_t)c[cp + 1] << 8 | (size_t)c[cp + 2] << 16 |
                 (size_t)c[cp + 3] << 24;
    const char *doc = (const char *)c + cp + 4;
    char *line = (char *)e + ep, *nl = strchr(line, '\n');
    nd_json_doc *d = NULL, *d2 = NULL;
    nd_err err, err2;
    int rc, rc2;
    cp += 4 + len;
    if (!nl) {
      fprintf(stderr, "expected file too short\n");
      return 1;
    }
    *nl = 0;
    ep = (size_t)(nl - (char *)e) + 1;
    ndocs++;
    rc = nd_json_parse(doc, len, &serde_lim, &d, &err);
    rc2 = nd_json_parse(doc, len, &def, &d2, &err2);
    if (strncmp(line, "ERR ", 4) == 0) {
      nbad++;
      if (rc == ND_OK) {
        mism++;
        if (mism < 20) fprintf(stderr, "doc %lu: accepted, serde rejects: %.*s\n",
                               (unsigned long)ndocs, (int)(len < 200 ? len : 200), doc);
      } else if (strcmp(err.msg, line + 4) != 0) {
        /* the message must be serde_json's Display, byte for byte */
        mism++;
        nmsg++;
        if (mism < 20)
          fprintf(stderr, "doc %lu: message\n  want %s\n  got  %s\n  doc  %.*s\n",
                  (unsigned long)ndocs, line + 4, err.msg, (int)(len < 200 ? len : 200), doc);
      }
      /* under the default limits: serde's message, or one of ours, marked as such */
      if (rc2 == ND_OK || (rc2 == ND_E_BOUNDS ? strncmp(err2.msg, "nd_json limit: ", 15) != 0
                                              : strcmp(err2.msg, line + 4) != 0)) {
        mism++;
        if (mism < 20)
          fprintf(stderr, "doc %lu: default-limit result %d %s\n", (unsigned long)ndocs, rc2,
                  err2.msg);
      }
    } else {
      char *sp1 = strchr(line + 3, ' '), *sp2 = sp1 ? strchr(sp1 + 1, ' ') : NULL;
      nok++;
      if (rc != ND_OK) {
        mism++;
        if (mism < 20) fprintf(stderr, "doc %lu: rejected (%s), serde accepts: %.*s\n",
                               (unsigned long)ndocs, err.msg, (int)(len < 200 ? len : 200), doc);
      } else {
        char *s1 = NULL, *s2 = NULL;
        size_t l1 = 0, l2 = 0;
        if (!sp1 || !sp2) {
          fprintf(stderr, "bad expected line\n");
          return 1;
        }
        want.n = 0;
        got.n = 0;
        dump(nd_json_root(d), &got);
        check_dom(nd_json_root(d));
        CHECK(nd_json_to_string(nd_json_root(d), ND_JSON_KEYS_SORTED, &s1, &l1) == 0);
        CHECK(nd_json_to_string(nd_json_root(d), ND_JSON_KEYS_INSERTION, &s2, &l2) == 0);
        sb_str(&got, " ");
        sb_hex(&got, s1, l1);
        sb_str(&got, " ");
        sb_hex(&got, s2, l2);
        free(s1);
        free(s2);
        if (strcmp(got.p, line + 3) != 0) {
          mism++;
          if (mism < 20)
            fprintf(stderr, "doc %lu: mismatch\n  doc  %.*s\n  want %s\n  got  %s\n",
                    (unsigned long)ndocs, (int)(len < 300 ? len : 300), doc, line + 3, got.p);
        }
      }
      /* the default limits may refuse (depth 64, 1 MiB, 65536 elements) but never differ */
      if (rc2 != ND_OK && (rc2 != ND_E_BOUNDS || strncmp(err2.msg, "nd_json limit: ", 15) != 0)) {
        mism++;
        fprintf(stderr, "doc %lu: default limits gave %d\n", (unsigned long)ndocs, rc2);
      }
    }
    nd_json_doc_free(d);
    nd_json_doc_free(d2);
  }
  g_checks += (long)ndocs;
  g_fail += (long)mism;
  printf("docs: %lu documents (%lu accepted, %lu rejected by serde_json, messages compared), "
         "%lu mismatches (%lu in messages)\n",
         (unsigned long)ndocs, (unsigned long)nok, (unsigned long)nbad, (unsigned long)mism,
         (unsigned long)nmsg);
  free(c);
  free(e);
  free(want.p);
  free(got.p);
  return mism != 0;
}

static uint64_t rd64(const unsigned char *p) {
  uint64_t v = 0;
  int k;
  for (k = 7; k >= 0; k--) v = v << 8 | p[k];
  return v;
}

static int run_floats(const char *path, int is64) {
  size_t n, i, cnt = 0, mism = 0;
  unsigned char *b = read_file(path, &n);
  char out[ND_JSON_NUMBUF], out2[ND_JSON_NUMBUF];
  for (i = 0; i + 24 <= n; i += 24) {
    uint64_t bits = rd64(b + i), h1 = rd64(b + i + 8), h2 = rd64(b + i + 16);
    size_t l1, l2;
    if (is64) {
      l1 = nd_json_write_f64(dbits(bits), out);
      l2 = nd_json_write_f64(dbits(bits), out2);
    } else {
      float f = fbits((uint32_t)bits);
      l1 = nd_json_write_f32(f, out);
      l2 = nd_json_write_f64((double)f, out2);
    }
    cnt++;
    if (fnv(out, l1) != h1 || fnv(out2, l2) != h2) {
      mism++;
      if (mism < 20)
        fprintf(stderr, "%s bits %08lx%08lx: got %s / %s\n", is64 ? "f64" : "f32",
                (unsigned long)(bits >> 32), (unsigned long)(bits & 0xffffffffu), out, out2);
    }
  }
  g_checks += (long)cnt;
  g_fail += (long)mism;
  printf("%s: %lu values, %lu mismatches\n", is64 ? "floats64" : "floats32", (unsigned long)cnt,
         (unsigned long)mism);
  free(b);
  return mism != 0;
}

static int run_f32all(const char *path) {
  size_t n;
  unsigned char *b = read_file(path, &n);
  unsigned block, bad = 0;
  char out[ND_JSON_NUMBUF];
  for (block = 0; block < 256; block++) {
    uint64_t h = 0xcbf29ce484222325u, want;
    uint32_t lo;
    char hx[17];
    for (lo = 0; lo < (1u << 24); lo++) {
      uint32_t bits = (uint32_t)block << 24 | lo;
      size_t l = nd_json_write_f32(fbits(bits), out), k;
      for (k = 0; k < l; k++) {
        h ^= (unsigned char)out[k];
        h *= 0x100000001b3u;
      }
      h ^= 0x0a;
      h *= 0x100000001b3u;
    }
    memcpy(hx, b + block * 17u, 16);
    hx[16] = 0;
    want = strtoull(hx, NULL, 16);
    if (want != h) {
      bad++;
      fprintf(stderr, "f32 block %02x mismatch\n", block);
    }
  }
  g_checks += 256;
  g_fail += (long)bad;
  printf("f32all: 256 blocks (all 2^32 bit patterns), %u mismatching blocks\n", bad);
  free(b);
  return bad != 0;
}

/* ─── built-in cases ─── */

static void f64_is(double v, const char *want) {
  char b[ND_JSON_NUMBUF];
  nd_json_write_f64(v, b);
  g_checks++;
  if (strcmp(b, want)) {
    g_fail++;
    fprintf(stderr, "f64: got %s want %s\n", b, want);
  }
}
static void f32_is(float v, const char *want) {
  char b[ND_JSON_NUMBUF];
  nd_json_write_f32(v, b);
  g_checks++;
  if (strcmp(b, want)) {
    g_fail++;
    fprintf(stderr, "f32: got %s want %s\n", b, want);
  }
}

static const nd_json_value *parse_ok(const char *s, nd_json_doc **d) {
  nd_err err;
  int rc = nd_json_parse(s, strlen(s), NULL, d, &err);
  g_checks++;
  if (rc) {
    g_fail++;
    fprintf(stderr, "parse failed (%s): %s\n", err.msg, s);
    return NULL;
  }
  return nd_json_root(*d);
}

static void roundtrip_is(const char *in, nd_json_keyorder order, const char *want) {
  nd_json_doc *d = NULL;
  const nd_json_value *v = parse_ok(in, &d);
  char *s = NULL;
  size_t n;
  if (!v) return;
  CHECK(nd_json_to_string(v, order, &s, &n) == 0);
  g_checks++;
  if (!s || strcmp(s, want)) {
    g_fail++;
    fprintf(stderr, "serialise %s: got %s want %s\n", in, s ? s : "(null)", want);
  }
  free(s);
  nd_json_doc_free(d);
}

static void builtins(void) {
  nd_json_doc *d = NULL;
  const nd_json_value *v;
  nd_json_writer w;
  nd_json_limits lim;
  nd_err err;
  int64_t i64;
  uint64_t u64;
  double f;
  const char *s;
  size_t n;
  char *deep;
  int k;

  /* floats: serde_json 1.0.149 (zmij) output */
  f64_is(1e21, "1e+21");
  f64_is(1e15, "1000000000000000.0");
  f64_is(1e16, "1e+16");
  f64_is(0.1, "0.1");
  f64_is(450.0, "450.0");
  f64_is(1e-5, "0.00001");
  f64_is(1e-6, "1e-6");
  f64_is(1e-7, "1e-7");
  f64_is(5e-324, "5e-324");
  f64_is(1.7976931348623157e308, "1.7976931348623157e+308");
  f64_is(-0.0, "-0.0");
  f64_is(0.0, "0.0");
  f64_is(123456789.0, "123456789.0");
  f64_is(dbits(0x7ff0000000000000u), "null");
  f64_is(dbits(0x7ff8000000000000u), "null");
  f32_is(1e21f, "1e+21");
  f32_is(1e12f, "1000000000000.0");
  f32_is(1e13f, "1e+13");
  f32_is(0.1f, "0.1");
  f32_is(450.0f, "450.0");
  f32_is(1e-6f, "0.000001");
  f32_is(1e-7f, "1e-7");
  f32_is(fbits(1), "1e-45");
  f32_is(-0.0f, "-0.0");
  f64_is((double)0.1f, "0.10000000149011612");
  f64_is((double)1e21f, "1.0000000200408773e+21");
  f64_is((double)fbits(1), "1.401298464324817e-45");

  /* numbers: as_i64 / as_u64 / as_f64 semantics */
  v = parse_ok("[5, 5.0, 1e3, -0, -5, 18446744073709551615, 18446744073709551616, "
               "-9223372036854775808, -9223372036854775809, 9223372036854775808]",
               &d);
  if (v) {
    CHECK(nd_json_as_i64(nd_json_at(v, 0), &i64) && i64 == 5);
    CHECK(nd_json_as_u64(nd_json_at(v, 0), &u64) && u64 == 5);
    CHECK(!nd_json_as_i64(nd_json_at(v, 1), &i64) && nd_json_as_f64(nd_json_at(v, 1), &f) &&
          f == 5.0);
    CHECK(!nd_json_as_i64(nd_json_at(v, 2), &i64) && !nd_json_as_u64(nd_json_at(v, 2), &u64));
    CHECK(!nd_json_as_i64(nd_json_at(v, 3), &i64));
    CHECK(nd_json_as_i64(nd_json_at(v, 4), &i64) && i64 == -5 &&
          !nd_json_as_u64(nd_json_at(v, 4), &u64));
    CHECK(nd_json_as_u64(nd_json_at(v, 5), &u64) && u64 == UINT64_MAX &&
          !nd_json_as_i64(nd_json_at(v, 5), &i64));
    CHECK(!nd_json_as_u64(nd_json_at(v, 6), &u64) && nd_json_as_f64(nd_json_at(v, 6), &f) &&
          f == 18446744073709551616.0);
    CHECK(nd_json_as_i64(nd_json_at(v, 7), &i64) && i64 == INT64_MIN);
    CHECK(!nd_json_as_i64(nd_json_at(v, 8), &i64));
    CHECK(nd_json_at(v, 2)->u.num.text_len == 3 && memcmp(nd_json_at(v, 2)->u.num.text, "1e3", 3) == 0);
  }
  nd_json_doc_free(d);
  roundtrip_is("[5,5.0,1e3,-0,-5,0.1,1e-7,1e21]", ND_JSON_KEYS_SORTED,
               "[5,5.0,1000.0,-0.0,-5,0.1,1e-7,1e+21]");

  /* duplicates: kept in order, lookup returns the last; sorted output collapses like serde */
  v = parse_ok("{\"b\":1,\"a\":2,\"b\":3,\"\":4,\"b\":{\"x\":[]}}", &d);
  if (v) {
    CHECK(v->u.obj.len == 5);
    CHECK(nd_json_get_cstr(v, "b") == &v->u.obj.members[4].value);
    CHECK(nd_json_get_cstr(v, "a") == &v->u.obj.members[1].value);
    CHECK(nd_json_get(v, "", 0) == &v->u.obj.members[3].value);
    CHECK(nd_json_get_cstr(v, "c") == NULL);
  }
  nd_json_doc_free(d);
  roundtrip_is("{\"b\":1,\"a\":2,\"b\":3,\"\":4,\"b\":{\"x\":[]}}", ND_JSON_KEYS_SORTED,
               "{\"\":4,\"a\":2,\"b\":{\"x\":[]}}");
  roundtrip_is("{\"b\":1,\"a\":2,\"b\":3,\"\":4,\"b\":{\"x\":[]}}", ND_JSON_KEYS_INSERTION,
               "{\"b\":1,\"a\":2,\"b\":3,\"\":4,\"b\":{\"x\":[]}}");

  /* strings */
  v = parse_ok("\"a\\u0000b\\ud83d\\ude00\\u00e9\\/\"", &d);
  if (v) {
    CHECK(nd_json_as_str(v, &s, &n) && n == 10 && memcmp(s, "a\0b\xf0\x9f\x98\x80\xc3\xa9/", 10) == 0);
  }
  nd_json_doc_free(d);
  roundtrip_is("\"\\u0001\\u001f\\b\\t\\n\\f\\r\\\"\\\\/\\u007f\\u00e9\\u2028\"",
               ND_JSON_KEYS_SORTED, "\"\\u0001\\u001f\\b\\t\\n\\f\\r\\\"\\\\/\x7f\xc3\xa9\xe2\x80\xa8\"");
  CHECK(nd_json_parse("\"\\ud800\"", 8, NULL, &d, &err) == ND_E_FORMAT);
  CHECK(nd_json_parse("\"\\udc00\"", 8, NULL, &d, &err) == ND_E_FORMAT);
  CHECK(nd_json_parse("\"\xed\xa0\x80\"", 5, NULL, &d, &err) == ND_E_FORMAT);
  CHECK(nd_json_parse("[1,]", 4, NULL, &d, &err) == ND_E_FORMAT);
  CHECK(nd_json_parse("1e400", 5, NULL, &d, &err) == ND_E_FORMAT);
  CHECK(nd_json_parse("", 0, NULL, &d, &err) == ND_E_FORMAT);
  CHECK(strcmp(err.msg, "EOF while parsing a value at line 1 column 0") == 0);
  CHECK(nd_json_parse("{\n \"a\":\"x\n", 11, NULL, &d, &err) == ND_E_FORMAT);
  CHECK(strcmp(err.msg, "control character (\\u0000-\\u001F) found while parsing a string at line 3 column 0") == 0);
  CHECK(nd_json_parse("[1] x", 5, NULL, &d, &err) == ND_E_FORMAT);
  CHECK(strcmp(err.msg, "trailing characters at line 1 column 5") == 0);
  CHECK(nd_json_parse("1 ", 2, NULL, &d, &err) == ND_OK);
  nd_json_doc_free(d);

  /* limits */
  deep = (char *)malloc(400);
  for (k = 0; k < 200; k++) {
    deep[k] = '[';
    deep[399 - k] = ']';
  }
  CHECK(nd_json_parse(deep + 200 - 64, 128, NULL, &d, &err) == ND_OK);
  nd_json_doc_free(d);
  CHECK(nd_json_parse(deep + 200 - 65, 130, NULL, &d, &err) == ND_E_BOUNDS);
  lim = nd_json_limits_default();
  lim.max_depth = 500;
  CHECK(nd_json_parse(deep + 200 - 127, 254, &lim, &d, &err) == ND_OK);
  nd_json_doc_free(d);
  CHECK(nd_json_parse(deep + 200 - 128, 256, &lim, &d, &err) == ND_E_FORMAT);
  CHECK(strcmp(err.msg, "recursion limit exceeded at line 1 column 128") == 0);
  CHECK(nd_json_parse(deep + 200 - 65, 130, NULL, &d, &err) == ND_E_BOUNDS);
  CHECK(strncmp(err.msg, "nd_json limit: ", 15) == 0);
  free(deep);
  lim = nd_json_limits_default();
  lim.max_input = 4;
  CHECK(nd_json_parse("[1,2]", 5, &lim, &d, &err) == ND_E_BOUNDS);
  lim = nd_json_limits_default();
  lim.max_elements = 3;
  CHECK(nd_json_parse("[1,2]", 5, &lim, &d, &err) == ND_OK);
  nd_json_doc_free(d);
  CHECK(nd_json_parse("[1,2,3]", 7, &lim, &d, &err) == ND_E_BOUNDS);
  CHECK(nd_json_parse("{\"a\":1}", 7, &lim, &d, &err) == ND_OK); /* object, key, value */
  nd_json_doc_free(d);
  CHECK(nd_json_parse("{\"a\":[1]}", 9, &lim, &d, &err) == ND_E_BOUNDS);

  /* writer */
  nd_json_writer_init(&w, 0);
  nd_json_writer_begin_array(&w);
  nd_json_writer_begin_object(&w);
  nd_json_writer_key_cstr(&w, "name");
  nd_json_writer_string_cstr(&w, "start_timer");
  nd_json_writer_key_cstr(&w, "arguments");
  nd_json_writer_begin_object(&w);
  nd_json_writer_key_cstr(&w, "seconds");
  nd_json_writer_i64(&w, 450);
  nd_json_writer_key_cstr(&w, "x");
  nd_json_writer_f32(&w, 0.1f);
  nd_json_writer_key_cstr(&w, "y");
  nd_json_writer_f64(&w, (double)0.1f);
  nd_json_writer_key_cstr(&w, "z");
  nd_json_writer_begin_array(&w);
  nd_json_writer_null(&w);
  nd_json_writer_bool(&w, 1);
  nd_json_writer_u64(&w, UINT64_MAX);
  nd_json_writer_i64(&w, INT64_MIN);
  nd_json_writer_end_array(&w);
  nd_json_writer_end_object(&w);
  nd_json_writer_end_object(&w);
  nd_json_writer_end_array(&w);
  CHECK(w.err == 0);
  CHECK(w.data && strcmp(w.data, "[{\"name\":\"start_timer\",\"arguments\":{\"seconds\":450,\"x\":0.1,"
                                  "\"y\":0.10000000149011612,\"z\":[null,true,18446744073709551615,"
                                  "-9223372036854775808]}}]") == 0);
  nd_json_writer_null(&w); /* a second root value */
  CHECK(w.err == ND_E_ARG);
  nd_json_writer_reset(&w);
  nd_json_writer_begin_object(&w);
  nd_json_writer_i64(&w, 1); /* value without key */
  CHECK(w.err == ND_E_ARG);
  nd_json_writer_free(&w);
  nd_json_writer_init(&w, 8);
  nd_json_writer_string_cstr(&w, "123456789");
  CHECK(w.err == ND_E_BOUNDS);
  nd_json_writer_free(&w);
}

int main(int argc, char **argv) {
  int rc = 0, i;
  builtins();
  printf("builtins: %ld checks, %ld failures\n", g_checks, g_fail);
  for (i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--docs") && i + 2 < argc) {
      rc |= run_docs(argv[i + 1], argv[i + 2]);
      i += 2;
    } else if (!strcmp(argv[i], "--floats64") && i + 1 < argc) {
      rc |= run_floats(argv[++i], 1);
    } else if (!strcmp(argv[i], "--floats32") && i + 1 < argc) {
      rc |= run_floats(argv[++i], 0);
    } else if (!strcmp(argv[i], "--f32all") && i + 1 < argc) {
      rc |= run_f32all(argv[++i]);
    } else {
      fprintf(stderr, "unknown argument %s\n", argv[i]);
      return 2;
    }
  }
  printf("total: %ld checks, %ld failures\n", g_checks, g_fail);
  return (g_fail || rc) ? 1 : 0;
}
