/* test_tok.c: parity test of nd_tok / nd_prompt against a reference dump produced by the Rust
 * implementation (SpTokenizer, build_prompt, compact_json, to_snake_case).
 *
 *   test_tok <needle3.cact> <ref.bin> [stride]
 *
 * ref.bin is written by a throwaway Rust program linked against needle-infer; it holds the
 * inputs and the Rust outputs. Every record is checked byte for byte; with stride > 1 only
 * every stride-th encode/decode/prompt/compact/snake/id_of record is run (for slow emulation).
 * The test also extracts the tokenizer from the container itself, feeds nd_tok_load the
 * truncated, mutated and crafted blobs the reference recorded (comparing accept/reject and the
 * error message), and fuzzes nd_tok_load with random corruptions (must not crash).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nd_prompt.h"
#include "nd_tok.h"

typedef struct {
  const uint8_t *p;
  size_t n, i;
  int bad;
} cur;

static uint8_t g8(cur *c) {
  if (c->bad || c->n - c->i < 1) {
    c->bad = 1;
    return 0;
  }
  return c->p[c->i++];
}
static uint32_t g32(cur *c) {
  uint32_t v;
  if (c->bad || c->n - c->i < 4) {
    c->bad = 1;
    return 0;
  }
  v = (uint32_t)c->p[c->i] | ((uint32_t)c->p[c->i + 1] << 8) | ((uint32_t)c->p[c->i + 2] << 16) |
      ((uint32_t)c->p[c->i + 3] << 24);
  c->i += 4;
  return v;
}
static int64_t g64(cur *c) {
  uint64_t lo = g32(c), hi = g32(c);
  return (int64_t)(lo | (hi << 32));
}
/* A length-prefixed byte string, pointing into the buffer. */
static const uint8_t *gbytes(cur *c, size_t *len) {
  uint32_t n = g32(c);
  const uint8_t *p;
  if (c->bad || c->n - c->i < n) {
    c->bad = 1;
    *len = 0;
    return NULL;
  }
  p = c->p + c->i;
  c->i += n;
  *len = n;
  return p;
}
/* A u32 id array, copied out (the buffer may be unaligned). */
static uint32_t *gids(cur *c, size_t *n) {
  uint32_t k = g32(c), j;
  uint32_t *v;
  if (c->bad || (c->n - c->i) / 4 < k) {
    c->bad = 1;
    *n = 0;
    return NULL;
  }
  v = (uint32_t *)malloc(k ? (size_t)k * 4 : 1);
  if (!v) {
    c->bad = 1;
    return NULL;
  }
  for (j = 0; j < k; j++) v[j] = g32(c);
  *n = k;
  return v;
}

static int read_file(const char *path, uint8_t **buf, size_t *len) {
  FILE *f = fopen(path, "rb");
  long sz;
  if (!f) return -1;
  if (fseek(f, 0, SEEK_END) != 0 || (sz = ftell(f)) < 0 || fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return -1;
  }
  *buf = (uint8_t *)malloc(sz ? (size_t)sz : 1);
  if (!*buf || fread(*buf, 1, (size_t)sz, f) != (size_t)sz) {
    fclose(f);
    return -1;
  }
  fclose(f);
  *len = (size_t)sz;
  return 0;
}

static uint32_t le32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t le64(const uint8_t *p) { return (uint64_t)le32(p) | ((uint64_t)le32(p + 4) << 32); }

/* The single RAW record of a v3 container: 49 header words, codebook_len (word 2) f32s, then
 * num_tensors (word 1) 44-byte records (u8 dtype, u8 ndim, u16, u32 shape[4], u64 offset,
 * u64 nbytes, u32 group, u32 bits). */
static int extract_tokenizer(const uint8_t *f, size_t n, const uint8_t **blob, size_t *blen) {
  size_t nt, cb, dir, t, found = 0;
  if (n < 196) return -1;
  nt = le32(f + 4);
  cb = le32(f + 8);
  if (!nd_mul_ok(cb, 4, &dir) || !nd_add_ok(dir, 196, &dir)) return -1;
  {
    size_t recs, end;
    if (!nd_mul_ok(nt, 44, &recs) || !nd_add_ok(dir, recs, &end) || end > n) return -1;
  }
  for (t = 0; t < nt; t++) {
    const uint8_t *r = f + dir + t * 44;
    if (r[0] == 4) {
      uint64_t off = le64(r + 20), nb = le64(r + 28);
      if (off > n || nb > n - off) return -1;
      *blob = f + off;
      *blen = (size_t)nb;
      found++;
    }
  }
  return found == 1 ? 0 : -1;
}

static long n_fail = 0;
static long counts[16];

static void fail(const char *what, long rec, const uint8_t *in, size_t inlen) {
  n_fail++;
  if (n_fail <= 20) {
    fprintf(stderr, "MISMATCH %s at record %ld, input (%lu bytes): ", what, rec, (unsigned long)inlen);
    fwrite(in, 1, inlen > 200 ? 200 : inlen, stderr);
    fputc('\n', stderr);
  }
}

static int has_nul(const uint8_t *p, size_t n) { return n && memchr(p, 0, n) != NULL; }

/* Copy to a NUL-terminated heap string. */
static char *dupz(const uint8_t *p, size_t n) {
  char *s = (char *)malloc(n + 1);
  if (!s) {
    fprintf(stderr, "out of memory\n");
    exit(2);
  }
  if (n) memcpy(s, p, n);
  s[n] = '\0';
  return s;
}

static void check_encode(const nd_tok *t, cur *c, long rec, int run) {
  size_t tl, nids, dl, got_n, got_dl;
  const uint8_t *text = gbytes(c, &tl);
  uint32_t *ids = gids(c, &nids), *got = NULL;
  const uint8_t *dec = gbytes(c, &dl);
  char *gd = NULL;
  if (c->bad || !run) {
    free(ids);
    return;
  }
  counts[3]++;
  if (nd_tok_encode(t, (const char *)text, tl, &got, &got_n) != ND_OK || got_n != nids ||
      (nids && memcmp(got, ids, nids * 4) != 0)) {
    fail("encode", rec, text, tl);
  } else if (nd_tok_decode(t, got, got_n, &gd, &got_dl) != ND_OK || got_dl != dl ||
             (dl && memcmp(gd, dec, dl) != 0) || gd[got_dl] != '\0') {
    fail("decode(encode)", rec, text, tl);
  }
  free(got);
  free(gd);
  free(ids);
}

static void check_decode(const nd_tok *t, cur *c, long rec, int run) {
  size_t nids, dl, got_dl;
  uint32_t *ids = gids(c, &nids);
  const uint8_t *dec = gbytes(c, &dl);
  char *gd = NULL;
  if (c->bad || !run) {
    free(ids);
    return;
  }
  counts[4]++;
  if (nd_tok_decode(t, ids, nids, &gd, &got_dl) != ND_OK || got_dl != dl ||
      (dl && memcmp(gd, dec, dl) != 0) || gd[got_dl] != '\0') {
    fail("decode", rec, dec, dl);
  }
  free(gd);
  free(ids);
}

/* The sub-records the reference wrote for a mutated or crafted tokenizer (t == NULL: skip them). */
static void check_sub(const nd_tok *t, cur *c, long rec) {
  uint32_t k = g32(c), j;
  for (j = 0; j < k && !c->bad; j++) {
    uint8_t kind = g8(c);
    if (kind == 3)
      check_encode(t, c, rec, t != NULL);
    else if (kind == 4)
      check_decode(t, c, rec, t != NULL);
    else
      c->bad = 1;
  }
}

static const char *ref_kind_prefix(uint8_t st) {
  switch (st) {
    case 1: return "tokenizer blob truncated";
    case 2: return "piece ";
    case 3: return "tokenizer blob declares zero pieces";
    case 4: return "header ";
    case 5: return "byte piece ";
    default: return "";
  }
}

static void check_blob(const uint8_t *b, size_t bl, cur *c, long rec) {
  uint8_t st = g8(c);
  size_t ml;
  const uint8_t *msg = gbytes(c, &ml);
  nd_tok *t = NULL;
  nd_err e;
  int rc;
  if (c->bad) return;
  e.msg[0] = '\0';
  rc = nd_tok_load(b, bl, &t, &e);
  counts[st == 0 ? 9 : 10]++;
  if (st == 0) {
    if (rc != ND_OK) {
      fprintf(stderr, "blob record %ld: Rust accepted, C rejected: %s\n", rec, e.msg);
      fail("blob accept", rec, msg, ml);
      check_sub(NULL, c, rec);
      return;
    }
    check_sub(t, c, rec);
    nd_tok_free(t);
  } else {
    int same;
    if (rc != ND_E_FORMAT || t != NULL) {
      fail("blob reject", rec, msg, ml);
      nd_tok_free(t);
      return;
    }
    /* Exact message on 64-bit (the reference's usize); on 32-bit a saturated `need` in a
     * truncation message differs, so only the error kind is compared there. */
    if (sizeof(size_t) == 8 || st != 1)
      same = strlen(e.msg) == ml && memcmp(e.msg, msg, ml) == 0;
    else
      same = strncmp(e.msg, ref_kind_prefix(st), strlen(ref_kind_prefix(st))) == 0;
    if (!same) {
      fprintf(stderr, "blob record %ld: C says \"%s\"\n", rec, e.msg);
      fail("blob error message", rec, msg, ml);
    }
  }
}

static uint64_t rng_state = 0x243F6A8885A308D3ull;
static uint32_t rnd(void) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return (uint32_t)(rng_state >> 16);
}

/* Random corruptions: nd_tok_load must return cleanly; an accepted table must encode and
 * decode without memory errors (checked under ASan/valgrind). */
static void fuzz_load(const uint8_t *blob, size_t bl, int iters) {
  uint8_t *b = (uint8_t *)malloc(bl);
  int it, ok = 0, rej = 0;
  static const char *const s[] = {"Set a timer for 90 seconds.", "<tool_call>{\"a\":1}</tool_call>",
                                  "日本 😀 \x7f zz", ""};
  if (!b) exit(2);
  for (it = 0; it < iters; it++) {
    size_t len = bl, k, np = 1 + rnd() % 6;
    nd_tok *t = NULL;
    memcpy(b, blob, bl);
    for (k = 0; k < np; k++) {
      size_t off = (rnd() % 2) ? rnd() % 64 : rnd() % bl;
      b[off] = (uint8_t)rnd();
    }
    if (rnd() % 3 == 0) len = rnd() % (bl + 1);
    if (nd_tok_load(b, len, &t, NULL) == ND_OK) {
      size_t j;
      int skip = 0;
      ok++;
      for (j = 0; j < nd_tok_size(t); j++) {
        size_t pl;
        if (nd_tok_type(t, (uint32_t)j) == ND_TK_USER_DEFINED) {
          nd_tok_piece(t, (uint32_t)j, &pl);
          if (pl == 0) skip = 1;
        }
      }
      (void)skip; /* the C encoder handles empty markers (skips them), so always encode */
      for (j = 0; j < 4; j++) {
        uint32_t *ids = NULL, rid[16];
        size_t n = 0, dl, q;
        char *d = NULL;
        if (nd_tok_encode(t, s[j], strlen(s[j]), &ids, &n) == ND_OK &&
            nd_tok_decode(t, ids, n, &d, &dl) == ND_OK) {
          free(d);
        }
        free(ids);
        for (q = 0; q < 16; q++) rid[q] = rnd() % (uint32_t)(nd_tok_size(t) + 3);
        if (nd_tok_decode(t, rid, 16, &d, &dl) == ND_OK) free(d);
      }
      nd_tok_free(t);
    } else {
      rej++;
    }
  }
  /* every prefix of the first 4 KiB, and the last few */
  {
    size_t l;
    for (l = 0; l <= bl; l++) {
      nd_tok *t = NULL;
      if (l == 4096 && bl > 4200) l = bl - 100;
      if (nd_tok_load(blob, l, &t, NULL) == ND_OK) {
        if (l != bl) {
          fprintf(stderr, "prefix of %lu bytes accepted\n", (unsigned long)l);
          n_fail++;
        }
        nd_tok_free(t);
      }
    }
  }
  printf("fuzz: %d random corruptions (%d accepted, %d rejected), all prefixes checked\n", iters, ok,
         rej);
  free(b);
}

static void self_checks(const nd_tok *t) {
  uint32_t m[ND_N_CHAT_MARKERS];
  uint32_t *ids = NULL;
  size_t n;
  char *s;
  /* chat markers resolve, in order */
  if (nd_tok_chat_marker_ids(t, m) != ND_OK) {
    fprintf(stderr, "chat markers missing\n");
    n_fail++;
  }
  /* invalid UTF-8 cannot be encoded (Rust's &str cannot hold it) */
  if (nd_tok_encode(t, "\xff", 1, &ids, &n) != ND_E_ARG || ids != NULL) {
    fprintf(stderr, "invalid UTF-8 accepted by encode\n");
    n_fail++;
  }
  if (nd_tok_encode(t, "", 0, &ids, &n) != ND_OK || n != 0) n_fail++;
  if (nd_to_snake_case("\xc3") != NULL) {
    fprintf(stderr, "invalid UTF-8 accepted by snake_case\n");
    n_fail++;
  }
  s = nd_to_snake_case("HTMLParser");
  if (!s || strcmp(s, "html_parser") != 0) n_fail++;
  free(s);
  if (nd_tok_type(t, (uint32_t)nd_tok_size(t)) != -1 || nd_tok_piece(t, 0xFFFFFFFFu, NULL) != NULL)
    n_fail++;
}

int main(int argc, char **argv) {
  uint8_t *cact = NULL, *ref = NULL;
  size_t cact_len = 0, ref_len = 0, blen = 0;
  const uint8_t *blob = NULL;
  nd_tok *t = NULL;
  nd_err e;
  cur c;
  long rec = 0, stride = 1, seen[16];
  int done = 0;
  memset(seen, 0, sizeof seen);

  if (argc < 3) {
    fprintf(stderr, "usage: %s <needle3.cact> <ref.bin> [stride]\n", argv[0]);
    return 2;
  }
  if (argc > 3) stride = atol(argv[3]);
  if (stride < 1) stride = 1;
  if (read_file(argv[1], &cact, &cact_len) || read_file(argv[2], &ref, &ref_len)) {
    fprintf(stderr, "cannot read inputs\n");
    return 2;
  }
  if (extract_tokenizer(cact, cact_len, &blob, &blen)) {
    fprintf(stderr, "no single RAW tokenizer record in %s\n", argv[1]);
    return 2;
  }
  if (nd_tok_load(blob, blen, &t, &e) != ND_OK) {
    fprintf(stderr, "nd_tok_load: %s\n", e.msg);
    return 1;
  }
  printf("tokenizer: %lu pieces, %lu-byte blob, add_dummy_prefix=%d byte_fallback=%d\n",
         (unsigned long)nd_tok_size(t), (unsigned long)blen, nd_tok_add_dummy_prefix(t),
         nd_tok_byte_fallback(t));
  self_checks(t);

  c.p = ref;
  c.n = ref_len;
  c.i = 0;
  c.bad = 0;
  if (ref_len < 10 || memcmp(ref, "NDTOKREF1\0", 10) != 0) {
    fprintf(stderr, "bad reference file\n");
    return 2;
  }
  c.i = 10;
  while (!c.bad && !done) {
    uint8_t kind = g8(&c);
    int run;
    rec++;
    if (kind < 16) seen[kind]++;
    run = kind < 16 && (seen[kind] - 1) % stride == 0;
    switch (kind) {
      case 0:
        done = 1;
        break;
      case 1: {
        uint32_t v[5];
        int k, ad, bf;
        for (k = 0; k < 5; k++) v[k] = g32(&c);
        ad = g8(&c);
        bf = g8(&c);
        counts[1]++;
        if (v[0] != nd_tok_size(t) || v[1] != nd_tok_pad_id(t) || v[2] != nd_tok_eos_id(t) ||
            v[3] != nd_tok_bos_id(t) || v[4] != nd_tok_unk_id(t) || ad != nd_tok_add_dummy_prefix(t) ||
            bf != nd_tok_byte_fallback(t))
          fail("header", rec, (const uint8_t *)"", 0);
        break;
      }
      case 2: { /* piece table entry */
        uint32_t id = g32(&c), bits, gbits;
        size_t pl, tbl, gl;
        const uint8_t *p = gbytes(&c, &pl);
        int ty = g8(&c);
        const uint8_t *tb;
        const char *gp;
        uint8_t tbuf[1024];
        long gtl;
        float sc;
        bits = g32(&c);
        tb = gbytes(&c, &tbl);
        if (c.bad) break;
        counts[2]++;
        gp = nd_tok_piece(t, id, &gl);
        sc = nd_tok_score(t, id);
        memcpy(&gbits, &sc, 4);
        gtl = nd_tok_token_bytes(t, id, tbuf, sizeof tbuf);
        if (!gp || gl != pl || memcmp(gp, p, pl) != 0 || gp[gl] != '\0' || nd_tok_type(t, id) != ty ||
            gbits != bits || gtl < 0 || (size_t)gtl != tbl || (tbl && memcmp(tbuf, tb, tbl) != 0))
          fail("piece", rec, p, pl);
        break;
      }
      case 3:
        check_encode(t, &c, rec, run);
        break;
      case 4:
        check_decode(t, &c, rec, run);
        break;
      case 5: { /* build_prompt */
        size_t ql, tl, sl, ol;
        const uint8_t *q = gbytes(&c, &ql), *tj = gbytes(&c, &tl);
        int has_sys = g8(&c);
        const uint8_t *sy = gbytes(&c, &sl), *o = gbytes(&c, &ol);
        char *qz, *tz, *sz, *got;
        if (c.bad || !run) break;
        if (has_nul(q, ql) || has_nul(tj, tl) || has_nul(sy, sl)) {
          counts[11]++;
          break;
        }
        counts[5]++;
        qz = dupz(q, ql);
        tz = dupz(tj, tl);
        sz = dupz(sy, sl);
        got = nd_build_prompt(qz, tz, has_sys ? sz : NULL);
        if (!got || strlen(got) != ol || memcmp(got, o, ol) != 0) fail("build_prompt", rec, q, ql);
        free(got);
        free(qz);
        free(tz);
        free(sz);
        break;
      }
      case 6:
      case 7: { /* compact_json, to_snake_case */
        size_t il, ol;
        const uint8_t *in = gbytes(&c, &il), *o = gbytes(&c, &ol);
        char *iz, *got;
        if (c.bad || !run) break;
        if (has_nul(in, il)) { /* not expressible as a C string */
          counts[11]++;
          break;
        }
        counts[kind]++;
        iz = dupz(in, il);
        got = kind == 6 ? nd_compact_json(iz) : nd_to_snake_case(iz);
        if (!got || strlen(got) != ol || memcmp(got, o, ol) != 0)
          fail(kind == 6 ? "compact_json" : "to_snake_case", rec, in, il);
        free(got);
        free(iz);
        break;
      }
      case 8: { /* id_of */
        size_t sl;
        const uint8_t *s = gbytes(&c, &sl);
        int64_t id = g64(&c);
        if (c.bad || !run) break;
        counts[8]++;
        if (nd_tok_id_of_n(t, (const char *)s, sl) != (int32_t)id) fail("id_of_n", rec, s, sl);
        if (!has_nul(s, sl)) {
          char *z = dupz(s, sl);
          if (nd_tok_id_of(t, z) != (int32_t)id) fail("id_of", rec, s, sl);
          free(z);
        }
        break;
      }
      case 9: { /* mutation of the real blob */
        uint32_t trunc = g32(&c), np = g32(&c), k;
        uint8_t *b = (uint8_t *)malloc(blen);
        if (!b) return 2;
        memcpy(b, blob, blen);
        for (k = 0; k < np && !c.bad; k++) {
          uint32_t off = g32(&c);
          uint8_t v = g8(&c);
          if (off < blen) b[off] = v;
        }
        if (trunc > blen) c.bad = 1;
        if (!c.bad) check_blob(b, trunc, &c, rec);
        free(b);
        break;
      }
      case 10: { /* crafted blob */
        size_t bl;
        const uint8_t *b = gbytes(&c, &bl);
        uint8_t *copy;
        if (c.bad) break;
        /* exact-size heap copy, so ASan/valgrind see any over-read */
        copy = (uint8_t *)malloc(bl ? bl : 1);
        if (!copy) return 2;
        if (bl) memcpy(copy, b, bl);
        check_blob(copy, bl, &c, rec);
        free(copy);
        break;
      }
      default:
        c.bad = 1;
        break;
    }
  }
  if (c.bad || !done) {
    fprintf(stderr, "reference file malformed at record %ld\n", rec);
    return 2;
  }

  fuzz_load(blob, blen, stride > 1 ? 300 : 3000);

  printf("checked: %ld piece entries, %ld encode, %ld decode, %ld build_prompt, %ld compact_json, "
         "%ld to_snake_case, %ld id_of\n",
         counts[2], counts[3], counts[4], counts[5], counts[6], counts[7], counts[8]);
  printf("blobs: %ld accepted and %ld rejected by Rust, C agreeing on each unless reported; %ld records skipped (NUL in a C "
         "string argument)\n",
         counts[9], counts[10], counts[11]);
  printf("%s: %ld mismatches\n", n_fail ? "FAIL" : "PASS", n_fail);
  nd_tok_free(t);
  free(cact);
  free(ref);
  return n_fail ? 1 : 0;
}
