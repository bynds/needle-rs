/* test_corrupt: every single-field corruption of runner/tests/container_robustness.rs, loaded
 * through nd_engine_from_bytes, must come back as accepted or refused (never a crash, which the
 * sanitizer builds turn any memory error into), and must never accept what the Rust loader
 * refuses. The C loader is stricter: it also refuses containers whose FP16 vectors or geometry do
 * not fit the forward pass, which Rust loads and then either aborts on at the first request or
 * runs on wrong-sized tensors. Those are listed as "stricter" and counted, not failed.
 *   test_corrupt MODEL VERDICTS.tsv   (VERDICTS from runner/examples/corrupt_cases.rs) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nd_engine.h"

#define HEADER_WORDS 49
#define CODEBOOK_LEN 28
#define REC 44

static const uint32_t EXTREMES[6] = {0, 1, 0x0000FFFFu, 0x40000000u, 0x7FFFFFFFu, 0xFFFFFFFFu};

static uint8_t *base;
static size_t blen;
static char *verdicts, *vcur;
static size_t ncase, nmismatch, nstricter;

static void put32(uint8_t *b, size_t at, uint32_t v) {
  int i;
  for (i = 0; i < 4; i++) b[at + (size_t)i] = (uint8_t)(v >> (8 * i));
}

static void put64(uint8_t *b, size_t at, uint64_t v) {
  int i;
  for (i = 0; i < 8; i++) b[at + (size_t)i] = (uint8_t)(v >> (8 * i));
}

static uint8_t *copy_base(size_t n) {
  uint8_t *b = malloc(n ? n : 1);
  if (!b) {
    fprintf(stderr, "out of memory\n");
    exit(1);
  }
  memcpy(b, base, n);
  return b;
}

/* Rust's {:#x}. */
static void hex(char *out, size_t cap, uint64_t v) { snprintf(out, cap, "0x%llx", (unsigned long long)v); }

static void run_case(const char *what, uint8_t *bytes, size_t n) {
  nd_engine *g = NULL;
  nd_err e = {""};
  int ok = nd_engine_from_bytes(bytes, n, 0, &g, &e) == ND_OK;
  char *line = vcur, *tab, *nl;
  const char *want = "?";
  ncase++;
  if (getenv("ND_TRACE")) fprintf(stderr, "case: %s -> %s %s\n", what, ok ? "ok" : "err", e.msg);
  nd_engine_free(g);
  if (line && *line) {
    nl = strchr(line, '\n');
    if (nl) *nl = 0;
    tab = strchr(line, '\t');
    if (tab) {
      *tab = 0;
      want = tab + 1;
      if (strcmp(line, what) != 0) {
        fprintf(stderr, "case list diverged: C '%s' vs Rust '%s'\n", what, line);
        exit(1);
      }
    }
    vcur = nl ? nl + 1 : NULL;
  }
  if (!strcmp(want, "ok") && !ok) {
    nstricter++;
    printf("stricter %s: %s\n", what, e.msg);
  } else if (strcmp(want, ok ? "ok" : "err") != 0) {
    nmismatch++;
    printf("MISMATCH %s: C %s (%s), Rust %s\n", what, ok ? "ok" : "err", e.msg, want);
  }
}

static char *slurp(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  long l;
  char *p;
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  l = ftell(f);
  fseek(f, 0, SEEK_SET);
  p = malloc((size_t)l + 1);
  if (!p || fread(p, 1, (size_t)l, f) != (size_t)l) exit(1);
  fclose(f);
  p[l] = 0;
  *n = (size_t)l;
  return p;
}

int main(int argc, char **argv) {
  size_t n, num_tensors, dir, recs[34], nrecs = 0, r, i, w, tok, tok_off;
  char what[160], hx[32];
  uint8_t *b;
  if (argc < 3) {
    fprintf(stderr, "usage: test_corrupt MODEL VERDICTS.tsv\n");
    return 2;
  }
  base = (uint8_t *)slurp(argv[1], &blen);
  verdicts = slurp(argv[2], &n);
  if (!base || !verdicts) return 2;
  vcur = verdicts;
  num_tensors = (size_t)base[4] | (size_t)base[5] << 8 | (size_t)base[6] << 16 | (size_t)base[7] << 24;
  dir = HEADER_WORDS * 4 + CODEBOOK_LEN * 4;

  for (w = 1; w < HEADER_WORDS - 1; w++)
    for (i = 0; i < 6; i++) {
      b = copy_base(blen);
      put32(b, w * 4, EXTREMES[i]);
      hex(hx, sizeof hx, EXTREMES[i]);
      snprintf(what, sizeof what, "header word %zu = %s", w, hx);
      run_case(what, b, blen);
    }

  for (r = 0; r < 30; r++) recs[nrecs++] = r;
  recs[nrecs++] = num_tensors / 2;
  recs[nrecs++] = num_tensors - 12;
  recs[nrecs++] = num_tensors - 2;
  recs[nrecs++] = num_tensors - 1;
  for (r = 0; r < nrecs; r++) {
    static const char *f32n[4] = {"shape0", "shape1", "group", "bits"};
    static const size_t f32o[4] = {4, 8, 36, 40};
    static const char *f64n[2] = {"offset", "nbytes"};
    static const size_t f64o[2] = {20, 28};
    size_t at = dir + recs[r] * REC, f;
    for (f = 0; f < 4; f++)
      for (i = 0; i < 6; i++) {
        b = copy_base(blen);
        put32(b, at + f32o[f], EXTREMES[i]);
        hex(hx, sizeof hx, EXTREMES[i]);
        snprintf(what, sizeof what, "record %zu %s = %s", recs[r], f32n[f], hx);
        run_case(what, b, blen);
      }
    for (f = 0; f < 2; f++) {
      uint64_t vals[5];
      vals[0] = 0, vals[1] = 1, vals[2] = (uint64_t)blen - 1, vals[3] = 0xFFFFFFFFu, vals[4] = UINT64_MAX;
      for (i = 0; i < 5; i++) {
        b = copy_base(blen);
        put64(b, at + f64o[f], vals[i]);
        hex(hx, sizeof hx, vals[i]);
        snprintf(what, sizeof what, "record %zu %s = %s", recs[r], f64n[f], hx);
        run_case(what, b, blen);
      }
    }
    b = copy_base(blen);
    b[at] = 0xEE;
    snprintf(what, sizeof what, "record %zu dtype = 0xEE", recs[r]);
    run_case(what, b, blen);
  }

  tok = dir + (num_tensors - 1) * REC;
  tok_off = 0;
  for (i = 0; i < 8; i++) tok_off |= (size_t)base[tok + 20 + i] << (8 * i);
  {
    static const char *names[5] = {"n_pieces", "pad", "eos", "bos", "unk"};
    size_t f;
    for (f = 0; f < 5; f++)
      for (i = 0; i < 6; i++) {
        b = copy_base(blen);
        put32(b, tok_off + 4 * f, EXTREMES[i]);
        hex(hx, sizeof hx, EXTREMES[i]);
        snprintf(what, sizeof what, "tokenizer %s = %s", names[f], hx);
        run_case(what, b, blen);
      }
  }
  for (i = 0; i < 2; i++) {
    uint16_t v = i ? 0xFFFF : 0;
    b = copy_base(blen);
    b[tok_off + 24 + 5] = (uint8_t)v;
    b[tok_off + 24 + 6] = (uint8_t)(v >> 8);
    hex(hx, sizeof hx, v);
    snprintf(what, sizeof what, "tokenizer piece 0 len = %s", hx);
    run_case(what, b, blen);
  }
  {
    size_t cuts[7];
    cuts[0] = 0, cuts[1] = 3, cuts[2] = 100, cuts[3] = dir - 1, cuts[4] = dir + REC * 3 + 5, cuts[5] = blen / 2,
    cuts[6] = blen - 1;
    for (i = 0; i < 7; i++) {
      b = copy_base(cuts[i]);
      snprintf(what, sizeof what, "truncated to %zu", cuts[i]);
      run_case(what, b, cuts[i]);
    }
  }
  run_case("untouched", copy_base(blen), blen);
  printf("%zu cases, %zu refused only by C, %zu accepted by C but refused by Rust\n", ncase, nstricter, nmismatch);
  free(base);
  free(verdicts);
  return nmismatch ? 1 : 0;
}
