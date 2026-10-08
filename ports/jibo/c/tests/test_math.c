/* test_math.c: bit-equality test of nd_math.c against the Rust libm 0.2.16 reference.
 *
 * The reference files come from tests/math_ref.rs (see its header for how to build it and the
 * file formats), produced on the same target as this test runs on.
 *
 *   test_math cases FILE              compare every input/output record (NaNs compare equal)
 *   test_math exh FILE [LO HI]        recompute the 2^32-input block hashes of one unary function
 *                                     for blocks LO..HI-1 (default all 65536) and compare
 *
 * Exit status 0 when everything matches, 1 on any mismatch, 2 on a usage or I/O error.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nd_math.h"

static const char *const NAMES[6] = {"expf", "logf", "sinf", "cosf", "powf", "tanhf"};

static uint32_t f2u(float x) {
  uint32_t u;
  memcpy(&u, &x, sizeof u);
  return u;
}
static float u2f(uint32_t u) {
  float x;
  memcpy(&x, &u, sizeof x);
  return x;
}
static int is_nan_bits(uint32_t u) { return (u & 0x7fffffffu) > 0x7f800000u; }
static uint32_t canon(uint32_t u) { return is_nan_bits(u) ? 0x7fc00000u : u; }

static uint32_t rd32(const unsigned char *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static float eval(uint32_t f, float a, float b) {
  switch (f) {
    case 0: return nd_expf(a);
    case 1: return nd_logf(a);
    case 2: return nd_sinf(a);
    case 3: return nd_cosf(a);
    case 4: return nd_powf(a, b);
    default: return nd_tanhf(a);
  }
}

static int run_cases(const char *path) {
  FILE *fp = fopen(path, "rb");
  unsigned char hdr[4], rec[16 * 4096];
  unsigned long n[6] = {0}, bad[6] = {0}, shown = 0;
  size_t got, i;
  int f;
  if (!fp || fread(hdr, 1, 4, fp) != 4 || memcmp(hdr, "NDMC", 4) != 0) {
    fprintf(stderr, "%s: not a cases file\n", path);
    return 2;
  }
  while ((got = fread(rec, 16, 4096, fp)) > 0) {
    for (i = 0; i < got; i++) {
      const unsigned char *p = rec + 16 * i;
      uint32_t fn = rd32(p), a = rd32(p + 4), b = rd32(p + 8), want = rd32(p + 12);
      uint32_t have;
      if (fn > 5) {
        fprintf(stderr, "%s: bad function id %lu\n", path, (unsigned long)fn);
        fclose(fp);
        return 2;
      }
      have = f2u(eval(fn, u2f(a), u2f(b)));
      n[fn]++;
      if (canon(have) != canon(want)) {
        bad[fn]++;
        if (shown++ < 20) {
          printf("MISMATCH %s(0x%08lx", NAMES[fn], (unsigned long)a);
          if (fn == 4) printf(", 0x%08lx", (unsigned long)b);
          printf("): rust 0x%08lx c 0x%08lx\n", (unsigned long)want, (unsigned long)have);
        }
      }
    }
  }
  fclose(fp);
  {
    unsigned long tot = 0, totbad = 0;
    for (f = 0; f < 6; f++) {
      printf("%-6s %10lu cases, %lu mismatches\n", NAMES[f], n[f], bad[f]);
      tot += n[f];
      totbad += bad[f];
    }
    printf("total  %10lu cases, %lu mismatches\n", tot, totbad);
    return totbad ? 1 : 0;
  }
}

static uint64_t block_hash(uint32_t f, uint32_t block) {
  uint64_t h = 0xcbf29ce484222325ull;
  uint32_t lo;
  for (lo = 0; lo <= 0xffffu; lo++) {
    float x = u2f(block << 16 | lo);
    h ^= canon(f2u(eval(f, x, 0.0f)));
    h *= 0x00000100000001b3ull;
  }
  return h;
}

static int run_exh(const char *path, uint32_t lo, uint32_t hi) {
  FILE *fp = fopen(path, "rb");
  unsigned char hdr[8], *buf;
  uint32_t f, k;
  unsigned long bad = 0;
  if (!fp || fread(hdr, 1, 8, fp) != 8 || memcmp(hdr, "NDMX", 4) != 0) {
    fprintf(stderr, "%s: not an exhaustive file\n", path);
    return 2;
  }
  f = rd32(hdr + 4);
  buf = malloc(65536u * 8u);
  if (!buf || f > 5 || f == 4 || fread(buf, 8, 65536, fp) != 65536) {
    fprintf(stderr, "%s: short or bad file\n", path);
    free(buf);
    fclose(fp);
    return 2;
  }
  fclose(fp);
  for (k = lo; k < hi; k++) {
    const unsigned char *p = buf + 8u * k;
    uint64_t want = (uint64_t)rd32(p) | (uint64_t)rd32(p + 4) << 32;
    if (block_hash(f, k) != want) {
      if (bad++ < 20) printf("MISMATCH %s block 0x%04lx (inputs 0x%04lx0000..0x%04lxffff)\n",
                             NAMES[f], (unsigned long)k, (unsigned long)k, (unsigned long)k);
    }
  }
  free(buf);
  printf("%s exhaustive blocks 0x%04lx..0x%04lx (%llu inputs): %lu mismatching blocks\n", NAMES[f],
         (unsigned long)lo, (unsigned long)hi, (unsigned long long)(hi - lo) * 65536ull, bad);
  return bad ? 1 : 0;
}

int main(int argc, char **argv) {
  if (argc >= 3 && strcmp(argv[1], "cases") == 0) return run_cases(argv[2]);
  if ((argc == 3 || argc == 5) && strcmp(argv[1], "exh") == 0) {
    uint32_t lo = 0, hi = 65536;
    if (argc == 5) {
      lo = (uint32_t)strtoul(argv[3], NULL, 0);
      hi = (uint32_t)strtoul(argv[4], NULL, 0);
      if (hi > 65536 || lo > hi) return 2;
    }
    return run_exh(argv[2], lo, hi);
  }
  fprintf(stderr, "usage: test_math cases FILE | test_math exh FILE [LO HI]\n");
  return 2;
}
