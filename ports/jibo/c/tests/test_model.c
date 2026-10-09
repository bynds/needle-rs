/* test_model: the C forward pass against a Rust trace (runner/examples/trace.rs).
 *   test_model MODEL TRACEDIR [DEPTH]
 * Every stage must be bit-identical: cells, all-position logits, prefill, decode steps with f32
 * and int8 caches, and the confidence head. For each stage it also prints the number of
 * differing floats, the max |diff|, the cosine, and whether the argmax agrees. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nd_cact.h"
#include "nd_model.h"

static void *slurp(const char *dir, const char *name, size_t *n) {
  char path[1024];
  FILE *f;
  long l;
  void *p;
  snprintf(path, sizeof path, "%s%s%s", dir, name ? "/" : "", name ? name : "");
  f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  l = ftell(f);
  fseek(f, 0, SEEK_SET);
  p = malloc((size_t)l + 1);
  if (!p || fread(p, 1, (size_t)l, f) != (size_t)l) {
    fclose(f);
    free(p);
    return NULL;
  }
  fclose(f);
  ((char *)p)[l] = 0;
  *n = (size_t)l;
  return p;
}

static size_t argmax(const float *v, size_t n) {
  size_t i, b = 0;
  for (i = 1; i < n; i++)
    if (v[i] > v[b]) b = i;
  return b;
}

/* Compare `n` floats in rows of `row` (argmax per row). Returns 1 when bit-identical. */
static int check(const char *stage, const float *got, const float *want, size_t n, size_t row) {
  size_t i, diff = 0, r, am = 0;
  double dot = 0, ga = 0, wa = 0, maxd = 0;
  for (i = 0; i < n; i++) {
    uint32_t a, b;
    memcpy(&a, &got[i], 4);
    memcpy(&b, &want[i], 4);
    if (a != b) diff++;
    if (fabs((double)got[i] - (double)want[i]) > maxd) maxd = fabs((double)got[i] - (double)want[i]);
    dot += (double)got[i] * (double)want[i];
    ga += (double)got[i] * (double)got[i];
    wa += (double)want[i] * (double)want[i];
  }
  for (r = 0; row && r < n / row; r++)
    if (argmax(got + r * row, row) != argmax(want + r * row, row)) am++;
  printf("%-10s n=%-9zu differing=%-7zu max|d|=%.3g cos=%.9f argmax_mismatch=%zu %s\n", stage, n, diff, maxd,
         dot / (sqrt(ga) * sqrt(wa) + 1e-300), am, diff ? "FAIL" : "ok");
  return diff == 0;
}

int main(int argc, char **argv) {
  size_t mlen, n, seq, rows, steps, L, d, i, depth = 0;
  uint8_t *raw;
  uint32_t *ids, *fed;
  float *want, *got;
  nd_cact c;
  nd_model *m;
  nd_cache *cache;
  nd_err e = {""};
  int ok = 1;
  const char *dir;
  if (argc < 3) {
    fprintf(stderr, "usage: test_model MODEL TRACEDIR [DEPTH]\n");
    return 2;
  }
  dir = argv[2];
  if (argc > 3) depth = (size_t)strtoul(argv[3], NULL, 10);
  raw = slurp(argv[1], NULL, &mlen);
  if (!raw || nd_cact_parse(raw, mlen, &c, &e) || nd_model_load(&c, depth, &m, &e)) {
    fprintf(stderr, "load: %s\n", e.msg);
    return 1;
  }
  rows = nd_logit_rows(&m->cfg);
  L = m->cfg.num_layers;
  d = m->cfg.d_model;
  ids = slurp(dir, "ids.bin", &n);
  seq = n / 4;
  fed = slurp(dir, "steps.bin", &n);
  steps = n / 4;
  printf("seq=%zu rows=%zu layers=%zu steps=%zu\n", seq, rows, L, steps);

  want = slurp(dir, "cells.bin", &n);
  got = malloc(n);
  if (nd_forward_cells(m, ids, seq, got)) return 1;
  ok &= check("cells", got, want, n / 4, 0);
  free(want), free(got);

  want = slurp(dir, "seq.bin", &n);
  got = malloc(n);
  if (nd_forward_sequence(m, ids, seq, got)) return 1;
  ok &= check("sequence", got, want, n / 4, rows);
  free(want), free(got);

  {
    float *pre = slurp(dir, "prefill.bin", &n), *dec = slurp(dir, "decode.bin", &n);
    float *d8 = slurp(dir, "decode_i8.bin", &n);
    got = malloc((steps + 1) * rows * 4);
    cache = nd_cache_new(&m->cfg, seq + steps, ND_KV_F32);
    if (!cache || nd_prefill(m, ids, seq, cache, got)) return 1;
    ok &= check("prefill", got, pre, rows, rows);
    for (i = 0; i < steps; i++)
      if (nd_decode_step(m, cache, fed[i], got + i * rows)) return 1;
    ok &= check("decode", got, dec, steps * rows, rows);
    nd_cache_free(cache);
    cache = nd_cache_new(&m->cfg, seq + steps, ND_KV_INT8);
    if (!cache || nd_prefill(m, ids, seq, cache, got)) return 1;
    for (i = 0; i < steps; i++)
      if (nd_decode_step(m, cache, fed[i], got + (i + 1) * rows)) return 1;
    ok &= check("decode_i8", got, d8, (steps + 1) * rows, rows);
    nd_cache_free(cache);

    /* The tool-prefix cache's path: a prefill of ids[0..split) into a cache sized to it, a copy
     * (nd_cache_clone, as the engine stores and restores it), then decode steps over the rest and
     * the same continuation. It must land on Rust's one-prefill logits and decode steps. */
    {
      size_t splits[6], ns, k, prec;
      char name[32];
      splits[0] = 1, splits[1] = 63, splits[2] = 64, splits[3] = 65, splits[4] = seq / 2, splits[5] = seq - 1;
      for (prec = 0; prec < 2; prec++)
        for (k = 0; k < 6; k++) {
          size_t split = splits[k];
          nd_cache *part, *copy;
          float *ref = prec ? d8 : NULL;
          if (split == 0 || split >= seq) continue;
          part = nd_cache_new(&m->cfg, split, prec ? ND_KV_INT8 : ND_KV_F32);
          if (!part || nd_prefill(m, ids, split, part, got)) return 1;
          copy = nd_cache_clone(part);
          nd_cache_free(part);
          if (!copy) return 1;
          for (i = split; i < seq; i++)
            if (nd_decode_step(m, copy, ids[i], got)) return 1;
          for (ns = 0; ns < steps; ns++)
            if (nd_decode_step(m, copy, fed[ns], got + (ns + 1) * rows)) return 1;
          snprintf(name, sizeof name, "prefix%zu%s", split, prec ? "_i8" : "");
          if (prec) {
            ok &= check(name, got, ref, (steps + 1) * rows, rows);
          } else {
            ok &= check(name, got, pre, rows, rows) & check(name, got + rows, dec, steps * rows, rows);
          }
          nd_cache_free(copy);
        }
    }
    free(pre), free(dec), free(d8), free(got);
  }

  want = slurp(dir, "head.bin", &n);
  if (want && m->confidence) {
    got = malloc(n);
    if (nd_forward_head(m, m->confidence, ids, seq, got)) return 1;
    ok &= check("head", got, want, n / 4, 0);
    free(got);
  }
  free(want);
  free(ids), free(fed);
  nd_model_free(m);
  nd_cact_free(&c);
  free(raw);
  printf("%s\n", ok ? "PASS" : "FAIL");
  (void)L, (void)d;
  return ok ? 0 : 1;
}
