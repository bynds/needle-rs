/* test_cq: nd_cq against `needle-jibo dump-op` vectors of the real container.
 *   test_cq MODEL OPDIR...   (each OPDIR from dump-op with op=cq) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nd_cact.h"
#include "nd_cq.h"

static void *slurp(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  long l;
  void *p;
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  l = ftell(f);
  fseek(f, 0, SEEK_SET);
  p = malloc((size_t)l + 1);
  if (fread(p, 1, (size_t)l, f) != (size_t)l) l = 0;
  fclose(f);
  ((char *)p)[l] = 0;
  *n = (size_t)l;
  return p;
}

static long key(const char *txt, const char *k) {
  char pat[64];
  const char *p;
  snprintf(pat, sizeof pat, "\n%s=", k);
  p = strstr(txt, pat);
  return p ? strtol(p + strlen(pat), NULL, 10) : -1;
}

int main(int argc, char **argv) {
  size_t mlen, i;
  uint8_t *model;
  nd_cact c;
  nd_err e = {""};
  int fails = 0, a;
  if (argc < 3) return 2;
  model = slurp(argv[1], &mlen);
  if (!model || nd_cact_parse(model, mlen, &c, &e)) {
    fprintf(stderr, "parse: %s\n", e.msg);
    return 1;
  }
  for (a = 2; a < argc; a++) {
    char path[4096];
    size_t tl, xl, yl, blen, tokens, rec, n;
    char *txt;
    float *x, *want, *got, *acc;
    nd_cq w;
    const uint8_t *blob;
    long differ = 0;
    snprintf(path, sizeof path, "%s/op.txt", argv[a]);
    txt = slurp(path, &tl);
    rec = (size_t)key(txt, "record");
    tokens = (size_t)key(txt, "tokens");
    blob = nd_cact_blob(&c, rec, &blen);
    if (nd_cq_from_blob(blob, blen, c.recs[rec].shape[0], c.recs[rec].shape[1], c.recs[rec].group,
                        c.recs[rec].bits, c.codebook, &w, &e)) {
      fprintf(stderr, "%s: %s\n", argv[a], e.msg);
      return 1;
    }
    snprintf(path, sizeof path, "%s/x.bin", argv[a]);
    x = slurp(path, &xl);
    snprintf(path, sizeof path, "%s/y.bin", argv[a]);
    want = slurp(path, &yl);
    n = tokens * w.out_feat;
    got = calloc(n, sizeof(float));
    acc = calloc(tokens, sizeof(float));
    nd_cq_matmul_rows_prepared(&w, x, tokens, 0, w.out_feat, got, acc);
    for (i = 0; i < n; i++) differ += memcmp(&got[i], &want[i], 4) != 0;
    printf("%-40s matmul  %zu x %zu x %zu: %ld differ\n", argv[a], tokens, w.out_feat, w.in_feat, differ);
    fails += differ != 0;
    /* matvec per token must equal the batched product bit for bit */
    differ = 0;
    for (i = 0; i < tokens; i++) {
      size_t r;
      float *yv = calloc(w.out_feat, sizeof(float));
      nd_cq_matvec_rows_prepared(&w, x + i * w.in_padded, 0, w.out_feat, yv);
      for (r = 0; r < w.out_feat; r++) differ += memcmp(&yv[r], &want[i * w.out_feat + r], 4) != 0;
      free(yv);
    }
    printf("%-40s matvec: %ld differ\n", argv[a], differ);
    fails += differ != 0;
    free(txt); free(x); free(want); free(got); free(acc);
    nd_cq_free(&w);
  }
  nd_cact_free(&c);
  free(model);
  printf(fails ? "FAIL\n" : "ok\n");
  return fails != 0;
}
