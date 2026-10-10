/* embed MODEL MANIFEST: each manifest line "in.f32 out.f32" (16 kHz mono float32 PCM in);
   writes needle_embed's rows (frames x 512 float32) to out. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "needle.h"
static void *slurp(const char *p, long *n) {
  FILE *f = fopen(p, "rb"); void *b; if (!f) return NULL;
  fseek(f, 0, SEEK_END); *n = ftell(f); rewind(f); b = malloc(*n ? *n : 1);
  if (fread(b, 1, *n, f) != (size_t)*n) { fclose(f); free(b); return NULL; } fclose(f); return b;
}
int main(int argc, char **argv) {
  long n; unsigned char *m = slurp(argv[1], &n); char in[4096], out[4096]; FILE *man = fopen(argv[2], "r"); int done = 0;
  if (!m || needle_load(m, (unsigned long long)n) < 0) { fprintf(stderr, "load: %s\n", needle_last_error()); return 1; }
  while (fscanf(man, "%4095s %4095s", in, out) == 2) {
    long b; float *pcm = slurp(in, &b); int k, got; float *e; FILE *o;
    if (!pcm) { fprintf(stderr, "read %s\n", in); return 1; }
    k = needle_embed(NULL, pcm, (int)(b / 4), NULL, 0);
    if (k <= 0) { fprintf(stderr, "%s: embed size %d: %s\n", in, k, needle_last_error()); free(pcm); continue; }
    e = malloc(k * sizeof(float)); got = needle_embed(NULL, pcm, (int)(b / 4), e, k);
    o = fopen(out, "wb"); fwrite(e, sizeof(float), got, o); fclose(o); free(e); free(pcm);
    if (++done % 500 == 0) fprintf(stderr, "%d\n", done);
  }
  return 0;
}
