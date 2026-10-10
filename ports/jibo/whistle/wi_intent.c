/* wi_intent.c: see wi_intent.h. File format (little-endian): "WIM1", int32 classes, int32 dim,
 * then per class an int32 length and that many name bytes, then mean[dim], scale[dim],
 * w[classes][dim], b[classes] as float32. */
#include "wi_intent.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

int wi_load(const char *path, wi_model *m) {
  FILE *f = fopen(path, "rb");
  char magic[4];
  int32_t c, d, i;
  memset(m, 0, sizeof *m);
  if (!f) return -1;
  if (!rd(f, magic, 4) || memcmp(magic, "WIM1", 4) || !rd(f, &c, 4) || !rd(f, &d, 4) || c < 1 || c > 4096 || d < 1 ||
      d > 1 << 16)
    goto bad;
  m->classes = c, m->dim = d;
  m->names = calloc((size_t)c, sizeof *m->names);
  m->mean = malloc((size_t)d * sizeof(float));
  m->scale = malloc((size_t)d * sizeof(float));
  m->w = malloc((size_t)c * d * sizeof(float));
  m->b = malloc((size_t)c * sizeof(float));
  if (!m->names || !m->mean || !m->scale || !m->w || !m->b) goto bad;
  for (i = 0; i < c; i++) {
    int32_t n;
    if (!rd(f, &n, 4) || n < 0 || n > 4096 || !(m->names[i] = calloc((size_t)n + 1, 1)) || !rd(f, m->names[i], (size_t)n))
      goto bad;
  }
  if (!rd(f, m->mean, (size_t)d * 4) || !rd(f, m->scale, (size_t)d * 4) || !rd(f, m->w, (size_t)c * d * 4) ||
      !rd(f, m->b, (size_t)c * 4))
    goto bad;
  fclose(f);
  return 0;
bad:
  fclose(f);
  wi_free(m);
  return -1;
}

void wi_free(wi_model *m) {
  int i;
  if (m->names)
    for (i = 0; i < m->classes; i++) free(m->names[i]);
  free(m->names), free(m->mean), free(m->scale), free(m->w), free(m->b);
  memset(m, 0, sizeof *m);
}

int wi_classify(const wi_model *m, const float *emb, int frames, int width, float *prob) {
  float *x = malloc((size_t)m->dim * sizeof(float)), top = -INFINITY, sum = 0.0f;
  int t, j, k, best = 0;
  if (!x || m->dim != 3 * width || frames < 1) {
    free(x);
    return -1;
  }
  /* mean, max, population standard deviation over frames */
  for (j = 0; j < width; j++) {
    double s = 0.0, s2 = 0.0;
    float mx = emb[j];
    for (t = 0; t < frames; t++) {
      float v = emb[(size_t)t * width + j];
      s += v, s2 += (double)v * v;
      if (v > mx) mx = v;
    }
    s /= frames;
    s2 = s2 / frames - s * s;
    x[j] = (float)s, x[width + j] = mx, x[2 * width + j] = (float)sqrt(s2 > 0 ? s2 : 0);
  }
  for (j = 0; j < m->dim; j++) x[j] = (x[j] - m->mean[j]) / m->scale[j];
  for (k = 0; k < m->classes; k++) {
    const float *w = m->w + (size_t)k * m->dim;
    float z = m->b[k];
    for (j = 0; j < m->dim; j++) z += w[j] * x[j];
    prob[k] = z;
    if (z > top) top = z, best = k;
  }
  for (k = 0; k < m->classes; k++) sum += prob[k] = expf(prob[k] - top);
  for (k = 0; k < m->classes; k++) prob[k] /= sum;
  free(x);
  return best;
}
