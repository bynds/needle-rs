/* nd_cq.c: see nd_cq.h. Transcribes needle-core/src/cq.rs (CqWeight) and hadamard.rs. */
#include "nd_cq.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "nd_cact.h"

#if defined(ND_NEON) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#define ND_HAVE_NEON 1
#else
#define ND_HAVE_NEON 0
#endif

#define TERNARY_CENTROID 1.2240064f

static void fwht(float *x, size_t n) {
  size_t h = 1, i, j;
  while (h < n) {
    for (i = 0; i < n; i += h << 1) {
      for (j = i; j < i + h; j++) {
        float a = x[j], b = x[j + h];
        x[j] = a + b;
        x[j + h] = a - b;
      }
    }
    h <<= 1;
  }
}

void nd_fwht_normalized(float *x, size_t n) {
  size_t i;
  float inv;
  fwht(x, n);
  inv = 1.0f / sqrtf((float)n);
  for (i = 0; i < n; i++) x[i] *= inv;
}

static size_t packed_row_bytes(size_t in_padded, uint8_t bits) {
  size_t eff = bits == ND_TERNARY_RECORD_BITS ? 2 : bits;
  return in_padded * eff / 8;
}

static int is_pow2(size_t v) { return v && !(v & (v - 1)); }

int nd_cq_from_blob(const uint8_t *blob, size_t blen, size_t out_feat, size_t in_feat, size_t group,
                    uint8_t bits, const float *codebook, nd_cq *w, nd_err *e) {
  size_t in_padded, num_groups, row_bytes, n_packed, n_norms, n_norm_bytes, total, i, k;
  size_t width;
  memset(w, 0, sizeof *w);
  if (out_feat == 0 || in_feat == 0) {
    nd_seterr(e, "cq: empty shape");
    return ND_E_FORMAT;
  }
  if (group == 0 || group > ND_CQ_MAX_GROUP || group % 8 || !is_pow2(group)) {
    nd_seterr(e, "cq: bad group %zu", group);
    return ND_E_FORMAT;
  }
  if (bits == ND_TERNARY_RECORD_BITS) {
    float s = TERNARY_CENTROID / sqrtf((float)group);
    w->levels[0] = -s;
    w->levels[1] = 0.0f;
    w->levels[2] = s;
    w->nlevels = 3;
  } else if (bits == 2 || bits == 3 || bits == 4) {
    size_t off = bits == 2 ? 0 : bits == 3 ? 4 : 12;
    w->nlevels = (size_t)1 << bits;
    memcpy(w->levels, codebook + off, w->nlevels * sizeof(float));
  } else {
    nd_seterr(e, "cq: unsupported bits %u", bits);
    return ND_E_FORMAT;
  }
  /* Checked: the shapes come from the file. */
  if (!nd_mul_ok((in_feat + group - 1) / group, group, &in_padded) || in_padded > (size_t)-1 / 8) goto shortblob;
  num_groups = in_padded / group;
  row_bytes = packed_row_bytes(in_padded, bits);
  if (!nd_mul_ok(out_feat, row_bytes, &n_packed) || !nd_mul_ok(out_feat, num_groups, &n_norms) ||
      !nd_mul_ok(n_norms, 2, &n_norm_bytes) || !nd_add_ok(n_packed, n_norm_bytes, &total))
    goto shortblob;
  if (blen < total) goto shortblob;

  w->packed = malloc(n_packed ? n_packed : 1);
  w->norms = nd_calloc(n_norms, sizeof(float));
  if (!w->packed || !w->norms) {
    nd_cq_free(w);
    return ND_E_NOMEM;
  }
  memcpy(w->packed, blob, n_packed);
  for (i = 0; i < n_norms; i++) {
    const uint8_t *p = blob + n_packed + 2 * i;
    w->norms[i] = nd_f16_to_f32((uint16_t)(p[0] | p[1] << 8));
  }

  /* build_lut */
  width = bits == ND_TERNARY_RECORD_BITS ? 2 : bits;
  w->per_byte = width == 2 ? 4 : width == 4 ? 2 : 0;
  if (w->per_byte) {
    uint32_t mask = (1u << width) - 1;
    w->lut = nd_calloc(256 * w->per_byte, sizeof(float));
    if (!w->lut) {
      nd_cq_free(w);
      return ND_E_NOMEM;
    }
    for (i = 0; i < 256; i++) {
      for (k = 0; k < w->per_byte; k++) {
        uint32_t code = ((uint32_t)i >> (k * width)) & mask;
        size_t idx;
        if (bits == ND_TERNARY_RECORD_BITS)
          idx = code == 3 ? 0 : code == 0 ? 1 : code == 1 ? 2 : 1;
        else
          idx = code;
        w->lut[i * w->per_byte + k] = w->levels[idx];
      }
    }
  }
  w->row_bytes = row_bytes;
  w->num_groups = num_groups;
  w->out_feat = out_feat;
  w->in_feat = in_feat;
  w->in_padded = in_padded;
  w->group = group;
  w->bits = bits;
  return ND_OK;
shortblob:
  nd_seterr(e, "cq: blob of %zu bytes is short for [%zu, %zu] at %u bits", blen, out_feat, in_feat, bits);
  return ND_E_FORMAT;
}

void nd_cq_free(nd_cq *w) {
  free(w->packed);
  free(w->norms);
  free(w->lut);
  memset(w, 0, sizeof *w);
}

int nd_cq_select_rows(const nd_cq *w, const size_t *rows, size_t n, nd_cq *out, nd_err *e) {
  size_t i, pb, nb;
  *out = *w;
  out->packed = NULL;
  out->norms = NULL;
  out->lut = NULL;
  if (!nd_mul_ok(n, w->row_bytes, &pb) || !nd_mul_ok(n, w->num_groups, &nb)) return ND_E_BOUNDS;
  out->packed = malloc(pb ? pb : 1);
  out->norms = nd_calloc(nb, sizeof(float));
  out->lut = w->lut ? nd_calloc(256 * w->per_byte, sizeof(float)) : NULL;
  if (!out->packed || !out->norms || (w->lut && !out->lut)) {
    nd_cq_free(out);
    return ND_E_NOMEM;
  }
  if (w->lut) memcpy(out->lut, w->lut, 256 * w->per_byte * sizeof(float));
  for (i = 0; i < n; i++) {
    if (rows[i] >= w->out_feat) {
      nd_seterr(e, "cq: row %zu out of range", rows[i]);
      nd_cq_free(out);
      return ND_E_BOUNDS;
    }
    memcpy(out->packed + i * w->row_bytes, w->packed + rows[i] * w->row_bytes, w->row_bytes);
    memcpy(out->norms + i * w->num_groups, w->norms + rows[i] * w->num_groups, w->num_groups * sizeof(float));
  }
  out->out_feat = n;
  return ND_OK;
}

void nd_cq_prepare(const nd_cq *w, const float *x, float *xh) {
  size_t g;
  memcpy(xh, x, w->in_feat * sizeof(float));
  memset(xh + w->in_feat, 0, (w->in_padded - w->in_feat) * sizeof(float));
  for (g = 0; g < w->num_groups; g++) nd_fwht_normalized(xh + g * w->group, w->group);
}

static uint32_t read_bits(const uint8_t *row, size_t bit, size_t bits, uint32_t mask) {
  size_t byte = bit / 8, shift = bit % 8;
  uint32_t lo = row[byte];
  uint32_t hi = shift + bits > 8 ? row[byte + 1] : 0;
  return ((lo | hi << 8) >> shift) & mask;
}

/* decode_group: one group's levels (times norm) into buf. */
static void decode_group(const nd_cq *w, const uint8_t *row, size_t g, float norm, float *buf) {
  size_t k;
  if (w->per_byte) {
    size_t p = w->per_byte, bpg = w->group / p, bi;
    const uint8_t *gb = row + g * bpg;
    for (bi = 0; bi < bpg; bi++)
      for (k = 0; k < p; k++) buf[bi * p + k] = w->lut[gb[bi] * p + k] * norm;
  } else {
    size_t bits = w->bits;
    uint32_t mask = (1u << bits) - 1;
    for (k = 0; k < w->group; k++) buf[k] = w->levels[read_bits(row, (g * w->group + k) * bits, bits, mask)] * norm;
  }
}

/* Eight lanes over chunks of eight, summed in order from -0.0 (Rust `lanes.iter().sum()`). */
static float sum_lanes(const float *l) {
  float s = -0.0f;
  int k;
  for (k = 0; k < ND_LANES; k++) s += l[k];
  return s;
}

/* group_dot_lanes: a decoded group u against x. len is a multiple of 8 here (group >= 8). */
static float group_dot_lanes(const float *u, const float *x, size_t len) {
  size_t chunks = len / ND_LANES, c, k;
  float lanes[ND_LANES];
#if ND_HAVE_NEON
  float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
  for (c = 0; c < chunks; c++) {
    a0 = vmlaq_f32(a0, vld1q_f32(u + c * 8), vld1q_f32(x + c * 8));
    a1 = vmlaq_f32(a1, vld1q_f32(u + c * 8 + 4), vld1q_f32(x + c * 8 + 4));
  }
  vst1q_f32(lanes, a0);
  vst1q_f32(lanes + 4, a1);
  (void)k;
#else
  for (k = 0; k < ND_LANES; k++) lanes[k] = 0.0f;
  for (c = 0; c < chunks; c++)
    for (k = 0; k < ND_LANES; k++) lanes[k] += u[c * ND_LANES + k] * x[c * ND_LANES + k];
#endif
  return sum_lanes(lanes);
}

static float group_dot_serial(const float *u, const float *x, size_t len) {
  float s = 0.0f;
  size_t i;
  for (i = 0; i < len; i++) s += u[i] * x[i];
  return s;
}

/* dot_group<P>: LUT decode fused with the 8-lane dot (the matvec path). Instantiated per P, as
 * Rust's const generic is: with P a constant the compiler unrolls the decode. */
#if ND_HAVE_NEON
#define DOT_GROUP_LANES(P)                                                                     \
  float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);                                  \
  for (; bi < full; bi += per_iter) {                                                          \
    float32x4_t v0, v1;                                                                        \
    if ((P) == 4) {                                                                            \
      v0 = vld1q_f32(lut + gbytes[bi] * 4);                                                    \
      v1 = vld1q_f32(lut + gbytes[bi + 1] * 4);                                                \
    } else {                                                                                   \
      v0 = vcombine_f32(vld1_f32(lut + gbytes[bi] * 2), vld1_f32(lut + gbytes[bi + 1] * 2));   \
      v1 = vcombine_f32(vld1_f32(lut + gbytes[bi + 2] * 2), vld1_f32(lut + gbytes[bi + 3] * 2)); \
    }                                                                                          \
    a0 = vmlaq_f32(a0, v0, vld1q_f32(gx + bi * (P)));                                          \
    a1 = vmlaq_f32(a1, v1, vld1q_f32(gx + bi * (P) + 4));                                      \
  }                                                                                            \
  vst1q_f32(lanes, a0);                                                                        \
  vst1q_f32(lanes + 4, a1);
#else
#define DOT_GROUP_LANES(P)                                                                     \
  for (k = 0; k < ND_LANES; k++) lanes[k] = 0.0f;                                              \
  for (; bi < full; bi += per_iter) {                                                          \
    float vals[ND_LANES];                                                                      \
    size_t t;                                                                                  \
    for (t = 0; t < per_iter; t++)                                                             \
      for (k = 0; k < (P); k++) vals[t * (P) + k] = lut[gbytes[bi + t] * (P) + k];             \
    for (k = 0; k < ND_LANES; k++) lanes[k] += vals[k] * gx[bi * (P) + k];                     \
  }
#endif

#define DEFINE_DOT_GROUP(P)                                                                    \
  static float dot_group_##P(const float *lut, const uint8_t *gbytes, size_t nbytes,           \
                             const float *gx) {                                                \
    const size_t per_iter = ND_LANES / (P);                                                    \
    size_t full = nbytes - nbytes % per_iter, bi = 0, k;                                       \
    float lanes[ND_LANES], acc;                                                                \
    DOT_GROUP_LANES(P)                                                                         \
    acc = sum_lanes(lanes);                                                                    \
    for (; bi < nbytes; bi++)                                                                  \
      for (k = 0; k < (P); k++) acc += lut[gbytes[bi] * (P) + k] * gx[bi * (P) + k];           \
    return acc;                                                                                \
  }

DEFINE_DOT_GROUP(4)
DEFINE_DOT_GROUP(2)

void nd_cq_matvec_rows_prepared(const nd_cq *w, const float *xh, size_t row_start, size_t rows, float *y) {
  size_t yi, g;
  for (yi = 0; yi < rows; yi++) {
    size_t o = row_start + yi;
    const uint8_t *row = w->packed + o * w->row_bytes;
    const float *norms = w->norms + o * w->num_groups;
    float total = 0.0f;
    if (w->per_byte) {
      size_t bpg = w->group / w->per_byte;
      if (w->per_byte == 4)
        for (g = 0; g < w->num_groups; g++)
          total += norms[g] * dot_group_4(w->lut, row + g * bpg, bpg, xh + g * w->group);
      else
        for (g = 0; g < w->num_groups; g++)
          total += norms[g] * dot_group_2(w->lut, row + g * bpg, bpg, xh + g * w->group);
    } else {
      size_t bits = w->bits, k;
      uint32_t mask = (1u << bits) - 1;
      for (g = 0; g < w->num_groups; g++) {
        const float *gx = xh + g * w->group;
        float acc = 0.0f;
        for (k = 0; k < w->group; k++)
          acc += w->levels[read_bits(row, (g * w->group + k) * bits, bits, mask)] * gx[k];
        total += norms[g] * acc;
      }
    }
    y[yi] = total;
  }
}

int nd_cq_matvec(const nd_cq *w, const float *x, float *y) {
  float *xh = nd_calloc(w->in_padded, sizeof(float));
  if (!xh) return ND_E_NOMEM;
  nd_cq_prepare(w, x, xh);
  nd_cq_matvec_rows_prepared(w, xh, 0, w->out_feat, y);
  free(xh);
  return ND_OK;
}

void nd_cq_matmul_rows_prepared(const nd_cq *w, const float *xh, size_t batch, size_t row_start, size_t rows,
                                float *y, float *acc) {
  float ug[ND_CQ_MAX_GROUP];
  size_t r, g, b, stride = w->in_padded;
  int lanewise = w->per_byte != 0;
  for (r = 0; r < rows; r++) {
    size_t o = row_start + r;
    const uint8_t *row = w->packed + o * w->row_bytes;
    const float *norms = w->norms + o * w->num_groups;
    for (b = 0; b < batch; b++) acc[b] = 0.0f;
    for (g = 0; g < w->num_groups; g++) {
      size_t base = g * w->group;
      decode_group(w, row, g, 1.0f, ug);
      for (b = 0; b < batch; b++) {
        const float *gx = xh + b * stride + base;
        float s = lanewise ? group_dot_lanes(ug, gx, w->group) : group_dot_serial(ug, gx, w->group);
        acc[b] += norms[g] * s;
      }
    }
    for (b = 0; b < batch; b++) y[b * rows + r] = acc[b];
  }
}

void nd_cq_dequantize_row(const nd_cq *w, size_t o, float *out) {
  float buf[ND_CQ_MAX_GROUP];
  const uint8_t *row = w->packed + o * w->row_bytes;
  const float *norms = w->norms + o * w->num_groups;
  size_t g, k;
  for (g = 0; g < w->num_groups; g++) {
    size_t base = g * w->group;
    decode_group(w, row, g, norms[g], buf);
    nd_fwht_normalized(buf, w->group);
    for (k = 0; k < w->group; k++)
      if (base + k < w->in_feat) out[base + k] = buf[k];
  }
}
