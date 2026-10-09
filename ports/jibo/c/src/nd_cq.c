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

#if ND_HAVE_NEON && defined(__GNUC__)
/* The NEON inner loops of needle-core's cq_neon.rs, instruction for instruction: post-increment
 * 32-byte loads into two q registers and two non-fused VMLA per eight lanes. GCC's code for the
 * same loops from arm_neon.h intrinsics split each load in two and kept extra counters (perfvm:
 * 11 instructions per eight lanes against 7, about 50% more for every CQ product). q8-q13 are
 * d16-d27, caller-saved in the AAPCS. Each needs n >= 1. */
static void neon_lanes8(const float *u, const float *x, size_t n, float out[8]) {
  __asm__ __volatile__(
      "vmov.i32 q12, #0\n\t"
      "vmov.i32 q13, #0\n\t"
      "1:\n\t"
      "vld1.32 {d16-d19}, [%[u]]!\n\t"
      "vld1.32 {d20-d23}, [%[x]]!\n\t"
      "vmla.f32 q12, q8, q10\n\t"
      "vmla.f32 q13, q9, q11\n\t"
      "subs %[n], %[n], #1\n\t"
      "bne 1b\n\t"
      "vst1.32 {d24-d27}, [%[o]]\n\t"
      : [u] "+r"(u), [x] "+r"(x), [n] "+r"(n)
      : [o] "r"(out)
      : "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory");
}

/* neon_lanes8 for two inputs against the same u (needle-core's lanes8x2): u is loaded once for
 * both positions, each position's lanes and order exactly as neon_lanes8 gives them. */
static void neon_lanes8x2(const float *u, const float *x0, const float *x1, size_t n, float out0[8], float out1[8]) {
  __asm__ __volatile__(
      "vmov.i32 q12, #0\n\t"
      "vmov.i32 q13, #0\n\t"
      "vmov.i32 q14, #0\n\t"
      "vmov.i32 q15, #0\n\t"
      "1:\n\t"
      "vld1.32 {d16-d19}, [%[u]]!\n\t"
      "vld1.32 {d20-d23}, [%[x0]]!\n\t"
      "vld1.32 {d0-d3}, [%[x1]]!\n\t"
      "vmla.f32 q12, q8, q10\n\t"
      "vmla.f32 q13, q9, q11\n\t"
      "vmla.f32 q14, q8, q0\n\t"
      "vmla.f32 q15, q9, q1\n\t"
      "subs %[n], %[n], #1\n\t"
      "bne 1b\n\t"
      "vst1.32 {d24-d27}, [%[o0]]\n\t"
      "vst1.32 {d28-d31}, [%[o1]]\n\t"
      : [u] "+r"(u), [x0] "+r"(x0), [x1] "+r"(x1), [n] "+r"(n)
      : [o0] "r"(out0), [o1] "r"(out1)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "d30", "d31", "cc", "memory");
}

/* neon_lanes8 for four inputs against the same u (needle-core's lanes8x4): u is loaded once for
 * four positions, each position's lanes and order exactly as neon_lanes8 gives them. The four
 * inputs share q2-q3 in turn; out is 32 floats, position-major. */
static void neon_lanes8x4(const float *u, const float *x0, const float *x1, const float *x2, const float *x3,
                          size_t n, float out[32]) {
  __asm__ __volatile__(
      "vmov.i32 q8, #0\n\t"
      "vmov.i32 q9, #0\n\t"
      "vmov.i32 q10, #0\n\t"
      "vmov.i32 q11, #0\n\t"
      "vmov.i32 q12, #0\n\t"
      "vmov.i32 q13, #0\n\t"
      "vmov.i32 q14, #0\n\t"
      "vmov.i32 q15, #0\n\t"
      "1:\n\t"
      "vld1.32 {d0-d3}, [%[u]]!\n\t"
      "vld1.32 {d4-d7}, [%[x0]]!\n\t"
      "vmla.f32 q8, q0, q2\n\t"
      "vmla.f32 q9, q1, q3\n\t"
      "vld1.32 {d4-d7}, [%[x1]]!\n\t"
      "vmla.f32 q10, q0, q2\n\t"
      "vmla.f32 q11, q1, q3\n\t"
      "vld1.32 {d4-d7}, [%[x2]]!\n\t"
      "vmla.f32 q12, q0, q2\n\t"
      "vmla.f32 q13, q1, q3\n\t"
      "vld1.32 {d4-d7}, [%[x3]]!\n\t"
      "vmla.f32 q14, q0, q2\n\t"
      "vmla.f32 q15, q1, q3\n\t"
      "subs %[n], %[n], #1\n\t"
      "bne 1b\n\t"
      "vst1.32 {d16-d19}, [%[o]]!\n\t"
      "vst1.32 {d20-d23}, [%[o]]!\n\t"
      "vst1.32 {d24-d27}, [%[o]]!\n\t"
      "vst1.32 {d28-d31}, [%[o]]\n\t"
      : [u] "+r"(u), [x0] "+r"(x0), [x1] "+r"(x1), [x2] "+r"(x2), [x3] "+r"(x3), [n] "+r"(n), [o] "+r"(out)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23",
        "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31", "cc", "memory");
}

static void neon_lut4_lanes(const float *lut, const uint8_t *g, const float *x, size_t pairs, float out[8]) {
  uint32_t t0, t1;
  __asm__ __volatile__(
      "vmov.i32 q12, #0\n\t"
      "vmov.i32 q13, #0\n\t"
      "1:\n\t"
      "ldrb %[t0], [%[g]], #1\n\t"
      "ldrb %[t1], [%[g]], #1\n\t"
      "add %[t0], %[lut], %[t0], lsl #4\n\t"
      "add %[t1], %[lut], %[t1], lsl #4\n\t"
      "vld1.32 {d16-d17}, [%[t0]]\n\t"
      "vld1.32 {d18-d19}, [%[t1]]\n\t"
      "vld1.32 {d20-d23}, [%[x]]!\n\t"
      "vmla.f32 q12, q8, q10\n\t"
      "vmla.f32 q13, q9, q11\n\t"
      "subs %[n], %[n], #1\n\t"
      "bne 1b\n\t"
      "vst1.32 {d24-d27}, [%[o]]\n\t"
      : [g] "+r"(g), [x] "+r"(x), [n] "+r"(pairs), [t0] "=&r"(t0), [t1] "=&r"(t1)
      : [lut] "r"(lut), [o] "r"(out)
      : "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory");
}

/* neon_lut4_lanes for two rows at once (needle-core's lut4_lanes2): each row keeps its own two
 * accumulators and lane order; the rows share the loads of x and the loop control. The second
 * row's levels use q0-q1 (d0-d3), caller-saved. */
static void neon_lut4_lanes2(const float *lut, const uint8_t *g0, const uint8_t *g1, const float *x, size_t pairs,
                             float out0[8], float out1[8]) {
  uint32_t t0, t1;
  __asm__ __volatile__(
      "vmov.i32 q12, #0\n\t"
      "vmov.i32 q13, #0\n\t"
      "vmov.i32 q14, #0\n\t"
      "vmov.i32 q15, #0\n\t"
      "1:\n\t"
      "ldrb %[t0], [%[g0]], #1\n\t"
      "ldrb %[t1], [%[g0]], #1\n\t"
      "add %[t0], %[lut], %[t0], lsl #4\n\t"
      "add %[t1], %[lut], %[t1], lsl #4\n\t"
      "vld1.32 {d16-d17}, [%[t0]]\n\t"
      "vld1.32 {d18-d19}, [%[t1]]\n\t"
      "ldrb %[t0], [%[g1]], #1\n\t"
      "ldrb %[t1], [%[g1]], #1\n\t"
      "add %[t0], %[lut], %[t0], lsl #4\n\t"
      "add %[t1], %[lut], %[t1], lsl #4\n\t"
      "vld1.32 {d0-d1}, [%[t0]]\n\t"
      "vld1.32 {d2-d3}, [%[t1]]\n\t"
      "vld1.32 {d20-d23}, [%[x]]!\n\t"
      "vmla.f32 q12, q8, q10\n\t"
      "vmla.f32 q13, q9, q11\n\t"
      "vmla.f32 q14, q0, q10\n\t"
      "vmla.f32 q15, q1, q11\n\t"
      "subs %[n], %[n], #1\n\t"
      "bne 1b\n\t"
      "vst1.32 {d24-d27}, [%[o0]]\n\t"
      "vst1.32 {d28-d31}, [%[o1]]\n\t"
      : [g0] "+r"(g0), [g1] "+r"(g1), [x] "+r"(x), [n] "+r"(pairs), [t0] "=&r"(t0), [t1] "=&r"(t1)
      : [lut] "r"(lut), [o0] "r"(out0), [o1] "r"(out1)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "d30", "d31", "cc", "memory");
}

/* neon_lut2_lanes for two rows at once (needle-core's lut2_lanes2). */
static void neon_lut2_lanes2(const float *lut, const uint8_t *g0, const uint8_t *g1, const float *x, size_t quads,
                             float out0[8], float out1[8]) {
  uint32_t t0, t1;
  __asm__ __volatile__(
      "vmov.i32 q12, #0\n\t"
      "vmov.i32 q13, #0\n\t"
      "vmov.i32 q14, #0\n\t"
      "vmov.i32 q15, #0\n\t"
      "1:\n\t"
      "ldrb %[t0], [%[g0]], #1\n\t"
      "ldrb %[t1], [%[g0]], #1\n\t"
      "add %[t0], %[lut], %[t0], lsl #3\n\t"
      "add %[t1], %[lut], %[t1], lsl #3\n\t"
      "vld1.32 {d16}, [%[t0]]\n\t"
      "vld1.32 {d17}, [%[t1]]\n\t"
      "ldrb %[t0], [%[g0]], #1\n\t"
      "ldrb %[t1], [%[g0]], #1\n\t"
      "add %[t0], %[lut], %[t0], lsl #3\n\t"
      "add %[t1], %[lut], %[t1], lsl #3\n\t"
      "vld1.32 {d18}, [%[t0]]\n\t"
      "vld1.32 {d19}, [%[t1]]\n\t"
      "ldrb %[t0], [%[g1]], #1\n\t"
      "ldrb %[t1], [%[g1]], #1\n\t"
      "add %[t0], %[lut], %[t0], lsl #3\n\t"
      "add %[t1], %[lut], %[t1], lsl #3\n\t"
      "vld1.32 {d0}, [%[t0]]\n\t"
      "vld1.32 {d1}, [%[t1]]\n\t"
      "ldrb %[t0], [%[g1]], #1\n\t"
      "ldrb %[t1], [%[g1]], #1\n\t"
      "add %[t0], %[lut], %[t0], lsl #3\n\t"
      "add %[t1], %[lut], %[t1], lsl #3\n\t"
      "vld1.32 {d2}, [%[t0]]\n\t"
      "vld1.32 {d3}, [%[t1]]\n\t"
      "vld1.32 {d20-d23}, [%[x]]!\n\t"
      "vmla.f32 q12, q8, q10\n\t"
      "vmla.f32 q13, q9, q11\n\t"
      "vmla.f32 q14, q0, q10\n\t"
      "vmla.f32 q15, q1, q11\n\t"
      "subs %[n], %[n], #1\n\t"
      "bne 1b\n\t"
      "vst1.32 {d24-d27}, [%[o0]]\n\t"
      "vst1.32 {d28-d31}, [%[o1]]\n\t"
      : [g0] "+r"(g0), [g1] "+r"(g1), [x] "+r"(x), [n] "+r"(quads), [t0] "=&r"(t0), [t1] "=&r"(t1)
      : [lut] "r"(lut), [o0] "r"(out0), [o1] "r"(out1)
      : "d0", "d1", "d2", "d3", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "d30", "d31", "cc", "memory");
}

static void neon_lut2_lanes(const float *lut, const uint8_t *g, const float *x, size_t quads, float out[8]) {
  uint32_t t0, t1;
  __asm__ __volatile__(
      "vmov.i32 q12, #0\n\t"
      "vmov.i32 q13, #0\n\t"
      "1:\n\t"
      "ldrb %[t0], [%[g]], #1\n\t"
      "ldrb %[t1], [%[g]], #1\n\t"
      "add %[t0], %[lut], %[t0], lsl #3\n\t"
      "add %[t1], %[lut], %[t1], lsl #3\n\t"
      "vld1.32 {d16}, [%[t0]]\n\t"
      "vld1.32 {d17}, [%[t1]]\n\t"
      "ldrb %[t0], [%[g]], #1\n\t"
      "ldrb %[t1], [%[g]], #1\n\t"
      "add %[t0], %[lut], %[t0], lsl #3\n\t"
      "add %[t1], %[lut], %[t1], lsl #3\n\t"
      "vld1.32 {d18}, [%[t0]]\n\t"
      "vld1.32 {d19}, [%[t1]]\n\t"
      "vld1.32 {d20-d23}, [%[x]]!\n\t"
      "vmla.f32 q12, q8, q10\n\t"
      "vmla.f32 q13, q9, q11\n\t"
      "subs %[n], %[n], #1\n\t"
      "bne 1b\n\t"
      "vst1.32 {d24-d27}, [%[o]]\n\t"
      : [g] "+r"(g), [x] "+r"(x), [n] "+r"(quads), [t0] "=&r"(t0), [t1] "=&r"(t1)
      : [lut] "r"(lut), [o] "r"(out)
      : "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory");
}

static void neon_axpy8(float *y, float a, const float *x, size_t chunks) {
  uint32_t ab; /* a's bits, through a core register as cq_neon.rs passes them */
  memcpy(&ab, &a, sizeof ab);
  __asm__ __volatile__(
      "vdup.32 q14, %[a]\n\t"
      "1:\n\t"
      "vld1.32 {d16-d19}, [%[x]]!\n\t"
      "vld1.32 {d20-d23}, [%[y]]\n\t"
      "vmla.f32 q10, q8, q14\n\t"
      "vmla.f32 q11, q9, q14\n\t"
      "vst1.32 {d20-d23}, [%[y]]!\n\t"
      "subs %[n], %[n], #1\n\t"
      "bne 1b\n\t"
      : [y] "+r"(y), [x] "+r"(x), [n] "+r"(chunks)
      : [a] "r"(ab)
      : "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d28", "d29", "cc", "memory");
}
#define ND_NEON_ASM 1
#else
#define ND_NEON_ASM 0
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

void nd_axpy(float *y, float a, const float *x, size_t n) {
  size_t i = 0;
#if ND_NEON_ASM
  if (n >= 8) {
    neon_axpy8(y, a, x, n / 8);
    i = n / 8 * 8;
  }
#else
  /* Unrolled by four: GCC at -O2 does not unroll, and the loop overhead was most of each step. */
  for (; i + 4 <= n; i += 4) {
    y[i] += a * x[i];
    y[i + 1] += a * x[i + 1];
    y[i + 2] += a * x[i + 2];
    y[i + 3] += a * x[i + 3];
  }
#endif
  for (; i < n; i++) y[i] += a * x[i];
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
    const float *lut = w->lut;
    /* The batched path decodes unscaled (norm 1, which rounds nothing): a plain LUT copy,
     * specialised by indices per byte so the copy unrolls. */
    if (norm == 1.0f && p == 4)
      for (bi = 0; bi < bpg; bi++) {
        const float *e = lut + gb[bi] * 4;
        float *o = buf + bi * 4;
        o[0] = e[0], o[1] = e[1], o[2] = e[2], o[3] = e[3];
      }
    else if (norm == 1.0f && p == 2)
      for (bi = 0; bi < bpg; bi++) {
        const float *e = lut + gb[bi] * 2;
        buf[bi * 2] = e[0], buf[bi * 2 + 1] = e[1];
      }
    else
      for (bi = 0; bi < bpg; bi++)
        for (k = 0; k < p; k++) buf[bi * p + k] = lut[gb[bi] * p + k] * norm;
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
  size_t chunks = len / ND_LANES, c;
  float lanes[ND_LANES];
#if ND_NEON_ASM
  if (chunks) {
    neon_lanes8(u, x, chunks, lanes);
  } else {
    for (c = 0; c < ND_LANES; c++) lanes[c] = 0.0f;
  }
#else
  /* Eight locals rather than a lane array, as in dot_group: GCC keeps them in registers. */
  float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f, l4 = 0.0f, l5 = 0.0f, l6 = 0.0f, l7 = 0.0f;
  for (c = 0; c < chunks; c++) {
    const float *uu = u + c * 8, *xx = x + c * 8;
    l0 += uu[0] * xx[0], l1 += uu[1] * xx[1], l2 += uu[2] * xx[2], l3 += uu[3] * xx[3];
    l4 += uu[4] * xx[4], l5 += uu[5] * xx[5], l6 += uu[6] * xx[6], l7 += uu[7] * xx[7];
  }
  lanes[0] = l0, lanes[1] = l1, lanes[2] = l2, lanes[3] = l3;
  lanes[4] = l4, lanes[5] = l5, lanes[6] = l6, lanes[7] = l7;
#endif
  return sum_lanes(lanes);
}

/* group_dot_lanes for two inputs against the same u, each exactly as group_dot_lanes computes
 * it; u is read once for both. */
static void group_dot_lanes2(const float *u, const float *x0, const float *x1, size_t len, float *s0, float *s1) {
  size_t chunks = len / ND_LANES, c;
  float l0[ND_LANES], l1[ND_LANES];
#if ND_NEON_ASM
  if (chunks) {
    neon_lanes8x2(u, x0, x1, chunks, l0, l1);
  } else {
    for (c = 0; c < ND_LANES; c++) l0[c] = l1[c] = 0.0f;
  }
#else
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
  float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f, b4 = 0.0f, b5 = 0.0f, b6 = 0.0f, b7 = 0.0f;
  for (c = 0; c < chunks; c++) {
    const float *uu = u + c * 8, *p = x0 + c * 8, *q = x1 + c * 8;
    a0 += uu[0] * p[0], a1 += uu[1] * p[1], a2 += uu[2] * p[2], a3 += uu[3] * p[3];
    a4 += uu[4] * p[4], a5 += uu[5] * p[5], a6 += uu[6] * p[6], a7 += uu[7] * p[7];
    b0 += uu[0] * q[0], b1 += uu[1] * q[1], b2 += uu[2] * q[2], b3 += uu[3] * q[3];
    b4 += uu[4] * q[4], b5 += uu[5] * q[5], b6 += uu[6] * q[6], b7 += uu[7] * q[7];
  }
  l0[0] = a0, l0[1] = a1, l0[2] = a2, l0[3] = a3, l0[4] = a4, l0[5] = a5, l0[6] = a6, l0[7] = a7;
  l1[0] = b0, l1[1] = b1, l1[2] = b2, l1[3] = b3, l1[4] = b4, l1[5] = b5, l1[6] = b6, l1[7] = b7;
#endif
  *s0 = sum_lanes(l0);
  *s1 = sum_lanes(l1);
}

static float group_dot_serial(const float *u, const float *x, size_t len) {
  float s = 0.0f;
  size_t i;
  for (i = 0; i < len; i++) s += u[i] * x[i];
  return s;
}

/* dot_group<P>: LUT decode fused with the 8-lane dot (the matvec path). Instantiated per P, as
 * Rust's const generic is: with P a constant the compiler unrolls the decode. */
#if ND_NEON_ASM
#define DOT_GROUP_LANES(P)                                                                     \
  if (full) {                                                                                  \
    if ((P) == 4)                                                                              \
      neon_lut4_lanes(lut, gbytes, gx, full / per_iter, lanes);                                \
    else                                                                                       \
      neon_lut2_lanes(lut, gbytes, gx, full / per_iter, lanes);                                \
    bi = full;                                                                                 \
  } else {                                                                                     \
    for (k = 0; k < ND_LANES; k++) lanes[k] = 0.0f;                                            \
  }
#else
/* Scalar: the eight lanes as eight locals, so the compiler keeps them in registers (an array
 * indexed in a loop measured ~17% slower on ARMv7 VFP), and the LUT entries for one step of eight
 * read straight into the products. Same products, same lane order: the same bits. */
#define DOT_GROUP_LANES(P)                                                                     \
  {                                                                                            \
    float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f, l4 = 0.0f, l5 = 0.0f, l6 = 0.0f, l7 = 0.0f; \
    for (; bi < full; bi += per_iter) {                                                        \
      const float *x8 = gx + bi * (P), *e0, *e1;                                               \
      if ((P) == 4) {                                                                          \
        e0 = lut + gbytes[bi] * 4;                                                             \
        e1 = lut + gbytes[bi + 1] * 4;                                                         \
        l0 += e0[0] * x8[0], l1 += e0[1] * x8[1], l2 += e0[2] * x8[2], l3 += e0[3] * x8[3];    \
        l4 += e1[0] * x8[4], l5 += e1[1] * x8[5], l6 += e1[2] * x8[6], l7 += e1[3] * x8[7];    \
      } else {                                                                                 \
        const float *e2 = lut + gbytes[bi + 2] * 2, *e3 = lut + gbytes[bi + 3] * 2;            \
        e0 = lut + gbytes[bi] * 2;                                                             \
        e1 = lut + gbytes[bi + 1] * 2;                                                         \
        l0 += e0[0] * x8[0], l1 += e0[1] * x8[1], l2 += e1[0] * x8[2], l3 += e1[1] * x8[3];    \
        l4 += e2[0] * x8[4], l5 += e2[1] * x8[5], l6 += e3[0] * x8[6], l7 += e3[1] * x8[7];    \
      }                                                                                        \
    }                                                                                          \
    lanes[0] = l0, lanes[1] = l1, lanes[2] = l2, lanes[3] = l3;                                \
    lanes[4] = l4, lanes[5] = l5, lanes[6] = l6, lanes[7] = l7;                                \
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
  size_t yi = 0, g;
#if ND_NEON_ASM
  /* Two rows per pass, each row's lanes and sums exactly as one row's. */
  if ((w->per_byte == 4 && w->group / 4 >= 2 && (w->group / 4) % 2 == 0) ||
      (w->per_byte == 2 && w->group / 2 >= 4 && (w->group / 2) % 4 == 0)) {
    size_t bpg = w->group / w->per_byte;
    for (; yi + 2 <= rows; yi += 2) {
      size_t o = row_start + yi;
      const uint8_t *r0 = w->packed + o * w->row_bytes, *r1 = r0 + w->row_bytes;
      const float *n0 = w->norms + o * w->num_groups, *n1 = n0 + w->num_groups;
      float t0 = 0.0f, t1 = 0.0f, l0[ND_LANES], l1[ND_LANES];
      for (g = 0; g < w->num_groups; g++) {
        if (w->per_byte == 4)
          neon_lut4_lanes2(w->lut, r0 + g * bpg, r1 + g * bpg, xh + g * w->group, bpg / 2, l0, l1);
        else
          neon_lut2_lanes2(w->lut, r0 + g * bpg, r1 + g * bpg, xh + g * w->group, bpg / 4, l0, l1);
        t0 += n0[g] * sum_lanes(l0);
        t1 += n1[g] * sum_lanes(l1);
      }
      y[yi] = t0;
      y[yi + 1] = t1;
    }
  }
#endif
  for (; yi < rows; yi++) {
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
      b = 0;
      /* Two positions per pass: the decoded group is read once for both, each position's lanes
       * and sums exactly as one position's (group widths are multiples of 8 here). */
#if ND_NEON_ASM
      /* Four positions per pass with NEON (16 q registers hold four positions' lanes). */
      if (lanewise && w->group % ND_LANES == 0 && w->group >= ND_LANES)
        for (; b + 4 <= batch; b += 4) {
          float l[4 * ND_LANES];
          const float *x = xh + b * stride + base;
          size_t q;
          neon_lanes8x4(ug, x, x + stride, x + 2 * stride, x + 3 * stride, w->group / ND_LANES, l);
          for (q = 0; q < 4; q++) acc[b + q] += norms[g] * sum_lanes(l + q * ND_LANES);
        }
#endif
      if (lanewise && w->group % ND_LANES == 0)
        for (; b + 2 <= batch; b += 2) {
          float s0, s1;
          group_dot_lanes2(ug, xh + b * stride + base, xh + (b + 1) * stride + base, w->group, &s0, &s1);
          acc[b] += norms[g] * s0;
          acc[b + 1] += norms[g] * s1;
        }
      for (; b < batch; b++) {
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
