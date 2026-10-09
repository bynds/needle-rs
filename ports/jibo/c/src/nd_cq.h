/* nd_cq.h: Cactus-Quants weights (needle-core/src/cq.rs, CqWeight).
 *
 * A logical [out, in] matrix stored as packed codebook indices (LSB-first, `bits` per index) and
 * FP16 per-group norms, in a rotated domain: each group of `group` inputs was multiplied by the
 * normalised Walsh-Hadamard matrix before quantisation, so a product needs the activation rotated
 * the same way (nd_cq_prepare) and no inverse transform of the weight.
 *
 * Every product keeps the Rust kernels' summation order: eight lanes per group (lane k
 * accumulates elements k, k+8, ...), the lanes summed in order from -0.0, then `total += norm *
 * group_sum` across groups. 3-bit records use a single sequential accumulator, as Rust's
 * matvec_generic. With ND_NEON (ARM NEON builds) the same lanes run in two q registers with
 * non-fused VMLA: the same bits.
 */
#ifndef ND_CQ_H
#define ND_CQ_H

#include "nd_common.h"

#define ND_CQ_MAX_GROUP 1024
#define ND_TERNARY_RECORD_BITS 5
#define ND_LANES 8

typedef struct {
  uint8_t *packed;   /* out_feat * row_bytes */
  float *norms;      /* out_feat * num_groups */
  float *lut;        /* 256 * per_byte, or NULL when per_byte == 0 */
  float levels[16];
  size_t nlevels;
  size_t per_byte, row_bytes, num_groups;
  size_t out_feat, in_feat, in_padded, group;
  uint8_t bits;
} nd_cq;

/* CqWeight::from_blob. `codebook` is the container's 28-float cb2|cb3|cb4 block. */
int nd_cq_from_blob(const uint8_t *blob, size_t blen, size_t out_feat, size_t in_feat, size_t group,
                    uint8_t bits, const float *codebook, nd_cq *w, nd_err *e);
void nd_cq_free(nd_cq *w);

/* CqWeight::select_rows: a new weight holding rows[0..n] of w. */
int nd_cq_select_rows(const nd_cq *w, const size_t *rows, size_t n, nd_cq *out, nd_err *e);

static inline size_t nd_cq_prepared_len(const nd_cq *w) { return w->in_padded; }

/* x[in_feat] -> xh[in_padded]: copy, zero-pad, normalised FWHT per group. */
void nd_cq_prepare(const nd_cq *w, const float *x, float *xh);

/* y[rows] = W[row_start .. row_start+rows] . xh */
void nd_cq_matvec_rows_prepared(const nd_cq *w, const float *xh, size_t row_start, size_t rows, float *y);

/* y[out_feat] = W . x, with its own rotation scratch (returns ND_E_NOMEM on allocation failure). */
int nd_cq_matvec(const nd_cq *w, const float *x, float *y);

/* Batched: xh is batch x in_padded, y is batch-major (y[b * rows + r]); acc has >= batch floats.
 * Bit-identical to batch calls of nd_cq_matvec_rows_prepared. */
void nd_cq_matmul_rows_prepared(const nd_cq *w, const float *xh, size_t batch, size_t row_start, size_t rows,
                                float *y, float *acc);

/* Row o of the dequantised matrix, in_feat floats (inverse rotation applied). */
void nd_cq_dequantize_row(const nd_cq *w, size_t o, float *out);

/* y[i] += a * x[i] for i < n: one product and one sum per element, in element order (any split
 * of the work is the same bits). NEON builds do eight elements per step with needle-core's axpy8
 * kernel; y and x must not overlap. */
void nd_axpy(float *y, float a, const float *x, size_t n);

/* The normalised Walsh-Hadamard transform in place, n a power of two (hadamard.rs). */
void nd_fwht_normalized(float *x, size_t n);

#endif
