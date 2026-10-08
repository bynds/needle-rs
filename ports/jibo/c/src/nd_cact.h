/* nd_cact.h: the Needle 3 `.cact` container (needle-infer/src/cact.rs, CactV3).
 *
 * 49 u32 header words (tag 0x05E12A84, geometry, rope_theta as f32), the cb2|cb3|cb4 codebook
 * (28 f32), num_tensors 44-byte directory records, then 64-byte aligned blobs. Every geometry
 * field and every record is bounds-checked against the file before anything is sized from it
 * (CactV3Geometry::check_bounds on the Rust side).
 */
#ifndef ND_CACT_H
#define ND_CACT_H

#include "nd_common.h"

#define ND_TAG_V3 0x05E12A84u
#define ND_HEADER_BYTES 196
#define ND_REC_BYTES 44
#define ND_CODEBOOK_LEN 28

#define ND_DT_FP16 1
#define ND_DT_FP32 2
#define ND_DT_CQ 3
#define ND_DT_RAW 4

typedef struct {
  uint8_t dtype, ndim;
  size_t shape[4];
  uint64_t offset, nbytes;
  size_t group;
  uint8_t bits;
} nd_record;

typedef struct {
  size_t num_tensors, codebook_len, kv_window;
  uint32_t kv_bits;
  size_t vocab_size, out_vocab, d_model, num_heads, num_kv_heads, num_layers;
  size_t qk_head_dim, v_head_dim, max_seq_len, hada_n, mhc_lanes, sliding_window;
  uint64_t global_mask;
  size_t qkv_conv_taps, engram_slots, engram_sub_dim, num_engram_tables;
  size_t engram_conv_taps, engram_conv_dilation, engram_seed_heads;
  size_t engram_orders[4], n_orders;
  size_t engram_sites[16], n_sites;
  float rope_theta;
} nd_geom;

typedef struct {
  const uint8_t *raw; /* not owned */
  size_t len;
  nd_geom g;
  float codebook[ND_CODEBOOK_LEN];
  nd_record *recs;
  size_t nrec;
} nd_cact;

/* Parse `raw` (kept, not copied: it must outlive the nd_cact). */
int nd_cact_parse(const uint8_t *raw, size_t len, nd_cact *out, nd_err *e);
void nd_cact_free(nd_cact *c);

/* The blob of record i. */
const uint8_t *nd_cact_blob(const nd_cact *c, size_t i, size_t *len);

/* Decode an FP16 or FP32 record into a new float array of exactly `want` elements (refused if
 * the record holds a different count). want == 0 accepts any count and reports it in *n. */
int nd_cact_floats(const nd_cact *c, size_t i, size_t want, float **out, size_t *n, nd_err *e);

/* IEEE half to float, as needle-core's math::f16_to_f32. */
float nd_f16_to_f32(uint16_t bits);

#endif
