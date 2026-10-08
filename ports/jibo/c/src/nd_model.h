/* nd_model.h: the Needle 3 model (needle-core/src/v3/, needle-infer/src/v3.rs).
 *
 * Weights stay packed (nd_cq); norms, gates, diagonals, Kronecker factors and permutations are
 * decoded to float at load. The forward pass is a transcription of V3Model::forward_impl_pooled
 * (batched prefill, optionally filling a cache and/or a probe-head pool) and decode_step, with
 * the cache of v3/cache.rs. Results are bit-identical to needle-core.
 */
#ifndef ND_MODEL_H
#define ND_MODEL_H

#include "nd_cact.h"
#include "nd_cq.h"

typedef struct {
  size_t vocab_size, out_vocab, d_model, num_heads, num_kv_heads, num_layers;
  size_t qk_head_dim, v_head_dim, max_seq_len, hada_n, mhc_lanes, sliding_window;
  float rope_theta;
  uint8_t is_global[64];
  size_t qkv_conv_taps;
  /* Engram */
  size_t orders[4], n_orders, heads, slots, sub_dim, conv_taps, conv_dilation, seed_heads;
  size_t sites[16], n_sites;
} nd_cfg;

typedef struct {
  float *d1, *d2, *b2, *d3, *d4;
  float *w1a, *w1b, *w2a, *w2b, *w3a, *w3b;
  float *cond_v, *cond_u; /* [d_model, rank], [rank, hada_n] */
  size_t cond_rank;
} nd_hmlp;

typedef struct {
  float *norm_in;
  nd_cq q_proj, k_proj, v_proj;
  float *q_taps, *k_taps, *v_taps; /* (taps, dim); NULL without conv */
  float *q_norm, *k_norm;
  nd_cq gate_proj, out_proj;
  float *post_norm;
  float attn_gate; /* pre-sigmoid */
  float *pre_hada;
  nd_hmlp mlp;
} nd_layer;

typedef struct {
  nd_cq tables, key_proj, value_proj;
  float *taps; /* (conv_taps, d_model) */
} nd_engram_site;

typedef struct {
  float *probes, *gain, *query, *row_bias, *proj, *bias;
  size_t cells, probes_per_cell, queries, out_dim, nbias;
} nd_head;

typedef struct {
  nd_cfg cfg;
  nd_cq embedding;
  nd_layer *layers;
  float *a_pre, *a_post, *a_res, *b_pre, *b_post, *b_res;
  nd_cq phi_pre, phi_post, phi_res;
  nd_engram_site *engrams;
  float *final_norm;
  uint32_t *p1, *p2;
  float embed_scale;
  nd_head *confidence; /* NULL when the container exports none */
} nd_model;

/* Load from a parsed container. depth 0 (or the container's own) is the full model; otherwise a
 * ladder rung (2..=num_layers), as needle-infer's model_from_cact_at_depth. */
int nd_model_load(const nd_cact *c, size_t depth, nd_model **out, nd_err *e);
void nd_model_free(nd_model *m);

static inline size_t nd_logit_rows(const nd_cfg *c) { return c->out_vocab ? c->out_vocab : c->vocab_size; }

/* ---- cache (v3/cache.rs) ---- */
typedef enum { ND_KV_F32 = 0, ND_KV_INT8 = 1 } nd_kv_precision;
typedef struct nd_cache nd_cache;
nd_cache *nd_cache_new(const nd_cfg *cfg, size_t hint, nd_kv_precision p);
void nd_cache_free(nd_cache *c);
size_t nd_cache_pos(const nd_cache *c);

/* Batched prefill filling `cache`; writes the last position's logits (nd_logit_rows floats).
 * Only the last position's head is computed (each row is independent, so this is the same bits
 * as the full batch). Returns ND_OK, or ND_E_NOMEM. tokens may not be empty. */
int nd_prefill(const nd_model *m, const uint32_t *tokens, size_t seq, nd_cache *cache, float *logits);

/* One decode step at the cache's position. logits: nd_logit_rows floats. */
int nd_decode_step(const nd_model *m, nd_cache *cache, uint32_t token, float *logits);

/* Logits for every position, (seq, rows) row-major, no cache (forward_sequence). */
int nd_forward_sequence(const nd_model *m, const uint32_t *tokens, size_t seq, float *logits);

/* Per-layer cells (seq, num_layers + 1, d_model) (forward_cells). */
int nd_forward_cells(const nd_model *m, const uint32_t *tokens, size_t seq, float *cells);

/* The probe head over a sequence, streamed (forward_head); out has head->out_dim floats. */
int nd_forward_head(const nd_model *m, const nd_head *head, const uint32_t *tokens, size_t seq, float *out);

#endif
