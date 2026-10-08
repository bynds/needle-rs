/* nd_load.c: building an nd_model from a container. Transcribes needle-infer/src/v3.rs
 * (config_from_geometry, V3Layout::derive and validate, layer_from_layout, model_from_cact,
 * model_from_cact_at_depth, heads_from_cact, confidence_head(_at_depth)) and
 * needle-core/src/v3/ladder.rs.
 *
 * Stricter than the Rust loader in one way, deliberately: every float tensor's length and every
 * permutation value is checked against the geometry here, because the C forward pass, unlike
 * Rust's, has no bounds-checked slices to stop a short tensor. A valid container loads the same.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "nd_model.h"

#define HEAD_CONFIDENCE 2
#define HEAD_ROUTER 3

/* ---- ladder.rs ---- */
static size_t ladder_order(size_t L, size_t *order) {
  size_t sel[64], ns = 0, no = 0, i, j;
  if (L == 0) return 0;
  if (L == 1) {
    order[0] = 0;
    return 1;
  }
  sel[ns++] = 0;
  sel[ns++] = L - 1;
  order[no++] = 0;
  order[no++] = L - 1;
  while (no < L) {
    size_t bg = 0, bl = 0, br = 0;
    int have = 0;
    /* sort_unstable on distinct values == sort */
    for (i = 1; i < ns; i++)
      for (j = i; j > 0 && sel[j - 1] > sel[j]; j--) {
        size_t t = sel[j];
        sel[j] = sel[j - 1];
        sel[j - 1] = t;
      }
    for (i = 0; i + 1 < ns; i++) {
      size_t left = sel[i], right = sel[i + 1], gap;
      if (right - left <= 1) continue;
      gap = right - left;
      if (!have || gap > bg || (gap == bg && left < bl)) {
        have = 1;
        bg = gap;
        bl = left;
        br = right;
      }
    }
    if (!have) break;
    sel[ns++] = (bl + br) / 2;
    order[no++] = (bl + br) / 2;
  }
  return no;
}

/* ladder_layer_indices: the kept blocks, sorted. 0 when depth is outside 2..=L. */
static size_t ladder_kept(size_t L, size_t depth, size_t *kept) {
  size_t order[64], n, i, j;
  if (depth < 2 || depth > L || L > 64) return 0;
  n = ladder_order(L, order);
  if (n < depth) return 0;
  for (i = 0; i < depth; i++) kept[i] = order[i];
  for (i = 1; i < depth; i++)
    for (j = i; j > 0 && kept[j - 1] > kept[j]; j--) {
      size_t t = kept[j];
      kept[j] = kept[j - 1];
      kept[j - 1] = t;
    }
  return depth;
}

/* ---- config_from_geometry ---- */
static int config_from_geometry(const nd_geom *g, nd_cfg *c, nd_err *e) {
  size_t i, heads;
  memset(c, 0, sizeof *c);
  if (g->n_orders == 0) {
    nd_seterr(e, "container declares no engram orders");
    return ND_E_FORMAT;
  }
  if (g->num_engram_tables % g->n_orders) {
    nd_seterr(e, "engram table count %zu not divisible by %zu orders", g->num_engram_tables, g->n_orders);
    return ND_E_FORMAT;
  }
  heads = g->num_engram_tables / g->n_orders;
  if (heads == 0) {
    nd_seterr(e, "container declares no engram tables");
    return ND_E_FORMAT;
  }
  if (g->d_model / (g->n_orders * heads) != g->engram_sub_dim) {
    nd_seterr(e, "engram sub_dim %zu does not match the geometry", g->engram_sub_dim);
    return ND_E_FORMAT;
  }
  if (g->num_kv_heads == 0 || g->num_heads % g->num_kv_heads) {
    nd_seterr(e, "heads %zu not divisible by kv heads %zu", g->num_heads, g->num_kv_heads);
    return ND_E_FORMAT;
  }
  c->vocab_size = g->vocab_size;
  c->out_vocab = g->out_vocab;
  c->d_model = g->d_model;
  c->num_heads = g->num_heads;
  c->num_kv_heads = g->num_kv_heads;
  c->num_layers = g->num_layers;
  c->qk_head_dim = g->qk_head_dim;
  c->v_head_dim = g->v_head_dim;
  c->max_seq_len = g->max_seq_len;
  c->hada_n = g->hada_n;
  c->mhc_lanes = g->mhc_lanes;
  c->rope_theta = g->rope_theta;
  c->sliding_window = g->sliding_window;
  for (i = 0; i < g->num_layers && i < 64; i++) c->is_global[i] = (uint8_t)(g->global_mask >> i & 1);
  c->qkv_conv_taps = g->qkv_conv_taps;
  c->n_orders = g->n_orders;
  memcpy(c->orders, g->engram_orders, sizeof c->orders);
  c->heads = heads;
  c->slots = g->engram_slots;
  c->sub_dim = g->engram_sub_dim;
  c->conv_taps = g->engram_conv_taps;
  c->conv_dilation = g->engram_conv_dilation;
  c->seed_heads = g->engram_seed_heads;
  c->n_sites = g->n_sites;
  memcpy(c->sites, g->engram_sites, sizeof c->sites);
  return ND_OK;
}

/* ---- the canon walk (V3Layout::derive) ---- */
typedef struct {
  size_t norm_in, q, k, v, taps[3], q_norm, k_norm, gate, out, post_norm, attn_gate, pre_hada, mlp[13];
  int has_taps;
} layer_idx;

typedef struct {
  size_t embedding;
  layer_idx *layers;
  size_t scalars[6], phi[3], perms[2];
  size_t (*engrams)[4];
  size_t final_norm;
  int has_manifest;
  size_t manifest, head_lo, head_hi;
} layout;

static void layout_free(layout *l) {
  free(l->layers);
  free(l->engrams);
}

static int want_shape(const nd_cact *c, size_t idx, const char *what, size_t r0, size_t r1, nd_err *e) {
  const nd_record *r = &c->recs[idx];
  if (r->shape[0] != r0 || r->shape[1] != r1) {
    nd_seterr(e, "tensor %zu (%s): shape [%zu, %zu] does not match geometry [%zu, %zu]", idx, what, r->shape[0],
              r->shape[1], r0, r1);
    return ND_E_FORMAT;
  }
  return ND_OK;
}

static int derive(const nd_cact *c, const nd_cfg *cfg, layout *lo, nd_err *e) {
  size_t n = c->nrec, cur = 0, li, s, k, tok = n;
  size_t qd = cfg->num_heads * cfg->qk_head_dim, kd = cfg->num_kv_heads * cfg->qk_head_dim;
  size_t vd = cfg->num_kv_heads * cfg->v_head_dim, od = cfg->num_heads * cfg->v_head_dim;
  int rc;
#define TAKE(dst)                                                                                \
  do {                                                                                           \
    if (cur >= n) {                                                                              \
      nd_seterr(e, "the canon for this geometry needs more tensors than the directory's %zu", n); \
      return ND_E_FORMAT;                                                                        \
    }                                                                                            \
    (dst) = cur++;                                                                               \
  } while (0)
  memset(lo, 0, sizeof *lo);
  lo->layers = nd_calloc(cfg->num_layers, sizeof(layer_idx));
  lo->engrams = nd_calloc(cfg->n_sites, sizeof *lo->engrams);
  if (!lo->layers || !lo->engrams) return ND_E_NOMEM;
  TAKE(lo->embedding);
  for (li = 0; li < cfg->num_layers; li++) {
    layer_idx *l = &lo->layers[li];
    TAKE(l->norm_in);
    TAKE(l->q);
    TAKE(l->k);
    TAKE(l->v);
    if (cfg->qkv_conv_taps > 0) {
      l->has_taps = 1;
      TAKE(l->taps[0]);
      TAKE(l->taps[1]);
      TAKE(l->taps[2]);
    }
    TAKE(l->q_norm);
    TAKE(l->k_norm);
    TAKE(l->gate);
    TAKE(l->out);
    TAKE(l->post_norm);
    TAKE(l->attn_gate);
    TAKE(l->pre_hada);
    for (k = 0; k < 13; k++) TAKE(l->mlp[k]);
  }
  for (k = 0; k < 6; k++) TAKE(lo->scalars[k]);
  for (k = 0; k < 3; k++) TAKE(lo->phi[k]);
  TAKE(lo->perms[0]);
  TAKE(lo->perms[1]);
  for (s = 0; s < cfg->n_sites; s++)
    for (k = 0; k < 4; k++) TAKE(lo->engrams[s][k]);
  TAKE(lo->final_norm);
  for (k = 0; k < n; k++)
    if (c->recs[k].dtype == ND_DT_RAW) {
      tok = k;
      break;
    }
  if (cur < tok) {
    lo->has_manifest = 1;
    TAKE(lo->manifest);
  }
  lo->head_lo = cur;
  lo->head_hi = tok > cur ? tok : cur;
#undef TAKE
  /* validate */
  if ((rc = want_shape(c, lo->embedding, "embedding", cfg->vocab_size, cfg->d_model, e))) return rc;
  for (li = 0; li < cfg->num_layers; li++) {
    layer_idx *l = &lo->layers[li];
    if ((rc = want_shape(c, l->q, "q_proj", qd, cfg->d_model, e)) || (rc = want_shape(c, l->k, "k_proj", kd, cfg->d_model, e)) ||
        (rc = want_shape(c, l->v, "v_proj", vd, cfg->d_model, e)) ||
        (rc = want_shape(c, l->gate, "gate_proj", od, cfg->d_model, e)) ||
        (rc = want_shape(c, l->out, "out_proj", cfg->d_model, od, e)))
      return rc;
  }
  for (s = 0; s < cfg->n_sites; s++) {
    if ((rc = want_shape(c, lo->engrams[s][0], "engram tables", cfg->n_orders * cfg->heads * cfg->slots, cfg->sub_dim, e)) ||
        (rc = want_shape(c, lo->engrams[s][1], "engram key_proj", cfg->d_model, cfg->d_model, e)) ||
        (rc = want_shape(c, lo->engrams[s][2], "engram value_proj", cfg->d_model, cfg->d_model, e)))
      return rc;
  }
  return ND_OK;
}

/* ---- loading ---- */
static int load_cq(const nd_cact *c, size_t idx, nd_cq *w, nd_err *e) {
  const nd_record *r = &c->recs[idx];
  size_t blen;
  const uint8_t *b;
  if (r->dtype != ND_DT_CQ) {
    nd_seterr(e, "tensor %zu has dtype %u, want %d", idx, r->dtype, ND_DT_CQ);
    return ND_E_FORMAT;
  }
  b = nd_cact_blob(c, idx, &blen);
  return nd_cq_from_blob(b, blen, r->shape[0], r->shape[1], r->group, r->bits, c->codebook, w, e);
}

static void hada_blocks(size_t n, size_t *ba, size_t *bb) {
  size_t bits = 0, v = n - 1;
  while (v) {
    bits++;
    v >>= 1;
  }
  *ba = (size_t)1 << (bits / 2);
  *bb = n / *ba;
}

static int load_layer(const nd_cact *c, const nd_cfg *cfg, const layer_idx *l, nd_layer *out, nd_err *e) {
  size_t d = cfg->d_model, qk = cfg->qk_head_dim, hn = cfg->hada_n, ba, bb, gate_n;
  size_t qd = cfg->num_heads * qk, kd = cfg->num_kv_heads * qk, vd = cfg->num_kv_heads * cfg->v_head_dim;
  float *gate;
  int rc;
  const nd_record *cv;
  memset(out, 0, sizeof *out);
  hada_blocks(hn, &ba, &bb);
#define F(dst, idx, n) \
  if ((rc = nd_cact_floats(c, (idx), (n), &(dst), NULL, e))) return rc
  F(out->norm_in, l->norm_in, d);
  if ((rc = load_cq(c, l->q, &out->q_proj, e)) || (rc = load_cq(c, l->k, &out->k_proj, e)) ||
      (rc = load_cq(c, l->v, &out->v_proj, e)))
    return rc;
  if (l->has_taps) {
    F(out->q_taps, l->taps[0], cfg->qkv_conv_taps * qd);
    F(out->k_taps, l->taps[1], cfg->qkv_conv_taps * kd);
    F(out->v_taps, l->taps[2], cfg->qkv_conv_taps * vd);
  }
  F(out->q_norm, l->q_norm, qk);
  F(out->k_norm, l->k_norm, qk);
  if ((rc = load_cq(c, l->gate, &out->gate_proj, e)) || (rc = load_cq(c, l->out, &out->out_proj, e))) return rc;
  F(out->post_norm, l->post_norm, d);
  if ((rc = nd_cact_floats(c, l->attn_gate, 0, &gate, &gate_n, e))) return rc;
  if (gate_n == 0) {
    free(gate);
    nd_seterr(e, "attn_gate is empty");
    return ND_E_FORMAT;
  }
  out->attn_gate = gate[0];
  free(gate);
  F(out->pre_hada, l->pre_hada, d);
  F(out->mlp.d1, l->mlp[0], hn);
  F(out->mlp.d2, l->mlp[1], hn);
  F(out->mlp.b2, l->mlp[2], hn);
  F(out->mlp.d3, l->mlp[3], hn);
  F(out->mlp.d4, l->mlp[4], hn);
  F(out->mlp.w1a, l->mlp[5], ba * ba);
  F(out->mlp.w1b, l->mlp[6], bb * bb);
  F(out->mlp.w2a, l->mlp[7], ba * ba);
  F(out->mlp.w2b, l->mlp[8], bb * bb);
  F(out->mlp.w3a, l->mlp[9], ba * ba);
  F(out->mlp.w3b, l->mlp[10], bb * bb);
  cv = &c->recs[l->mlp[11]];
  out->mlp.cond_rank = cv->shape[1];
  if (cv->shape[0] != d || out->mlp.cond_rank == 0 || out->mlp.cond_rank > 1024) {
    nd_seterr(e, "cond_v shape [%zu, %zu] does not fit d_model %zu", cv->shape[0], cv->shape[1], d);
    return ND_E_FORMAT;
  }
  F(out->mlp.cond_v, l->mlp[11], d * out->mlp.cond_rank);
  F(out->mlp.cond_u, l->mlp[12], out->mlp.cond_rank * hn);
#undef F
  return ND_OK;
}

static void free_layer(nd_layer *l) {
  float **f[] = {&l->norm_in, &l->q_taps, &l->k_taps, &l->v_taps, &l->q_norm, &l->k_norm, &l->post_norm,
                 &l->pre_hada, &l->mlp.d1, &l->mlp.d2, &l->mlp.b2, &l->mlp.d3, &l->mlp.d4, &l->mlp.w1a,
                 &l->mlp.w1b, &l->mlp.w2a, &l->mlp.w2b, &l->mlp.w3a, &l->mlp.w3b, &l->mlp.cond_v, &l->mlp.cond_u};
  size_t i;
  for (i = 0; i < sizeof f / sizeof f[0]; i++) free(*f[i]);
  nd_cq_free(&l->q_proj);
  nd_cq_free(&l->k_proj);
  nd_cq_free(&l->v_proj);
  nd_cq_free(&l->gate_proj);
  nd_cq_free(&l->out_proj);
}

static void free_head(nd_head *h) {
  if (!h) return;
  free(h->probes);
  free(h->gain);
  free(h->query);
  free(h->row_bias);
  free(h->proj);
  free(h->bias);
  free(h);
}

void nd_model_free(nd_model *m) {
  size_t i;
  if (!m) return;
  nd_cq_free(&m->embedding);
  if (m->layers)
    for (i = 0; i < m->cfg.num_layers; i++) free_layer(&m->layers[i]);
  free(m->layers);
  free(m->a_pre);
  free(m->a_post);
  free(m->a_res);
  free(m->b_pre);
  free(m->b_post);
  free(m->b_res);
  nd_cq_free(&m->phi_pre);
  nd_cq_free(&m->phi_post);
  nd_cq_free(&m->phi_res);
  if (m->engrams)
    for (i = 0; i < m->cfg.n_sites; i++) {
      nd_cq_free(&m->engrams[i].tables);
      nd_cq_free(&m->engrams[i].key_proj);
      nd_cq_free(&m->engrams[i].value_proj);
      free(m->engrams[i].taps);
    }
  free(m->engrams);
  free(m->final_norm);
  free(m->p1);
  free(m->p2);
  free_head(m->confidence);
  free(m);
}

/* A CQ head matrix, dequantised (needle-infer `dense`). */
static int dense(const nd_cact *c, size_t idx, float **out, size_t *rows, size_t *cols, nd_err *e) {
  nd_cq w;
  size_t o, n;
  int rc = load_cq(c, idx, &w, e);
  if (rc) return rc;
  if (!nd_mul_ok(w.out_feat, w.in_feat, &n) || !(*out = nd_calloc(n, sizeof(float)))) {
    nd_cq_free(&w);
    return ND_E_NOMEM;
  }
  for (o = 0; o < w.out_feat; o++) nd_cq_dequantize_row(&w, o, *out + o * w.in_feat);
  *rows = w.out_feat;
  *cols = w.in_feat;
  nd_cq_free(&w);
  return ND_OK;
}

/* heads_from_cact, keeping only the confidence head. */
static int load_confidence(const nd_cact *c, const nd_cfg *cfg, const layout *lo, nd_head **out, nd_err *e) {
  float *codes = NULL;
  size_t ncodes, ci, it = lo->head_lo;
  int rc;
  *out = NULL;
  if (!lo->has_manifest) return ND_OK;
  if ((rc = nd_cact_floats(c, lo->manifest, 0, &codes, &ncodes, e))) return rc;
  for (ci = 0; ci < ncodes; ci++) {
    /* Rust's `c as u8` saturates (NaN to 0); a C cast of an out-of-range float is undefined. */
    float cf = codes[ci];
    int code = !(cf > 0.0f) ? 0 : cf >= 255.0f ? 255 : (int)(uint8_t)cf;
    size_t idx[6], k, need = code == HEAD_ROUTER ? 7 : 6, prow, pcol, qrow, qcol, orow, ocol, cells, gain_n;
    nd_head *h;
    if (it + need > lo->head_hi) {
      free(codes);
      nd_seterr(e, "head tensors ran out");
      return ND_E_FORMAT;
    }
    for (k = 0; k < 6; k++) idx[k] = it + k;
    it += need;
    /* Walk every head, as Rust does (it fails if any runs out of tensors); load the first
     * confidence head only. */
    if (code != HEAD_CONFIDENCE || *out) continue;
    h = calloc(1, sizeof *h);
    if (!h) {
      free(codes);
      return ND_E_NOMEM;
    }
    cells = cfg->num_layers + 1;
    if ((rc = dense(c, idx[0], &h->probes, &prow, &pcol, e)) || (rc = nd_cact_floats(c, idx[1], 0, &h->gain, &gain_n, e)) ||
        (rc = dense(c, idx[2], &h->query, &qrow, &qcol, e)) || (rc = dense(c, idx[4], &h->proj, &orow, &ocol, e)) ||
        (rc = nd_cact_floats(c, idx[5], 0, &h->bias, &h->nbias, e))) {
      free_head(h);
      free(codes);
      return rc;
    }
    if (prow % cells || gain_n != prow || pcol != cfg->d_model || qcol != cfg->d_model ||
        ocol != qrow * cfg->d_model) {
      free_head(h);
      free(codes);
      nd_seterr(e, "probe head shapes disagree with the geometry");
      return ND_E_FORMAT;
    }
    h->cells = cells;
    h->probes_per_cell = prow / cells;
    h->queries = qrow;
    h->out_dim = orow;
    if ((rc = nd_cact_floats(c, idx[3], qrow * prow, &h->row_bias, NULL, e))) {
      free_head(h);
      free(codes);
      return rc;
    }
    *out = h;
  }
  free(codes);
  return ND_OK;
}

/* confidence_head_at_depth: keep cell 0 and the kept blocks' cells. */
static int slice_head(nd_head *h, const size_t *kept, size_t depth, size_t d) {
  size_t k = h->probes_per_cell, q = h->queries, m = h->cells * k, nc = depth + 1, i, qi;
  size_t cells[65];
  float *probes = nd_calloc(nc * k * d, sizeof(float)), *gain = nd_calloc(nc * k, sizeof(float));
  float *rb = nd_calloc(q * nc * k, sizeof(float));
  if (!probes || !gain || !rb) {
    free(probes);
    free(gain);
    free(rb);
    return ND_E_NOMEM;
  }
  cells[0] = 0;
  for (i = 0; i < depth; i++) cells[i + 1] = kept[i] + 1;
  for (i = 0; i < nc; i++) {
    memcpy(probes + i * k * d, h->probes + cells[i] * k * d, k * d * sizeof(float));
    memcpy(gain + i * k, h->gain + cells[i] * k, k * sizeof(float));
  }
  for (qi = 0; qi < q; qi++)
    for (i = 0; i < nc; i++) memcpy(rb + (qi * nc + i) * k, h->row_bias + qi * m + cells[i] * k, k * sizeof(float));
  free(h->probes);
  free(h->gain);
  free(h->row_bias);
  h->probes = probes;
  h->gain = gain;
  h->row_bias = rb;
  h->cells = nc;
  return ND_OK;
}

static int load_perm(const nd_cact *c, size_t idx, size_t n, uint32_t **out, nd_err *e) {
  float *v;
  size_t i;
  int rc = nd_cact_floats(c, idx, n, &v, NULL, e);
  if (rc) return rc;
  *out = nd_calloc(n, sizeof(uint32_t));
  if (!*out) {
    free(v);
    return ND_E_NOMEM;
  }
  for (i = 0; i < n; i++) {
    if (!(v[i] >= 0.0f && v[i] < (float)n) || v[i] != floorf(v[i])) {
      free(v);
      nd_seterr(e, "Hadamard permutation entry %zu is not an index below %zu", i, n);
      return ND_E_FORMAT;
    }
    (*out)[i] = (uint32_t)v[i];
  }
  free(v);
  return ND_OK;
}

int nd_model_load(const nd_cact *c, size_t depth, nd_model **out, nd_err *e) {
  nd_cfg parent;
  layout lo;
  nd_model *m;
  size_t kept[64], L, n, li, s, ns = 0, i;
  int rc, full;
  *out = NULL;
  if ((rc = config_from_geometry(&c->g, &parent, e))) return rc;
  if (parent.num_layers == 0 || parent.num_layers > 64 || parent.d_model == 0 || parent.mhc_lanes == 0 ||
      parent.num_heads == 0 || parent.qk_head_dim == 0 || parent.qk_head_dim % 2 || parent.v_head_dim == 0 ||
      parent.max_seq_len == 0 || parent.hada_n < parent.d_model || parent.hada_n & (parent.hada_n - 1) ||
      parent.vocab_size == 0 || parent.slots == 0 || parent.sub_dim == 0) {
    nd_seterr(e, "geometry the forward pass cannot run");
    return ND_E_FORMAT;
  }
  if ((rc = derive(c, &parent, &lo, e))) {
    layout_free(&lo);
    return rc;
  }
  L = parent.num_layers;
  if (depth == 0) depth = L;
  full = depth == L;
  if (full) {
    for (i = 0; i < L; i++) kept[i] = i;
  } else if (!ladder_kept(L, depth, kept)) {
    layout_free(&lo);
    nd_seterr(e, "depth %zu is outside the ladder 2..=%zu", depth, L);
    return ND_E_ARG;
  }
  m = calloc(1, sizeof *m);
  if (!m) {
    layout_free(&lo);
    return ND_E_NOMEM;
  }
  m->cfg = parent;
  m->cfg.num_layers = depth;
  /* config_at_depth: global layers and Engram sites renumbered by rank */
  memset(m->cfg.is_global, 0, sizeof m->cfg.is_global);
  for (i = 0; i < depth; i++) m->cfg.is_global[i] = parent.is_global[kept[i]];
  for (s = 0; s < parent.n_sites; s++)
    for (i = 0; i < depth; i++)
      if (kept[i] == parent.sites[s]) m->cfg.sites[ns++] = i;
  m->cfg.n_sites = ns;
  n = parent.mhc_lanes;

  if ((rc = load_cq(c, lo.embedding, &m->embedding, e))) goto fail;
  m->layers = nd_calloc(depth, sizeof(nd_layer));
  m->engrams = nd_calloc(ns, sizeof(nd_engram_site));
  if (!m->layers || !m->engrams) {
    rc = ND_E_NOMEM;
    goto fail;
  }
  for (li = 0; li < depth; li++)
    if ((rc = load_layer(c, &parent, &lo.layers[kept[li]], &m->layers[li], e))) goto fail;
  {
    float *sc[6] = {0};
    size_t width[6] = {1, 1, 1, n, n, n * n}, k;
    nd_cq phi[3];
    size_t pw[3] = {n, n, n * n};
    memset(phi, 0, sizeof phi);
    for (k = 0; k < 6; k++) {
      float *dst;
      if ((rc = nd_cact_floats(c, lo.scalars[k], L * width[k], &sc[k], NULL, e))) goto fail_sc;
      dst = nd_calloc(depth * width[k], sizeof(float));
      if (!dst) {
        rc = ND_E_NOMEM;
        goto fail_sc;
      }
      for (i = 0; i < depth; i++) memcpy(dst + i * width[k], sc[k] + kept[i] * width[k], width[k] * sizeof(float));
      free(sc[k]);
      sc[k] = dst;
    }
    m->a_pre = sc[0];
    m->a_post = sc[1];
    m->a_res = sc[2];
    m->b_pre = sc[3];
    m->b_post = sc[4];
    m->b_res = sc[5];
    for (k = 0; k < 3; k++) {
      size_t *rows, nr = depth * pw[k], r = 0, j;
      if ((rc = load_cq(c, lo.phi[k], &phi[k], e))) goto fail;
      if (phi[k].out_feat != L * pw[k] || phi[k].in_feat != n * parent.d_model) {
        nd_cq_free(&phi[k]);
        nd_seterr(e, "mHC projection %zu has shape [%zu, %zu]", k, phi[k].out_feat, phi[k].in_feat);
        rc = ND_E_FORMAT;
        goto fail;
      }
      if (full) {
        nd_cq *dst = k == 0 ? &m->phi_pre : k == 1 ? &m->phi_post : &m->phi_res;
        *dst = phi[k];
        continue;
      }
      rows = nd_calloc(nr, sizeof(size_t));
      if (!rows) {
        nd_cq_free(&phi[k]);
        rc = ND_E_NOMEM;
        goto fail;
      }
      for (i = 0; i < depth; i++)
        for (j = 0; j < pw[k]; j++) rows[r++] = kept[i] * pw[k] + j;
      rc = nd_cq_select_rows(&phi[k], rows, nr, k == 0 ? &m->phi_pre : k == 1 ? &m->phi_post : &m->phi_res, e);
      free(rows);
      nd_cq_free(&phi[k]);
      if (rc) goto fail;
    }
    goto sc_done;
  fail_sc:
    for (k = 0; k < 6; k++) free(sc[k]);
    goto fail;
  sc_done:;
  }
  ns = 0;
  for (s = 0; s < parent.n_sites; s++) {
    int keep = 0;
    for (i = 0; i < depth; i++) keep |= kept[i] == parent.sites[s];
    if (!keep) continue;
    if ((rc = load_cq(c, lo.engrams[s][0], &m->engrams[ns].tables, e)) ||
        (rc = load_cq(c, lo.engrams[s][1], &m->engrams[ns].key_proj, e)) ||
        (rc = load_cq(c, lo.engrams[s][2], &m->engrams[ns].value_proj, e)) ||
        (rc = nd_cact_floats(c, lo.engrams[s][3], parent.conv_taps * parent.d_model, &m->engrams[ns].taps, NULL, e)))
      goto fail;
    ns++;
  }
  if ((rc = load_perm(c, lo.perms[0], parent.hada_n, &m->p1, e)) || (rc = load_perm(c, lo.perms[1], parent.hada_n, &m->p2, e)) ||
      (rc = nd_cact_floats(c, lo.final_norm, parent.d_model, &m->final_norm, NULL, e)))
    goto fail;
  m->embed_scale = sqrtf((float)parent.d_model);
  if ((rc = load_confidence(c, &parent, &lo, &m->confidence, e))) goto fail;
  if (m->confidence && !full && (rc = slice_head(m->confidence, kept, depth, parent.d_model))) goto fail;
  layout_free(&lo);
  *out = m;
  return ND_OK;
fail:
  layout_free(&lo);
  nd_model_free(m);
  return rc ? rc : ND_E_FORMAT;
}
