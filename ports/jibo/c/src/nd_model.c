/* nd_model.c: the Needle 3 forward pass and cache. Transcribes needle-core/src/v3/model.rs
 * (forward_impl_pooled, decode_step), cache.rs, attention.rs, engram.rs, mhc.rs, kernels.rs
 * (hadamard_mlp, kron_apply), heads.rs (ProbePool, pool_and_project), norm.rs and
 * kernels.rs (rms_unit, sinkhorn). Summation orders, operand orders and the start value of every
 * sum follow the Rust code line for line; see nd_common.h for the rules. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "nd_math.h"
#include "nd_prof.h"
#include "nd_model.h"

#define EPS 1e-6f
#define SINKHORN_ITERS 20
#define ENGRAM_SEED 0x9E3779B9u
#define ENGRAM_PRIME 0x01000193u
#define CHUNK 64

/* ---- small kernels ---- */
static float sigmoid(float x) { return 1.0f / (1.0f + nd_expf(-x)); }
static float silu(float x) { return x * sigmoid(x); }

/* Rust f32::max: the non-NaN operand when one is NaN. */
static float fmax32(float a, float b) { return a != a ? b : b != b ? a : a >= b ? a : b; }

/* zc_rms_norm_vec: x * (1 + s) / rms, eps inside the root. */
static void zc_rms_norm(float *x, const float *scale, size_t n) {
  float sq = -0.0f, inv;
  size_t i;
  for (i = 0; i < n; i++) sq += x[i] * x[i];
  inv = 1.0f / sqrtf(sq / (float)n + EPS);
  for (i = 0; i < n; i++) x[i] = (1.0f + scale[i]) * x[i] * inv;
}

static void rms_unit(float *x, size_t n) {
  float sq = -0.0f, inv;
  size_t i;
  for (i = 0; i < n; i++) sq += x[i] * x[i];
  inv = 1.0f / sqrtf(sq / (float)n + EPS);
  for (i = 0; i < n; i++) x[i] *= inv;
}

static void rms_unit_to(const float *x, float *out, size_t n) {
  float sq = -0.0f, inv;
  size_t i;
  for (i = 0; i < n; i++) sq += x[i] * x[i];
  inv = 1.0f / sqrtf(sq / (float)n + EPS);
  for (i = 0; i < n; i++) out[i] = x[i] * inv;
}

static float logsumexp(const float *x, size_t n) {
  float max = -INFINITY, sum = -0.0f;
  size_t i;
  for (i = 0; i < n; i++) max = fmax32(max, x[i]);
  if (!isfinite(max)) return max;
  for (i = 0; i < n; i++) sum += nd_expf(x[i] - max);
  return max + nd_logf(sum);
}

static void sinkhorn(float *m, size_t n) {
  size_t it, i, j;
  for (it = 0; it < SINKHORN_ITERS; it++) {
    for (i = 0; i < n; i++) {
      float lse = logsumexp(m + i * n, n);
      for (j = 0; j < n; j++) m[i * n + j] -= lse;
    }
    for (j = 0; j < n; j++) {
      float max = -INFINITY, sum = 0.0f, lse;
      for (i = 0; i < n; i++) max = fmax32(max, m[i * n + j]);
      for (i = 0; i < n; i++) sum += nd_expf(m[i * n + j] - max);
      lse = max + nd_logf(sum);
      for (i = 0; i < n; i++) m[i * n + j] -= lse;
    }
  }
  for (i = 0; i < n * n; i++) m[i] = nd_expf(m[i]);
}

/* ---- mhc.rs ---- */
static float pre_off(size_t layer, size_t lane, size_t lanes) { return lane == layer % lanes ? 4.0f : -4.0f; }
static float post_off(size_t layer, size_t lane, size_t lanes) { return lane == layer % lanes ? 0.0f : -4.0f; }

static void mix_down(const nd_model *m, const float *lanes_buf, float *h, size_t li, float *u) {
  size_t n = m->cfg.mhc_lanes, d = m->cfg.d_model, lane, c;
  for (lane = 0; lane < n; lane++)
    h[lane] = sigmoid(m->a_pre[li] * h[lane] + m->b_pre[li * n + lane] + pre_off(li, lane, n));
  /* Lane-outer over whole rows: the same additions from 0 in lane order for every c, with long
   * inner loops (mhc.rs does the same). */
  for (c = 0; c < d; c++) u[c] = 0.0f;
  for (lane = 0; lane < n; lane++) nd_axpy(u, h[lane], lanes_buf + lane * d, d);
}

static void scatter_up(const nd_model *m, float *lanes_buf, float *hpost, float *hres, const float *y, size_t li,
                       float *scratch) {
  size_t n = m->cfg.mhc_lanes, d = m->cfg.d_model, lane, i, j, c;
  for (lane = 0; lane < n; lane++)
    hpost[lane] = 2.0f * sigmoid(m->a_post[li] * hpost[lane] + m->b_post[li * n + lane] + post_off(li, lane, n));
  for (i = 0; i < n * n; i++) hres[i] = m->a_res[li] * hres[i] + m->b_res[li * n * n + i];
  sinkhorn(hres, n);
  memcpy(scratch, lanes_buf, n * d * sizeof(float));
  for (i = 0; i < n; i++) {
    float *out = lanes_buf + i * d;
    float hp = hpost[i];
    for (c = 0; c < d; c++) out[c] = 0.0f;
    for (j = 0; j < n; j++) nd_axpy(out, hres[i * n + j], scratch + j * d, d);
    nd_axpy(out, hp, y, d);
  }
}

/* ---- engram.rs ---- */
static void engram_indices(const nd_cfg *cfg, const uint32_t *tokens, size_t seq, uint32_t *out) {
  size_t stride = cfg->seed_heads == 0 ? cfg->heads : cfg->seed_heads;
  size_t num_tables = cfg->n_orders * cfg->heads, oi, h, p, j;
  for (oi = 0; oi < cfg->n_orders; oi++) {
    size_t order = cfg->orders[oi];
    for (h = 0; h < cfg->heads; h++) {
      size_t table = oi * cfg->heads + h;
      uint32_t seed = ENGRAM_SEED * (uint32_t)(oi * stride + h + 1);
      for (p = 0; p < seq; p++) {
        uint32_t acc = seed;
        for (j = 0; j < order; j++) {
          uint32_t tok = j > p ? 0 : tokens[p - j];
          acc = (acc ^ tok) * ENGRAM_PRIME;
        }
        acc ^= acc >> 15;
        out[p * num_tables + table] = acc % (uint32_t)cfg->slots;
      }
    }
  }
}

static int ngram_valid(const nd_cfg *cfg, size_t p, size_t table) { return p + 1 >= cfg->orders[table / cfg->heads]; }

static size_t max_order(const nd_cfg *cfg) {
  size_t i, m = 0;
  for (i = 0; i < cfg->n_orders; i++)
    if (cfg->orders[i] > m) m = cfg->orders[i];
  return m ? m : 1;
}

static void value_conv(const nd_cfg *cfg, float *buf, const float *taps, size_t seq) {
  size_t dim = cfg->d_model, mo = max_order(cfg), p, c, j;
  for (p = seq; p-- > 0;) {
    for (c = 0; c < dim; c++) {
      float acc = 0.0f;
      for (j = 0; j < cfg->conv_taps; j++) {
        size_t back = j * cfg->conv_dilation;
        if (back > p || p < j * mo) continue;
        acc += taps[j * dim + c] * buf[(p - back) * dim + c];
      }
      buf[p * dim + c] = acc;
    }
  }
}

static void apply_site(float *x, const float *k, const float *v, size_t d) {
  float sx = 0.0f, sk = 0.0f, ux, uk, dot = 0.0f, alpha;
  size_t i;
  for (i = 0; i < d; i++) sx += x[i] * x[i];
  ux = 1.0f / sqrtf(sx / (float)d + 1e-6f);
  for (i = 0; i < d; i++) sk += k[i] * k[i];
  uk = 1.0f / sqrtf(sk / (float)d + 1e-6f);
  for (i = 0; i < d; i++) dot += x[i] * k[i];
  alpha = sigmoid(dot * ux * uk / sqrtf((float)d));
  for (i = 0; i < d; i++) x[i] += alpha * v[i];
}

/* One site's fetched rows for a position: tables (num_tables x sub_dim), zero where invalid. */
static void engram_fetch(const nd_model *m, const nd_engram_site *site, const uint32_t *idx_row, size_t abs_pos,
                         float *fetched, float *row) {
  const nd_cfg *cfg = &m->cfg;
  size_t nt = cfg->n_orders * cfg->heads, t;
  for (t = 0; t < nt; t++) {
    float *dst = fetched + t * cfg->sub_dim;
    if (ngram_valid(cfg, abs_pos, t)) {
      nd_cq_dequantize_row(&site->tables, t * cfg->slots + idx_row[t], row);
      memcpy(dst, row, cfg->sub_dim * sizeof(float));
    } else {
      memset(dst, 0, cfg->sub_dim * sizeof(float));
    }
  }
}

/* ---- kernels.rs: the learned Kronecker MLP ---- */
static void kron_apply(const float *z, const float *a, const float *b, size_t ba, size_t bb, float *out, float *t) {
  size_t i, k, j;
  memset(t, 0, ba * bb * sizeof(float));
  for (i = 0; i < ba; i++) {
    const float *zi = z + i * bb, *ai = a + i * ba;
    for (k = 0; k < ba; k++) {
      float aik = ai[k];
      float *tk;
      if (aik == 0.0f) continue;
      tk = t + k * bb;
      nd_axpy(tk, aik, zi, bb);
    }
  }
  for (k = 0; k < ba; k++) {
    const float *tk = t + k * bb;
    float *ok = out + k * bb;
    memset(ok, 0, bb * sizeof(float));
    for (j = 0; j < bb; j++) {
      float tkj = tk[j];
      const float *bj;
      if (tkj == 0.0f) continue;
      bj = b + j * bb;
      nd_axpy(ok, tkj, bj, bb);
    }
  }
}

typedef struct {
  float *proj, *cond, *z, *buf, *t;
} mlp_scratch;

static void hadamard_mlp(const nd_model *m, const float *x, const nd_hmlp *w, float *out, mlp_scratch *s) {
  size_t d = m->cfg.d_model, hn = m->cfg.hada_n, r = w->cond_rank, i, j, c, bits = 0, v = hn - 1, ba, bb;
  float max = -INFINITY, sum = 0.0f;
  while (v) {
    bits++;
    v >>= 1;
  }
  ba = (size_t)1 << (bits / 2);
  bb = hn / ba;
  for (j = 0; j < r; j++) {
    float acc = 0.0f;
    for (i = 0; i < d; i++) acc += x[i] * w->cond_v[i * r + j];
    s->proj[j] = acc;
  }
  for (j = 0; j < r; j++) max = fmax32(max, s->proj[j]);
  for (j = 0; j < r; j++) {
    s->proj[j] = nd_expf(s->proj[j] - max);
    sum += s->proj[j];
  }
  for (j = 0; j < r; j++) s->proj[j] /= sum;
  for (c = 0; c < hn; c++) s->cond[c] = 1.0f;
  for (j = 0; j < r; j++) {
    float pj = s->proj[j];
    const float *row;
    if (pj == 0.0f) continue;
    row = w->cond_u + j * hn;
    nd_axpy(s->cond, pj, row, hn);
  }
  memset(s->z, 0, hn * sizeof(float));
  memcpy(s->z, x, d * sizeof(float));
  for (i = 0; i < hn; i++) s->z[i] *= w->d1[i];
  kron_apply(s->z, w->w1a, w->w1b, ba, bb, s->buf, s->t);
  for (i = 0; i < hn; i++) s->z[i] = s->buf[m->p1[i]];
  for (i = 0; i < hn; i++) s->z[i] = silu(w->d2[i] * s->cond[i] * s->z[i] + w->b2[i]);
  kron_apply(s->z, w->w2a, w->w2b, ba, bb, s->buf, s->t);
  for (i = 0; i < hn; i++) s->z[i] = s->buf[m->p2[i]];
  for (i = 0; i < hn; i++) s->z[i] *= w->d3[i];
  kron_apply(s->z, w->w3a, w->w3b, ba, bb, s->buf, s->t);
  for (i = 0; i < d; i++) out[i] = s->buf[i] * w->d4[i];
}

/* ---- attention.rs ---- */
static void causal_depthwise_conv(float *buf, const float *taps, size_t seq, size_t dim, size_t n_taps) {
  size_t t, c, j;
  if (n_taps == 0) return;
  for (t = seq; t-- > 0;) {
    for (c = 0; c < dim; c++) {
      float acc = 0.0f;
      for (j = 0; j < n_taps; j++) {
        if (j > t) break;
        acc += taps[j * dim + c] * buf[(t - j) * dim + c];
      }
      buf[t * dim + c] = acc;
    }
  }
}

static void norm_and_rope(float *buf, const float *scale, const float *cs, const float *sn, size_t seq, size_t heads,
                          size_t hd) {
  size_t half = hd / 2, t, h, i;
  for (t = 0; t < seq; t++)
    for (h = 0; h < heads; h++) {
      float *x = buf + (t * heads + h) * hd, sq = 0.0f, rms;
      for (i = 0; i < hd; i++) sq += x[i] * x[i];
      rms = sqrtf(sq / (float)hd + 1e-6f);
      for (i = 0; i < hd; i++) x[i] = (1.0f + scale[i]) * x[i] / rms;
      for (i = 0; i < half; i++) {
        float c = cs[t * half + i], s = sn[t * half + i], x1 = x[i], x2 = x[half + i];
        x[i] = x1 * c - x2 * s;
        x[half + i] = x2 * c + x1 * s;
      }
    }
}

/* A key/value store: f32, or int8 with one scale per (position, kv head). */
typedef struct {
  const float *k, *v;
  const int8_t *kq, *vq;
  const float *ks, *vs;
} kv_view;

static float kv_dot(const kv_view *kv, const float *q, size_t off, size_t len, size_t sat) {
  float a = 0.0f;
  size_t i;
  if (kv->k) {
    for (i = 0; i < len; i++) a += q[i] * kv->k[off + i];
    return a;
  }
  for (i = 0; i < len; i++) a += q[i] * (float)kv->kq[off + i];
  return a * kv->ks[sat];
}

static void kv_accum(const kv_view *kv, float w, size_t off, float *out, size_t n, size_t sat) {
  size_t i;
  if (kv->v) {
    nd_axpy(out, w, kv->v + off, n);
    return;
  }
  w = w * kv->vs[sat];
  for (i = 0; i < n; i++) out[i] += w * (float)kv->vq[off + i];
}

/* attend: the whole sequence, causal, optional window (0 = none). */
static void attend(const nd_cfg *cfg, const float *q, const kv_view *kv, size_t seq, size_t window, float *out,
                   float *scores) {
  size_t H = cfg->num_heads, KV = cfg->num_kv_heads, qk = cfg->qk_head_dim, vd = cfg->v_head_dim;
  size_t rep = H / KV, t, h, nn;
  float scale = 1.0f / sqrtf((float)qk);
  for (t = 0; t < seq; t++) {
    size_t lo = window && t + 1 > window ? t + 1 - window : 0;
    for (h = 0; h < H; h++) {
      size_t kvh = h / rep;
      const float *qv = q + (t * H + h) * qk;
      float max = -INFINITY, sum = 0.0f, inv, *o;
      for (nn = lo; nn <= t; nn++) {
        float acc = kv_dot(kv, qv, (nn * KV + kvh) * qk, qk, nn * KV + kvh);
        scores[nn] = acc * scale;
        if (scores[nn] > max) max = scores[nn];
      }
      for (nn = lo; nn <= t; nn++) {
        scores[nn] = nd_expf(scores[nn] - max);
        sum += scores[nn];
      }
      inv = 1.0f / sum;
      o = out + (t * H + h) * vd;
      memset(o, 0, vd * sizeof(float));
      for (nn = lo; nn <= t; nn++) kv_accum(kv, scores[nn] * inv, (nn * KV + kvh) * vd, o, vd, nn * KV + kvh);
    }
  }
}

/* attend_step: one query against a ring of `slots`, logical positions lo..=hi. */
static void attend_step(const nd_cfg *cfg, const float *q, const kv_view *kv, size_t slots, size_t lo, size_t hi,
                        float *out, float *scores) {
  size_t H = cfg->num_heads, KV = cfg->num_kv_heads, qk = cfg->qk_head_dim, vd = cfg->v_head_dim;
  size_t rep = H / KV, span = hi - lo + 1, h, i;
  float scale = 1.0f / sqrtf((float)qk);
  for (h = 0; h < H; h++) {
    size_t kvh = h / rep;
    const float *qv = q + h * qk;
    float max = -INFINITY, sum = 0.0f, inv, *o;
    /* The ring slot of position lo + i, stepped rather than taken modulo slots (no divide
     * instruction in the ARMv7 baseline: % was a library call per position). */
    size_t slot = lo % slots;
    for (i = 0; i < span; i++) {
      float acc = kv_dot(kv, qv, (slot * KV + kvh) * qk, qk, slot * KV + kvh);
      scores[i] = acc * scale;
      if (scores[i] > max) max = scores[i];
      if (++slot == slots) slot = 0;
    }
    for (i = 0; i < span; i++) {
      scores[i] = nd_expf(scores[i] - max);
      sum += scores[i];
    }
    inv = 1.0f / sum;
    o = out + h * vd;
    memset(o, 0, vd * sizeof(float));
    slot = lo % slots;
    for (i = 0; i < span; i++) {
      kv_accum(kv, scores[i] * inv, (slot * KV + kvh) * vd, o, vd, slot * KV + kvh);
      if (++slot == slots) slot = 0;
    }
  }
}

/* ---- cache.rs ---- */
static void quant_scale(const float *x, size_t n, float *scale, float *inv, float *qmax) {
  float absmax = 0.0f;
  size_t i;
  *qmax = 127.0f; /* bits 8: (1 << 7) - 1 */
  for (i = 0; i < n; i++) {
    float a = x[i] < 0.0f ? -x[i] : x[i];
    if (a > absmax) absmax = a;
  }
  *scale = absmax > 0.0f ? absmax / *qmax : 1.0f;
  *inv = 1.0f / *scale;
}

/* Rust f32::clamp(lo, hi). */
static float clamp32(float x, float lo, float hi) { return x < lo ? lo : x > hi ? hi : x; }

static void fake_quant_vec(float *x, size_t n) {
  float scale, inv, qmax;
  size_t i;
  quant_scale(x, n, &scale, &inv, &qmax);
  for (i = 0; i < n; i++) x[i] = clamp32(roundf(x[i] * inv), -qmax - 1.0f, qmax) * scale;
}

static int8_t f32_to_i8_sat(float v) {
  if (v != v) return 0;
  if (v <= -128.0f) return -128;
  if (v >= 127.0f) return 127;
  return (int8_t)v;
}

static float quant_into(const float *src, int8_t *dst, size_t n) {
  float scale, inv, qmax;
  size_t i;
  quant_scale(src, n, &scale, &inv, &qmax);
  for (i = 0; i < n; i++) dst[i] = f32_to_i8_sat(clamp32(roundf(src[i] * inv), -qmax - 1.0f, qmax));
  return scale;
}

static void quant_rows(float *x, size_t rows, size_t hd) {
  size_t r;
  for (r = 0; r < rows; r++) fake_quant_vec(x + r * hd, hd);
}

/* quantize_rows over a (rows, hd) buffer. */
static void quantize_rows(const float *x, size_t rows, size_t hd, int8_t *ints, float *scales) {
  size_t r;
  for (r = 0; r < rows; r++) scales[r] = quant_into(x + r * hd, ints + r * hd, hd);
}

typedef struct {
  float *k, *v;
  int8_t *kq, *vq;
  float *ks, *vs;
  size_t slots, cap;
  float *q_tail, *k_tail, *v_tail;
} layer_cache;

struct nd_cache {
  nd_cfg cfg;
  nd_kv_precision prec;
  layer_cache *layers;
  uint32_t tokens[16];
  size_t ntok;
  float **engram_tail;
  size_t pos;
  int failed; /* an allocation failed while growing */
};

static size_t span_of(const nd_cfg *c, size_t li) {
  return c->is_global[li] || c->sliding_window == 0 ? 0 : c->sliding_window;
}

void nd_cache_free(nd_cache *c) {
  size_t i;
  if (!c) return;
  if (c->layers)
    for (i = 0; i < c->cfg.num_layers; i++) {
      layer_cache *l = &c->layers[i];
      free(l->k);
      free(l->v);
      free(l->kq);
      free(l->vq);
      free(l->ks);
      free(l->vs);
      free(l->q_tail);
      free(l->k_tail);
      free(l->v_tail);
    }
  free(c->layers);
  if (c->engram_tail)
    for (i = 0; i < c->cfg.n_sites; i++) free(c->engram_tail[i]);
  free(c->engram_tail);
  free(c);
}

nd_cache *nd_cache_new(const nd_cfg *cfg, size_t hint, nd_kv_precision p) {
  nd_cache *c = calloc(1, sizeof *c);
  size_t kd = cfg->num_kv_heads * cfg->qk_head_dim, vd = cfg->num_kv_heads * cfg->v_head_dim;
  size_t qd = cfg->num_heads * cfg->qk_head_dim, tail = cfg->qkv_conv_taps ? cfg->qkv_conv_taps - 1 : 0;
  size_t reach = (cfg->conv_taps ? cfg->conv_taps - 1 : 0) * cfg->conv_dilation, li, s;
  if (!c) return NULL;
  c->cfg = *cfg;
  c->prec = p;
  if (hint < 1) hint = 1;
  if (hint > cfg->max_seq_len) hint = cfg->max_seq_len;
  c->layers = nd_calloc(cfg->num_layers, sizeof(layer_cache));
  c->engram_tail = nd_calloc(cfg->n_sites, sizeof(float *));
  if (!c->layers || !c->engram_tail) goto fail;
  for (li = 0; li < cfg->num_layers; li++) {
    layer_cache *l = &c->layers[li];
    size_t w = span_of(cfg, li);
    l->cap = w ? (w < cfg->max_seq_len ? w : cfg->max_seq_len) : cfg->max_seq_len;
    l->slots = hint < l->cap ? hint : l->cap;
    if (p == ND_KV_INT8) {
      size_t heads = l->slots * cfg->num_kv_heads, i;
      l->kq = nd_calloc(l->slots * kd, 1);
      l->vq = nd_calloc(l->slots * vd, 1);
      l->ks = nd_calloc(heads, sizeof(float));
      l->vs = nd_calloc(heads, sizeof(float));
      if (!l->kq || !l->vq || !l->ks || !l->vs) goto fail;
      for (i = 0; i < heads; i++) l->ks[i] = l->vs[i] = 1.0f;
    } else {
      l->k = nd_calloc(l->slots * kd, sizeof(float));
      l->v = nd_calloc(l->slots * vd, sizeof(float));
      if (!l->k || !l->v) goto fail;
    }
    l->q_tail = nd_calloc(tail * qd, sizeof(float));
    l->k_tail = nd_calloc(tail * kd, sizeof(float));
    l->v_tail = nd_calloc(tail * vd, sizeof(float));
    if (!l->q_tail || !l->k_tail || !l->v_tail) goto fail;
  }
  for (s = 0; s < cfg->n_sites; s++)
    if (!(c->engram_tail[s] = nd_calloc(reach * cfg->d_model, sizeof(float)))) goto fail;
  return c;
fail:
  nd_cache_free(c);
  return NULL;
}

static void *dup_bytes(const void *src, size_t n) {
  void *d = malloc(n ? n : 1);
  if (d && n) memcpy(d, src, n);
  return d;
}

nd_cache *nd_cache_clone(const nd_cache *src) {
  const nd_cfg *cfg = &src->cfg;
  size_t kd = cfg->num_kv_heads * cfg->qk_head_dim, vd = cfg->num_kv_heads * cfg->v_head_dim;
  size_t qd = cfg->num_heads * cfg->qk_head_dim, tail = cfg->qkv_conv_taps ? cfg->qkv_conv_taps - 1 : 0;
  size_t reach = (cfg->conv_taps ? cfg->conv_taps - 1 : 0) * cfg->conv_dilation, li, s;
  nd_cache *c = calloc(1, sizeof *c);
  if (!c) return NULL;
  *c = *src;
  c->layers = nd_calloc(cfg->num_layers, sizeof(layer_cache));
  c->engram_tail = nd_calloc(cfg->n_sites, sizeof(float *));
  if (!c->layers || !c->engram_tail) goto fail;
  for (li = 0; li < cfg->num_layers; li++) {
    const layer_cache *a = &src->layers[li];
    layer_cache *b = &c->layers[li];
    size_t heads = a->slots * cfg->num_kv_heads;
    b->slots = a->slots;
    b->cap = a->cap;
    if (src->prec == ND_KV_INT8) {
      b->kq = dup_bytes(a->kq, a->slots * kd);
      b->vq = dup_bytes(a->vq, a->slots * vd);
      b->ks = dup_bytes(a->ks, heads * sizeof(float));
      b->vs = dup_bytes(a->vs, heads * sizeof(float));
      if (!b->kq || !b->vq || !b->ks || !b->vs) goto fail;
    } else {
      b->k = dup_bytes(a->k, a->slots * kd * sizeof(float));
      b->v = dup_bytes(a->v, a->slots * vd * sizeof(float));
      if (!b->k || !b->v) goto fail;
    }
    b->q_tail = dup_bytes(a->q_tail, tail * qd * sizeof(float));
    b->k_tail = dup_bytes(a->k_tail, tail * kd * sizeof(float));
    b->v_tail = dup_bytes(a->v_tail, tail * vd * sizeof(float));
    if (!b->q_tail || !b->k_tail || !b->v_tail) goto fail;
  }
  for (s = 0; s < cfg->n_sites; s++)
    if (!(c->engram_tail[s] = dup_bytes(src->engram_tail[s], reach * cfg->d_model * sizeof(float)))) goto fail;
  return c;
fail:
  nd_cache_free(c);
  return NULL;
}

size_t nd_cache_pos(const nd_cache *c) { return c->pos; }

static void push_token(nd_cache *c, uint32_t tok) {
  size_t keep = max_order(&c->cfg);
  if (c->ntok == keep) {
    memmove(c->tokens, c->tokens + 1, (keep - 1) * sizeof(uint32_t));
    c->ntok--;
  }
  c->tokens[c->ntok++] = tok;
}

static int grow(void **p, size_t n, size_t elem) {
  size_t bytes = n * elem;
  void *q = realloc(*p, bytes ? bytes : 1);
  if (!q) return 0;
  *p = q;
  return 1;
}

static int ensure(nd_cache *c, size_t li, size_t pos) {
  layer_cache *l = &c->layers[li];
  size_t kd = c->cfg.num_kv_heads * c->cfg.qk_head_dim, vd = c->cfg.num_kv_heads * c->cfg.v_head_dim;
  size_t heads = c->cfg.num_kv_heads, want, old = l->slots, i;
  if (pos < l->slots || l->slots >= l->cap) return ND_OK;
  want = l->slots * 2 > pos + 1 ? l->slots * 2 : pos + 1;
  if (want > l->cap) want = l->cap;
  if (c->prec == ND_KV_F32) {
    if (!grow((void **)&l->k, want * kd, sizeof(float)) || !grow((void **)&l->v, want * vd, sizeof(float))) return ND_E_NOMEM;
    memset(l->k + old * kd, 0, (want - old) * kd * sizeof(float));
    memset(l->v + old * vd, 0, (want - old) * vd * sizeof(float));
  } else {
    if (!grow((void **)&l->kq, want * kd, 1) || !grow((void **)&l->vq, want * vd, 1) ||
        !grow((void **)&l->ks, want * heads, sizeof(float)) || !grow((void **)&l->vs, want * heads, sizeof(float)))
      return ND_E_NOMEM;
    memset(l->kq + old * kd, 0, (want - old) * kd);
    memset(l->vq + old * vd, 0, (want - old) * vd);
    for (i = old * heads; i < want * heads; i++) l->ks[i] = l->vs[i] = 1.0f;
  }
  l->slots = want;
  return ND_OK;
}

static void cache_span(const nd_cache *c, size_t li, size_t pos, size_t *lo, size_t *hi) {
  size_t w = span_of(&c->cfg, li), slots = c->layers[li].slots, l = 0, floor;
  if (w && pos + 1 > w) l = pos + 1 - w;
  floor = pos + 1 > slots ? pos + 1 - slots : 0;
  *lo = l > floor ? l : floor;
  *hi = pos;
}

static int write_kv(nd_cache *c, size_t li, size_t pos, const float *k, const float *v) {
  size_t kd = c->cfg.num_kv_heads * c->cfg.qk_head_dim, vd = c->cfg.num_kv_heads * c->cfg.v_head_dim;
  size_t qk = c->cfg.qk_head_dim, vh = c->cfg.v_head_dim, h, slot;
  layer_cache *l;
  if (ensure(c, li, pos)) return ND_E_NOMEM;
  l = &c->layers[li];
  slot = pos % l->slots;
  if (c->prec == ND_KV_F32) {
    memcpy(l->k + slot * kd, k, kd * sizeof(float));
    memcpy(l->v + slot * vd, v, vd * sizeof(float));
  } else {
    size_t base = slot * c->cfg.num_kv_heads;
    for (h = 0; h < c->cfg.num_kv_heads; h++) {
      l->ks[base + h] = quant_into(k + h * qk, l->kq + slot * kd + h * qk, qk);
      l->vs[base + h] = quant_into(v + h * vh, l->vq + slot * vd + h * vh, vh);
    }
  }
  return ND_OK;
}

static kv_view cache_kv(const nd_cache *c, size_t li) {
  const layer_cache *l = &c->layers[li];
  kv_view kv;
  memset(&kv, 0, sizeof kv);
  if (c->prec == ND_KV_F32) {
    kv.k = l->k;
    kv.v = l->v;
  } else {
    kv.kq = l->kq;
    kv.vq = l->vq;
    kv.ks = l->ks;
    kv.vs = l->vs;
  }
  return kv;
}

/* conv_step for Q (0), K (1) or V (2). */
static void conv_step(nd_cache *c, size_t li, int which, const float *taps, float *raw, size_t dim, float *out) {
  size_t n = c->cfg.qkv_conv_taps, pos = c->pos, cc, j;
  layer_cache *l = &c->layers[li];
  float *tail = which == 0 ? l->q_tail : which == 1 ? l->k_tail : l->v_tail;
  if (n == 0) return;
  for (cc = 0; cc < dim; cc++) {
    float acc = taps[cc] * raw[cc];
    for (j = 1; j < n; j++) {
      size_t slot;
      if (j > pos) break;
      slot = n - 1 - j;
      acc += taps[j * dim + cc] * tail[slot * dim + cc];
    }
    out[cc] = acc;
  }
  if (n > 1) {
    memmove(tail, tail + dim, (n - 2) * dim * sizeof(float));
    memcpy(tail + (n - 2) * dim, raw, dim * sizeof(float));
  }
  memcpy(raw, out, dim * sizeof(float));
}

static void engram_conv_step(nd_cache *c, size_t site, const float *taps, float *raw, float *out) {
  const nd_cfg *cfg = &c->cfg;
  size_t n = cfg->conv_taps, dil = cfg->conv_dilation, mo = max_order(cfg), d = cfg->d_model, pos = c->pos;
  size_t reach = (n ? n - 1 : 0) * dil, cc, j;
  float *tail = c->engram_tail[site];
  for (cc = 0; cc < d; cc++) {
    float acc = 0.0f;
    for (j = 0; j < n; j++) {
      size_t back = j * dil;
      float val;
      if (back > pos || pos < j * mo) continue;
      val = j == 0 ? raw[cc] : tail[(reach - back) * d + cc];
      acc += taps[j * d + cc] * val;
    }
    out[cc] = acc;
  }
  if (reach > 0) {
    memmove(tail, tail + d, (reach - 1) * d * sizeof(float));
    memcpy(tail + (reach - 1) * d, raw, d * sizeof(float));
  }
  memcpy(raw, out, d * sizeof(float));
}

/* last_rows: the last n rows of (seq, dim), oldest first, zero-padded at the front. */
static void last_rows(const float *buf, size_t seq, size_t dim, size_t n, float *out) {
  size_t m = n < seq ? n : seq, j;
  memset(out, 0, n * dim * sizeof(float));
  for (j = 0; j < m; j++) memcpy(out + (n - m + j) * dim, buf + (seq - m + j) * dim, dim * sizeof(float));
}

/* ---- the probe-head pool (heads.rs) ---- */
typedef struct {
  const nd_head *head;
  size_t d;
  float *max, *den, *num;
} probe_pool;

static int pool_init(probe_pool *p, const nd_head *h, size_t d) {
  size_t m = h->cells * h->probes_per_cell, i;
  p->head = h;
  p->d = d;
  p->max = nd_calloc(m, sizeof(float));
  p->den = nd_calloc(m, sizeof(float));
  p->num = nd_calloc(m * d, sizeof(float));
  if (!p->max || !p->den || !p->num) return ND_E_NOMEM;
  for (i = 0; i < m; i++) p->max[i] = -INFINITY;
  return ND_OK;
}

static void pool_free(probe_pool *p) {
  free(p->max);
  free(p->den);
  free(p->num);
}

static void pool_observe(probe_pool *p, size_t cell_index, const float *cell) {
  size_t k = p->head->probes_per_cell, d = p->d, kk, i;
  float scale = 1.0f / sqrtf((float)d);
  for (kk = 0; kk < k; kk++) {
    size_t idx = cell_index * k + kk;
    const float *pr = p->head->probes + idx * d;
    float dot = 0.0f, s, m0, m1, shift, w, *dst;
    for (i = 0; i < d; i++) dot += cell[i] * pr[i];
    s = dot * scale;
    m0 = p->max[idx];
    m1 = s > m0 ? s : m0;
    shift = isfinite(m0) ? nd_expf(m0 - m1) : 0.0f;
    w = nd_expf(s - m1);
    p->den[idx] = p->den[idx] * shift + w;
    dst = p->num + idx * d;
    for (i = 0; i < d; i++) dst[i] = dst[i] * shift + w * cell[i];
    p->max[idx] = m1;
  }
}

static int pool_and_project(const nd_head *h, const float *r, size_t d, float *out) {
  size_t l1 = h->cells, k = h->probes_per_cell, q = h->queries, m = l1 * k, qi, idx, i, o;
  float scale = 1.0f / sqrtf((float)d);
  float *pooled = nd_calloc(q * d, sizeof(float)), *u = nd_calloc(m, sizeof(float));
  if (!pooled || !u) {
    free(pooled);
    free(u);
    return ND_E_NOMEM;
  }
  for (qi = 0; qi < q; qi++) {
    const float *qv = h->query + qi * d;
    float max = -INFINITY, sum = 0.0f, inv, *dst;
    for (idx = 0; idx < m; idx++) {
      const float *rv = r + idx * d;
      float acc = 0.0f;
      for (i = 0; i < d; i++) acc += rv[i] * qv[i];
      u[idx] = acc * scale + h->row_bias[qi * m + idx];
      if (u[idx] > max) max = u[idx];
    }
    for (idx = 0; idx < m; idx++) {
      u[idx] = nd_expf(u[idx] - max);
      sum += u[idx];
    }
    inv = 1.0f / sum;
    dst = pooled + qi * d;
    memset(dst, 0, d * sizeof(float));
    for (idx = 0; idx < m; idx++) {
      float w = u[idx] * inv;
      const float *rv = r + idx * d;
      for (i = 0; i < d; i++) dst[i] += w * rv[i];
    }
  }
  for (o = 0; o < h->out_dim; o++) {
    const float *row = h->proj + o * q * d;
    float acc = 0.0f;
    for (i = 0; i < q * d; i++) acc += row[i] * pooled[i];
    out[o] = acc + (o < h->nbias ? h->bias[o] : 0.0f);
  }
  free(pooled);
  free(u);
  return ND_OK;
}

static int pool_finish(probe_pool *p, float *out) {
  const nd_head *h = p->head;
  size_t idx, i, d = p->d;
  for (idx = 0; idx < h->cells * h->probes_per_cell; idx++) {
    float inv = 1.0f / p->den[idx], g, *dst = p->num + idx * d;
    for (i = 0; i < d; i++) dst[i] *= inv;
    rms_unit(dst, d);
    g = h->gain[idx];
    for (i = 0; i < d; i++) dst[i] *= g;
  }
  return pool_and_project(h, p->num, d, out);
}

/* ---- RoPE ---- */
static void rope_tables(const nd_cfg *cfg, size_t first, size_t seq, float *cs, float *sn) {
  size_t half = cfg->qk_head_dim / 2, i, t;
  for (i = 0; i < half; i++) {
    float freq = 1.0f / nd_powf(cfg->rope_theta, (float)(2 * i) / (float)cfg->qk_head_dim);
    for (t = 0; t < seq; t++) {
      float angle = (float)(first + t) * freq;
      cs[t * half + i] = nd_cosf(angle);
      sn[t * half + i] = nd_sinf(angle);
    }
  }
}

static size_t clamp_token(const nd_model *m, uint32_t tok) {
  return (size_t)tok < m->cfg.vocab_size ? (size_t)tok : m->cfg.vocab_size - 1;
}

/* ---- the batched forward (forward_impl_pooled) ----
 * logits: NULL, or (want_all ? seq : 1) x rows. cells: NULL or (seq, L+1, d). */
/* Whether the three mHC gate projections take the same prepared input (same width and group),
 * so one lane norm and one Hadamard preparation serve all three: the lanes only change in
 * scatter_up, after all three are read. */
static int phi_shared(const nd_model *m) {
  const nd_cq *a = &m->phi_pre, *b = &m->phi_post, *c = &m->phi_res;
  return a->in_feat == b->in_feat && a->in_feat == c->in_feat && a->group == b->group && a->group == c->group &&
         a->in_padded == b->in_padded && a->in_padded == c->in_padded;
}

static int forward(const nd_model *m, const uint32_t *tokens, size_t seq, float *cells, nd_cache *cache,
                   probe_pool *pool, float *logits, int want_all) {
  const nd_cfg *cfg = &m->cfg;
  size_t d = cfg->d_model, n = cfg->mhc_lanes, rows = nd_logit_rows(cfg), L = cfg->num_layers, l1 = L + 1;
  size_t H = cfg->num_heads, KV = cfg->num_kv_heads, qk = cfg->qk_head_dim, vh = cfg->v_head_dim;
  size_t qd = H * qk, kd = KV * qk, vd = KV * vh, od = H * vh, half = qk / 2;
  size_t nt = cfg->n_orders * cfg->heads, ns = cfg->n_sites, chunk = seq < CHUNK ? seq : CHUNK;
  size_t prep = m->layers[0].q_proj.in_padded, oprep = m->layers[0].out_proj.in_padded;
  size_t lmprep = m->embedding.in_padded, li, t, s, c0, i, c, lane, tail = cfg->qkv_conv_taps ? cfg->qkv_conv_taps - 1 : 0;
  size_t reach = (cfg->conv_taps ? cfg->conv_taps - 1 : 0) * cfg->conv_dilation;
  int quantised = cache && cache->prec == ND_KV_INT8, rc = ND_E_NOMEM;
  /* All buffers in one list, freed together. */
  float *rc_, *rs_, *ek, *ev, *evraw = NULL, *lanes, *emb, *nx, *nxp, *hpre, *hpost, *hres, *scratch, *u, *bx, *h;
  float *q, *k, *v, *attn, *gates, *xh, *oh, *proj, *acc, *mlpo, *y, *fetched, *row, *scores, *pooled, *lh, *cell;
  float *qt = NULL, *kt = NULL, *vt = NULL, *hpost_all = NULL, *hres_all = NULL;
  int shared = phi_shared(m);
  int8_t *kq = NULL, *vq = NULL;
  float *ksc = NULL, *vsc = NULL;
  uint32_t *idx;
  mlp_scratch ms;
  void *all[64];
  size_t na = 0;
#define A(p, cnt, sz)                     \
  do {                                    \
    all[na++] = (p) = nd_calloc((cnt), (sz)); \
    if (!(p)) goto out;                   \
  } while (0)
  memset(&ms, 0, sizeof ms);
  memset(all, 0, sizeof all);
  A(rc_, seq * half, sizeof(float));
  A(rs_, seq * half, sizeof(float));
  A(ek, ns * seq * d, sizeof(float));
  A(ev, ns * seq * d, sizeof(float));
  if (cache) A(evraw, ns * seq * d, sizeof(float));
  A(idx, seq * nt, sizeof(uint32_t));
  A(fetched, nt * cfg->sub_dim, sizeof(float));
  A(row, cfg->sub_dim, sizeof(float));
  A(lanes, seq * n * d, sizeof(float));
  A(emb, d, sizeof(float));
  A(nx, n * d, sizeof(float));
  A(nxp, m->phi_pre.in_padded, sizeof(float));
  A(hpre, n, sizeof(float));
  A(hpost, n, sizeof(float));
  A(hres, n * n, sizeof(float));
  A(scratch, n * d, sizeof(float));
  A(u, seq * d, sizeof(float));
  A(bx, seq * d, sizeof(float));
  A(h, seq * d, sizeof(float));
  A(q, seq * qd, sizeof(float));
  A(k, seq * kd, sizeof(float));
  A(v, seq * vd, sizeof(float));
  A(attn, seq * od, sizeof(float));
  A(gates, seq * od, sizeof(float));
  A(xh, chunk * prep, sizeof(float));
  A(oh, chunk * oprep, sizeof(float));
  A(proj, seq * d, sizeof(float));
  A(acc, chunk, sizeof(float));
  A(mlpo, d, sizeof(float));
  A(y, d, sizeof(float));
  A(scores, seq, sizeof(float));
  A(cell, d, sizeof(float));
  if (shared) {
    A(hpost_all, seq * n, sizeof(float));
    A(hres_all, seq * n * n, sizeof(float));
  }
  A(ms.proj, 1024, sizeof(float));
  A(ms.cond, cfg->hada_n, sizeof(float));
  A(ms.z, cfg->hada_n, sizeof(float));
  A(ms.buf, cfg->hada_n, sizeof(float));
  A(ms.t, cfg->hada_n, sizeof(float));
  if (cache && tail) {
    A(qt, tail * qd, sizeof(float));
    A(kt, tail * kd, sizeof(float));
    A(vt, tail * vd, sizeof(float));
  }
  if (quantised) {
    A(kq, seq * kd, 1);
    A(vq, seq * vd, 1);
    A(ksc, seq * KV, sizeof(float));
    A(vsc, seq * KV, sizeof(float));
  }

  ND_PB(ND_P_CONV_ROPE);
  rope_tables(cfg, 0, seq, rc_, rs_);
  ND_PE();

  /* Engram keys and values (engram_kv_with_raw) */
  ND_PB(ND_P_ENGRAM);
  engram_indices(cfg, tokens, seq, idx);
  /* Key and value projections batched over chunks of positions, as the block's Q/K/V are (the
   * matmul is bit-identical to a matvec per position, and decodes each weight group once per
   * chunk); one preparation of the fetched rows serves both when their geometry matches. */
  for (s = 0; s < ns; s++) {
    const nd_engram_site *site = &m->engrams[s];
    const nd_cq *kp = &site->key_proj, *vp = &site->value_proj;
    int kv_shared = kp->in_feat == vp->in_feat && kp->group == vp->group && kp->in_padded == vp->in_padded &&
                 kp->in_padded <= prep;
    for (c0 = 0; c0 < seq; c0 += chunk) {
      size_t b = seq - c0 < chunk ? seq - c0 : chunk;
      for (i = 0; i < b; i++) {
        size_t base = (s * seq + c0 + i) * d;
        engram_fetch(m, site, idx + (c0 + i) * nt, c0 + i, fetched, row);
        if (kv_shared)
          nd_cq_prepare(kp, fetched, xh + i * kp->in_padded);
        else if (nd_cq_matvec(kp, fetched, ek + base) || nd_cq_matvec(vp, fetched, ev + base))
          goto out;
      }
      if (kv_shared) {
        nd_cq_matmul_rows_prepared(kp, xh, b, 0, d, ek + (s * seq + c0) * d, acc);
        nd_cq_matmul_rows_prepared(vp, xh, b, 0, d, ev + (s * seq + c0) * d, acc);
      }
    }
    if (evraw) memcpy(evraw + s * seq * d, ev + s * seq * d, seq * d * sizeof(float));
    value_conv(cfg, ev + s * seq * d, site->taps, seq);
  }
  ND_PE();

  ND_PB(ND_P_EMBED);
  for (t = 0; t < seq; t++) {
    nd_cq_dequantize_row(&m->embedding, clamp_token(m, tokens[t]), emb);
    for (i = 0; i < d; i++) emb[i] *= m->embed_scale;
    if (cells) memcpy(cells + t * l1 * d, emb, d * sizeof(float));
    if (pool) pool_observe(pool, 0, emb);
    for (lane = 0; lane < n; lane++) memcpy(lanes + (t * n + lane) * d, emb, d * sizeof(float));
  }
  ND_PE();

  for (li = 0; li < L; li++) {
    const nd_layer *ly = &m->layers[li];
    size_t site = (size_t)-1;
    float agate;
    for (s = 0; s < ns; s++)
      if (cfg->sites[s] == li) {
        site = s;
        break;
      }
    ND_PB(ND_P_MHC);
    for (t = 0; t < seq; t++) {
      const float *ls = lanes + t * n * d;
      rms_unit_to(ls, nx, n * d);
      nd_cq_prepare(&m->phi_pre, nx, nxp);
      nd_cq_matvec_rows_prepared(&m->phi_pre, nxp, li * n, n, hpre);
      if (shared) {
        nd_cq_matvec_rows_prepared(&m->phi_post, nxp, li * n, n, hpost_all + t * n);
        nd_cq_matvec_rows_prepared(&m->phi_res, nxp, li * n * n, n * n, hres_all + t * n * n);
      }
      mix_down(m, ls, hpre, li, u + t * d);
    }
    ND_PE();
    ND_PB(ND_P_ENGRAM);
    memcpy(bx, u, seq * d * sizeof(float));
    if (site != (size_t)-1)
      for (t = 0; t < seq; t++) apply_site(bx + t * d, ek + (site * seq + t) * d, ev + (site * seq + t) * d, d);
    ND_PE();

    ND_PB(ND_P_QKV);
    memcpy(h, bx, seq * d * sizeof(float));
    for (t = 0; t < seq; t++) zc_rms_norm(h + t * d, ly->norm_in, d);
    ND_PE();
    for (c0 = 0; c0 < seq; c0 += chunk) {
      size_t b = seq - c0 < chunk ? seq - c0 : chunk;
      ND_PB(ND_P_QKV);
      for (i = 0; i < b; i++) nd_cq_prepare(&ly->q_proj, h + (c0 + i) * d, xh + i * prep);
      nd_cq_matmul_rows_prepared(&ly->q_proj, xh, b, 0, qd, q + c0 * qd, acc);
      nd_cq_matmul_rows_prepared(&ly->k_proj, xh, b, 0, kd, k + c0 * kd, acc);
      nd_cq_matmul_rows_prepared(&ly->v_proj, xh, b, 0, vd, v + c0 * vd, acc);
      ND_PE();
      ND_PB(ND_P_GATE_OUT);
      nd_cq_matmul_rows_prepared(&ly->gate_proj, xh, b, 0, od, gates + c0 * od, acc);
      ND_PE();
    }
    ND_PB(ND_P_CONV_ROPE);
    if (cache && tail) {
      last_rows(q, seq, qd, tail, qt);
      last_rows(k, seq, kd, tail, kt);
      last_rows(v, seq, vd, tail, vt);
    }
    if (cfg->qkv_conv_taps) {
      causal_depthwise_conv(q, ly->q_taps, seq, qd, cfg->qkv_conv_taps);
      causal_depthwise_conv(k, ly->k_taps, seq, kd, cfg->qkv_conv_taps);
      causal_depthwise_conv(v, ly->v_taps, seq, vd, cfg->qkv_conv_taps);
    }
    norm_and_rope(q, ly->q_norm, rc_, rs_, seq, H, qk);
    norm_and_rope(k, ly->k_norm, rc_, rs_, seq, KV, qk);
    if (quantised) {
      quant_rows(q, seq * H, qk);
      quant_rows(k, seq * KV, qk);
      quant_rows(v, seq * KV, vh);
    }
    ND_PE();
    ND_PB(ND_P_ATTN);
    if (cache) {
      for (t = 0; t < seq; t++)
        if (write_kv(cache, li, t, k + t * kd, v + t * vd)) goto out;
      if (tail) {
        memcpy(cache->layers[li].q_tail, qt, tail * qd * sizeof(float));
        memcpy(cache->layers[li].k_tail, kt, tail * kd * sizeof(float));
        memcpy(cache->layers[li].v_tail, vt, tail * vd * sizeof(float));
      }
    }
    {
      kv_view kv;
      memset(&kv, 0, sizeof kv);
      if (quantised) {
        quantize_rows(k, seq * KV, qk, kq, ksc);
        quantize_rows(v, seq * KV, vh, vq, vsc);
        kv.kq = kq;
        kv.vq = vq;
        kv.ks = ksc;
        kv.vs = vsc;
      } else {
        kv.k = k;
        kv.v = v;
      }
      attend(cfg, q, &kv, seq, span_of(cfg, li), attn, scores);
    }
    ND_PE();
    ND_PB(ND_P_GATE_OUT);
    agate = sigmoid(ly->attn_gate);
    for (t = 0; t < seq; t++)
      for (i = 0; i < od; i++) attn[t * od + i] *= sigmoid(gates[t * od + i]);
    for (c0 = 0; c0 < seq; c0 += chunk) {
      size_t b = seq - c0 < chunk ? seq - c0 : chunk;
      for (i = 0; i < b; i++) nd_cq_prepare(&ly->out_proj, attn + (c0 + i) * od, oh + i * oprep);
      nd_cq_matmul_rows_prepared(&ly->out_proj, oh, b, 0, d, proj + c0 * d, acc);
    }
    ND_PE();
    for (t = 0; t < seq; t++) {
      float *pr = proj + t * d, *s1 = bx + t * d;
      ND_PB(ND_P_GATE_OUT);
      zc_rms_norm(pr, ly->post_norm, d);
      for (i = 0; i < d; i++) s1[i] += agate * pr[i];
      ND_PE();
      ND_PB(ND_P_MLP);
      memcpy(pr, s1, d * sizeof(float));
      zc_rms_norm(pr, ly->pre_hada, d);
      hadamard_mlp(m, pr, &ly->mlp, mlpo, &ms);
      for (i = 0; i < d; i++) s1[i] += mlpo[i];
      ND_PE();
    }
    ND_PB(ND_P_MHC);
    for (t = 0; t < seq; t++) {
      float *ls = lanes + t * n * d;
      if (shared) {
        memcpy(hpost, hpost_all + t * n, n * sizeof(float));
        memcpy(hres, hres_all + t * n * n, n * n * sizeof(float));
      } else {
        rms_unit_to(ls, nx, n * d);
        nd_cq_prepare(&m->phi_post, nx, nxp);
        nd_cq_matvec_rows_prepared(&m->phi_post, nxp, li * n, n, hpost);
        nd_cq_prepare(&m->phi_res, nx, nxp);
        nd_cq_matvec_rows_prepared(&m->phi_res, nxp, li * n * n, n * n, hres);
      }
      for (c = 0; c < d; c++) y[c] = bx[t * d + c] - u[t * d + c];
      scatter_up(m, ls, hpost, hres, y, li, scratch);
    }
    ND_PE();
    if (cells || pool)
      for (t = 0; t < seq; t++) {
        for (c = 0; c < d; c++) {
          float a = 0.0f;
          for (lane = 0; lane < n; lane++) a += lanes[(t * n + lane) * d + c];
          cell[c] = a / (float)n;
        }
        if (cells) memcpy(cells + (t * l1 + li + 1) * d, cell, d * sizeof(float));
        if (pool) pool_observe(pool, li + 1, cell);
      }
  }

  if (cache) {
    cache->pos = seq;
    {
      size_t keep = max_order(cfg), start = seq > keep ? seq - keep : 0;
      cache->ntok = seq - start;
      memcpy(cache->tokens, tokens + start, cache->ntok * sizeof(uint32_t));
    }
    if (reach > 0)
      for (s = 0; s < ns; s++) last_rows(evraw + s * seq * d, seq, d, reach, cache->engram_tail[s]);
  }

  if (logits) ND_PB(ND_P_HEAD);
  if (logits) {
    size_t first = want_all ? 0 : seq - 1, cnt = seq - first;
    A(pooled, cnt * d, sizeof(float));
    A(lh, (cnt < CHUNK ? cnt : CHUNK) * lmprep, sizeof(float));
    for (t = first; t < seq; t++) {
      float *o = pooled + (t - first) * d;
      for (c = 0; c < d; c++) {
        float a = 0.0f;
        for (lane = 0; lane < n; lane++) a += lanes[(t * n + lane) * d + c];
        o[c] = a / (float)n;
      }
      zc_rms_norm(o, m->final_norm, d);
    }
    for (c0 = 0; c0 < cnt; c0 += CHUNK) {
      size_t b = cnt - c0 < CHUNK ? cnt - c0 : CHUNK;
      for (i = 0; i < b; i++) nd_cq_prepare(&m->embedding, pooled + (c0 + i) * d, lh + i * lmprep);
      nd_cq_matmul_rows_prepared(&m->embedding, lh, b, 0, rows, logits + c0 * rows, acc);
    }
  }
  ND_PE();
  rc = ND_OK;
out:
  for (i = 0; i < na; i++) free(all[i]);
#undef A
  return rc;
}

int nd_prefill(const nd_model *m, const uint32_t *tokens, size_t seq, nd_cache *cache, float *logits) {
  if (seq == 0) return ND_E_ARG;
  return forward(m, tokens, seq, NULL, cache, NULL, logits, 0);
}

int nd_forward_sequence(const nd_model *m, const uint32_t *tokens, size_t seq, float *logits) {
  if (seq == 0) return ND_OK;
  return forward(m, tokens, seq, NULL, NULL, NULL, logits, 1);
}

int nd_forward_cells(const nd_model *m, const uint32_t *tokens, size_t seq, float *cells) {
  if (seq == 0) return ND_OK;
  return forward(m, tokens, seq, cells, NULL, NULL, NULL, 0);
}

int nd_forward_head(const nd_model *m, const nd_head *head, const uint32_t *tokens, size_t seq, float *out) {
  probe_pool p;
  int rc;
  memset(&p, 0, sizeof p);
  if ((rc = pool_init(&p, head, m->cfg.d_model)) == ND_OK && (rc = forward(m, tokens, seq, NULL, NULL, &p, NULL, 0)) == ND_OK)
    rc = pool_finish(&p, out);
  pool_free(&p);
  return rc;
}

/* decode_step */
int nd_decode_step(const nd_model *m, nd_cache *cache, uint32_t token, float *logits) {
  const nd_cfg *cfg = &m->cfg;
  size_t d = cfg->d_model, n = cfg->mhc_lanes, pos = cache->pos, rows = nd_logit_rows(cfg), L = cfg->num_layers;
  size_t H = cfg->num_heads, KV = cfg->num_kv_heads, qk = cfg->qk_head_dim, vh = cfg->v_head_dim;
  size_t qd = H * qk, kd = KV * qk, vd = KV * vh, od = H * vh, half = qk / 2;
  size_t nt = cfg->n_orders * cfg->heads, ns = cfg->n_sites, li, s, i, c, lane, here;
  float *ek, *ev, *fetched, *row, *lanes, *emb, *rc_, *rs_, *nx, *nxp, *hpre, *hpost, *hres, *scratch, *u, *bx, *hh;
  float *q, *k, *v, *attn, *gate, *proj, *mlpo, *y, *x, *lmp, *tmp, *scores;
  uint32_t idx[16 * 64];
  uint32_t hist[16];
  mlp_scratch ms;
  void *all[48];
  size_t na = 0, nh;
  int rc = ND_E_NOMEM, shared = phi_shared(m);
#define A(p, cnt, sz)                     \
  do {                                    \
    all[na++] = (p) = nd_calloc((cnt), (sz)); \
    if (!(p)) goto out;                   \
  } while (0)
  memset(&ms, 0, sizeof ms);
  if (nt > 64) return ND_E_BOUNDS;
  A(ek, ns * d, sizeof(float));
  A(ev, ns * d, sizeof(float));
  A(fetched, nt * cfg->sub_dim, sizeof(float));
  A(row, cfg->sub_dim, sizeof(float));
  A(lanes, n * d, sizeof(float));
  A(emb, d, sizeof(float));
  A(rc_, half, sizeof(float));
  A(rs_, half, sizeof(float));
  A(nx, n * d, sizeof(float));
  A(nxp, m->phi_pre.in_padded, sizeof(float));
  A(hpre, n, sizeof(float));
  A(hpost, n, sizeof(float));
  A(hres, n * n, sizeof(float));
  A(scratch, n * d, sizeof(float));
  A(u, d, sizeof(float));
  A(bx, d, sizeof(float));
  A(hh, d, sizeof(float));
  A(q, qd, sizeof(float));
  A(k, kd, sizeof(float));
  A(v, vd, sizeof(float));
  A(attn, od, sizeof(float));
  A(gate, od, sizeof(float));
  A(proj, d, sizeof(float));
  A(mlpo, d, sizeof(float));
  A(y, d, sizeof(float));
  A(x, d, sizeof(float));
  A(lmp, m->embedding.in_padded, sizeof(float));
  A(tmp, qd > d ? qd : d, sizeof(float));
  A(scores, pos + 1, sizeof(float));
  A(ms.proj, 1024, sizeof(float));
  A(ms.cond, cfg->hada_n, sizeof(float));
  A(ms.z, cfg->hada_n, sizeof(float));
  A(ms.buf, cfg->hada_n, sizeof(float));
  A(ms.t, cfg->hada_n, sizeof(float));

  ND_PB(ND_P_ENGRAM);
  push_token(cache, token);
  nh = cache->ntok;
  memcpy(hist, cache->tokens, nh * sizeof(uint32_t));
  here = nh - 1;
  engram_indices(cfg, hist, nh, idx);
  for (s = 0; s < ns; s++) {
    const nd_engram_site *site = &m->engrams[s];
    engram_fetch(m, site, idx + here * nt, pos, fetched, row);
    if (nd_cq_matvec(&site->key_proj, fetched, ek + s * d) || nd_cq_matvec(&site->value_proj, fetched, ev + s * d))
      goto out;
    engram_conv_step(cache, s, site->taps, ev + s * d, tmp);
  }
  ND_PE();
  ND_PB(ND_P_EMBED);

  nd_cq_dequantize_row(&m->embedding, clamp_token(m, token), emb);
  for (i = 0; i < d; i++) emb[i] *= m->embed_scale;
  for (lane = 0; lane < n; lane++) memcpy(lanes + lane * d, emb, d * sizeof(float));
  ND_PE();
  rope_tables(cfg, pos, 1, rc_, rs_);

  for (li = 0; li < L; li++) {
    const nd_layer *ly = &m->layers[li];
    size_t site = (size_t)-1, lo, hi;
    kv_view kv;
    float agate;
    for (s = 0; s < ns; s++)
      if (cfg->sites[s] == li) {
        site = s;
        break;
      }
    ND_PB(ND_P_MHC);
    rms_unit_to(lanes, nx, n * d);
    nd_cq_prepare(&m->phi_pre, nx, nxp);
    nd_cq_matvec_rows_prepared(&m->phi_pre, nxp, li * n, n, hpre);
    if (shared) {
      nd_cq_matvec_rows_prepared(&m->phi_post, nxp, li * n, n, hpost);
      nd_cq_matvec_rows_prepared(&m->phi_res, nxp, li * n * n, n * n, hres);
    }
    mix_down(m, lanes, hpre, li, u);
    ND_PE();
    ND_PB(ND_P_ENGRAM);
    memcpy(bx, u, d * sizeof(float));
    if (site != (size_t)-1) apply_site(bx, ek + site * d, ev + site * d, d);
    ND_PE();
    ND_PB(ND_P_QKV);
    memcpy(hh, bx, d * sizeof(float));
    zc_rms_norm(hh, ly->norm_in, d);
    if (nd_cq_matvec(&ly->q_proj, hh, q) || nd_cq_matvec(&ly->k_proj, hh, k) || nd_cq_matvec(&ly->v_proj, hh, v))
      goto out;
    ND_PE();
    ND_PB(ND_P_CONV_ROPE);
    if (cfg->qkv_conv_taps) {
      conv_step(cache, li, 0, ly->q_taps, q, qd, tmp);
      conv_step(cache, li, 1, ly->k_taps, k, kd, tmp);
      conv_step(cache, li, 2, ly->v_taps, v, vd, tmp);
    }
    norm_and_rope(q, ly->q_norm, rc_, rs_, 1, H, qk);
    norm_and_rope(k, ly->k_norm, rc_, rs_, 1, KV, qk);
    if (cache->prec == ND_KV_INT8) quant_rows(q, H, qk);
    ND_PE();
    ND_PB(ND_P_ATTN);
    if (write_kv(cache, li, pos, k, v)) goto out;
    cache_span(cache, li, pos, &lo, &hi);
    kv = cache_kv(cache, li);
    attend_step(cfg, q, &kv, cache->layers[li].slots, lo, hi, attn, scores);
    ND_PE();
    ND_PB(ND_P_GATE_OUT);
    if (nd_cq_matvec(&ly->gate_proj, hh, gate)) goto out;
    for (i = 0; i < od; i++) attn[i] *= sigmoid(gate[i]);
    if (nd_cq_matvec(&ly->out_proj, attn, proj)) goto out;
    zc_rms_norm(proj, ly->post_norm, d);
    agate = sigmoid(ly->attn_gate);
    for (i = 0; i < d; i++) bx[i] += agate * proj[i];
    ND_PE();
    ND_PB(ND_P_MLP);
    memcpy(proj, bx, d * sizeof(float));
    zc_rms_norm(proj, ly->pre_hada, d);
    hadamard_mlp(m, proj, &ly->mlp, mlpo, &ms);
    for (i = 0; i < d; i++) bx[i] += mlpo[i];
    ND_PE();
    ND_PB(ND_P_MHC);
    for (c = 0; c < d; c++) y[c] = bx[c] - u[c];
    if (!shared) {
      rms_unit_to(lanes, nx, n * d);
      nd_cq_prepare(&m->phi_post, nx, nxp);
      nd_cq_matvec_rows_prepared(&m->phi_post, nxp, li * n, n, hpost);
      nd_cq_prepare(&m->phi_res, nx, nxp);
      nd_cq_matvec_rows_prepared(&m->phi_res, nxp, li * n * n, n * n, hres);
    }
    scatter_up(m, lanes, hpost, hres, y, li, scratch);
    ND_PE();
  }
  cache->pos++;
  ND_PB(ND_P_HEAD);
  for (c = 0; c < d; c++) {
    float a = 0.0f;
    for (lane = 0; lane < n; lane++) a += lanes[lane * d + c];
    x[c] = a / (float)n;
  }
  zc_rms_norm(x, m->final_norm, d);
  nd_cq_prepare(&m->embedding, x, lmp);
  nd_cq_matvec_rows_prepared(&m->embedding, lmp, 0, rows, logits);
  ND_PE();
  rc = ND_OK;
out:
  for (i = 0; i < na; i++) free(all[i]);
#undef A
  return rc;
}
