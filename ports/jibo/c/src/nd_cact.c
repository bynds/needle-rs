/* nd_cact.c: see nd_cact.h. Transcribes CactV3::from_bytes, CactV3Geometry::from_words and
 * check_bounds, parse_directory and decode_floats. */
#include "nd_cact.h"

#include <stdlib.h>
#include <string.h>

static uint32_t rd32(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static uint64_t rd64(const uint8_t *p) { return (uint64_t)rd32(p) | (uint64_t)rd32(p + 4) << 32; }

float nd_f16_to_f32(uint16_t bits) {
  uint32_t sign = (uint32_t)(bits >> 15) << 31;
  uint32_t ex = (bits >> 10) & 0x1Fu;
  uint32_t mant = bits & 0x3FFu;
  uint32_t out;
  float f;
  if (ex == 0) {
    float val = (float)mant / (float)(1u << 24);
    return sign ? -val : val;
  }
  if (ex == 31) {
    out = sign | 0x7F800000u | (mant << 13);
  } else {
    out = sign | ((ex + 127 - 15) << 23) | (mant << 13);
  }
  memcpy(&f, &out, 4);
  return f;
}

/* CactV3Geometry::check_bounds: products in u64, before anything is sized from a field. */
static int check_bounds(const nd_geom *g, size_t file_len, nd_err *e) {
  const uint64_t DIM = 1u << 16;
  uint64_t records = file_len > ND_HEADER_BYTES ? (file_len - ND_HEADER_BYTES) / ND_REC_BYTES : 0;
  struct {
    const char *name;
    uint64_t value, max;
  } t[] = {
      {"num_tensors", g->num_tensors, records},
      {"num_layers", g->num_layers, 64},
      {"vocab_size", g->vocab_size, 1u << 20},
      {"out_vocab", g->out_vocab, g->vocab_size},
      {"d_model", g->d_model, DIM},
      {"num_heads", g->num_heads, DIM},
      {"num_kv_heads", g->num_kv_heads, DIM},
      {"qk_head_dim", g->qk_head_dim, DIM},
      {"v_head_dim", g->v_head_dim, DIM},
      {"hada_n", g->hada_n, DIM},
      {"engram_sub_dim", g->engram_sub_dim, DIM},
      {"mhc_lanes", g->mhc_lanes, 64},
      {"num_heads * qk_head_dim", (uint64_t)g->num_heads * g->qk_head_dim, DIM},
      {"num_heads * v_head_dim", (uint64_t)g->num_heads * g->v_head_dim, DIM},
      {"mhc_lanes * d_model", (uint64_t)g->mhc_lanes * g->d_model, DIM},
      {"max_seq_len", g->max_seq_len, 1u << 20},
      {"sliding_window", g->sliding_window, 1u << 20},
      {"kv_window", g->kv_window, 1u << 20},
      {"qkv_conv_taps", g->qkv_conv_taps, 64},
      {"engram_conv_taps", g->engram_conv_taps, 64},
      {"engram_conv_dilation", g->engram_conv_dilation, 64},
      {"engram_seed_heads", g->engram_seed_heads, 1u << 10},
      {"num_engram_tables", g->num_engram_tables, 1u << 10},
      {"engram_slots", g->engram_slots, 1u << 24},
      {"num_engram_tables * engram_slots", (uint64_t)g->num_engram_tables * g->engram_slots, 1u << 28},
  };
  size_t i;
  for (i = 0; i < sizeof t / sizeof t[0]; i++) {
    if (t[i].value > t[i].max) {
      nd_seterr(e, "header field %s is %llu, above the bound %llu", t[i].name,
                (unsigned long long)t[i].value, (unsigned long long)t[i].max);
      return ND_E_BOUNDS;
    }
  }
  for (i = 0; i < g->n_orders; i++) {
    if (g->engram_orders[i] > 16) {
      nd_seterr(e, "engram order %zu above 16", g->engram_orders[i]);
      return ND_E_BOUNDS;
    }
  }
  for (i = 0; i < g->n_sites; i++) {
    if (g->num_layers == 0 || g->engram_sites[i] > g->num_layers - 1) {
      nd_seterr(e, "engram site %zu beyond the layers", g->engram_sites[i]);
      return ND_E_BOUNDS;
    }
  }
  return ND_OK;
}

int nd_cact_parse(const uint8_t *raw, size_t len, nd_cact *out, nd_err *e) {
  uint32_t w[49];
  nd_geom *g;
  size_t i, dir_start, dir_end;
  memset(out, 0, sizeof *out);
  if (len < ND_HEADER_BYTES) {
    nd_seterr(e, "cact truncated: need at least %d bytes, got %zu", ND_HEADER_BYTES, len);
    return ND_E_FORMAT;
  }
  for (i = 0; i < 49; i++) w[i] = rd32(raw + i * 4);
  if (w[0] != ND_TAG_V3) {
    nd_seterr(e, "not a Needle 3 cact blob: tag 0x%08x", w[0]);
    return ND_E_FORMAT;
  }
  g = &out->g;
  g->num_tensors = w[1];
  g->codebook_len = w[2];
  g->kv_window = w[3];
  g->kv_bits = w[4];
  g->vocab_size = w[5];
  g->out_vocab = w[6];
  g->d_model = w[7];
  g->num_heads = w[8];
  g->num_kv_heads = w[9];
  g->num_layers = w[10];
  g->qk_head_dim = w[11];
  g->v_head_dim = w[12];
  g->max_seq_len = w[13];
  g->hada_n = w[14];
  g->mhc_lanes = w[15];
  g->sliding_window = w[16];
  g->global_mask = (uint64_t)w[17] | (uint64_t)w[18] << 32;
  g->qkv_conv_taps = w[19];
  g->engram_slots = w[20];
  g->engram_sub_dim = w[21];
  g->num_engram_tables = w[22];
  g->engram_conv_taps = w[23];
  g->engram_conv_dilation = w[24];
  g->engram_seed_heads = w[25];
  g->n_orders = w[26] < 4 ? w[26] : 4;
  for (i = 0; i < g->n_orders; i++) g->engram_orders[i] = w[27 + i];
  g->n_sites = w[31] < 16 ? w[31] : 16;
  for (i = 0; i < g->n_sites; i++) g->engram_sites[i] = w[32 + i];
  memcpy(&g->rope_theta, &w[48], 4);

  if (g->codebook_len != ND_CODEBOOK_LEN) {
    nd_seterr(e, "header codebook is %zu floats, want %d (cb2|cb3|cb4)", g->codebook_len, ND_CODEBOOK_LEN);
    return ND_E_FORMAT;
  }
  if (check_bounds(g, len, e)) return ND_E_BOUNDS;
  dir_start = ND_HEADER_BYTES + ND_CODEBOOK_LEN * 4;
  dir_end = dir_start + g->num_tensors * ND_REC_BYTES; /* bounded by check_bounds */
  if (len < dir_end) {
    nd_seterr(e, "cact truncated: need at least %zu bytes, got %zu", dir_end, len);
    return ND_E_FORMAT;
  }
  for (i = 0; i < ND_CODEBOOK_LEN; i++) {
    uint32_t b = rd32(raw + ND_HEADER_BYTES + i * 4);
    memcpy(&out->codebook[i], &b, 4);
  }
  out->recs = nd_calloc(g->num_tensors, sizeof(nd_record));
  if (!out->recs) return ND_E_NOMEM;
  out->nrec = g->num_tensors;
  for (i = 0; i < g->num_tensors; i++) {
    const uint8_t *b = raw + dir_start + i * ND_REC_BYTES;
    nd_record *r = &out->recs[i];
    uint64_t end;
    r->dtype = b[0];
    r->ndim = b[1];
    if (r->ndim > 4) {
      nd_seterr(e, "tensor %zu has ndim %u, want 0..=4", i, r->ndim);
      nd_cact_free(out);
      return ND_E_FORMAT;
    }
    r->shape[0] = rd32(b + 4);
    r->shape[1] = rd32(b + 8);
    r->shape[2] = rd32(b + 12);
    r->shape[3] = rd32(b + 16);
    r->offset = rd64(b + 20);
    r->nbytes = rd64(b + 28);
    r->group = rd32(b + 36);
    r->bits = (uint8_t)rd32(b + 40);
    end = r->offset + r->nbytes;
    if (end < r->offset) end = UINT64_MAX; /* saturating, as Rust */
    if (end > (uint64_t)len) {
      nd_seterr(e, "tensor %zu spans %llu..+%llu but the file is %zu bytes", i,
                (unsigned long long)r->offset, (unsigned long long)r->nbytes, len);
      nd_cact_free(out);
      return ND_E_FORMAT;
    }
  }
  out->raw = raw;
  out->len = len;
  return ND_OK;
}

void nd_cact_free(nd_cact *c) {
  free(c->recs);
  c->recs = NULL;
  c->nrec = 0;
}

const uint8_t *nd_cact_blob(const nd_cact *c, size_t i, size_t *len) {
  const nd_record *r = &c->recs[i];
  *len = (size_t)r->nbytes; /* r->offset + r->nbytes <= c->len, checked at parse */
  return c->raw + (size_t)r->offset;
}

int nd_cact_floats(const nd_cact *c, size_t i, size_t want, float **out, size_t *n, nd_err *e) {
  const nd_record *r;
  const uint8_t *b;
  size_t blen, count, k;
  float *v;
  *out = NULL;
  if (i >= c->nrec) {
    nd_seterr(e, "tensor %zu is past the directory", i);
    return ND_E_BOUNDS;
  }
  r = &c->recs[i];
  b = nd_cact_blob(c, i, &blen);
  if (r->dtype == ND_DT_FP16) {
    if (blen % 2) goto ragged;
    count = blen / 2;
  } else if (r->dtype == ND_DT_FP32) {
    if (blen % 4) goto ragged;
    count = blen / 4;
  } else {
    nd_seterr(e, "tensor %zu has dtype %u, want %d", i, r->dtype, ND_DT_FP16);
    return ND_E_FORMAT;
  }
  if (want && count != want) {
    nd_seterr(e, "tensor %zu holds %zu values, the geometry needs %zu", i, count, want);
    return ND_E_FORMAT;
  }
  v = nd_calloc(count, sizeof(float));
  if (!v) return ND_E_NOMEM;
  for (k = 0; k < count; k++) {
    if (r->dtype == ND_DT_FP16) {
      v[k] = nd_f16_to_f32((uint16_t)(b[2 * k] | b[2 * k + 1] << 8));
    } else {
      uint32_t u = rd32(b + 4 * k);
      memcpy(&v[k], &u, 4);
    }
  }
  *out = v;
  if (n) *n = count;
  return ND_OK;
ragged:
  nd_seterr(e, "tensor %zu has %zu bytes, not a whole element count", i, blen);
  return ND_E_FORMAT;
}
