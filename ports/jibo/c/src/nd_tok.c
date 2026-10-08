/* nd_tok.c: transcription of crates/needle-infer/src/sp_tokenizer.rs. See nd_tok.h. */
#include "nd_tok.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define HDR_BYTES 24u
#define REC_FIXED 7u
#define NONE ((size_t)-1)

struct nd_tok {
  uint32_t n;
  char *pool;        /* every surface, each followed by a NUL */
  size_t *off;       /* surface i is pool[off[i] .. off[i] + len[i]) */
  uint16_t *len;
  float *scores;
  uint8_t *types;
  /* Surface -> id, open addressing; slots hold id + 1, 0 is empty. Later duplicates win. */
  uint32_t *slots;
  size_t mask;
  /* Byte value -> id + 1 (0: none), for byte fallback. */
  uint32_t byte_to_id[256];
  /* User-defined piece ids, longest surface first, equal lengths in table order. */
  uint32_t *markers;
  size_t n_markers;
  uint32_t pad_id, eos_id, bos_id, unk_id;
  int add_dummy_prefix, byte_fallback;
};

/* ---- UTF-8 ------------------------------------------------------------------------------ */

/* One step of core::str::lossy::Utf8Chunks over s[i..len): returns the number of bytes the
 * step consumes and sets *ok to whether they form a valid character. An invalid step consumes
 * exactly the maximal subpart (at least one byte), as Rust does. */
static size_t utf8_step(const uint8_t *s, size_t len, size_t i, int *ok) {
  uint8_t b = s[i], c;
#define GET(k) ((i + (k)) < len ? s[i + (k)] : 0u)
#define CONT(x) ((x) >= 0x80u && (x) <= 0xBFu)
  *ok = 0;
  if (b < 0x80u) {
    *ok = 1;
    return 1;
  }
  if (b >= 0xC2u && b <= 0xDFu) {
    if (!CONT(GET(1))) return 1;
    *ok = 1;
    return 2;
  }
  if (b >= 0xE0u && b <= 0xEFu) {
    c = (uint8_t)GET(1);
    if (b == 0xE0u) {
      if (!(c >= 0xA0u && c <= 0xBFu)) return 1;
    } else if (b == 0xEDu) {
      if (!(c >= 0x80u && c <= 0x9Fu)) return 1;
    } else if (!CONT(c)) {
      return 1;
    }
    if (!CONT(GET(2))) return 2;
    *ok = 1;
    return 3;
  }
  if (b >= 0xF0u && b <= 0xF4u) {
    c = (uint8_t)GET(1);
    if (b == 0xF0u) {
      if (!(c >= 0x90u && c <= 0xBFu)) return 1;
    } else if (b == 0xF4u) {
      if (!(c >= 0x80u && c <= 0x8Fu)) return 1;
    } else if (!CONT(c)) {
      return 1;
    }
    if (!CONT(GET(2))) return 2;
    if (!CONT(GET(3))) return 3;
    *ok = 1;
    return 4;
  }
  return 1;
#undef GET
#undef CONT
}

int nd_utf8_valid(const uint8_t *s, size_t len) {
  size_t i = 0;
  int ok;
  while (i < len) {
    i += utf8_step(s, len, i, &ok);
    if (!ok) return 0;
  }
  return 1;
}

char *nd_utf8_lossy(const uint8_t *s, size_t len, size_t *out_len) {
  size_t cap, i = 0, o = 0, k;
  int ok;
  char *out;
  /* Each invalid byte becomes at most 3 bytes. */
  if (!nd_mul_ok(len, 3, &cap) || !nd_add_ok(cap, 1, &cap)) return NULL;
  out = (char *)malloc(cap);
  if (!out) return NULL;
  while (i < len) {
    k = utf8_step(s, len, i, &ok);
    if (ok) {
      memcpy(out + o, s + i, k);
      o += k;
    } else {
      out[o++] = (char)0xEF;
      out[o++] = (char)0xBF;
      out[o++] = (char)0xBD;
    }
    i += k;
  }
  out[o] = '\0';
  if (out_len) *out_len = o;
  return out;
}

/* Length of the (valid) UTF-8 character starting with lead byte b. */
static size_t utf8_len(uint8_t b) {
  if (b < 0x80u) return 1;
  if (b < 0xE0u) return 2;
  if (b < 0xF0u) return 3;
  return 4;
}

/* ---- hash table ------------------------------------------------------------------------- */

static size_t hash_bytes(const char *p, size_t n) {
  uint64_t h = 1469598103934665603ull;
  size_t i;
  for (i = 0; i < n; i++) {
    h ^= (uint8_t)p[i];
    h *= 1099511628211ull;
  }
  return (size_t)(h ^ (h >> 32));
}

static int32_t lookup(const nd_tok *t, const char *p, size_t n) {
  size_t h = hash_bytes(p, n) & t->mask;
  for (;;) {
    uint32_t s = t->slots[h];
    if (s == 0) return -1;
    if (t->len[s - 1] == n && memcmp(t->pool + t->off[s - 1], p, n) == 0) return (int32_t)(s - 1);
    h = (h + 1) & t->mask;
  }
}

static void insert(nd_tok *t, uint32_t id) {
  const char *p = t->pool + t->off[id];
  size_t n = t->len[id];
  size_t h = hash_bytes(p, n) & t->mask;
  for (;;) {
    uint32_t s = t->slots[h];
    if (s == 0 || (t->len[s - 1] == n && memcmp(t->pool + t->off[s - 1], p, n) == 0)) {
      t->slots[h] = id + 1; /* a later duplicate overwrites the earlier id */
      return;
    }
    h = (h + 1) & t->mask;
  }
}

/* ---- loading ---------------------------------------------------------------------------- */

static uint32_t rd_u32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int hexval(uint8_t c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

/* `p.get(3..5).and_then(|h| u8::from_str_radix(h, 16).ok())`. from_str_radix on an unsigned
 * type accepts a leading '+' (but not a lone one, and not '-'). The char-boundary condition of
 * `get` only matters for non-ASCII bytes, which never parse. */
static int parse_byte_piece(const char *p, size_t n, uint8_t *out) {
  int a, b;
  if (n < 5) return 0;
  if (p[3] == '+') {
    b = hexval((uint8_t)p[4]);
    if (b < 0) return 0;
    *out = (uint8_t)b;
    return 1;
  }
  a = hexval((uint8_t)p[3]);
  b = hexval((uint8_t)p[4]);
  if (a < 0 || b < 0) return 0;
  *out = (uint8_t)(a * 16 + b);
  return 1;
}

static size_t sat_add(size_t a, size_t b) {
  size_t r;
  return nd_add_ok(a, b, &r) ? r : (size_t)-1;
}
static size_t sat_mul(size_t a, size_t b) {
  size_t r;
  return nd_mul_ok(a, b, &r) ? r : (size_t)-1;
}

/* Stable sort of marker ids by surface length, longest first (bottom-up merge sort). */
static void sort_markers(const nd_tok *t, uint32_t *a, uint32_t *tmp, size_t n) {
  size_t w, lo, mid, hi, i, j, k;
  for (w = 1; w < n; w *= 2) {
    for (lo = 0; lo < n; lo += 2 * w) {
      mid = lo + w < n ? lo + w : n;
      hi = lo + 2 * w < n ? lo + 2 * w : n;
      i = lo;
      j = mid;
      k = lo;
      while (i < mid && j < hi) {
        /* take from the right only when strictly longer: keeps equal lengths in order */
        if (t->len[a[j]] > t->len[a[i]])
          tmp[k++] = a[j++];
        else
          tmp[k++] = a[i++];
      }
      while (i < mid) tmp[k++] = a[i++];
      while (j < hi) tmp[k++] = a[j++];
    }
    memcpy(a, tmp, n * sizeof *a);
    if (w > n / 2) break;
  }
}

void nd_tok_free(nd_tok *t) {
  if (!t) return;
  free(t->pool);
  free(t->off);
  free(t->len);
  free(t->scores);
  free(t->types);
  free(t->slots);
  free(t->markers);
  free(t);
}

int nd_tok_load(const uint8_t *blob, size_t len, nd_tok **out, nd_err *e) {
  size_t n, most, off, i, pool_bytes, cap, o;
  uint32_t ids[4];
  static const char *const names[4] = {"pad", "eos", "bos", "unk"};
  nd_tok *t = NULL;
  uint32_t *tmp = NULL;

  if (out) *out = NULL;
  if (!out || (!blob && len)) {
    nd_seterr(e, "nd_tok_load: bad arguments");
    return ND_E_ARG;
  }
  if (len < HDR_BYTES) {
    nd_seterr(e, "tokenizer blob truncated: need %u bytes, got %lu", HDR_BYTES, (unsigned long)len);
    return ND_E_FORMAT;
  }
  n = rd_u32(blob);
  ids[0] = rd_u32(blob + 4);
  ids[1] = rd_u32(blob + 8);
  ids[2] = rd_u32(blob + 12);
  ids[3] = rd_u32(blob + 16);
  if (n == 0) {
    nd_seterr(e, "tokenizer blob declares zero pieces");
    return ND_E_FORMAT;
  }
  /* Every piece needs at least its fixed record, so the blob bounds the count; checked
   * before anything is sized from it. */
  most = (len - HDR_BYTES) / REC_FIXED;
  if (n > most) {
    nd_seterr(e, "tokenizer blob truncated: need %lu bytes, got %lu",
              (unsigned long)sat_add(HDR_BYTES, sat_mul(n, REC_FIXED)), (unsigned long)len);
    return ND_E_FORMAT;
  }

  /* First pass: bounds and UTF-8, in the order Rust reports them. */
  off = HDR_BYTES;
  pool_bytes = 0;
  for (i = 0; i < n; i++) {
    size_t plen;
    if (off + REC_FIXED > len) { /* off <= len, so no wrap */
      nd_seterr(e, "tokenizer blob truncated: need %lu bytes, got %lu",
                (unsigned long)(off + REC_FIXED), (unsigned long)len);
      return ND_E_FORMAT;
    }
    plen = (size_t)blob[off + 5] | ((size_t)blob[off + 6] << 8);
    off += REC_FIXED;
    if (plen > len - off) {
      nd_seterr(e, "tokenizer blob truncated: need %lu bytes, got %lu",
                (unsigned long)sat_add(off, plen), (unsigned long)len);
      return ND_E_FORMAT;
    }
    if (!nd_utf8_valid(blob + off, plen)) {
      nd_seterr(e, "piece %lu is not valid UTF-8", (unsigned long)i);
      return ND_E_FORMAT;
    }
    off += plen;
    /* pool_bytes <= len - HDR_BYTES + n <= 2 * len, but check anyway */
    if (!nd_add_ok(pool_bytes, plen + 1, &pool_bytes)) {
      nd_seterr(e, "tokenizer blob too large");
      return ND_E_BOUNDS;
    }
  }
  for (i = 0; i < 4; i++) {
    if ((size_t)ids[i] >= n) {
      nd_seterr(e, "header %s=%lu is outside the %lu-piece table", names[i], (unsigned long)ids[i],
                (unsigned long)n);
      return ND_E_FORMAT;
    }
  }

  /* Table size: a power of two at least 2n. n <= len / 7. */
  cap = 2;
  while (cap < n) {
    if (cap > ((size_t)-1) / 4) {
      nd_seterr(e, "tokenizer blob too large");
      return ND_E_BOUNDS;
    }
    cap *= 2;
  }
  cap *= 2;

  t = (nd_tok *)calloc(1, sizeof *t);
  if (!t) goto nomem;
  t->n = (uint32_t)n;
  t->pool = (char *)nd_calloc(pool_bytes, 1);
  t->off = (size_t *)nd_calloc(n, sizeof *t->off);
  t->len = (uint16_t *)nd_calloc(n, sizeof *t->len);
  t->scores = (float *)nd_calloc(n, sizeof *t->scores);
  t->types = (uint8_t *)nd_calloc(n, 1);
  t->slots = (uint32_t *)nd_calloc(cap, sizeof *t->slots);
  if (!t->pool || !t->off || !t->len || !t->scores || !t->types || !t->slots) goto nomem;
  t->mask = cap - 1;
  t->pad_id = ids[0];
  t->eos_id = ids[1];
  t->bos_id = ids[2];
  t->unk_id = ids[3];
  t->add_dummy_prefix = blob[20] != 0;
  t->byte_fallback = blob[21] != 0;

  off = HDR_BYTES;
  o = 0;
  for (i = 0; i < n; i++) {
    uint32_t bits = rd_u32(blob + off);
    size_t plen = (size_t)blob[off + 5] | ((size_t)blob[off + 6] << 8);
    memcpy(&t->scores[i], &bits, sizeof bits);
    t->types[i] = blob[off + 4];
    t->len[i] = (uint16_t)plen;
    off += REC_FIXED;
    t->off[i] = o;
    memcpy(t->pool + o, blob + off, plen);
    t->pool[o + plen] = '\0';
    o += plen + 1;
    off += plen;
  }

  for (i = 0; i < n; i++) insert(t, (uint32_t)i);

  for (i = 0; i < n; i++) {
    uint8_t b;
    if (t->types[i] != ND_TK_BYTE) continue;
    if (!parse_byte_piece(t->pool + t->off[i], t->len[i], &b)) {
      nd_seterr(e, "byte piece %lu is not of the form <0xNN>", (unsigned long)i);
      nd_tok_free(t);
      return ND_E_FORMAT;
    }
    t->byte_to_id[b] = (uint32_t)i + 1;
  }

  for (i = 0; i < n; i++)
    if (t->types[i] == ND_TK_USER_DEFINED) t->n_markers++;
  t->markers = (uint32_t *)nd_calloc(t->n_markers, sizeof *t->markers);
  tmp = (uint32_t *)nd_calloc(t->n_markers, sizeof *tmp);
  if (!t->markers || !tmp) goto nomem;
  o = 0;
  for (i = 0; i < n; i++)
    if (t->types[i] == ND_TK_USER_DEFINED) t->markers[o++] = (uint32_t)i;
  sort_markers(t, t->markers, tmp, t->n_markers);
  free(tmp);

  *out = t;
  return ND_OK;

nomem:
  free(tmp);
  nd_tok_free(t);
  nd_seterr(e, "out of memory loading tokenizer");
  return ND_E_NOMEM;
}

/* ---- accessors -------------------------------------------------------------------------- */

size_t nd_tok_size(const nd_tok *t) { return t->n; }
uint32_t nd_tok_pad_id(const nd_tok *t) { return t->pad_id; }
uint32_t nd_tok_eos_id(const nd_tok *t) { return t->eos_id; }
uint32_t nd_tok_bos_id(const nd_tok *t) { return t->bos_id; }
uint32_t nd_tok_unk_id(const nd_tok *t) { return t->unk_id; }
int nd_tok_add_dummy_prefix(const nd_tok *t) { return t->add_dummy_prefix; }
int nd_tok_byte_fallback(const nd_tok *t) { return t->byte_fallback; }

const char *nd_tok_piece(const nd_tok *t, uint32_t id, size_t *len) {
  if (id >= t->n) return NULL;
  if (len) *len = t->len[id];
  return t->pool + t->off[id];
}

int nd_tok_type(const nd_tok *t, uint32_t id) { return id < t->n ? (int)t->types[id] : -1; }

float nd_tok_score(const nd_tok *t, uint32_t id) { return id < t->n ? t->scores[id] : 0.0f; }

int32_t nd_tok_id_of_n(const nd_tok *t, const char *piece, size_t len) {
  if (!piece && len) return -1;
  return lookup(t, piece ? piece : "", len);
}

int32_t nd_tok_id_of(const nd_tok *t, const char *piece) {
  if (!piece) return -1;
  return lookup(t, piece, strlen(piece));
}

long nd_tok_token_bytes(const nd_tok *t, uint32_t id, uint8_t *buf, size_t cap) {
  const char *p;
  size_t n, i, o = 0;
  uint8_t b;
  if (id >= t->n) return -1;
  p = t->pool + t->off[id];
  n = t->len[id];
  switch (t->types[id]) {
    case ND_TK_BYTE:
      if (!parse_byte_piece(p, n, &b)) return 0;
      if (cap > 0) buf[0] = b;
      return 1;
    case ND_TK_CONTROL:
    case ND_TK_UNKNOWN:
      return 0;
    default:
      for (i = 0; i < n;) {
        if (n - i >= 3 && memcmp(p + i, ND_META_SPACE, 3) == 0) {
          if (o < cap) buf[o] = ' ';
          o++;
          i += 3;
        } else {
          if (o < cap) buf[o] = (uint8_t)p[i];
          o++;
          i++;
        }
      }
      return (long)o;
  }
}

const char *const nd_chat_markers[ND_N_CHAT_MARKERS] = {
    "<|im_start|>", "<|im_end|>", "<think>",       "</think>",       "<tools>",
    "</tools>",     "<tool_call>", "</tool_call>", "<tool_result>", "</tool_result>"};

int nd_tok_chat_marker_ids(const nd_tok *t, uint32_t out[ND_N_CHAT_MARKERS]) {
  int i;
  for (i = 0; i < ND_N_CHAT_MARKERS; i++) {
    int32_t id = nd_tok_id_of(t, nd_chat_markers[i]);
    if (id < 0) return ND_E_FORMAT;
    out[i] = (uint32_t)id;
  }
  return ND_OK;
}

/* ---- encode ----------------------------------------------------------------------------- */

typedef struct {
  uint32_t *v;
  size_t n, cap;
  int err;
} idvec;

static void push(idvec *a, uint32_t id) {
  if (a->err) return;
  if (a->n == a->cap) {
    size_t nc, bytes;
    uint32_t *nv;
    nc = a->cap ? a->cap : 16;
    if (a->cap && !nd_mul_ok(a->cap, 2, &nc)) {
      a->err = ND_E_BOUNDS;
      return;
    }
    if (!nd_mul_ok(nc, sizeof *nv, &bytes)) {
      a->err = ND_E_BOUNDS;
      return;
    }
    nv = (uint32_t *)realloc(a->v, bytes);
    if (!nv) {
      a->err = ND_E_NOMEM;
      return;
    }
    a->v = nv;
    a->cap = nc;
  }
  a->v[a->n++] = id;
}

/* Greedy highest-score pairwise merging over the characters of seg[0..len) (RefTokenizer._bpe),
 * keeping the leftmost pair at the maximum score. starts/ends/next have room for every char. */
static void bpe_into(const nd_tok *t, const char *seg, size_t len, size_t *starts, size_t *ends,
                     size_t *next, idvec *out) {
  size_t n = 0, b = 0, live, j, k, i;
  if (len == 0) return;
  while (b < len) {
    size_t cl = utf8_len((uint8_t)seg[b]);
    starts[n] = b;
    ends[n] = b + cl;
    n++;
    b += cl;
  }
  for (i = 0; i < n; i++) next[i] = i + 1;
  next[n - 1] = NONE;
  live = n;

  while (live > 1) {
    float best_score = -INFINITY;
    size_t best = NONE;
    j = 0;
    while (next[j] != NONE) {
      int32_t id;
      k = next[j];
      id = lookup(t, seg + starts[j], ends[k] - starts[j]);
      if (id >= 0) {
        float s = t->scores[id];
        if (best == NONE || s > best_score) {
          best_score = s;
          best = j;
        }
      }
      j = k;
    }
    if (best == NONE) break;
    k = next[best];
    ends[best] = ends[k];
    next[best] = next[k];
    live--;
  }

  j = 0;
  while (j != NONE) {
    const char *sym = seg + starts[j];
    size_t sl = ends[j] - starts[j];
    int32_t id = lookup(t, sym, sl);
    if (id >= 0) {
      push(out, (uint32_t)id);
    } else if (t->byte_fallback) {
      for (i = 0; i < sl; i++) {
        uint32_t bid = t->byte_to_id[(uint8_t)sym[i]];
        push(out, bid ? bid - 1 : t->unk_id);
      }
    } else {
      push(out, t->unk_id);
    }
    j = next[j];
  }
}

int nd_tok_encode(const nd_tok *t, const char *text, size_t len, uint32_t **ids, size_t *n) {
  size_t spaces = 0, esc_len, extra, i, o, nchars, seg, bytes;
  char *esc = NULL;
  size_t *starts = NULL, *ends = NULL, *next = NULL;
  idvec out = {NULL, 0, 0, 0};

  if (!ids || !n) return ND_E_ARG;
  *ids = NULL;
  *n = 0;
  if (!t || (!text && len)) return ND_E_ARG;
  if (len == 0) return ND_OK;
  if (!nd_utf8_valid((const uint8_t *)text, len)) return ND_E_ARG;

  /* text.replace(' ', META_SPACE), with META_SPACE inserted in front for add_dummy_prefix. */
  for (i = 0; i < len; i++)
    if (text[i] == ' ') spaces++;
  if (!nd_mul_ok(spaces, 2, &extra) || !nd_add_ok(len, extra, &esc_len) ||
      !nd_add_ok(esc_len, t->add_dummy_prefix ? 3 : 0, &esc_len) || !nd_add_ok(esc_len, 1, &bytes))
    return ND_E_BOUNDS;
  esc = (char *)malloc(bytes);
  if (!esc) return ND_E_NOMEM;
  o = 0;
  if (t->add_dummy_prefix) {
    memcpy(esc, ND_META_SPACE, 3);
    o = 3;
  }
  for (i = 0; i < len; i++) {
    if (text[i] == ' ') {
      memcpy(esc + o, ND_META_SPACE, 3);
      o += 3;
    } else {
      esc[o++] = text[i];
    }
  }
  esc[o] = '\0';

  /* Scratch for the merge lists, sized for the whole escaped text. */
  nchars = 0;
  for (i = 0; i < esc_len; i += utf8_len((uint8_t)esc[i])) nchars++;
  starts = (size_t *)nd_calloc(nchars, sizeof *starts);
  ends = (size_t *)nd_calloc(nchars, sizeof *ends);
  next = (size_t *)nd_calloc(nchars, sizeof *next);
  if (!starts || !ends || !next) {
    out.err = ND_E_NOMEM;
    goto done;
  }

  /* Split off the user-defined pieces, longest match first; the text between them is one
   * contiguous segment of `esc`. */
  i = 0;
  seg = 0;
  while (i < esc_len) {
    size_t m, hit = NONE, rem = esc_len - i;
    for (m = 0; m < t->n_markers; m++) {
      uint32_t id = t->markers[m];
      size_t ml = t->len[id];
      /* An empty marker would match everywhere and make Rust loop forever; skip it. */
      if (ml == 0 || ml > rem) continue;
      if (memcmp(esc + i, t->pool + t->off[id], ml) == 0) {
        hit = m;
        break;
      }
    }
    if (hit != NONE) {
      uint32_t id = t->markers[hit];
      bpe_into(t, esc + seg, i - seg, starts, ends, next, &out);
      push(&out, id);
      i += t->len[id];
      seg = i;
    } else {
      i += utf8_len((uint8_t)esc[i]);
    }
  }
  bpe_into(t, esc + seg, esc_len - seg, starts, ends, next, &out);

done:
  free(esc);
  free(starts);
  free(ends);
  free(next);
  if (out.err) {
    free(out.v);
    return out.err;
  }
  *ids = out.v;
  *n = out.n;
  return ND_OK;
}

/* ---- decode ----------------------------------------------------------------------------- */

int nd_tok_decode(const nd_tok *t, const uint32_t *ids, size_t n, char **text, size_t *len) {
  size_t total = 0, i, o, tl, r, w;
  uint8_t *buf;
  char *s;

  if (!text) return ND_E_ARG;
  *text = NULL;
  if (len) *len = 0;
  if (!t || (!ids && n)) return ND_E_ARG;

  for (i = 0; i < n; i++) {
    uint32_t id = ids[i];
    if (id >= t->n) continue;
    if (t->types[id] == ND_TK_BYTE) {
      if (!nd_add_ok(total, 1, &total)) return ND_E_BOUNDS;
    } else if (t->types[id] != ND_TK_CONTROL && t->types[id] != ND_TK_UNKNOWN) {
      if (!nd_add_ok(total, t->len[id], &total)) return ND_E_BOUNDS;
    }
  }
  buf = (uint8_t *)malloc(total ? total : 1);
  if (!buf) return ND_E_NOMEM;
  o = 0;
  for (i = 0; i < n; i++) {
    uint32_t id = ids[i];
    uint8_t b;
    if (id >= t->n) continue;
    switch (t->types[id]) {
      case ND_TK_BYTE:
        if (parse_byte_piece(t->pool + t->off[id], t->len[id], &b)) buf[o++] = b;
        break;
      case ND_TK_CONTROL:
      case ND_TK_UNKNOWN:
        break;
      default:
        memcpy(buf + o, t->pool + t->off[id], t->len[id]);
        o += t->len[id];
        break;
    }
  }

  s = nd_utf8_lossy(buf, o, &tl);
  free(buf);
  if (!s) return ND_E_NOMEM;

  /* .replace(META_SPACE, " "): in valid UTF-8 the bytes E2 96 81 can only be U+2581. */
  for (r = 0, w = 0; r < tl;) {
    if (tl - r >= 3 && (uint8_t)s[r] == 0xE2u && (uint8_t)s[r + 1] == 0x96u &&
        (uint8_t)s[r + 2] == 0x81u) {
      s[w++] = ' ';
      r += 3;
    } else {
      s[w++] = s[r++];
    }
  }
  s[w] = '\0';
  tl = w;

  if (t->add_dummy_prefix && tl > 0 && s[0] == ' ') {
    memmove(s, s + 1, tl); /* includes the NUL */
    tl--;
  }
  *text = s;
  if (len) *len = tl;
  return ND_OK;
}
