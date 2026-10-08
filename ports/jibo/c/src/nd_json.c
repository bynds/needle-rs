/* nd_json.c: see nd_json.h.
 *
 * The number parser is a transcription of serde_json 1.0.149 src/de.rs (parse_integer,
 * parse_number, parse_decimal, parse_exponent, parse_long_integer, parse_decimal_overflow,
 * parse_exponent_overflow and the non-`float_roundtrip` f64_from_parts). The float formatter is a
 * transcription of zmij 1.0.21 src/lib.rs (`write`, `to_decimal_fast`, `to_decimal_schubfach`,
 * the power-of-ten significand computation), with the SIMD digit writers replaced by the scalar
 * equivalent and 128-bit products done in portable 64-bit arithmetic.
 */
#include "nd_json.h"

#include <float.h>
#include <stdlib.h>
#include <string.h>

/* ─── Arena ────────────────────────────────────────────────────────────────────────────────── */

typedef struct arena_chunk {
  struct arena_chunk *next;
  size_t used, cap;
  /* data follows, aligned */
} arena_chunk;

#define ARENA_ALIGN 16u
#define ARENA_HDR ((sizeof(arena_chunk) + ARENA_ALIGN - 1) / ARENA_ALIGN * ARENA_ALIGN)

typedef struct {
  arena_chunk *head;
} arena;

static void *arena_alloc(arena *a, size_t n) {
  size_t need, cap, total;
  arena_chunk *c = a->head;
  if (!nd_add_ok(n, ARENA_ALIGN - 1, &need)) return NULL;
  need = need / ARENA_ALIGN * ARENA_ALIGN;
  if (need == 0) need = ARENA_ALIGN;
  if (c && c->cap - c->used >= need) {
    void *p = (unsigned char *)c + ARENA_HDR + c->used;
    c->used += need;
    return p;
  }
  cap = need > 16384u ? need : 16384u;
  if (!nd_add_ok(cap, ARENA_HDR, &total)) return NULL;
  c = (arena_chunk *)malloc(total);
  if (!c) return NULL;
  c->cap = cap;
  c->used = need;
  c->next = a->head;
  a->head = c;
  return (unsigned char *)c + ARENA_HDR;
}

static void arena_free(arena *a) {
  arena_chunk *c = a->head;
  while (c) {
    arena_chunk *n = c->next;
    free(c);
    c = n;
  }
  a->head = NULL;
}

struct nd_json_doc {
  arena a;
  nd_json_value root;
};

/* ─── UTF-8 ────────────────────────────────────────────────────────────────────────────────── */

int nd_json_utf8_valid(const unsigned char *s, size_t n) {
  size_t i = 0;
  while (i < n) {
    unsigned c = s[i];
    if (c < 0x80u) {
      i++;
      continue;
    }
    if (c >= 0xC2u && c <= 0xDFu) {
      if (n - i < 2 || (s[i + 1] & 0xC0u) != 0x80u) return 0;
      i += 2;
    } else if (c >= 0xE0u && c <= 0xEFu) {
      unsigned c1;
      if (n - i < 3) return 0;
      c1 = s[i + 1];
      if (c == 0xE0u && (c1 < 0xA0u || c1 > 0xBFu)) return 0;
      if (c == 0xEDu && (c1 < 0x80u || c1 > 0x9Fu)) return 0;
      if ((c1 & 0xC0u) != 0x80u || (s[i + 2] & 0xC0u) != 0x80u) return 0;
      i += 3;
    } else if (c >= 0xF0u && c <= 0xF4u) {
      unsigned c1;
      if (n - i < 4) return 0;
      c1 = s[i + 1];
      if (c == 0xF0u && (c1 < 0x90u || c1 > 0xBFu)) return 0;
      if (c == 0xF4u && (c1 < 0x80u || c1 > 0x8Fu)) return 0;
      if ((c1 & 0xC0u) != 0x80u || (s[i + 2] & 0xC0u) != 0x80u || (s[i + 3] & 0xC0u) != 0x80u)
        return 0;
      i += 4;
    } else {
      return 0;
    }
  }
  return 1;
}

/* ─── 128-bit helpers ──────────────────────────────────────────────────────────────────────── */

typedef struct {
  uint64_t hi, lo;
} u128p;

static u128p umul128(uint64_t x, uint64_t y) {
  uint64_t x0 = x & 0xffffffffu, x1 = x >> 32, y0 = y & 0xffffffffu, y1 = y >> 32;
  uint64_t p00 = x0 * y0, p01 = x0 * y1, p10 = x1 * y0, p11 = x1 * y1;
  uint64_t mid = (p00 >> 32) + (p01 & 0xffffffffu) + (p10 & 0xffffffffu);
  u128p r;
  r.lo = (mid << 32) | (p00 & 0xffffffffu);
  r.hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
  return r;
}

static uint64_t umul128_hi64(uint64_t x, uint64_t y) { return umul128(x, y).hi; }

static u128p umul192_hi128(uint64_t x_hi, uint64_t x_lo, uint64_t y) {
  u128p p = umul128(x_hi, y);
  uint64_t lo = p.lo + umul128(x_lo, y).hi;
  u128p r;
  r.hi = p.hi + (uint64_t)(lo < p.lo);
  r.lo = lo;
  return r;
}

static uint64_t umulhi_inexact_to_odd64(uint64_t x_hi, uint64_t x_lo, uint64_t y) {
  u128p p = umul192_hi128(x_hi, x_lo, y);
  return p.hi | (uint64_t)((p.lo >> 1) != 0);
}

static uint32_t umulhi_inexact_to_odd32(uint64_t x_hi, uint32_t y) {
  u128p q = umul128(x_hi, (uint64_t)y);
  uint64_t p = (q.hi << 32) | (q.lo >> 32); /* (x_hi * y) >> 32, truncated to 64 bits */
  return (uint32_t)(p >> 32) | (uint32_t)(((uint32_t)p >> 1) != 0);
}

/* Arithmetic (flooring) shift right, independent of how the compiler shifts negatives. */
static int32_t asr32(int32_t v, int n) {
  if (v >= 0) return v >> n;
  return -(int32_t)((((uint32_t)(-(v + 1))) >> n)) - 1;
}

/* ─── zmij tables ──────────────────────────────────────────────────────────────────────────── */

static const uint64_t ZM_POW10S[28] = {
    0x8000000000000000u, 0xa000000000000000u, 0xc800000000000000u, 0xfa00000000000000u,
    0x9c40000000000000u, 0xc350000000000000u, 0xf424000000000000u, 0x9896800000000000u,
    0xbebc200000000000u, 0xee6b280000000000u, 0x9502f90000000000u, 0xba43b74000000000u,
    0xe8d4a51000000000u, 0x9184e72a00000000u, 0xb5e620f480000000u, 0xe35fa931a0000000u,
    0x8e1bc9bf04000000u, 0xb1a2bc2ec5000000u, 0xde0b6b3a76400000u, 0x8ac7230489e80000u,
    0xad78ebc5ac620000u, 0xd8d726b7177a8000u, 0x878678326eac9000u, 0xa968163f0a57b400u,
    0xd3c21bcecceda100u, 0x84595161401484a0u, 0xa56fa5b99019a5c8u, 0xcecb8f27f4200f3au,
};

static const u128p ZM_HIGH_PARTS[23] = {
    {0xaf8e5410288e1b6fu, 0x07ecf0ae5ee44ddau}, {0xb1442798f49ffb4au, 0x99cd11cfdf41779du},
    {0xb2fe3f0b8599ef07u, 0x861fa7e6dcb4aa15u}, {0xb4bca50b065abe63u, 0x0fed077a756b53aau},
    {0xb67f6455292cbf08u, 0x1a3bc84c17b1d543u}, {0xb84687c269ef3bfbu, 0x3d5d514f40eea742u},
    {0xba121a4650e4ddebu, 0x92f34d62616ce413u}, {0xbbe226efb628afeau, 0x890489f70a55368cu},
    {0xbdb6b8e905cb600fu, 0x5400e987bbc1c921u}, {0xbf8fdb78849a5f96u, 0xde98520472bdd034u},
    {0xc16d9a0095928a27u, 0x75b7053c0f178294u}, {0xc350000000000000u, 0x0000000000000000u},
    {0xc5371912364ce305u, 0x6c28000000000000u}, {0xc722f0ef9d80aad6u, 0x424d3ad2b7b97ef6u},
    {0xc913936dd571c84cu, 0x03bc3a19cd1e38eau}, {0xcb090c8001ab551cu, 0x5cadf5bfd3072cc6u},
    {0xcd036837130890a1u, 0x36dba887c37a8c10u}, {0xcf02b2c21207ef2eu, 0x94f967e45e03f4bcu},
    {0xd106f86e69d785c7u, 0xe13336d701beba52u}, {0xd31045a8341ca07cu, 0x1ede48111209a051u},
    {0xd51ea6fa85785631u, 0x552a74227f3ea566u}, {0xd732290fbacaf133u, 0xa97c177947ad4096u},
    {0xd94ad8b1c7380874u, 0x18375281ae7822bcu},
};

static const uint32_t ZM_FIXUPS[20] = {
    0x05271b1fu, 0x00000c20u, 0x00003200u, 0x12100020u, 0x00000000u, 0x06000000u, 0xc16409c0u,
    0xaf26700fu, 0xeb987b07u, 0x0000000du, 0x00000000u, 0x66fbfffeu, 0xb74100ecu, 0xa0669fe8u,
    0xedb21280u, 0x00000686u, 0x0a021200u, 0x29b89c20u, 0x08bc0edau, 0x00000000u,
};

/* Pow10SignificandsTable::compute(i): the 128-bit significand of 10**(i - 292), rounded down. */
static u128p zm_pow10_compute(uint32_t i) {
  uint64_t m = ZM_POW10S[(i + 11u) % 28u];
  u128p h = ZM_HIGH_PARTS[(i + 11u) / 28u];
  uint64_t h1 = umul128_hi64(h.lo, m);
  uint64_t c0 = h.lo * m;
  uint64_t c1 = h1 + h.hi * m;
  uint64_t c2 = (uint64_t)(c1 < h1) + umul128_hi64(h.hi, m);
  u128p r;
  if ((c2 >> 63) != 0) {
    r.hi = c2;
    r.lo = c1;
  } else {
    r.hi = (c2 << 1) | (c1 >> 63);
    r.lo = (c1 << 1) | (c0 >> 63);
  }
  r.lo -= (uint64_t)((ZM_FIXUPS[i >> 5] >> (i & 31u)) & 1u);
  return r;
}

/* POW10_SIGNIFICANDS.get_unchecked(dec_exp) */
static u128p zm_pow10(int32_t dec_exp) { return zm_pow10_compute((uint32_t)(dec_exp + 292)); }

static int32_t zm_compute_dec_exp(int32_t bin_exp, int regular) {
  return asr32(bin_exp * 315653 - (regular ? 0 : 1) * 131072, 20);
}

static unsigned zm_compute_exp_shift(int32_t bin_exp, int32_t dec_exp) {
  int32_t pow10_bin_exp = asr32(-dec_exp * 217707, 16);
  return (unsigned)(uint8_t)(bin_exp + pow10_bin_exp + 1);
}

typedef struct {
  int64_t sig;
  int32_t exp;
} zm_dec;

static zm_dec zm_schubfach64(uint64_t bin_sig, int32_t bin_exp, int regular) {
  int32_t dec_exp = zm_compute_dec_exp(bin_exp, regular);
  unsigned exp_shift = zm_compute_exp_shift(bin_exp, dec_exp);
  u128p pow10 = zm_pow10(-dec_exp);
  uint64_t bin_sig_shifted, lsb, lower, upper, shorter, scaled_sig, longer_below, longer_above;
  uint64_t cmp, dec_sig;
  int below_closer, below_in;
  zm_dec r;
  pow10.lo += 1;
  bin_sig_shifted = bin_sig << 2;
  lsb = bin_sig & 1u;
  lower = (bin_sig_shifted - ((uint64_t)(regular != 0) + 1u)) << exp_shift;
  lower = umulhi_inexact_to_odd64(pow10.hi, pow10.lo, lower) + lsb;
  upper = (bin_sig_shifted + 2u) << exp_shift;
  upper = umulhi_inexact_to_odd64(pow10.hi, pow10.lo, upper) - lsb;
  shorter = (upper >> 2) / 10u * 10u;
  if ((shorter << 2) >= lower) {
    r.sig = (int64_t)shorter;
    r.exp = dec_exp;
    return r;
  }
  scaled_sig = umulhi_inexact_to_odd64(pow10.hi, pow10.lo, bin_sig_shifted << exp_shift);
  longer_below = scaled_sig >> 2;
  longer_above = longer_below + 1u;
  cmp = scaled_sig - ((longer_below + longer_above) << 1);
  below_closer = (cmp >> 63) != 0 || (cmp == 0 && (longer_below & 1u) == 0);
  below_in = (longer_below << 2) >= lower;
  dec_sig = (below_closer && below_in) ? longer_below : longer_above;
  r.sig = (int64_t)dec_sig;
  r.exp = dec_exp;
  return r;
}

static zm_dec zm_schubfach32(uint32_t bin_sig, int32_t bin_exp, int regular) {
  int32_t dec_exp = zm_compute_dec_exp(bin_exp, regular);
  unsigned exp_shift = zm_compute_exp_shift(bin_exp, dec_exp);
  u128p pow10 = zm_pow10(-dec_exp);
  uint32_t bin_sig_shifted, lsb, lower, upper, shorter, scaled_sig, longer_below, longer_above;
  uint32_t cmp, dec_sig;
  int below_closer, below_in;
  zm_dec r;
  pow10.hi += 1;
  bin_sig_shifted = bin_sig << 2;
  lsb = bin_sig & 1u;
  lower = (uint32_t)((bin_sig_shifted - ((uint32_t)(regular != 0) + 1u)) << exp_shift);
  lower = umulhi_inexact_to_odd32(pow10.hi, lower) + lsb;
  upper = (uint32_t)((bin_sig_shifted + 2u) << exp_shift);
  upper = umulhi_inexact_to_odd32(pow10.hi, upper) - lsb;
  shorter = (upper >> 2) / 10u * 10u;
  if ((uint32_t)(shorter << 2) >= lower) {
    r.sig = (int64_t)shorter;
    r.exp = dec_exp;
    return r;
  }
  scaled_sig = umulhi_inexact_to_odd32(pow10.hi, (uint32_t)(bin_sig_shifted << exp_shift));
  longer_below = scaled_sig >> 2;
  longer_above = longer_below + 1u;
  cmp = scaled_sig - (uint32_t)((longer_below + longer_above) << 1);
  below_closer = (cmp >> 31) != 0 || (cmp == 0 && (longer_below & 1u) == 0);
  below_in = (uint32_t)(longer_below << 2) >= lower;
  dec_sig = (below_closer && below_in) ? longer_below : longer_above;
  r.sig = (int64_t)dec_sig;
  r.exp = dec_exp;
  return r;
}

#define ZM_HALF_ULP ((uint64_t)1 << 63)
#define ZM_DIV10_SIG64 (((uint64_t)1 << 63) / 5u + 1u)

/* to_decimal_fast::<f64, u64> */
static zm_dec zm_fast64(uint64_t bin_sig, int32_t raw_exp, int regular) {
  int32_t bin_exp = raw_exp - 1075;
  if (regular) {
    int32_t dec_exp = zm_compute_dec_exp(bin_exp, 1);
    unsigned exp_shift = zm_compute_exp_shift(bin_exp, dec_exp);
    u128p pow10 = zm_pow10(-dec_exp);
    u128p p = umul192_hi128(pow10.hi, pow10.lo, bin_sig << exp_shift);
    uint64_t integral = p.hi, fractional = p.lo;
    uint64_t cmp = fractional - ZM_HALF_ULP;
    if (cmp != 0) {
      uint64_t div10 = umul128_hi64(integral, ZM_DIV10_SIG64);
      uint64_t digit = integral - div10 * 10u;
      const unsigned nib = 4, nfb = 60;
      uint64_t ten = (uint64_t)10 << nfb;
      uint64_t scaled_sig_mod10 = (digit << nfb) | (fractional >> nib);
      uint64_t scaled_half_ulp = pow10.hi >> (nib - exp_shift + 1u);
      uint64_t upper = scaled_sig_mod10 + scaled_half_ulp;
      if (!(ten - upper <= 1u || scaled_sig_mod10 == scaled_half_ulp)) {
        int64_t shorter = (int64_t)(integral - digit);
        int64_t longer = (int64_t)(integral + (uint64_t)((cmp >> 63) == 0));
        int64_t dec_sig = scaled_sig_mod10 < scaled_half_ulp ? shorter : longer;
        zm_dec r;
        r.sig = ten < upper ? shorter + 10 : dec_sig;
        r.exp = dec_exp;
        return r;
      }
    }
  }
  return zm_schubfach64(bin_sig, bin_exp, regular);
}

/* to_decimal_fast::<f32, u32> */
static zm_dec zm_fast32(uint32_t bin_sig, int32_t raw_exp, int regular) {
  int32_t bin_exp = raw_exp - 150;
  if (regular) {
    int32_t dec_exp = zm_compute_dec_exp(bin_exp, 1);
    unsigned exp_shift = zm_compute_exp_shift(bin_exp, dec_exp);
    u128p pow10 = zm_pow10(-dec_exp);
    u128p p = umul128(pow10.hi, (uint64_t)(uint32_t)(bin_sig << exp_shift));
    uint32_t integral = (uint32_t)p.hi;
    uint64_t fractional = p.lo;
    uint64_t cmp = fractional - ZM_HALF_ULP;
    if (cmp != 0) {
      uint64_t div10 = umul128_hi64((uint64_t)integral, ZM_DIV10_SIG64);
      uint64_t digit = (uint64_t)integral - div10 * 10u;
      const unsigned nib = 32, nfb = 32;
      uint64_t ten = (uint64_t)10 << nfb;
      uint64_t scaled_sig_mod10 = (digit << nfb) | (fractional >> nib);
      uint64_t scaled_half_ulp = pow10.hi >> (nib - exp_shift + 1u);
      uint64_t upper = scaled_sig_mod10 + scaled_half_ulp;
      if (!(ten - upper <= 1u || scaled_sig_mod10 == scaled_half_ulp)) {
        int64_t shorter = (int64_t)((uint64_t)integral - digit);
        int64_t longer = (int64_t)((uint64_t)integral + (uint64_t)((cmp >> 63) == 0));
        int64_t dec_sig = scaled_sig_mod10 < scaled_half_ulp ? shorter : longer;
        zm_dec r;
        r.sig = ten < upper ? shorter + 10 : dec_sig;
        r.exp = dec_exp;
        return r;
      }
    }
  }
  return zm_schubfach32(bin_sig, bin_exp, regular);
}

/* The layout part of zmij's `write`: `digits` (ndig significant digits, trailing zeros already
 * removed) times 10**dec_exp, where dec_exp is the exponent of the first digit. */
static size_t zm_layout(char *out, int neg, const char *digits, int length, int32_t dec_exp,
                        int is64) {
  char *b = out;
  int lo = is64 ? -5 : -6, hi = is64 ? 15 : 12;
  if (neg) *b++ = '-';
  if (dec_exp >= lo && dec_exp <= hi) {
    if (length - 1 <= dec_exp) {
      int k;
      memcpy(b, digits, (size_t)length);
      for (k = length; k < dec_exp + 1; k++) b[k] = '0';
      b[dec_exp + 1] = '.';
      b[dec_exp + 2] = '0';
      b += dec_exp + 3;
    } else if (0 <= dec_exp) {
      memcpy(b, digits, (size_t)dec_exp + 1u);
      b[dec_exp + 1] = '.';
      memcpy(b + dec_exp + 2, digits + dec_exp + 1, (size_t)(length - dec_exp - 1));
      b += length + 1;
    } else {
      int zeros = 1 - dec_exp, k;
      for (k = 0; k < zeros; k++) b[k] = '0';
      b[1] = '.';
      memcpy(b + zeros, digits, (size_t)length);
      b += zeros + length;
    }
  } else {
    int32_t e;
    b[0] = digits[0];
    if (length > 1) {
      b[1] = '.';
      memcpy(b + 2, digits + 1, (size_t)length - 1u);
      b += length + 1;
    } else {
      b += 1;
    }
    *b++ = 'e';
    *b++ = dec_exp >= 0 ? '+' : '-';
    e = dec_exp >= 0 ? dec_exp : -dec_exp;
    if (e >= 100) *b++ = (char)('0' + e / 100);
    if (e >= 10) *b++ = (char)('0' + e / 10 % 10);
    *b++ = (char)('0' + e % 10);
  }
  *b = '\0';
  return (size_t)(b - out);
}

/* Digits of `sig`, zero-padded to `width` places, trailing zeros removed (write_significand). */
static int zm_digits(char *d, uint64_t sig, int width) {
  int k, len;
  for (k = width - 1; k >= 0; k--) {
    d[k] = (char)('0' + (int)(sig % 10u));
    sig /= 10u;
  }
  len = width;
  while (len > 0 && d[len - 1] == '0') len--;
  return len;
}

size_t nd_json_write_f64(double v, char out[ND_JSON_NUMBUF]) {
  uint64_t bits, bin_sig;
  int32_t bin_exp, dec_exp;
  int neg, extra, length;
  zm_dec dec;
  char digits[24];
  const int64_t threshold = 10000000000000000; /* 1e16 */
  memcpy(&bits, &v, sizeof bits);
  bin_exp = (int32_t)((bits << 1) >> 53);
  bin_sig = bits & (((uint64_t)1 << 52) - 1u);
  neg = (int)(bits >> 63);
  if (bin_exp == 0x7ff) {
    memcpy(out, "null", 5);
    return 4;
  }
  if (bin_exp == 0) {
    if (bin_sig == 0) {
      char *b = out;
      if (neg) *b++ = '-';
      memcpy(b, "0.0", 4);
      return (size_t)(b - out) + 3u;
    }
    dec = zm_schubfach64(bin_sig, 1 - 1075, 1);
    while (dec.sig < threshold) {
      dec.sig *= 10;
      dec.exp -= 1;
    }
  } else {
    dec = zm_fast64(bin_sig | ((uint64_t)1 << 52), bin_exp, bin_sig != 0);
  }
  dec_exp = dec.exp;
  extra = dec.sig >= threshold;
  dec_exp += 17 - 2 + extra;
  length = zm_digits(digits, (uint64_t)dec.sig, 16 + extra);
  return zm_layout(out, neg, digits, length, dec_exp, 1);
}

size_t nd_json_write_f32(float v, char out[ND_JSON_NUMBUF]) {
  uint32_t bits, bin_sig;
  int32_t bin_exp, dec_exp;
  int neg, extra, length;
  zm_dec dec;
  char digits[16];
  const int64_t threshold = 100000000; /* 1e8 */
  memcpy(&bits, &v, sizeof bits);
  bin_exp = (int32_t)((uint32_t)(bits << 1) >> 24);
  bin_sig = bits & ((1u << 23) - 1u);
  neg = (int)(bits >> 31);
  if (bin_exp == 0xff) {
    memcpy(out, "null", 5);
    return 4;
  }
  if (bin_exp == 0) {
    if (bin_sig == 0) {
      char *b = out;
      if (neg) *b++ = '-';
      memcpy(b, "0.0", 4);
      return (size_t)(b - out) + 3u;
    }
    dec = zm_schubfach32(bin_sig, 1 - 150, 1);
    while (dec.sig < threshold) {
      dec.sig *= 10;
      dec.exp -= 1;
    }
  } else {
    dec = zm_fast32(bin_sig | (1u << 23), bin_exp, bin_sig != 0);
  }
  dec_exp = dec.exp;
  extra = dec.sig >= threshold;
  dec_exp += 9 - 2 + extra;
  if (dec.sig < 10000000) {
    dec.sig *= 10;
    dec_exp -= 1;
  }
  length = zm_digits(digits, (uint64_t)dec.sig, 8 + extra);
  return zm_layout(out, neg, digits, length, dec_exp, 0);
}

static size_t u64_digits(uint64_t v, char *out) {
  char tmp[24];
  size_t n = 0, i;
  do {
    tmp[n++] = (char)('0' + (int)(v % 10u));
    v /= 10u;
  } while (v);
  for (i = 0; i < n; i++) out[i] = tmp[n - 1 - i];
  out[n] = '\0';
  return n;
}

size_t nd_json_write_u64(uint64_t v, char out[ND_JSON_NUMBUF]) { return u64_digits(v, out); }

size_t nd_json_write_i64(int64_t v, char out[ND_JSON_NUMBUF]) {
  if (v < 0) {
    uint64_t m = (uint64_t)0 - (uint64_t)v;
    out[0] = '-';
    return 1u + u64_digits(m, out + 1);
  }
  return u64_digits((uint64_t)v, out);
}

/* ─── Parser ───────────────────────────────────────────────────────────────────────────────── */

static const double POW10[309] = {
  1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 
  1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 
  1e20, 1e21, 1e22, 1e23, 1e24, 1e25, 1e26, 1e27, 1e28, 1e29, 
  1e30, 1e31, 1e32, 1e33, 1e34, 1e35, 1e36, 1e37, 1e38, 1e39, 
  1e40, 1e41, 1e42, 1e43, 1e44, 1e45, 1e46, 1e47, 1e48, 1e49, 
  1e50, 1e51, 1e52, 1e53, 1e54, 1e55, 1e56, 1e57, 1e58, 1e59, 
  1e60, 1e61, 1e62, 1e63, 1e64, 1e65, 1e66, 1e67, 1e68, 1e69, 
  1e70, 1e71, 1e72, 1e73, 1e74, 1e75, 1e76, 1e77, 1e78, 1e79, 
  1e80, 1e81, 1e82, 1e83, 1e84, 1e85, 1e86, 1e87, 1e88, 1e89, 
  1e90, 1e91, 1e92, 1e93, 1e94, 1e95, 1e96, 1e97, 1e98, 1e99, 
  1e100, 1e101, 1e102, 1e103, 1e104, 1e105, 1e106, 1e107, 1e108, 1e109, 
  1e110, 1e111, 1e112, 1e113, 1e114, 1e115, 1e116, 1e117, 1e118, 1e119, 
  1e120, 1e121, 1e122, 1e123, 1e124, 1e125, 1e126, 1e127, 1e128, 1e129, 
  1e130, 1e131, 1e132, 1e133, 1e134, 1e135, 1e136, 1e137, 1e138, 1e139, 
  1e140, 1e141, 1e142, 1e143, 1e144, 1e145, 1e146, 1e147, 1e148, 1e149, 
  1e150, 1e151, 1e152, 1e153, 1e154, 1e155, 1e156, 1e157, 1e158, 1e159, 
  1e160, 1e161, 1e162, 1e163, 1e164, 1e165, 1e166, 1e167, 1e168, 1e169, 
  1e170, 1e171, 1e172, 1e173, 1e174, 1e175, 1e176, 1e177, 1e178, 1e179, 
  1e180, 1e181, 1e182, 1e183, 1e184, 1e185, 1e186, 1e187, 1e188, 1e189, 
  1e190, 1e191, 1e192, 1e193, 1e194, 1e195, 1e196, 1e197, 1e198, 1e199, 
  1e200, 1e201, 1e202, 1e203, 1e204, 1e205, 1e206, 1e207, 1e208, 1e209, 
  1e210, 1e211, 1e212, 1e213, 1e214, 1e215, 1e216, 1e217, 1e218, 1e219, 
  1e220, 1e221, 1e222, 1e223, 1e224, 1e225, 1e226, 1e227, 1e228, 1e229, 
  1e230, 1e231, 1e232, 1e233, 1e234, 1e235, 1e236, 1e237, 1e238, 1e239, 
  1e240, 1e241, 1e242, 1e243, 1e244, 1e245, 1e246, 1e247, 1e248, 1e249, 
  1e250, 1e251, 1e252, 1e253, 1e254, 1e255, 1e256, 1e257, 1e258, 1e259, 
  1e260, 1e261, 1e262, 1e263, 1e264, 1e265, 1e266, 1e267, 1e268, 1e269, 
  1e270, 1e271, 1e272, 1e273, 1e274, 1e275, 1e276, 1e277, 1e278, 1e279, 
  1e280, 1e281, 1e282, 1e283, 1e284, 1e285, 1e286, 1e287, 1e288, 1e289, 
  1e290, 1e291, 1e292, 1e293, 1e294, 1e295, 1e296, 1e297, 1e298, 1e299, 
  1e300, 1e301, 1e302, 1e303, 1e304, 1e305, 1e306, 1e307, 1e308
};

nd_json_limits nd_json_limits_default(void) {
  nd_json_limits l;
  l.max_input = (size_t)1 << 20;
  l.max_depth = ND_JSON_DEFAULT_MAX_DEPTH;
  l.max_elements = 65536u;
  return l;
}

typedef struct {
  nd_json_value *v;
  size_t len, cap;
} vstack;
typedef struct {
  nd_json_member *m;
  size_t len, cap;
} mstack;

typedef struct {
  const unsigned char *s;
  size_t n, i;
  arena *a;
  size_t elements, max_elements;
  unsigned depth, max_depth;
  int serde_depth; /* max_depth is serde_json's own limit: report it as serde_json does */
  vstack vs;
  mstack ms;
  unsigned char *sb; /* string scratch */
  size_t sl, sc;
  int code; /* error code once failed */
  nd_err *err;
} parser;

/* serde_json's Display for a syntax error: "<what> at line L column C", where the position is
 * that of byte index `idx` as SliceRead::position_of_index computes it. */
static int p_fail_at(parser *p, size_t idx, const char *what) {
  size_t k, start_of_line = 0, line = 1;
  if (p->code) return p->code;
  p->code = ND_E_FORMAT;
  for (k = idx; k > 0; k--) {
    if (p->s[k - 1] == '\n') {
      start_of_line = k;
      break;
    }
  }
  for (k = 0; k < start_of_line; k++) line += p->s[k] == '\n';
  nd_seterr(p->err, "%s at line %lu column %lu", what, (unsigned long)line,
            (unsigned long)(idx - start_of_line));
  return ND_E_FORMAT;
}

/* Deserializer::error / read::error: the position of the next unread byte. */
static int p_err(parser *p, const char *what) { return p_fail_at(p, p->i, what); }

/* Deserializer::peek_error: one past the peeked byte, capped at the input length. */
static int p_perr(parser *p, const char *what) {
  return p_fail_at(p, p->i + 1 < p->n ? p->i + 1 : p->n, what);
}

/* Errors from this reader's own limits (not serde_json's). */
static int p_limit(parser *p, const char *what, unsigned long lim) {
  if (p->code) return p->code;
  p->code = ND_E_BOUNDS;
  nd_seterr(p->err, "nd_json limit: %s (limit %lu) at byte %lu", what, lim, (unsigned long)p->i);
  return ND_E_BOUNDS;
}

static int p_nomem(parser *p) {
  if (p->code) return p->code;
  p->code = ND_E_NOMEM;
  nd_seterr(p->err, "nd_json: out of memory");
  return ND_E_NOMEM;
}

static int p_peek_or_null(parser *p) { return p->i < p->n ? (int)p->s[p->i] : 0; }

/* parse_whitespace: skip and return the next byte (-1 at EOF) without consuming it. */
static int p_ws(parser *p) {
  while (p->i < p->n) {
    unsigned char c = p->s[p->i];
    if (c == ' ' || c == '\n' || c == '\t' || c == '\r')
      p->i++;
    else
      return c;
  }
  return -1;
}

static int p_count(parser *p) {
  if (p->elements >= p->max_elements)
    return p_limit(p, "too many elements", (unsigned long)p->max_elements);
  p->elements++;
  return 0;
}

/* overflow!(a * 10 + b, c) from serde_json */
static int ovf_u64(uint64_t a, uint64_t b) {
  const uint64_t c = UINT64_MAX;
  return a >= c / 10u && (a > c / 10u || b > c % 10u);
}
static int ovf_i32(int32_t a, int32_t b) {
  const int32_t c = INT32_MAX;
  return a >= c / 10 && (a > c / 10 || b > c % 10);
}

static int32_t sat_add(int32_t a, int32_t b) {
  int64_t s = (int64_t)a + (int64_t)b;
  if (s > INT32_MAX) return INT32_MAX;
  if (s < INT32_MIN) return INT32_MIN;
  return (int32_t)s;
}
static int32_t sat_sub(int32_t a, int32_t b) {
  int64_t s = (int64_t)a - (int64_t)b;
  if (s > INT32_MAX) return INT32_MAX;
  if (s < INT32_MIN) return INT32_MIN;
  return (int32_t)s;
}

static int is_digit(int c) { return c >= '0' && c <= '9'; }

#define E_EOF_VALUE "EOF while parsing a value"
#define E_EOF_STRING "EOF while parsing a string"
#define E_INVALID_NUMBER "invalid number"
#define E_OUT_OF_RANGE "number out of range"
#define E_INVALID_ESCAPE "invalid escape"
#define E_LONE_SURROGATE "lone leading surrogate in hex escape"
#define E_END_OF_HEX "unexpected end of hex escape"

static int f64_from_parts(parser *p, int positive, uint64_t significand, int32_t exponent,
                          double *out) {
  double f = (double)significand;
  for (;;) {
    int64_t ab = exponent < 0 ? -(int64_t)exponent : (int64_t)exponent;
    if (ab <= 308) {
      if (exponent >= 0) {
        f *= POW10[ab];
        if (f > DBL_MAX) return p_err(p, E_OUT_OF_RANGE);
      } else {
        f /= POW10[ab];
      }
      break;
    }
    if (f == 0.0) break;
    if (exponent >= 0) return p_err(p, E_OUT_OF_RANGE);
    f /= 1e308;
    exponent += 308;
  }
  *out = positive ? f : -f;
  return 0;
}

static int parse_exponent_overflow(parser *p, int positive, int zero_significand,
                                   int positive_exp, double *out) {
  if (!zero_significand && positive_exp) return p_err(p, E_OUT_OF_RANGE);
  while (is_digit(p_peek_or_null(p))) p->i++;
  *out = positive ? 0.0 : -0.0;
  return 0;
}

static int parse_exponent(parser *p, int positive, uint64_t significand, int32_t starting_exp,
                          double *out) {
  int positive_exp = 1, c;
  int32_t exp, final_exp;
  p->i++; /* 'e' / 'E' */
  c = p_peek_or_null(p);
  if (c == '+') {
    p->i++;
  } else if (c == '-') {
    p->i++;
    positive_exp = 0;
  }
  if (p->i >= p->n) return p_err(p, E_EOF_VALUE);
  c = p->s[p->i++];
  if (!is_digit(c)) return p_err(p, E_INVALID_NUMBER);
  exp = c - '0';
  while (is_digit(c = p_peek_or_null(p))) {
    int32_t digit = c - '0';
    p->i++;
    if (ovf_i32(exp, digit))
      return parse_exponent_overflow(p, positive, significand == 0, positive_exp, out);
    exp = exp * 10 + digit;
  }
  final_exp = positive_exp ? sat_add(starting_exp, exp) : sat_sub(starting_exp, exp);
  return f64_from_parts(p, positive, significand, final_exp, out);
}

static int parse_decimal_overflow(parser *p, int positive, uint64_t significand, int32_t exponent,
                                  double *out) {
  int c;
  while (is_digit(p_peek_or_null(p))) p->i++;
  c = p_peek_or_null(p);
  if (c == 'e' || c == 'E') return parse_exponent(p, positive, significand, exponent, out);
  return f64_from_parts(p, positive, significand, exponent, out);
}

static int parse_decimal(parser *p, int positive, uint64_t significand,
                         int32_t exponent_before_decimal_point, double *out) {
  int32_t exponent_after_decimal_point = 0, exponent;
  int c;
  p->i++; /* '.' */
  while (is_digit(c = p_peek_or_null(p))) {
    uint64_t digit = (uint64_t)(c - '0');
    if (ovf_u64(significand, digit))
      return parse_decimal_overflow(p, positive, significand,
                                    exponent_before_decimal_point + exponent_after_decimal_point,
                                    out);
    p->i++;
    significand = significand * 10u + digit;
    exponent_after_decimal_point -= 1;
  }
  if (exponent_after_decimal_point == 0) {
    if (p->i < p->n) return p_perr(p, E_INVALID_NUMBER);
    return p_perr(p, E_EOF_VALUE);
  }
  exponent = exponent_before_decimal_point + exponent_after_decimal_point;
  c = p_peek_or_null(p);
  if (c == 'e' || c == 'E') return parse_exponent(p, positive, significand, exponent, out);
  return f64_from_parts(p, positive, significand, exponent, out);
}

static int parse_long_integer(parser *p, int positive, uint64_t significand, double *out) {
  int32_t exponent = 0;
  for (;;) {
    int c = p_peek_or_null(p);
    if (is_digit(c)) {
      p->i++;
      exponent += 1;
    } else if (c == '.') {
      return parse_decimal(p, positive, significand, exponent, out);
    } else if (c == 'e' || c == 'E') {
      return parse_exponent(p, positive, significand, exponent, out);
    } else {
      return f64_from_parts(p, positive, significand, exponent, out);
    }
  }
}

static int parse_number(parser *p, int positive, uint64_t significand, nd_json_value *v) {
  int c = p_peek_or_null(p);
  if (c == '.') {
    v->u.num.kind = ND_JSON_NUM_F64;
    return parse_decimal(p, positive, significand, 0, &v->u.num.f);
  }
  if (c == 'e' || c == 'E') {
    v->u.num.kind = ND_JSON_NUM_F64;
    return parse_exponent(p, positive, significand, 0, &v->u.num.f);
  }
  if (positive) {
    v->u.num.kind = ND_JSON_NUM_U64;
    v->u.num.u = significand;
    v->u.num.f = (double)significand;
  } else if (significand == 0 || significand > ((uint64_t)1 << 63)) {
    /* (significand as i64).wrapping_neg() >= 0: -0 or below i64::MIN */
    v->u.num.kind = ND_JSON_NUM_F64;
    v->u.num.f = -(double)significand;
  } else {
    int64_t neg = significand == ((uint64_t)1 << 63) ? INT64_MIN : -(int64_t)significand;
    v->u.num.kind = ND_JSON_NUM_I64;
    v->u.num.i = neg;
    v->u.num.f = (double)neg;
  }
  return 0;
}

static int parse_integer(parser *p, int positive, nd_json_value *v) {
  int c;
  if (p->i >= p->n) return p_err(p, E_EOF_VALUE);
  c = p->s[p->i++];
  if (c == '0') {
    if (is_digit(p_peek_or_null(p))) return p_perr(p, E_INVALID_NUMBER);
    return parse_number(p, positive, 0, v);
  }
  if (c >= '1' && c <= '9') {
    uint64_t significand = (uint64_t)(c - '0');
    for (;;) {
      c = p_peek_or_null(p);
      if (is_digit(c)) {
        uint64_t digit = (uint64_t)(c - '0');
        if (ovf_u64(significand, digit)) {
          v->u.num.kind = ND_JSON_NUM_F64;
          return parse_long_integer(p, positive, significand, &v->u.num.f);
        }
        p->i++;
        significand = significand * 10u + digit;
      } else {
        return parse_number(p, positive, significand, v);
      }
    }
  }
  return p_err(p, E_INVALID_NUMBER);
}

static char *arena_copy(parser *p, const void *src, size_t n) {
  size_t m;
  char *d;
  if (!nd_add_ok(n, 1, &m)) return NULL;
  d = (char *)arena_alloc(p->a, m);
  if (!d) return NULL;
  if (n) memcpy(d, src, n);
  d[n] = '\0';
  return d;
}

static int sb_push(parser *p, const unsigned char *b, size_t n) {
  if (p->sc - p->sl < n) {
    size_t want, nc = p->sc ? p->sc : 64u;
    unsigned char *q;
    if (!nd_add_ok(p->sl, n, &want)) return p_nomem(p);
    while (nc < want) {
      if (!nd_mul_ok(nc, 2, &nc)) return p_nomem(p);
    }
    q = (unsigned char *)realloc(p->sb, nc);
    if (!q) return p_nomem(p);
    p->sb = q;
    p->sc = nc;
  }
  memcpy(p->sb + p->sl, b, n);
  p->sl += n;
  return 0;
}

static int push_cp(parser *p, uint32_t n) {
  unsigned char b[4];
  size_t k;
  if (n < 0x80u) {
    b[0] = (unsigned char)n;
    k = 1;
  } else if (n < 0x800u) {
    b[0] = (unsigned char)(0xC0u | (n >> 6));
    b[1] = (unsigned char)(0x80u | (n & 0x3Fu));
    k = 2;
  } else if (n < 0x10000u) {
    b[0] = (unsigned char)(0xE0u | (n >> 12));
    b[1] = (unsigned char)(0x80u | ((n >> 6) & 0x3Fu));
    b[2] = (unsigned char)(0x80u | (n & 0x3Fu));
    k = 3;
  } else {
    b[0] = (unsigned char)(0xF0u | (n >> 18));
    b[1] = (unsigned char)(0x80u | ((n >> 12) & 0x3Fu));
    b[2] = (unsigned char)(0x80u | ((n >> 6) & 0x3Fu));
    b[3] = (unsigned char)(0x80u | (n & 0x3Fu));
    k = 4;
  }
  return sb_push(p, b, k);
}

static int hexval(int c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

/* SliceRead::decode_hex_escape: four bytes are taken before they are validated. */
static int decode_hex(parser *p, uint32_t *out) {
  uint32_t n = 0;
  int k, bad = 0;
  if (p->n - p->i < 4) {
    p->i = p->n;
    return p_err(p, E_EOF_STRING);
  }
  for (k = 0; k < 4; k++) {
    int h = hexval(p->s[p->i + (size_t)k]);
    if (h < 0) bad = 1;
    n = n * 16u + (uint32_t)(h < 0 ? 0 : h);
  }
  p->i += 4;
  if (bad) return p_err(p, E_INVALID_ESCAPE);
  *out = n;
  return 0;
}

/* parse_escape / parse_unicode_escape with validate = true. At the byte after '\\'. */
static int parse_escape(parser *p) {
  unsigned char c, o;
  if (p->i >= p->n) return p_err(p, E_EOF_STRING);
  c = p->s[p->i++];
  switch (c) {
    case '"': o = '"'; break;
    case '\\': o = '\\'; break;
    case '/': o = '/'; break;
    case 'b': o = 0x08; break;
    case 'f': o = 0x0c; break;
    case 'n': o = '\n'; break;
    case 'r': o = '\r'; break;
    case 't': o = '\t'; break;
    case 'u': {
      uint32_t n, n2;
      if (decode_hex(p, &n)) return p->code;
      if (n >= 0xDC00u && n <= 0xDFFFu) return p_err(p, E_LONE_SURROGATE);
      if (n < 0xD800u || n > 0xDBFFu) return push_cp(p, n);
      if (p->i >= p->n) return p_err(p, E_EOF_STRING);
      if (p->s[p->i] != '\\') {
        p->i++;
        return p_err(p, E_END_OF_HEX);
      }
      p->i++;
      if (p->i >= p->n) return p_err(p, E_EOF_STRING);
      if (p->s[p->i] != 'u') {
        p->i++;
        return p_err(p, E_END_OF_HEX);
      }
      p->i++;
      if (decode_hex(p, &n2)) return p->code;
      if (n2 < 0xDC00u || n2 > 0xDFFFu) return p_err(p, E_LONE_SURROGATE);
      return push_cp(p, (((n - 0xD800u) << 10) | (n2 - 0xDC00u)) + 0x10000u);
    }
    default:
      return p_err(p, E_INVALID_ESCAPE);
  }
  return sb_push(p, &o, 1);
}

/* At an opening '"' (SliceRead::parse_str_bytes after the quote is eaten). */
static int parse_string(parser *p, const char **out, size_t *out_len) {
  size_t run;
  p->i++;
  p->sl = 0;
  for (;;) {
    unsigned char c;
    run = p->i;
    while (p->i < p->n) {
      c = p->s[p->i];
      if (c == '"' || c == '\\' || c < 0x20u) break;
      p->i++;
    }
    if (p->i > run && sb_push(p, p->s + run, p->i - run)) return p->code;
    if (p->i >= p->n) return p_err(p, E_EOF_STRING);
    c = p->s[p->i++];
    if (c == '"') break;
    if (c == '\\') {
      if (parse_escape(p)) return p->code;
    } else {
      return p_err(p, "control character (\\u0000-\\u001F) found while parsing a string");
    }
  }
  /* as_str (from_slice only: a &str input is valid UTF-8 already) */
  if (!nd_json_utf8_valid(p->sb, p->sl)) return p_err(p, "invalid unicode code point");
  *out = arena_copy(p, p->sb, p->sl);
  if (!*out) return p_nomem(p);
  *out_len = p->sl;
  return 0;
}

static int vs_push(parser *p, const nd_json_value *v) {
  if (p->vs.len == p->vs.cap) {
    size_t nc = p->vs.cap ? p->vs.cap : 32u, bytes;
    nd_json_value *q;
    if (p->vs.cap && !nd_mul_ok(nc, 2, &nc)) return p_nomem(p);
    if (!nd_mul_ok(nc, sizeof *q, &bytes)) return p_nomem(p);
    q = (nd_json_value *)realloc(p->vs.v, bytes);
    if (!q) return p_nomem(p);
    p->vs.v = q;
    p->vs.cap = nc;
  }
  p->vs.v[p->vs.len++] = *v;
  return 0;
}

static int ms_push(parser *p, const nd_json_member *m) {
  if (p->ms.len == p->ms.cap) {
    size_t nc = p->ms.cap ? p->ms.cap : 16u, bytes;
    nd_json_member *q;
    if (p->ms.cap && !nd_mul_ok(nc, 2, &nc)) return p_nomem(p);
    if (!nd_mul_ok(nc, sizeof *q, &bytes)) return p_nomem(p);
    q = (nd_json_member *)realloc(p->ms.m, bytes);
    if (!q) return p_nomem(p);
    p->ms.m = q;
    p->ms.cap = nc;
  }
  p->ms.m[p->ms.len++] = *m;
  return 0;
}

static int parse_value(parser *p, nd_json_value *v);

/* check_recursion! at the '[' / '{' (not yet eaten). */
static int enter_container(parser *p) {
  if (p->depth >= p->max_depth) {
    if (p->serde_depth) return p_perr(p, "recursion limit exceeded");
    return p_limit(p, "nesting too deep", (unsigned long)p->max_depth);
  }
  p->depth++;
  p->i++;
  return 0;
}

/* deserialize_any '[' with SeqAccess::has_next_element and end_seq. */
static int parse_array(parser *p, nd_json_value *v) {
  size_t base = p->vs.len, n, bytes;
  nd_json_value *items = NULL;
  int first = 1;
  if (enter_container(p)) return p->code;
  for (;;) {
    nd_json_value item;
    int c = p_ws(p);
    if (c < 0) return p_perr(p, "EOF while parsing a list");
    if (c == ']') break;
    if (first) {
      first = 0;
    } else if (c == ',') {
      p->i++;
      c = p_ws(p);
      if (c == ']') return p_perr(p, "trailing comma");
      if (c < 0) return p_perr(p, E_EOF_VALUE);
    } else {
      return p_perr(p, "expected `,` or `]`");
    }
    if (parse_value(p, &item)) return p->code;
    if (vs_push(p, &item)) return p->code;
  }
  p->i++; /* end_seq: the ']' */
  p->depth--;
  n = p->vs.len - base;
  if (n) {
    if (!nd_mul_ok(n, sizeof *items, &bytes)) return p_nomem(p);
    items = (nd_json_value *)arena_alloc(p->a, bytes);
    if (!items) return p_nomem(p);
    memcpy(items, p->vs.v + base, bytes);
  }
  p->vs.len = base;
  v->type = ND_JSON_ARRAY;
  v->u.arr.items = items;
  v->u.arr.len = n;
  return 0;
}

/* deserialize_any '{' with MapAccess::has_next_key, MapKey, parse_object_colon and end_map. */
static int parse_object(parser *p, nd_json_value *v) {
  size_t base = p->ms.len, n, bytes;
  nd_json_member *members = NULL;
  int first = 1;
  if (enter_container(p)) return p->code;
  for (;;) {
    nd_json_member m;
    int c = p_ws(p);
    if (c < 0) return p_perr(p, "EOF while parsing an object");
    if (c == '}') break;
    if (first) {
      first = 0;
      if (c != '"') return p_perr(p, "key must be a string");
    } else if (c == ',') {
      p->i++;
      c = p_ws(p);
      if (c == '}') return p_perr(p, "trailing comma");
      if (c < 0) return p_perr(p, E_EOF_VALUE);
      if (c != '"') return p_perr(p, "key must be a string");
    } else {
      return p_perr(p, "expected `,` or `}`");
    }
    if (p_count(p)) return p->code;
    if (parse_string(p, &m.key, &m.key_len)) return p->code;
    c = p_ws(p);
    if (c < 0) return p_perr(p, "EOF while parsing an object");
    if (c != ':') return p_perr(p, "expected `:`");
    p->i++;
    if (parse_value(p, &m.value)) return p->code;
    if (ms_push(p, &m)) return p->code;
  }
  p->i++; /* end_map: the '}' */
  p->depth--;
  n = p->ms.len - base;
  if (n) {
    if (!nd_mul_ok(n, sizeof *members, &bytes)) return p_nomem(p);
    members = (nd_json_member *)arena_alloc(p->a, bytes);
    if (!members) return p_nomem(p);
    memcpy(members, p->ms.m + base, bytes);
  }
  p->ms.len = base;
  v->type = ND_JSON_OBJECT;
  v->u.obj.members = members;
  v->u.obj.len = n;
  return 0;
}

static int parse_ident(parser *p, const char *rest) {
  for (; *rest; rest++) {
    if (p->i >= p->n) return p_err(p, E_EOF_VALUE);
    if (p->s[p->i++] != (unsigned char)*rest) return p_err(p, "expected ident");
  }
  return 0;
}

/* Deserializer::deserialize_any */
static int parse_value(parser *p, nd_json_value *v) {
  int c;
  memset(v, 0, sizeof *v);
  c = p_ws(p);
  if (c < 0) return p_perr(p, E_EOF_VALUE);
  if (p_count(p)) return p->code;
  switch (c) {
    case 'n':
      p->i++;
      v->type = ND_JSON_NULL;
      return parse_ident(p, "ull");
    case 't':
      p->i++;
      v->type = ND_JSON_BOOL;
      v->u.boolean = 1;
      return parse_ident(p, "rue");
    case 'f':
      p->i++;
      v->type = ND_JSON_BOOL;
      v->u.boolean = 0;
      return parse_ident(p, "alse");
    case '"':
      v->type = ND_JSON_STRING;
      return parse_string(p, &v->u.str.ptr, &v->u.str.len);
    case '[':
      return parse_array(p, v);
    case '{':
      return parse_object(p, v);
    default: {
      size_t start = p->i;
      int rc;
      v->type = ND_JSON_NUMBER;
      if (c == '-') {
        p->i++;
        rc = parse_integer(p, 0, v);
      } else if (is_digit(c)) {
        rc = parse_integer(p, 1, v);
      } else {
        return p_perr(p, "expected value");
      }
      if (rc) return rc;
      v->u.num.text_len = p->i - start;
      v->u.num.text = arena_copy(p, p->s + start, p->i - start);
      if (!v->u.num.text) return p_nomem(p);
      return 0;
    }
  }
}

int nd_json_parse(const char *text, size_t len, const nd_json_limits *lim, nd_json_doc **out,
                  nd_err *err) {
  parser p;
  nd_json_limits L = lim ? *lim : nd_json_limits_default();
  nd_json_doc *doc;
  int rc;
  if (!out || (!text && len)) {
    nd_seterr(err, "nd_json: bad arguments");
    return ND_E_ARG;
  }
  *out = NULL;
  if (len > L.max_input || len > (size_t)0x7fffffff) {
    nd_seterr(err, "nd_json limit: input of %lu bytes is too long (limit %lu)", (unsigned long)len,
              (unsigned long)(L.max_input < 0x7fffffff ? L.max_input : 0x7fffffff));
    return ND_E_BOUNDS;
  }
  doc = (nd_json_doc *)calloc(1, sizeof *doc);
  if (!doc) {
    nd_seterr(err, "nd_json: out of memory");
    return ND_E_NOMEM;
  }
  memset(&p, 0, sizeof p);
  p.s = (const unsigned char *)text;
  p.n = len;
  p.a = &doc->a;
  p.max_elements = L.max_elements;
  p.max_depth = L.max_depth < ND_JSON_SERDE_MAX_DEPTH ? L.max_depth : ND_JSON_SERDE_MAX_DEPTH;
  p.serde_depth = L.max_depth >= ND_JSON_SERDE_MAX_DEPTH;
  p.err = err;
  rc = parse_value(&p, &doc->root);
  if (!rc && p_ws(&p) >= 0) rc = p_perr(&p, "trailing characters"); /* Deserializer::end */
  free(p.vs.v);
  free(p.ms.m);
  free(p.sb);
  if (rc) {
    arena_free(&doc->a);
    free(doc);
    return rc;
  }
  *out = doc;
  return ND_OK;
}

void nd_json_doc_free(nd_json_doc *doc) {
  if (!doc) return;
  arena_free(&doc->a);
  free(doc);
}

const nd_json_value *nd_json_root(const nd_json_doc *doc) { return doc ? &doc->root : NULL; }

const nd_json_value *nd_json_get(const nd_json_value *obj, const char *key, size_t key_len) {
  size_t k;
  if (!obj || obj->type != ND_JSON_OBJECT) return NULL;
  for (k = obj->u.obj.len; k > 0; k--) {
    const nd_json_member *m = &obj->u.obj.members[k - 1];
    if (m->key_len == key_len && (key_len == 0 || memcmp(m->key, key, key_len) == 0))
      return &m->value;
  }
  return NULL;
}

const nd_json_value *nd_json_get_cstr(const nd_json_value *obj, const char *key) {
  return nd_json_get(obj, key, strlen(key));
}

const nd_json_value *nd_json_at(const nd_json_value *arr, size_t i) {
  if (!arr || arr->type != ND_JSON_ARRAY || i >= arr->u.arr.len) return NULL;
  return &arr->u.arr.items[i];
}

int nd_json_as_i64(const nd_json_value *v, int64_t *out) {
  if (!v || v->type != ND_JSON_NUMBER) return 0;
  if (v->u.num.kind == ND_JSON_NUM_I64) {
    *out = v->u.num.i;
    return 1;
  }
  if (v->u.num.kind == ND_JSON_NUM_U64 && v->u.num.u <= (uint64_t)INT64_MAX) {
    *out = (int64_t)v->u.num.u;
    return 1;
  }
  return 0;
}

int nd_json_as_u64(const nd_json_value *v, uint64_t *out) {
  if (!v || v->type != ND_JSON_NUMBER || v->u.num.kind != ND_JSON_NUM_U64) return 0;
  *out = v->u.num.u;
  return 1;
}

int nd_json_as_f64(const nd_json_value *v, double *out) {
  if (!v || v->type != ND_JSON_NUMBER) return 0;
  *out = v->u.num.f;
  return 1;
}

int nd_json_as_bool(const nd_json_value *v, int *out) {
  if (!v || v->type != ND_JSON_BOOL) return 0;
  *out = v->u.boolean;
  return 1;
}

int nd_json_as_str(const nd_json_value *v, const char **s, size_t *len) {
  if (!v || v->type != ND_JSON_STRING) return 0;
  *s = v->u.str.ptr;
  if (len) *len = v->u.str.len;
  return 1;
}

/* ─── Writer ───────────────────────────────────────────────────────────────────────────────── */

void nd_json_writer_init(nd_json_writer *w, size_t max_len) {
  memset(w, 0, sizeof *w);
  w->max_len = max_len;
}

void nd_json_writer_free(nd_json_writer *w) {
  free(w->data);
  w->data = NULL;
  w->len = w->cap = 0;
}

void nd_json_writer_reset(nd_json_writer *w) {
  size_t max_len = w->max_len, cap = w->cap;
  char *data = w->data;
  memset(w, 0, sizeof *w);
  w->max_len = max_len;
  w->cap = cap;
  w->data = data;
  if (data) data[0] = '\0';
}

void nd_json_writer_raw(nd_json_writer *w, const char *s, size_t n) {
  size_t need;
  if (w->err) return;
  if (!nd_add_ok(w->len, n, &need) || !nd_add_ok(need, 1, &need)) {
    w->err = ND_E_NOMEM;
    return;
  }
  if (w->max_len && need - 1u > w->max_len) {
    w->err = ND_E_BOUNDS;
    return;
  }
  if (need > w->cap) {
    size_t nc = w->cap ? w->cap : 256u;
    char *q;
    while (nc < need) {
      if (!nd_mul_ok(nc, 2, &nc)) {
        w->err = ND_E_NOMEM;
        return;
      }
    }
    q = (char *)realloc(w->data, nc);
    if (!q) {
      w->err = ND_E_NOMEM;
      return;
    }
    w->data = q;
    w->cap = nc;
  }
  if (n) memcpy(w->data + w->len, s, n);
  w->len += n;
  w->data[w->len] = '\0';
}

/* Bookkeeping before a value: comma inside arrays, refuse a bare value where a key is due. */
static int w_before_value(nd_json_writer *w) {
  if (w->err) return 0;
  if (w->depth == 0) {
    if (w->done) {
      w->err = ND_E_ARG;
      return 0;
    }
    return 1;
  }
  if (w->kind[w->depth - 1] == 'o') {
    if (!w->want_value) {
      w->err = ND_E_ARG;
      return 0;
    }
    w->want_value = 0;
    return 1;
  }
  if (w->count[w->depth - 1]) nd_json_writer_raw(w, ",", 1);
  w->count[w->depth - 1] = 1;
  return !w->err;
}

static void w_after_value(nd_json_writer *w) {
  if (w->depth == 0) w->done = 1;
}

static void w_escaped(nd_json_writer *w, const char *s, size_t n) {
  static const char hex[] = "0123456789abcdef";
  size_t start = 0, k;
  nd_json_writer_raw(w, "\"", 1);
  for (k = 0; k < n; k++) {
    unsigned char c = (unsigned char)s[k];
    const char *esc = NULL;
    char ubuf[6];
    size_t elen = 2;
    if (c >= 0x20u && c != '"' && c != '\\') continue;
    switch (c) {
      case '"': esc = "\\\""; break;
      case '\\': esc = "\\\\"; break;
      case 0x08: esc = "\\b"; break;
      case 0x09: esc = "\\t"; break;
      case 0x0a: esc = "\\n"; break;
      case 0x0c: esc = "\\f"; break;
      case 0x0d: esc = "\\r"; break;
      default:
        ubuf[0] = '\\';
        ubuf[1] = 'u';
        ubuf[2] = '0';
        ubuf[3] = '0';
        ubuf[4] = hex[c >> 4];
        ubuf[5] = hex[c & 0xFu];
        esc = ubuf;
        elen = 6;
        break;
    }
    if (k > start) nd_json_writer_raw(w, s + start, k - start);
    nd_json_writer_raw(w, esc, elen);
    start = k + 1;
  }
  if (n > start) nd_json_writer_raw(w, s + start, n - start);
  nd_json_writer_raw(w, "\"", 1);
}

static void w_open(nd_json_writer *w, unsigned char kind) {
  if (!w_before_value(w)) return;
  if (w->depth >= ND_JSON_WRITER_MAX_DEPTH) {
    w->err = ND_E_BOUNDS;
    return;
  }
  nd_json_writer_raw(w, kind == 'o' ? "{" : "[", 1);
  w->kind[w->depth] = kind;
  w->count[w->depth] = 0;
  w->depth++;
}

static void w_close(nd_json_writer *w, unsigned char kind) {
  if (w->err) return;
  if (w->depth == 0 || w->kind[w->depth - 1] != kind || w->want_value) {
    w->err = ND_E_ARG;
    return;
  }
  nd_json_writer_raw(w, kind == 'o' ? "}" : "]", 1);
  w->depth--;
  w_after_value(w);
}

void nd_json_writer_begin_object(nd_json_writer *w) { w_open(w, 'o'); }
void nd_json_writer_end_object(nd_json_writer *w) { w_close(w, 'o'); }
void nd_json_writer_begin_array(nd_json_writer *w) { w_open(w, 'a'); }
void nd_json_writer_end_array(nd_json_writer *w) { w_close(w, 'a'); }

void nd_json_writer_key(nd_json_writer *w, const char *k, size_t len) {
  if (w->err) return;
  if (w->depth == 0 || w->kind[w->depth - 1] != 'o' || w->want_value) {
    w->err = ND_E_ARG;
    return;
  }
  if (w->count[w->depth - 1]) nd_json_writer_raw(w, ",", 1);
  w->count[w->depth - 1] = 1;
  w_escaped(w, k, len);
  nd_json_writer_raw(w, ":", 1);
  w->want_value = 1;
}

void nd_json_writer_key_cstr(nd_json_writer *w, const char *k) {
  nd_json_writer_key(w, k, strlen(k));
}

static void w_scalar(nd_json_writer *w, const char *s, size_t n) {
  if (!w_before_value(w)) return;
  nd_json_writer_raw(w, s, n);
  w_after_value(w);
}

void nd_json_writer_null(nd_json_writer *w) { w_scalar(w, "null", 4); }
void nd_json_writer_bool(nd_json_writer *w, int b) {
  if (b)
    w_scalar(w, "true", 4);
  else
    w_scalar(w, "false", 5);
}
void nd_json_writer_i64(nd_json_writer *w, int64_t v) {
  char b[ND_JSON_NUMBUF];
  size_t n = nd_json_write_i64(v, b);
  w_scalar(w, b, n);
}
void nd_json_writer_u64(nd_json_writer *w, uint64_t v) {
  char b[ND_JSON_NUMBUF];
  size_t n = nd_json_write_u64(v, b);
  w_scalar(w, b, n);
}
void nd_json_writer_f64(nd_json_writer *w, double v) {
  char b[ND_JSON_NUMBUF];
  size_t n = nd_json_write_f64(v, b);
  w_scalar(w, b, n);
}
void nd_json_writer_f32(nd_json_writer *w, float v) {
  char b[ND_JSON_NUMBUF];
  size_t n = nd_json_write_f32(v, b);
  w_scalar(w, b, n);
}
void nd_json_writer_string(nd_json_writer *w, const char *s, size_t len) {
  if (!w_before_value(w)) return;
  w_escaped(w, s, len);
  w_after_value(w);
}
void nd_json_writer_string_cstr(nd_json_writer *w, const char *s) {
  nd_json_writer_string(w, s, strlen(s));
}

static int key_cmp(const nd_json_member *a, const nd_json_member *b) {
  size_t n = a->key_len < b->key_len ? a->key_len : b->key_len;
  int c = n ? memcmp(a->key, b->key, n) : 0;
  if (c) return c;
  return a->key_len < b->key_len ? -1 : a->key_len > b->key_len ? 1 : 0;
}

static void w_value(nd_json_writer *w, const nd_json_value *v, nd_json_keyorder order);

/* Members as a BTreeMap<String, Value> holds them: sorted by key bytes, the last duplicate wins.
 * A stable merge sort of indices keeps duplicates in source order, so the last of each run of
 * equal keys is the one serde_json keeps. */
static void w_sorted_object(nd_json_writer *w, const nd_json_value *v) {
  size_t n = v->u.obj.len, *idx, *tmp, width, k;
  const nd_json_member *m = v->u.obj.members;
  idx = (size_t *)nd_calloc(n, sizeof *idx);
  tmp = (size_t *)nd_calloc(n, sizeof *tmp);
  if (!idx || !tmp) {
    free(idx);
    free(tmp);
    w->err = ND_E_NOMEM;
    return;
  }
  for (k = 0; k < n; k++) idx[k] = k;
  for (width = 1; width < n; width *= 2) {
    size_t lo;
    for (lo = 0; lo < n; lo += 2 * width) {
      size_t mid = lo + width < n ? lo + width : n;
      size_t hi = lo + 2 * width < n ? lo + 2 * width : n;
      size_t a = lo, b = mid, o = lo;
      while (a < mid && b < hi) {
        if (key_cmp(&m[idx[b]], &m[idx[a]]) < 0)
          tmp[o++] = idx[b++];
        else
          tmp[o++] = idx[a++];
      }
      while (a < mid) tmp[o++] = idx[a++];
      while (b < hi) tmp[o++] = idx[b++];
    }
    memcpy(idx, tmp, n * sizeof *idx);
  }
  nd_json_writer_begin_object(w);
  for (k = 0; k < n && !w->err; k++) {
    if (k + 1 < n && key_cmp(&m[idx[k]], &m[idx[k + 1]]) == 0) continue;
    nd_json_writer_key(w, m[idx[k]].key, m[idx[k]].key_len);
    w_value(w, &m[idx[k]].value, ND_JSON_KEYS_SORTED);
  }
  nd_json_writer_end_object(w);
  free(idx);
  free(tmp);
}

static void w_value(nd_json_writer *w, const nd_json_value *v, nd_json_keyorder order) {
  size_t k;
  if (w->err) return;
  switch (v->type) {
    case ND_JSON_NULL: nd_json_writer_null(w); break;
    case ND_JSON_BOOL: nd_json_writer_bool(w, v->u.boolean); break;
    case ND_JSON_NUMBER:
      if (v->u.num.kind == ND_JSON_NUM_U64)
        nd_json_writer_u64(w, v->u.num.u);
      else if (v->u.num.kind == ND_JSON_NUM_I64)
        nd_json_writer_i64(w, v->u.num.i);
      else
        nd_json_writer_f64(w, v->u.num.f);
      break;
    case ND_JSON_STRING: nd_json_writer_string(w, v->u.str.ptr, v->u.str.len); break;
    case ND_JSON_ARRAY:
      nd_json_writer_begin_array(w);
      for (k = 0; k < v->u.arr.len && !w->err; k++) w_value(w, &v->u.arr.items[k], order);
      nd_json_writer_end_array(w);
      break;
    case ND_JSON_OBJECT:
      if (order == ND_JSON_KEYS_SORTED) {
        w_sorted_object(w, v);
        break;
      }
      nd_json_writer_begin_object(w);
      for (k = 0; k < v->u.obj.len && !w->err; k++) {
        const nd_json_member *m = &v->u.obj.members[k];
        nd_json_writer_key(w, m->key, m->key_len);
        w_value(w, &m->value, order);
      }
      nd_json_writer_end_object(w);
      break;
    default: w->err = ND_E_ARG; break;
  }
}

void nd_json_writer_value(nd_json_writer *w, const nd_json_value *v, nd_json_keyorder order) {
  if (!v) {
    if (!w->err) w->err = ND_E_ARG;
    return;
  }
  w_value(w, v, order);
}

int nd_json_to_string(const nd_json_value *v, nd_json_keyorder order, char **out, size_t *out_len) {
  nd_json_writer w;
  nd_json_writer_init(&w, 0);
  nd_json_writer_value(&w, v, order);
  if (!w.err && !w.data) nd_json_writer_raw(&w, "", 0);
  if (w.err) {
    nd_json_writer_free(&w);
    *out = NULL;
    return w.err;
  }
  *out = w.data;
  if (out_len) *out_len = w.len;
  return ND_OK;
}
