/* nd_json.h: a bounded JSON reader and a serde_json-compatible writer.
 *
 * Reader: an RFC 8259 DOM parser that accepts and rejects exactly what serde_json 1.0.149
 * (`serde_json::from_slice::<Value>`, default features, as pinned in needle-rs's Cargo.lock)
 * accepts and rejects, within the caller's limits:
 *
 *  - Whitespace is ' ', '\t', '\n', '\r' only; no BOM, no comments, no trailing commas, nothing
 *    after the root value but whitespace.
 *  - Strings: raw control bytes (< 0x20) are rejected; escapes \" \\ \/ \b \f \n \r \t \uXXXX;
 *    surrogate pairs combine; a lone leading or trailing surrogate is rejected (serde_json's
 *    validating mode, used for every `str`); \u0000 is accepted (strings carry a length and may
 *    contain NUL); the decoded bytes must be valid UTF-8.
 *  - Numbers follow serde_json's own parser exactly, including its (not always correctly rounded)
 *    float conversion: significand * or / 10^k in double precision, as serde_json does without
 *    the `float_roundtrip` feature. A literal whose magnitude overflows to infinity is rejected
 *    ("number out of range"), as serde_json rejects it. Each number keeps its source text and
 *    serde_json's classification:
 *      ND_JSON_NUM_U64  an integer literal (no '.', no exponent) that fits u64, sign '+'
 *      ND_JSON_NUM_I64  a negative integer literal that fits i64 ("-0" is NOT one: it is -0.0)
 *      ND_JSON_NUM_F64  everything else ("5.0", "1e3", "-0", integers beyond u64/i64 range)
 *    so nd_json_as_i64 / nd_json_as_u64 / nd_json_as_f64 match Value::as_i64 / as_u64 / as_f64.
 *  - Nesting: serde_json refuses a 128th nested array/object (127 is its limit). The effective
 *    limit here is min(limits.max_depth, 127); the default is 64.
 *  - Error messages: for every input serde_json rejects, the nd_err message is byte-identical to
 *    serde_json::Error's Display ("expected value at line 1 column 1", "EOF while parsing a string
 *    at line 1 column 7", ...), with serde_json's line/column rules (column counts bytes; the
 *    position each check reports, as its SliceRead/StrRead does), and the return code is
 *    ND_E_FORMAT. That includes serde_json's own nesting limit ("recursion limit exceeded"),
 *    reported when max_depth >= 127. Refusals caused by this reader's limits (max_input, a
 *    max_depth below 127, max_elements) return ND_E_BOUNDS and a message starting
 *    "nd_json limit: ", which serde_json never produces. "invalid unicode code point" can only
 *    arise from non-UTF-8 input (serde_json::from_slice); from_str inputs never produce it.
 *  - Objects keep every member in source order, duplicates included. nd_json_get returns the LAST
 *    member with the key, which is what serde_json keeps when it deserialises into a map
 *    (Map/BTreeMap/HashMap `insert` replaces the earlier value).
 *
 * Writer: compact output byte-identical to serde_json::to_string:
 *  - strings escape '"', '\\', \b \t \n \f \r, other bytes < 0x20 as \u00xx (lowercase hex);
 *    everything else, '/' and DEL and non-ASCII UTF-8 included, is copied through;
 *  - integers in decimal; floats exactly as serde_json 1.0.149 prints them. NOTE: that version
 *    formats with the `zmij` crate (1.0.21), not ryu: shortest round-trip digits, fixed notation
 *    for decimal exponents -5..=15 (f64) / -6..=12 (f32), integral values with ".0", and an
 *    exponent that always carries its sign: 1e21 -> "1e+21", 1e-7 -> "1e-7",
 *    f64::MAX -> "1.7976931348623157e+308". Non-finite values are written as `null`.
 *  - serialising an `f32` directly (serialize_f32) uses the f32 formatter (0.1f32 -> "0.1"),
 *    whereas an f32 put into a serde_json::Value first becomes an f64 (Value::from(0.1f32) ->
 *    "0.10000000149011612"); use nd_json_writer_f32 or nd_json_writer_f64((double)x) to match
 *    whichever the Rust code does.
 *  - objects: nd_json_writer_value(.., ND_JSON_KEYS_INSERTION) writes members in their stored
 *    order (duplicates included); ND_JSON_KEYS_SORTED writes them as a serde_json::Value would:
 *    keys in byte order with duplicates collapsed to the last one (serde_json's Map is a BTreeMap
 *    unless `preserve_order` is enabled, and it is not in needle-rs).
 */
#ifndef ND_JSON_H
#define ND_JSON_H

#include <stddef.h>
#include <stdint.h>

#include "nd_common.h"

#define ND_JSON_SERDE_MAX_DEPTH 127u /* serde_json's recursion limit (remaining_depth 128) */
#define ND_JSON_DEFAULT_MAX_DEPTH 64u

typedef struct {
  size_t max_input;    /* bytes; inputs longer than this are refused (0 means "no input") */
  unsigned max_depth;  /* nested arrays/objects; capped at ND_JSON_SERDE_MAX_DEPTH */
  size_t max_elements; /* values in the document (every scalar, array, object and key counts) */
} nd_json_limits;

/* max_input 1 MiB, max_depth 64, max_elements 65536. */
nd_json_limits nd_json_limits_default(void);

typedef enum {
  ND_JSON_NULL = 0,
  ND_JSON_BOOL,
  ND_JSON_NUMBER,
  ND_JSON_STRING,
  ND_JSON_ARRAY,
  ND_JSON_OBJECT
} nd_json_type;

typedef enum { ND_JSON_NUM_U64 = 0, ND_JSON_NUM_I64, ND_JSON_NUM_F64 } nd_json_numkind;

typedef struct nd_json_value nd_json_value;
typedef struct nd_json_member nd_json_member;

struct nd_json_value {
  nd_json_type type;
  union {
    int boolean;
    struct {
      const char *ptr; /* NUL-terminated for convenience; may also contain NUL */
      size_t len;
    } str;
    struct {
      const nd_json_value *items;
      size_t len;
    } arr;
    struct {
      const nd_json_member *members; /* source order, duplicates kept */
      size_t len;
    } obj;
    struct {
      const char *text; /* the literal as written (NUL-terminated copy) */
      size_t text_len;
      nd_json_numkind kind;
      uint64_t u; /* valid when kind == ND_JSON_NUM_U64 */
      int64_t i;  /* valid when kind == ND_JSON_NUM_I64 */
      double f;   /* always valid: serde_json's as_f64 */
    } num;
  } u;
};

struct nd_json_member {
  const char *key; /* decoded, NUL-terminated, may contain NUL */
  size_t key_len;
  nd_json_value value;
};

typedef struct nd_json_doc nd_json_doc;

/* Parse `len` bytes. On success *out owns the whole tree (free with nd_json_doc_free). Returns
 * ND_OK, ND_E_FORMAT (anything serde_json rejects; err->msg is serde_json's message), ND_E_BOUNDS
 * (one of this reader's limits; err->msg starts "nd_json limit: "), ND_E_NOMEM or ND_E_ARG.
 * `lim` may be NULL for the defaults. */
int nd_json_parse(const char *text, size_t len, const nd_json_limits *lim, nd_json_doc **out,
                  nd_err *err);
void nd_json_doc_free(nd_json_doc *doc);
const nd_json_value *nd_json_root(const nd_json_doc *doc);

/* Object lookup: the LAST member whose key equals key[0..key_len); NULL if absent or `obj` is not
 * an object. */
const nd_json_value *nd_json_get(const nd_json_value *obj, const char *key, size_t key_len);
const nd_json_value *nd_json_get_cstr(const nd_json_value *obj, const char *key);
/* Array element, NULL when out of range or not an array. */
const nd_json_value *nd_json_at(const nd_json_value *arr, size_t i);

/* serde_json's Value::as_i64 / as_u64 / as_f64 / as_bool / as_str: 1 and *out set on success, 0
 * otherwise. */
int nd_json_as_i64(const nd_json_value *v, int64_t *out);
int nd_json_as_u64(const nd_json_value *v, uint64_t *out);
int nd_json_as_f64(const nd_json_value *v, double *out);
int nd_json_as_bool(const nd_json_value *v, int *out);
int nd_json_as_str(const nd_json_value *v, const char **s, size_t *len);

/* Strict UTF-8 validation, as Rust's str::from_utf8: 1 when valid. */
int nd_json_utf8_valid(const unsigned char *s, size_t n);

/* ─── Float and integer formatting ─────────────────────────────────────────────────────────── */

#define ND_JSON_NUMBUF 32 /* enough for any output below, plus a NUL */

/* serde_json's text for an f64 / f32 (zmij's format_finite, or "null" for NaN and infinities).
 * Writes a NUL-terminated string to out and returns its length. */
size_t nd_json_write_f64(double v, char out[ND_JSON_NUMBUF]);
size_t nd_json_write_f32(float v, char out[ND_JSON_NUMBUF]);
size_t nd_json_write_i64(int64_t v, char out[ND_JSON_NUMBUF]);
size_t nd_json_write_u64(uint64_t v, char out[ND_JSON_NUMBUF]);

/* ─── Writer ───────────────────────────────────────────────────────────────────────────────── */

#define ND_JSON_WRITER_MAX_DEPTH 128

typedef enum { ND_JSON_KEYS_INSERTION = 0, ND_JSON_KEYS_SORTED = 1 } nd_json_keyorder;

/* A growable output buffer with structural bookkeeping: commas and colons are inserted
 * automatically. Errors are sticky in `err` (ND_E_NOMEM, ND_E_BOUNDS when max_len would be
 * exceeded, ND_E_ARG on a structurally invalid call sequence); once set, further calls do nothing.
 * `data` is always NUL-terminated after a successful call. */
typedef struct {
  char *data;
  size_t len, cap;
  size_t max_len; /* 0 = unlimited */
  int err;
  unsigned depth;
  unsigned char kind[ND_JSON_WRITER_MAX_DEPTH];  /* 'a' array, 'o' object */
  unsigned char count[ND_JSON_WRITER_MAX_DEPTH]; /* 0 = nothing written yet at this level */
  int want_value;                                /* a key was written; a value must follow */
  int done;                                      /* a complete root value was written */
} nd_json_writer;

void nd_json_writer_init(nd_json_writer *w, size_t max_len);
void nd_json_writer_free(nd_json_writer *w);
/* Forget the output, keep the buffer. */
void nd_json_writer_reset(nd_json_writer *w);

void nd_json_writer_begin_object(nd_json_writer *w);
void nd_json_writer_end_object(nd_json_writer *w);
void nd_json_writer_begin_array(nd_json_writer *w);
void nd_json_writer_end_array(nd_json_writer *w);
/* An object key (must be inside an object, where a key is expected). */
void nd_json_writer_key(nd_json_writer *w, const char *k, size_t len);
void nd_json_writer_key_cstr(nd_json_writer *w, const char *k);
void nd_json_writer_null(nd_json_writer *w);
void nd_json_writer_bool(nd_json_writer *w, int b);
void nd_json_writer_i64(nd_json_writer *w, int64_t v);
void nd_json_writer_u64(nd_json_writer *w, uint64_t v);
void nd_json_writer_f64(nd_json_writer *w, double v); /* serialize_f64 (NaN/inf -> null) */
void nd_json_writer_f32(nd_json_writer *w, float v);  /* serialize_f32 (NaN/inf -> null) */
/* A string value; bytes are escaped as serde_json does and otherwise copied (pass UTF-8). */
void nd_json_writer_string(nd_json_writer *w, const char *s, size_t len);
void nd_json_writer_string_cstr(nd_json_writer *w, const char *s);
/* A whole DOM value. */
void nd_json_writer_value(nd_json_writer *w, const nd_json_value *v, nd_json_keyorder order);
/* Append raw bytes with no bookkeeping (for callers that assemble text themselves). */
void nd_json_writer_raw(nd_json_writer *w, const char *s, size_t len);

/* Convenience: serialise a DOM value to a fresh malloc'd NUL-terminated string. */
int nd_json_to_string(const nd_json_value *v, nd_json_keyorder order, char **out, size_t *out_len);

#endif
