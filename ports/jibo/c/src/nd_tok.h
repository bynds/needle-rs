/* nd_tok.h: the Needle 3 SentencePiece tokenizer (transcription of
 * crates/needle-infer/src/sp_tokenizer.rs).
 *
 * The tokenizer is decoded from the RAW record a .cact container embeds:
 *
 *   header   u32 n_pieces, u32 pad_id, u32 eos_id, u32 bos_id, u32 unk_id,
 *            u8 add_dummy_prefix, u8 byte_fallback, u16 _pad              (24 bytes)
 *   pieces   n_pieces records in id order:
 *            f32 score, u8 type, u16 surface_len, surface_len UTF-8 bytes
 *
 * Encoding splits off the user-defined pieces (chat markers, longest first) and runs greedy
 * highest-score pairwise merging over the characters of each remaining segment, with byte
 * fallback. Decoding drops control/unknown pieces, re-assembles byte pieces, replaces invalid
 * UTF-8 the way Rust's String::from_utf8_lossy does, turns U+2581 into spaces and strips one
 * leading space when add_dummy_prefix is set. Token ids and decoded bytes are identical to the
 * Rust implementation.
 */
#ifndef ND_TOK_H
#define ND_TOK_H

#include "nd_common.h"

/* Piece types (export.TK_*). */
#define ND_TK_NORMAL 0
#define ND_TK_UNKNOWN 1
#define ND_TK_CONTROL 2
#define ND_TK_USER_DEFINED 3
#define ND_TK_BYTE 4

/* SentencePiece's visible space, U+2581, as UTF-8. */
#define ND_META_SPACE "\xE2\x96\x81"

typedef struct nd_tok nd_tok;

/* Decode a tokenizer blob. The blob is copied; it need not outlive the tokenizer.
 * Returns ND_OK, ND_E_FORMAT for a malformed blob (the message in *e matches the Rust
 * TokenizerError's Display text), or ND_E_NOMEM. */
int nd_tok_load(const uint8_t *blob, size_t len, nd_tok **out, nd_err *e);
void nd_tok_free(nd_tok *t);

/* Encode `len` bytes of UTF-8 text (which may contain NUL). On success *ids is a malloc'd
 * array of *n ids (NULL when *n == 0) the caller frees. ND_E_ARG if the text is not valid
 * UTF-8 (Rust's &str cannot be), ND_E_NOMEM / ND_E_BOUNDS on allocation or size failure. */
int nd_tok_encode(const nd_tok *t, const char *text, size_t len, uint32_t **ids, size_t *n);

/* Decode ids to text. *text is malloc'd, NUL-terminated, *len bytes long (excluding the NUL;
 * the text itself may contain NUL bytes). Ids outside the table are skipped, as in Rust. */
int nd_tok_decode(const nd_tok *t, const uint32_t *ids, size_t n, char **text, size_t *len);

/* Id of an exact surface (later duplicates win), or -1. */
int32_t nd_tok_id_of(const nd_tok *t, const char *piece);
int32_t nd_tok_id_of_n(const nd_tok *t, const char *piece, size_t len);

/* Surface of a piece (NUL-terminated; *len, if non-NULL, gets its byte length), or NULL. */
const char *nd_tok_piece(const nd_tok *t, uint32_t id, size_t *len);
/* Piece type, or -1 for an id outside the table. */
int nd_tok_type(const nd_tok *t, uint32_t id);
/* Piece score; 0 if the id is outside the table (check nd_tok_type first). */
float nd_tok_score(const nd_tok *t, uint32_t id);
size_t nd_tok_size(const nd_tok *t);

/* The bytes decoding `id` alone contributes before the lossy/meta-space pass, i.e. Rust's
 * SpTokenizer::token_bytes()[id]: TK_BYTE gives its byte, control/unknown nothing, anything
 * else its surface with U+2581 replaced by a space. Writes at most `cap` bytes into `buf` and
 * returns the full length, or -1 for an id outside the table. */
long nd_tok_token_bytes(const nd_tok *t, uint32_t id, uint8_t *buf, size_t cap);

/* The v2/v3 chat markers in the id order needle/model/tokenizer.py documents
 * (sp_tokenizer::CHAT_MARKERS), and their ids: ND_OK, or ND_E_FORMAT if the table lacks one. */
#define ND_N_CHAT_MARKERS 10
extern const char *const nd_chat_markers[ND_N_CHAT_MARKERS];
int nd_tok_chat_marker_ids(const nd_tok *t, uint32_t out[ND_N_CHAT_MARKERS]);

uint32_t nd_tok_pad_id(const nd_tok *t);
uint32_t nd_tok_eos_id(const nd_tok *t);
uint32_t nd_tok_bos_id(const nd_tok *t);
uint32_t nd_tok_unk_id(const nd_tok *t);
int nd_tok_add_dummy_prefix(const nd_tok *t);
int nd_tok_byte_fallback(const nd_tok *t);

/* UTF-8 helpers shared with nd_prompt.c. */
/* 1 if s[0..len) is valid UTF-8 by Rust's (= Unicode's) rules. */
int nd_utf8_valid(const uint8_t *s, size_t len);
/* String::from_utf8_lossy: each maximal invalid subpart becomes U+FFFD. Returns a malloc'd
 * NUL-terminated buffer of *out_len bytes, or NULL on allocation failure. */
char *nd_utf8_lossy(const uint8_t *s, size_t len, size_t *out_len);

#endif
