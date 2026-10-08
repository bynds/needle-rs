/* nd_grammar.h: grammar-constrained decoding of Needle tool calls.
 *
 * An exact transcription of needle-rs crates/needle-infer/src/constrained.rs:
 *
 *   nd_grammar_parse_tools            ToolDef::from_json / parse_tools_json (the ad-hoc scanner,
 *                                     not a JSON parser: it finds the first `"name":"` and the
 *                                     first `"parameters":` / `"properties":` substrings, keeps
 *                                     escapes raw, and so on, byte for byte as the Rust does)
 *   nd_grammar_new                    ConstrainedDecoder::new
 *   nd_grammar_set_unique_arg_keys    ConstrainedDecoder::with_unique_arg_keys (the v3 engine
 *                                     always enables it)
 *   nd_grammar_update                 ConstrainedDecoder::update
 *   nd_grammar_feed_bytes             ConstrainedDecoder::feed_bytes
 *   nd_grammar_logit_mask             ConstrainedDecoder::logit_mask: ND_GRAMMAR_ALLOWED (0.0f)
 *                                     or ND_GRAMMAR_BLOCKED (-1e9f) per token, all
 *                                     ND_GRAMMAR_ALLOWED outside a constrained region or when no
 *                                     token fits
 *   nd_grammar_byte_table_from_pieces constrained::byte_table (SpTokenizer::token_bytes)
 *
 * The tool name's snake form is needle-infer's tokenizer::to_snake_case, with Rust 1.87's Unicode
 * tables for is_alphanumeric / is_uppercase / is_lowercase.
 *
 * Divergences, all on inputs the Rust cannot be given or cannot survive:
 *  - The Rust takes `&str`; nd_grammar_parse_tools refuses input that is not valid UTF-8
 *    (ND_E_FORMAT) instead of having no defined behaviour for it.
 *  - Allocation failure is reported (ND_E_NOMEM) where Rust would abort; a token id so large that
 *    the id-indexed table cannot be allocated is ND_E_NOMEM / ND_E_BOUNDS.
 */
#ifndef ND_GRAMMAR_H
#define ND_GRAMMAR_H

#include <stddef.h>
#include <stdint.h>

#include "nd_common.h"

#define ND_GRAMMAR_ALLOWED 0.0f
#define ND_GRAMMAR_BLOCKED (-1e9f)

/* SentencePiece piece types (sp_tokenizer.rs TK_*). */
#define ND_TK_NORMAL 0
#define ND_TK_UNKNOWN 1
#define ND_TK_CONTROL 2
#define ND_TK_USER_DEFINED 3
#define ND_TK_BYTE 4

/* ToolDef. Strings are NUL-terminated copies (lengths given; names never contain NUL since the
 * input is UTF-8 JSON text, but lengths are authoritative). */
typedef struct {
  char *name;
  size_t name_len;
  char *snake_name;
  size_t snake_name_len;
  char **param_keys;
  size_t *param_key_lens;
  size_t n_param_keys;
} nd_tooldef;

typedef struct {
  nd_tooldef *tools;
  size_t n;
} nd_tooldefs;

/* ToolDef::from_json. On success *out holds zero or more tools (zero is not an error, as in
 * Rust). Free with nd_tooldefs_free. */
int nd_grammar_parse_tools(const char *json, size_t len, nd_tooldefs *out, nd_err *err);
void nd_tooldefs_free(nd_tooldefs *t);

/* One (id, bytes) entry of the token byte table, as byte_table() returns them. */
typedef struct {
  uint32_t id;
  const uint8_t *bytes;
  size_t len;
} nd_token_bytes;

/* byte_table(): entry i is token i; TK_BYTE pieces `<0xNN>` give their byte (or nothing if the
 * surface does not parse, as upstream's `p[3:5]`), TK_CONTROL / TK_UNKNOWN give nothing, and every
 * other piece gives its surface with U+2581 replaced by ' '. `pieces[i]` has `piece_lens[i]`
 * bytes (UTF-8, as the tokenizer validated them). The result (*table, n entries) is one
 * allocation; release it with free(). */
int nd_grammar_byte_table_from_pieces(const char *const *pieces, const size_t *piece_lens,
                                      const uint8_t *types, size_t n, nd_token_bytes **table,
                                      nd_err *err);

typedef enum {
  ND_GRAMMAR_FREE = 0,
  ND_GRAMMAR_IN_NAME = 1,
  ND_GRAMMAR_IN_ARG_KEY = 2
} nd_grammar_state;

typedef struct nd_grammar nd_grammar;

/* ConstrainedDecoder::new(tool_defs, token_bytes). The table is copied. */
int nd_grammar_new(const nd_tooldef *tools, size_t n_tools, const nd_token_bytes *table,
                   size_t n_entries, nd_grammar **out, nd_err *err);
void nd_grammar_free(nd_grammar *g);
void nd_grammar_set_unique_arg_keys(nd_grammar *g, int on);

/* Advance after emitting token_id (ids outside the table feed nothing). ND_E_NOMEM only. */
int nd_grammar_update(nd_grammar *g, uint32_t token_id);
int nd_grammar_feed_bytes(nd_grammar *g, const uint8_t *bytes, size_t len);

/* The additive mask for the current state into mask[0..vocab_size). ND_E_NOMEM only (the
 * duplicate-key filter builds a small trie). */
int nd_grammar_logit_mask(nd_grammar *g, float *mask, size_t vocab_size);

/* Introspection (tests, logging). */
nd_grammar_state nd_grammar_get_state(const nd_grammar *g);
const char *nd_grammar_current_function(const nd_grammar *g, size_t *len);
size_t nd_grammar_used_key_count(const nd_grammar *g);
const char *nd_grammar_used_key(const nd_grammar *g, size_t i, size_t *len);
const uint8_t *nd_grammar_constrained_buf(const nd_grammar *g, size_t *len);
/* Number of entries in the id-indexed token text table (max id + 1). */
size_t nd_grammar_table_size(const nd_grammar *g);

#endif
