/* nd_engine.h: Needle 3 generation (transcription of needle-infer/src/v3_engine.rs).
 *
 * A model, its embedded tokenizer and (when exported) its confidence head. Generation is
 * V3Engine::generate_controlled: prompt assembly, one batched prefill, greedy (or seeded
 * SplitMix64 sampling) decode, the stop rules, the optional tool-call grammar engaged only
 * between the <tool_call> markers, and streamed text deltas that never split a character.
 */
#ifndef ND_ENGINE_H
#define ND_ENGINE_H

#include "nd_model.h"
#include "nd_tok.h"
#include "nd_grammar.h"

#define ND_DEFAULT_MAX_NEW_TOKENS 256

typedef enum {
  ND_STOP_EOS = 0,
  ND_STOP_IM_END,
  ND_STOP_MAX_TOKENS,
  ND_STOP_MAX_SEQ_LEN,
  ND_STOP_CANCELLED
} nd_stop;

const char *nd_stop_name(nd_stop s); /* "Eos", "ImEnd", ... as Rust's Debug */

typedef struct {
  size_t max_new_tokens;
  float temperature; /* <= 0: greedy */
  uint64_t seed;
  const char *system; /* NULL: no system turn */
  int constrain;
  nd_kv_precision kv;
} nd_gen_opts;

void nd_gen_opts_default(nd_gen_opts *o);

typedef struct {
  char *text; /* malloc'd, NUL-terminated */
  size_t text_len;
  uint32_t *tokens;
  size_t ntokens;
  nd_stop stop;
  size_t positions;
  int prompt_truncated;
  size_t prompt_tokens;
  double t_tokenize, t_prefill, t_decode; /* seconds, CLOCK_MONOTONIC */
} nd_result;

void nd_result_free(nd_result *r);

typedef struct {
  nd_cact cact;
  uint8_t *raw; /* owned container bytes */
  nd_model *model;
  nd_tok *tok;
  uint32_t eos_id, bos_id;
  int32_t im_end_id, tc_start_id, tc_end_id;
  /* The grammar's token byte table (constrained::byte_table), built on first constrained use. */
  nd_token_bytes *byte_table;
  size_t n_byte_table;
  float *mask; /* rows floats, with the table */
} nd_engine;

/* Load from container bytes (taken over: freed by nd_engine_free, also on failure). depth 0 is
 * the container's own depth. */
int nd_engine_from_bytes(uint8_t *raw, size_t len, size_t depth, nd_engine **out, nd_err *e);
int nd_engine_load(const char *path, size_t depth, nd_engine **out, nd_err *e);
void nd_engine_free(nd_engine *g);

/* The ids generation would prefill, BOS included. *ids malloc'd. */
int nd_engine_prompt_ids(const nd_engine *g, const char *query, const char *tools_json, const char *system,
                         uint32_t **ids, size_t *n);

/* on_token gets each decoded delta (concatenating them gives r->text); keep_going is asked
 * before each decode step. Either may be NULL. Returns ND_OK or ND_E_NOMEM / ND_E_ARG. */
typedef void (*nd_on_token)(void *user, uint32_t token, const char *delta, size_t len);
typedef int (*nd_keep_going)(void *user);
int nd_generate(nd_engine *g, const char *query, const char *tools_json, const nd_gen_opts *o,
                nd_on_token on_token, nd_keep_going keep_going, void *user, nd_result *r);

/* confidence_for: the head over prompt + completion, as a probability. Returns 1 and sets *p,
 * 0 when the container exports no confidence head, or a negative error. */
int nd_confidence_for(const nd_engine *g, const char *query, const char *tools_json, const char *completion,
                      float *p);

/* extract_tool_call / reasoning: the text between the markers (an unterminated marker runs to
 * the end), trimmed (Rust str::trim). Returns a pointer into `text` and its length via *len, or
 * NULL when the opening marker is absent. */
const char *nd_extract_between(const char *text, const char *open, const char *close, size_t *len);
const char *nd_trim(const char *s, size_t n, size_t *out_len);

#endif
