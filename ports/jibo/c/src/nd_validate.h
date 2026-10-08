/* nd_validate.h: turning a completion into calls the application may consider, or an explicit
 * reason not to. Transcription of ports/jibo/runner/src/validate.rs; outcomes and their detail
 * strings are byte-identical to the Rust.
 *
 * A payload counts only if both markers closed it, it parses (serde_json rules) as a bounded JSON
 * array of {"name", "arguments"} objects, and every call satisfies the catalogue's schema. */
#ifndef ND_VALIDATE_H
#define ND_VALIDATE_H

#include <stddef.h>

#include "nd_catalog.h"
#include "nd_common.h"
#include "nd_json.h"

#define ND_MAX_PAYLOAD_BYTES 4096u
#define ND_MAX_CALLS 4u

typedef enum {
  ND_OUTCOME_CALLS = 0,          /* one or more calls, each valid against the catalogue */
  ND_OUTCOME_NO_CALL,            /* `[]`: the model decided no tool applies */
  ND_OUTCOME_NO_MARKER,          /* no <tool_call> marker at all */
  ND_OUTCOME_UNTERMINATED,       /* a <tool_call> that never closed */
  ND_OUTCOME_MALFORMED,          /* not a JSON array of call objects (or oversized); detail */
  ND_OUTCOME_UNSUPPORTED,        /* a tool outside the request's catalogue subset; detail */
  ND_OUTCOME_NEEDS_CLARIFICATION, /* a required argument is missing; detail */
  ND_OUTCOME_INVALID             /* wrong type, out of range, not in enum, undeclared; detail */
} nd_outcome_kind;

/* One argument of a call: a member of the payload's "arguments" object. */
typedef struct {
  const char *key; /* NUL-terminated; may also contain NUL */
  size_t key_len;
  const nd_json_value *value;
} nd_call_arg;

/* validate::Call. `name` is the catalogue's name for the tool (Call.name = tool.name). `args` is
 * the arguments Map as serde_json holds it: sorted by key bytes, the last duplicate kept. */
typedef struct {
  const nd_tool *tool;
  const char *name;
  size_t name_len;
  nd_call_arg *args;
  size_t n_args;
} nd_call;

typedef struct {
  nd_outcome_kind kind;
  char *detail; /* Malformed/Unsupported/NeedsClarification/Invalid: the Rust String; else NULL.
                 * NUL-terminated, but may also contain NUL (an argument key can): use detail_len */
  size_t detail_len;
  nd_call *calls; /* ND_OUTCOME_CALLS only */
  size_t n_calls;
  nd_json_doc *doc; /* owns the argument values */
} nd_outcome;

/* strict_payload: the text between the first <tool_call> and its </tool_call>, trimmed as
 * str::trim does. Returns ND_OUTCOME_CALLS (meaning Ok) with [*start, *start + *plen) set, or
 * ND_OUTCOME_NO_MARKER / ND_OUTCOME_UNTERMINATED. `text` should be valid UTF-8. */
nd_outcome_kind nd_strict_payload(const char *text, size_t len, size_t *start, size_t *plen);

/* validate(text, catalogue, allowed). allowed == NULL means None (any catalogue tool); otherwise
 * n_allowed names (Some, possibly empty). Fills *out (release with nd_outcome_free, also after a
 * failure). Returns ND_OK, ND_E_NOMEM, or ND_E_ARG (NULL arguments, or text not valid UTF-8). */
int nd_validate(const char *text, size_t len, const nd_catalogue *cat, const nd_strv *allowed,
                size_t n_allowed, nd_outcome *out);
void nd_outcome_free(nd_outcome *o);

/* Serialisation as the runner does it (serde_json, compact):
 *  nd_call_write_arguments: the arguments Map, `json!(c.arguments)` -> {"k":v,...}, sorted keys;
 *  nd_call_write: `json!({"name": c.name, "arguments": c.arguments})` -> {"arguments":{..},"name":".."}
 *    (a serde_json Map, so "arguments" sorts first);
 *  nd_calls_write: the `calls` / `rejected_calls` array of those. */
void nd_call_write_arguments(nd_json_writer *w, const nd_call *c);
void nd_call_write(nd_json_writer *w, const nd_call *c);
void nd_calls_write(nd_json_writer *w, const nd_call *calls, size_t n);

/* An nd_call over an arbitrary JSON object `args` (NULL for none), for callers that build calls
 * themselves (validate builds its own): args sorted/deduplicated as a serde_json Map. The call
 * borrows `args`' document; free c->args with nd_call_release. */
int nd_call_from_object(const nd_tool *tool, const nd_json_value *args, nd_call *c);
void nd_call_release(nd_call *c);

#endif
