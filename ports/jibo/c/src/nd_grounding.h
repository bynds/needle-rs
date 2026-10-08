/* nd_grounding.h: is each argument of a call something the user actually said?
 * Transcription of ports/jibo/runner/src/grounding.rs, with Rust's Unicode behaviour
 * (str::to_lowercase incl. Final_Sigma, char::is_alphanumeric, char::is_whitespace, str::parse::<f64>)
 * from tables generated with rustc 1.87.0. Results are identical to the Rust.
 *
 * The service uses it as `classify` does: unless grounding is Off, read the query once
 * (nd_mentions_read) and, for each validated call, append nd_ungrounded(call, tool, m,
 * strict = (mode == Strict)) to one list; grounded = (list is empty). */
#ifndef ND_GROUNDING_H
#define ND_GROUNDING_H

#include <stddef.h>

#include "nd_catalog.h"
#include "nd_common.h"
#include "nd_validate.h"

/* What a query states, read once (grounding::Mentions). */
typedef struct {
  double *numbers;
  size_t n_numbers;
  double *durations_s;
  size_t n_durations;
  double *clock; /* (hour, minute) pairs in 24-hour time: clock[2k], clock[2k+1] */
  size_t n_clock;
  int zero_alias, max_alias, min_alias;
  char *lower; /* q.to_lowercase() (private in Rust) */
  size_t lower_len;
} nd_mentions;

/* Mentions::read. q must be valid UTF-8 (ND_E_ARG otherwise). On failure *m is left empty. */
int nd_mentions_read(const char *q, size_t len, nd_mentions *m);
void nd_mentions_free(nd_mentions *m);

/* A growable list of owned strings (Vec<String>). Zero-initialise before first use. */
typedef struct {
  char **items; /* NUL-terminated; lens[] holds the byte lengths */
  size_t *lens;
  size_t n, cap;
} nd_strlist;
void nd_strlist_free(nd_strlist *l);

/* grounding::ungrounded: APPENDS to *out the arguments of `call` the query does not state, as
 * "tool.param=value" (value as serde_json prints it), in argument order. ND_OK or ND_E_NOMEM. */
int nd_ungrounded(const nd_call *call, const nd_tool *tool, const nd_mentions *m, int strict,
                  nd_strlist *out);

/* Rust helpers, exposed for tests and the service. */
/* str::to_lowercase; *out malloc'd and NUL-terminated. s must be valid UTF-8. */
int nd_rust_to_lowercase(const char *s, size_t len, char **out, size_t *out_len);
/* str::parse::<f64>: 1 and *out on success, 0 on a parse error. */
int nd_rust_parse_f64(const char *s, size_t len, double *out);
/* char::is_alphanumeric. */
int nd_rust_is_alphanumeric(uint32_t c);

#endif
