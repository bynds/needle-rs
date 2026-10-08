/* nd_catalog.h: the authorized tool catalogue and the subset of JSON Schema it may use.
 * Transcription of ports/jibo/runner/src/catalog.rs; behaviour and every error message are
 * byte-identical to the Rust (`Result<_, String>` errors become malloc'd messages).
 *
 * Also home to the small pieces of Rust/serde_json behaviour the runner modules share:
 *  - nd_serde_parse: serde_json::from_str::<Value>, including serde_json's error text
 *    ("expected value at line 1 column 2", ...), which the Rust embeds in its own messages;
 *  - nd_rust_debug_str: `format!("{:?}", s)` for a &str (Rust 1.87 escaping and Unicode tables);
 *  - nd_rust_is_whitespace / nd_rust_trim: char::is_whitespace and str::trim / trim_end.
 *
 * Strings are UTF-8 with an explicit length (Rust strings may contain NUL). The text passed to
 * nd_catalogue_parse must be valid UTF-8 (Rust's `&str`); invalid input is refused with ND_E_ARG.
 */
#ifndef ND_CATALOG_H
#define ND_CATALOG_H

#include <stddef.h>
#include <stdint.h>

#include "nd_common.h"
#include "nd_json.h"

#define ND_MAX_CATALOGUE_BYTES (16u * 1024u)
#define ND_MAX_TOOLS 32u
#define ND_MAX_PARAMS 16u

/* A borrowed string slice (pointer + length; need not be NUL-terminated). */
typedef struct {
  const char *ptr;
  size_t len;
} nd_strv;

typedef enum {
  ND_PARAM_STRING = 0,
  ND_PARAM_INTEGER,
  ND_PARAM_NUMBER,
  ND_PARAM_BOOLEAN
} nd_param_kind;

/* ParamType + Param. Owned, NUL-terminated copies (lengths kept: names may contain NUL). */
typedef struct {
  char *name;
  size_t name_len;
  nd_param_kind kind;
  /* ND_PARAM_STRING */
  int has_enum;
  char **enum_values;
  size_t *enum_lens;
  size_t n_enum;
  int has_max_length;
  size_t max_length; /* usize: on a 32-bit target a maxLength above SIZE_MAX is refused, as Rust */
  /* ND_PARAM_INTEGER */
  int has_imin, has_imax;
  int64_t imin, imax;
  /* ND_PARAM_NUMBER */
  int has_fmin, has_fmax;
  double fmin, fmax;
} nd_param;

typedef struct {
  char *name; /* [A-Za-z0-9_-]{1,64} */
  size_t name_len;
  char *snake_name; /* nd_to_snake_case(name) */
  size_t snake_len;
  nd_param *params; /* in serde_json Map order: keys sorted by bytes, last duplicate kept */
  size_t n_params;
  char **required; /* in file order, deduplicated */
  size_t *required_lens;
  size_t n_required;
  char *json; /* the tool's definition exactly as the file spells it */
  size_t json_len;
} nd_tool;

typedef struct {
  nd_tool *tools;
  size_t n_tools;
} nd_catalogue;

/* Catalogue::parse. ND_OK and *out (free with nd_catalogue_free); ND_E_FORMAT with *errmsg set
 * to the Rust error text (malloc'd, NUL-terminated; free it); ND_E_ARG for invalid UTF-8 or bad
 * arguments (*errmsg set when possible); ND_E_NOMEM (*errmsg NULL). errmsg may be NULL. */
int nd_catalogue_parse(const char *text, size_t len, nd_catalogue **out, char **errmsg);
void nd_catalogue_free(nd_catalogue *c);

/* Catalogue::get (exact name) and Catalogue::resolve (exact, then snake_case spelling). */
const nd_tool *nd_catalogue_get(const nd_catalogue *c, const char *name, size_t len);
const nd_tool *nd_catalogue_resolve(const nd_catalogue *c, const char *emitted, size_t len);

/* Catalogue::tools_json. names == NULL means None (every tool); otherwise n_names names (Some,
 * possibly empty). ND_OK with *out malloc'd (NUL-terminated, *out_len bytes); ND_E_FORMAT with
 * *errmsg = "tool \"x\" is not in the catalogue"; ND_E_NOMEM. */
int nd_catalogue_tools_json(const nd_catalogue *c, const nd_strv *names, size_t n_names,
                            char **out, size_t *out_len, char **errmsg);

/* ─── Shared Rust/serde_json behaviour ─────────────────────────────────────────────────────── */

/* serde_json::from_str::<Value>(text) on valid UTF-8. ND_OK and *doc; ND_E_FORMAT with *errmsg
 * = serde_json's Display of the error; ND_E_NOMEM. Nesting is serde_json's (127 levels). */
int nd_serde_parse(const char *text, size_t len, nd_json_doc **doc, char **errmsg);

/* Append `format!("{:?}", s)` (quotes included) for valid UTF-8 s to w (via nd_json_writer_raw). */
void nd_rust_debug_str(nd_json_writer *w, const char *s, size_t len);

/* char::is_whitespace. */
int nd_rust_is_whitespace(uint32_t c);
/* str::trim / str::trim_end on valid UTF-8: the kept range is [*start, *end). */
void nd_rust_trim(const char *s, size_t len, size_t *start, size_t *end);
void nd_rust_trim_end(const char *s, size_t len, size_t *end);

/* The members of a JSON object as a serde_json Map (BTreeMap<String, Value>) holds them: sorted
 * by key bytes, the last of duplicate keys kept. *out is a malloc'd array of *n pointers into
 * obj (free it); for an empty object *out may be NULL. ND_OK, ND_E_NOMEM or ND_E_ARG. */
int nd_sorted_members(const nd_json_value *obj, const nd_json_member ***out, size_t *n);

#endif
