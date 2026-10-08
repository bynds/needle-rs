/* nd_service.h: one loaded model answering bounded requests (runner/src/service.rs).
 *
 * Admission happens before compute: the request names tools from the authorised catalogue, its
 * prompt is tokenized and measured, and it is refused (`truncated`) rather than cut if prompt and
 * budget do not fit. Every response carries the model hash, depth, cache precision, grammar
 * setting, stop reason, token counts and phase timings. Responses are byte-identical to the Rust
 * runner's apart from measured times and resource readings.
 */
#ifndef ND_SERVICE_H
#define ND_SERVICE_H

#include "nd_catalog.h"
#include "nd_engine.h"
#include "nd_grounding.h"
#include "nd_obj.h"
#include "nd_validate.h"

typedef struct {
  size_t max_total_tokens, max_new_tokens, min_new_tokens, max_query_bytes;
  int has_deadline;
  uint64_t deadline_ms;
} nd_limits;

void nd_limits_default(nd_limits *l);

typedef enum { ND_GROUND_OFF = 0, ND_GROUND_REPORT, ND_GROUND_ENFORCE, ND_GROUND_STRICT } nd_grounding_mode;

int nd_grounding_parse(const char *s, nd_grounding_mode *out);
const char *nd_grounding_name(nd_grounding_mode g);

typedef struct {
  int constrain, kv_int8;
  const char *system;
  int confidence, debug_text;
  nd_grounding_mode grounding;
  int verify_model;
  const char *expect_sha256;
  int has_min_confidence;
  float min_confidence;
} nd_options;

void nd_options_default(nd_options *o);

typedef struct {
  char *id;
  char *query;
  size_t query_len; /* a JSON string may carry NUL; handle() refuses it */
  int has_tools;
  nd_strv *tools; /* each ptr malloc'd */
  size_t n_tools;
  int has_max_new;
  size_t max_new;
} nd_request;

void nd_request_free(nd_request *r);

typedef enum { ND_LINE_REQUEST = 0, ND_LINE_HEALTH = 1, ND_LINE_ERROR = 2 } nd_line_kind;

/* parse_line: ND_LINE_REQUEST fills *req; ND_LINE_ERROR sets *refusal (malloc'd JSON). */
nd_line_kind nd_parse_line(const char *line, size_t len, size_t max_line, nd_request *req, char **refusal);

/* {"detail","request_id","status"} as malloc'd JSON text, or into an object. */
char *nd_refusal(const char *id, const char *status, const char *detail);
void nd_refusal_obj(nd_obj *o, const char *id, const char *status, const char *detail, size_t detail_len);

typedef struct {
  const char *status;
  char *detail;
  size_t detail_len;
  char *calls, *rejected; /* serialised JSON arrays, or NULL */
  int grounded;           /* -1: not checked */
  nd_strlist ungrounded;
} nd_verdict;

void nd_verdict_free(nd_verdict *v);

/* classify: a pure function of its inputs, shared with `regrade`. */
int nd_classify(const nd_catalogue *cat, const nd_options *opts, const char *query, size_t query_len,
                const nd_strv *allowed, size_t n_allowed, int has_allowed, const char *text, size_t text_len,
                nd_stop stop, int prompt_truncated, size_t budget, nd_verdict *v);
/* gate_confidence: has_p/p is the confidence-head score. */
int nd_gate_confidence(nd_verdict *v, int has_min, float min, int has_p, float p);

const char *nd_stop_wire_name(nd_stop s); /* "eos", "im_end", ... */
int nd_stop_from_wire(const char *s, size_t len, nd_stop *out);

typedef struct {
  nd_engine *engine;
  nd_catalogue *cat;
  nd_limits limits;
  nd_options opts;
  char sha256[65];
  const char *sha256_source; /* "computed" or "cache" */
  size_t model_bytes, depth;
  double load_ms;
} nd_service;

/* Load the model at path (the catalogue is taken over). has_depth: a ladder rung. On failure
 * returns NULL and *err holds the message (malloc'd). */
nd_service *nd_service_load(const char *path, nd_catalogue *cat, int has_depth, size_t depth, const nd_limits *l,
                            const nd_options *o, char **err);
void nd_service_free(nd_service *s);

const char *nd_service_kv_name(const nd_service *s);
void nd_service_health(const nd_service *s, nd_obj *out);

/* Answer one request; `received` is a nd_now() reading. The response object is filled. */
void nd_service_handle(nd_service *s, const nd_request *r, double received, nd_obj *out);

/* CLOCK_MONOTONIC seconds, and Rust's ms(): (secs * 1e6).round() / 1e3. */
double nd_now(void);
double nd_ms(double seconds);

#endif
