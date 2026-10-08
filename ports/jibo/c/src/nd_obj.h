/* nd_obj.h: building JSON objects the way a serde_json::Value prints them.
 *
 * The runner assembles its responses as serde_json::Value objects, whose map is a BTreeMap (no
 * `preserve_order`): keys print in byte order, and inserting an existing key replaces its value.
 * nd_obj keeps (key, serialised value) pairs and sorts them when it is finished, so the C runner
 * prints the same bytes. Errors (allocation) are sticky; nd_obj_finish then returns NULL.
 */
#ifndef ND_OBJ_H
#define ND_OBJ_H

#include "nd_json.h"

typedef struct {
  char *key;
  char *val; /* serialised JSON */
} nd_obj_kv;

typedef struct {
  nd_obj_kv *kv;
  size_t n, cap;
  int err;
} nd_obj;

void nd_obj_init(nd_obj *o);
void nd_obj_free(nd_obj *o);
/* Insert or replace `key` with already-serialised JSON (copied). */
void nd_obj_raw(nd_obj *o, const char *key, const char *json, size_t len);
/* Insert `key` with a finished child object (consumed: o takes its text, child is freed). */
void nd_obj_child(nd_obj *o, const char *key, nd_obj *child);
void nd_obj_remove(nd_obj *o, const char *key);
const char *nd_obj_get(const nd_obj *o, const char *key); /* serialised value or NULL */
void nd_obj_str(nd_obj *o, const char *key, const char *s, size_t len);
void nd_obj_cstr(nd_obj *o, const char *key, const char *s); /* NULL s: null */
void nd_obj_u64(nd_obj *o, const char *key, uint64_t v);
void nd_obj_i64(nd_obj *o, const char *key, int64_t v);
void nd_obj_f64(nd_obj *o, const char *key, double v); /* serde: non-finite -> null */
void nd_obj_bool(nd_obj *o, const char *key, int b);
void nd_obj_null(nd_obj *o, const char *key);
/* Every member of a parsed object, each value serialised as a serde_json::Value would. */
void nd_obj_from_json(nd_obj *o, const nd_json_value *v);
/* Sorted, compact text (malloc'd, NUL-terminated), or NULL on error. o is reset. */
char *nd_obj_finish(nd_obj *o, size_t *len);

#endif
