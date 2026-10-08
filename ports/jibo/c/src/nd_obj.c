/* nd_obj.c: see nd_obj.h. */
#include "nd_obj.h"

#include <stdlib.h>
#include <string.h>

void nd_obj_init(nd_obj *o) { memset(o, 0, sizeof *o); }

void nd_obj_free(nd_obj *o) {
  size_t i;
  for (i = 0; i < o->n; i++) free(o->kv[i].key), free(o->kv[i].val);
  free(o->kv);
  memset(o, 0, sizeof *o);
}

static char *dup_n(const char *s, size_t n) {
  char *d = malloc(n + 1);
  if (d) {
    memcpy(d, s, n);
    d[n] = 0;
  }
  return d;
}

static void put_owned(nd_obj *o, const char *key, char *val) {
  size_t i;
  char *k;
  if (o->err || !val) {
    free(val);
    o->err = 1;
    return;
  }
  for (i = 0; i < o->n; i++)
    if (!strcmp(o->kv[i].key, key)) {
      free(o->kv[i].val);
      o->kv[i].val = val;
      return;
    }
  if (o->n == o->cap) {
    size_t cap = o->cap ? o->cap * 2 : 16;
    nd_obj_kv *g = realloc(o->kv, cap * sizeof *g);
    if (!g) {
      free(val);
      o->err = 1;
      return;
    }
    o->kv = g;
    o->cap = cap;
  }
  if (!(k = dup_n(key, strlen(key)))) {
    free(val);
    o->err = 1;
    return;
  }
  o->kv[o->n].key = k;
  o->kv[o->n].val = val;
  o->n++;
}

void nd_obj_raw(nd_obj *o, const char *key, const char *json, size_t len) { put_owned(o, key, dup_n(json, len)); }

void nd_obj_remove(nd_obj *o, const char *key) {
  size_t i;
  for (i = 0; i < o->n; i++)
    if (!strcmp(o->kv[i].key, key)) {
      free(o->kv[i].key), free(o->kv[i].val);
      o->kv[i] = o->kv[--o->n];
      return;
    }
}

const char *nd_obj_get(const nd_obj *o, const char *key) {
  size_t i;
  for (i = 0; i < o->n; i++)
    if (!strcmp(o->kv[i].key, key)) return o->kv[i].val;
  return NULL;
}

static char *writer_take(nd_json_writer *w) {
  char *s = NULL;
  if (!w->err && w->data) s = dup_n(w->data, w->len);
  nd_json_writer_free(w);
  return s;
}

void nd_obj_str(nd_obj *o, const char *key, const char *s, size_t len) {
  nd_json_writer w;
  nd_json_writer_init(&w, 0);
  nd_json_writer_string(&w, s, len);
  put_owned(o, key, writer_take(&w));
}

void nd_obj_cstr(nd_obj *o, const char *key, const char *s) {
  if (s)
    nd_obj_str(o, key, s, strlen(s));
  else
    nd_obj_null(o, key);
}

void nd_obj_u64(nd_obj *o, const char *key, uint64_t v) {
  char b[ND_JSON_NUMBUF];
  size_t n = nd_json_write_u64(v, b);
  nd_obj_raw(o, key, b, n);
}

void nd_obj_i64(nd_obj *o, const char *key, int64_t v) {
  char b[ND_JSON_NUMBUF];
  size_t n = nd_json_write_i64(v, b);
  nd_obj_raw(o, key, b, n);
}

void nd_obj_f64(nd_obj *o, const char *key, double v) {
  char b[ND_JSON_NUMBUF];
  size_t n = nd_json_write_f64(v, b);
  nd_obj_raw(o, key, b, n);
}

void nd_obj_bool(nd_obj *o, const char *key, int v) { nd_obj_raw(o, key, v ? "true" : "false", v ? 4 : 5); }
void nd_obj_null(nd_obj *o, const char *key) { nd_obj_raw(o, key, "null", 4); }

void nd_obj_child(nd_obj *o, const char *key, nd_obj *child) {
  size_t n;
  put_owned(o, key, nd_obj_finish(child, &n));
}

void nd_obj_from_json(nd_obj *o, const nd_json_value *v) {
  size_t i;
  if (v->type != ND_JSON_OBJECT) {
    o->err = 1;
    return;
  }
  for (i = 0; i < v->u.obj.len; i++) {
    const nd_json_member *m = &v->u.obj.members[i];
    char *text = NULL, *key;
    size_t tl;
    if (nd_json_to_string(&m->value, ND_JSON_KEYS_SORTED, &text, &tl) != ND_OK) {
      o->err = 1;
      return;
    }
    /* Keys with NUL cannot be C strings; a serde map keeps them, the runner never makes them. */
    if (!(key = dup_n(m->key, m->key_len))) {
      free(text);
      o->err = 1;
      return;
    }
    put_owned(o, key, text);
    free(key);
  }
}

static int by_key(const void *a, const void *b) {
  return strcmp(((const nd_obj_kv *)a)->key, ((const nd_obj_kv *)b)->key);
}

char *nd_obj_finish(nd_obj *o, size_t *len) {
  nd_json_writer w;
  size_t i;
  char *s;
  if (o->err) {
    nd_obj_free(o);
    return NULL;
  }
  qsort(o->kv, o->n, sizeof *o->kv, by_key);
  nd_json_writer_init(&w, 0);
  nd_json_writer_raw(&w, "{", 1);
  for (i = 0; i < o->n; i++) {
    if (i) nd_json_writer_raw(&w, ",", 1);
    {
      nd_json_writer k;
      nd_json_writer_init(&k, 0);
      nd_json_writer_string(&k, o->kv[i].key, strlen(o->kv[i].key));
      if (k.err) w.err = k.err;
      else nd_json_writer_raw(&w, k.data, k.len);
      nd_json_writer_free(&k);
    }
    nd_json_writer_raw(&w, ":", 1);
    nd_json_writer_raw(&w, o->kv[i].val, strlen(o->kv[i].val));
  }
  nd_json_writer_raw(&w, "}", 1);
  nd_obj_free(o);
  if (w.err) {
    nd_json_writer_free(&w);
    return NULL;
  }
  *len = w.len;
  s = writer_take(&w);
  return s;
}
