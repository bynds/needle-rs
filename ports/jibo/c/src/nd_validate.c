/* nd_validate.c: transcription of ports/jibo/runner/src/validate.rs (see nd_validate.h). */
#include "nd_validate.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nd_prompt.h"

static const char *find_bytes(const char *h, size_t hl, const char *n, size_t nl) {
  size_t k;
  if (nl == 0) return h;
  if (hl < nl) return NULL;
  for (k = 0; k + nl <= hl; k++)
    if (h[k] == n[0] && memcmp(h + k, n, nl) == 0) return h + k;
  return NULL;
}

nd_outcome_kind nd_strict_payload(const char *text, size_t len, size_t *start, size_t *plen) {
  const size_t sl = sizeof ND_TOOL_CALL_START - 1, el = sizeof ND_TOOL_CALL_END - 1;
  const char *open, *rest, *end;
  size_t rl, a, b;
  if (!text) return ND_OUTCOME_NO_MARKER;
  open = find_bytes(text, len, ND_TOOL_CALL_START, sl);
  if (!open) return ND_OUTCOME_NO_MARKER;
  rest = open + sl;
  rl = len - (size_t)(rest - text);
  end = find_bytes(rest, rl, ND_TOOL_CALL_END, el);
  if (!end) return ND_OUTCOME_UNTERMINATED;
  nd_rust_trim(rest, (size_t)(end - rest), &a, &b);
  if (start) *start = (size_t)(rest - text) + a;
  if (plen) *plen = b - a;
  return ND_OUTCOME_CALLS;
}

static int bytes_eq(const char *a, size_t al, const char *b, size_t bl) {
  return al == bl && (al == 0 || memcmp(a, b, al) == 0);
}

/* Message assembly into a writer (raw bytes); sticky errors become ND_E_NOMEM. */
static void w_fmt(nd_json_writer *w, const char *fmt, ...) {
  char buf[96];
  va_list ap;
  int n;
  va_start(ap, fmt);
  n = vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  if (n < 0 || (size_t)n >= sizeof buf) {
    if (!w->err) w->err = ND_E_ARG;
    return;
  }
  nd_json_writer_raw(w, buf, (size_t)n);
}
static void w_str(nd_json_writer *w, const char *s) { nd_json_writer_raw(w, s, strlen(s)); }

#define ZU(x) ((unsigned long long)(x))

/* Set o->kind and o->detail from the writer. */
static int set_detail(nd_outcome *o, nd_outcome_kind kind, nd_json_writer *w) {
  if (w->err || !w->data) {
    nd_json_writer_free(w);
    return ND_E_NOMEM;
  }
  o->kind = kind;
  o->detail = w->data;
  o->detail_len = w->len;
  return ND_OK;
}

int nd_call_from_object(const nd_tool *tool, const nd_json_value *args, nd_call *c) {
  const nd_json_member **mv = NULL;
  size_t n = 0, k;
  if (!c) return ND_E_ARG;
  memset(c, 0, sizeof *c);
  c->tool = tool;
  if (tool) {
    c->name = tool->name;
    c->name_len = tool->name_len;
  }
  if (!args) return ND_OK;
  if (args->type != ND_JSON_OBJECT) return ND_E_ARG;
  if (nd_sorted_members(args, &mv, &n)) return ND_E_NOMEM;
  if (n) {
    c->args = (nd_call_arg *)nd_calloc(n, sizeof *c->args);
    if (!c->args) {
      free(mv);
      return ND_E_NOMEM;
    }
    for (k = 0; k < n; k++) {
      c->args[k].key = mv[k]->key;
      c->args[k].key_len = mv[k]->key_len;
      c->args[k].value = &mv[k]->value;
    }
  }
  c->n_args = n;
  free(mv);
  return ND_OK;
}

void nd_call_release(nd_call *c) {
  if (!c) return;
  free(c->args);
  c->args = NULL;
  c->n_args = 0;
}

void nd_outcome_free(nd_outcome *o) {
  size_t k;
  if (!o) return;
  for (k = 0; k < o->n_calls; k++) nd_call_release(&o->calls[k]);
  free(o->calls);
  free(o->detail);
  nd_json_doc_free(o->doc);
  memset(o, 0, sizeof *o);
}

static size_t utf8_chars(const char *s, size_t n) {
  size_t k, c = 0;
  for (k = 0; k < n; k++)
    if (((unsigned char)s[k] & 0xC0u) != 0x80u) c++;
  return c;
}

/* check_arguments: ND_OK with o untouched when every argument passes; otherwise o->kind/detail. */
static int check_arguments(size_t i, const nd_tool *tool, const nd_call *c, nd_outcome *o,
                           int *failed) {
  size_t k, j;
  nd_json_writer w;
  *failed = 1;
  for (k = 0; k < c->n_args; k++) {
    const nd_call_arg *a = &c->args[k];
    const nd_param *p = NULL;
    const char *why = NULL;
    for (j = 0; j < tool->n_params; j++)
      if (bytes_eq(tool->params[j].name, tool->params[j].name_len, a->key, a->key_len)) {
        p = &tool->params[j];
        break;
      }
    if (!p) {
      nd_json_writer_init(&w, 0);
      w_fmt(&w, "call %llu: ", ZU(i));
      nd_json_writer_raw(&w, tool->name, tool->name_len);
      nd_json_writer_raw(&w, ".", 1);
      nd_json_writer_raw(&w, a->key, a->key_len);
      w_str(&w, " is not a declared argument");
      return set_detail(o, ND_OUTCOME_INVALID, &w);
    }
    switch (p->kind) {
      case ND_PARAM_STRING: {
        const char *s;
        size_t sl;
        if (!nd_json_as_str(a->value, &s, &sl)) {
          why = "must be a string";
          break;
        }
        if (p->has_enum) {
          int found = 0;
          for (j = 0; j < p->n_enum && !found; j++)
            found = bytes_eq(p->enum_values[j], p->enum_lens[j], s, sl);
          if (!found) {
            why = "is not one of the allowed values";
            break;
          }
        }
        if (p->has_max_length && utf8_chars(s, sl) > p->max_length) why = "is too long";
        break;
      }
      case ND_PARAM_INTEGER: {
        int64_t n;
        if (!nd_json_as_i64(a->value, &n)) {
          why = "must be an integer";
          break;
        }
        if ((p->has_imin && n < p->imin) || (p->has_imax && n > p->imax)) why = "is out of range";
        break;
      }
      case ND_PARAM_NUMBER: {
        double n;
        if (!nd_json_as_f64(a->value, &n)) {
          why = "must be a number";
          break;
        }
        if (!isfinite(n) || (p->has_fmin && n < p->fmin) || (p->has_fmax && n > p->fmax))
          why = "is out of range";
        break;
      }
      case ND_PARAM_BOOLEAN: {
        int b;
        if (!nd_json_as_bool(a->value, &b)) why = "must be a boolean";
        break;
      }
      default:
        why = "must be a boolean";
        break;
    }
    if (why) {
      nd_json_writer_init(&w, 0);
      w_fmt(&w, "call %llu: ", ZU(i));
      nd_json_writer_raw(&w, tool->name, tool->name_len);
      nd_json_writer_raw(&w, ".", 1);
      nd_json_writer_raw(&w, a->key, a->key_len);
      nd_json_writer_raw(&w, " ", 1);
      w_str(&w, why);
      return set_detail(o, ND_OUTCOME_INVALID, &w);
    }
  }
  for (k = 0; k < tool->n_required; k++) {
    int has = 0;
    for (j = 0; j < c->n_args && !has; j++)
      has = bytes_eq(c->args[j].key, c->args[j].key_len, tool->required[k], tool->required_lens[k]);
    if (!has) {
      nd_json_writer_init(&w, 0);
      w_fmt(&w, "call %llu: ", ZU(i));
      nd_json_writer_raw(&w, tool->name, tool->name_len);
      nd_json_writer_raw(&w, ".", 1);
      nd_json_writer_raw(&w, tool->required[k], tool->required_lens[k]);
      w_str(&w, " is required and was not given");
      return set_detail(o, ND_OUTCOME_NEEDS_CLARIFICATION, &w);
    }
  }
  *failed = 0;
  return ND_OK;
}

static int malformed(nd_outcome *o, const char *fmt, unsigned long long a) {
  nd_json_writer w;
  nd_json_writer_init(&w, 0);
  w_fmt(&w, fmt, a);
  return set_detail(o, ND_OUTCOME_MALFORMED, &w);
}

static int validate_inner(const char *text, size_t len, const nd_catalogue *cat,
                          const nd_strv *allowed, size_t n_allowed, nd_outcome *out) {
  size_t ps = 0, pl = 0, n, i;
  nd_outcome_kind k;
  const nd_json_value *root;
  char *serr = NULL;
  int rc;
  if (!out) return ND_E_ARG;
  memset(out, 0, sizeof *out);
  if (!cat || (!text && len) || (!allowed && n_allowed)) return ND_E_ARG;
  if (!text) text = "";
  if (!nd_json_utf8_valid((const unsigned char *)text, len)) return ND_E_ARG;
  k = nd_strict_payload(text, len, &ps, &pl);
  if (k != ND_OUTCOME_CALLS) {
    out->kind = k;
    return ND_OK;
  }
  if (pl > ND_MAX_PAYLOAD_BYTES) return malformed(out, "payload is %llu bytes", ZU(pl));
  rc = nd_serde_parse(text + ps, pl, &out->doc, &serr);
  if (rc == ND_E_FORMAT) {
    nd_json_writer w;
    nd_json_writer_init(&w, 0);
    w_str(&w, "payload is not JSON: ");
    w_str(&w, serr ? serr : "");
    free(serr);
    return set_detail(out, ND_OUTCOME_MALFORMED, &w);
  }
  if (rc != ND_OK) return rc;
  root = nd_json_root(out->doc);
  if (root->type != ND_JSON_ARRAY) {
    nd_json_writer w;
    nd_json_writer_init(&w, 0);
    w_str(&w, "payload is not an array");
    return set_detail(out, ND_OUTCOME_MALFORMED, &w);
  }
  n = root->u.arr.len;
  if (n == 0) {
    out->kind = ND_OUTCOME_NO_CALL;
    return ND_OK;
  }
  if (n > ND_MAX_CALLS) {
    nd_json_writer w;
    nd_json_writer_init(&w, 0);
    w_fmt(&w, "%llu calls (limit %u)", ZU(n), ND_MAX_CALLS);
    return set_detail(out, ND_OUTCOME_MALFORMED, &w);
  }
  out->calls = (nd_call *)nd_calloc(n, sizeof *out->calls);
  if (!out->calls) return ND_E_NOMEM;
  for (i = 0; i < n; i++) {
    const nd_json_value *item = &root->u.arr.items[i], *nv, *av;
    const nd_tool *tool;
    const char *name;
    size_t name_len, m;
    int failed;
    nd_call *c = &out->calls[i];
    if (item->type != ND_JSON_OBJECT) return malformed(out, "call %llu is not an object", ZU(i));
    for (m = 0; m < item->u.obj.len; m++) {
      const nd_json_member *mm = &item->u.obj.members[m];
      if (!bytes_eq(mm->key, mm->key_len, "name", 4) &&
          !bytes_eq(mm->key, mm->key_len, "arguments", 9))
        return malformed(out, "call %llu has keys other than name/arguments", ZU(i));
    }
    nv = nd_json_get_cstr(item, "name");
    if (!nv || !nd_json_as_str(nv, &name, &name_len))
      return malformed(out, "call %llu has no string name", ZU(i));
    tool = nd_catalogue_resolve(cat, name, name_len);
    if (tool && allowed) {
      size_t j;
      int ok = 0;
      for (j = 0; j < n_allowed && !ok; j++)
        ok = bytes_eq(allowed[j].ptr, allowed[j].len, tool->name, tool->name_len);
      if (!ok) tool = NULL;
    }
    if (!tool) {
      nd_json_writer w;
      nd_json_writer_init(&w, 0);
      w_fmt(&w, "call %llu: tool ", ZU(i));
      nd_rust_debug_str(&w, name, name_len);
      w_str(&w, " was not offered");
      return set_detail(out, ND_OUTCOME_UNSUPPORTED, &w);
    }
    av = nd_json_get_cstr(item, "arguments");
    if (av && av->type != ND_JSON_OBJECT)
      return malformed(out, "call %llu: arguments is not an object", ZU(i));
    rc = nd_call_from_object(tool, av, c);
    out->n_calls = i + 1;
    if (rc) return rc;
    rc = check_arguments(i, tool, c, out, &failed);
    if (rc || failed) return rc;
  }
  out->kind = ND_OUTCOME_CALLS;
  return ND_OK;
}

int nd_validate(const char *text, size_t len, const nd_catalogue *cat, const nd_strv *allowed,
                size_t n_allowed, nd_outcome *out) {
  int rc = validate_inner(text, len, cat, allowed, n_allowed, out);
  if (out && (rc != ND_OK || out->kind != ND_OUTCOME_CALLS)) {
    /* Only Calls carries calls (and needs the document they point into). */
    size_t k;
    for (k = 0; k < out->n_calls; k++) nd_call_release(&out->calls[k]);
    free(out->calls);
    out->calls = NULL;
    out->n_calls = 0;
    nd_json_doc_free(out->doc);
    out->doc = NULL;
  }
  return rc;
}

void nd_call_write_arguments(nd_json_writer *w, const nd_call *c) {
  size_t k;
  nd_json_writer_begin_object(w);
  for (k = 0; c && k < c->n_args; k++) {
    nd_json_writer_key(w, c->args[k].key, c->args[k].key_len);
    nd_json_writer_value(w, c->args[k].value, ND_JSON_KEYS_SORTED);
  }
  nd_json_writer_end_object(w);
}

void nd_call_write(nd_json_writer *w, const nd_call *c) {
  nd_json_writer_begin_object(w);
  nd_json_writer_key_cstr(w, "arguments");
  nd_call_write_arguments(w, c);
  nd_json_writer_key_cstr(w, "name");
  nd_json_writer_string(w, c->name, c->name_len);
  nd_json_writer_end_object(w);
}

void nd_calls_write(nd_json_writer *w, const nd_call *calls, size_t n) {
  size_t k;
  nd_json_writer_begin_array(w);
  for (k = 0; k < n; k++) nd_call_write(w, &calls[k]);
  nd_json_writer_end_array(w);
}
