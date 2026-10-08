#include "nd_common.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

void nd_seterr(nd_err *e, const char *fmt, ...) {
  va_list ap;
  if (!e) return;
  va_start(ap, fmt);
  vsnprintf(e->msg, sizeof e->msg, fmt, ap);
  va_end(ap);
}

void *nd_calloc(size_t n, size_t size) {
  size_t total;
  if (n == 0) n = 1;
  if (!nd_mul_ok(n, size, &total)) return NULL;
  return calloc(n, size);
}
