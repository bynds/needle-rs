/* glibc 2.38's C23 number parsers, referenced by a libc++ built against newer headers. glibc 2.21
   has the C99 versions, which differ only in not accepting "0b" binary prefixes. */
#define _GNU_SOURCE
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>
#include <locale.h>
long __isoc23_strtol(const char *s, char **e, int b) { return strtol(s, e, b); }
unsigned long __isoc23_strtoul(const char *s, char **e, int b) { return strtoul(s, e, b); }
long long __isoc23_strtoll(const char *s, char **e, int b) { return strtoll(s, e, b); }
unsigned long long __isoc23_strtoull(const char *s, char **e, int b) { return strtoull(s, e, b); }
long long __isoc23_strtoll_l(const char *s, char **e, int b, locale_t l) { return strtoll_l(s, e, b, l); }
unsigned long long __isoc23_strtoull_l(const char *s, char **e, int b, locale_t l) { return strtoull_l(s, e, b, l); }
long __isoc23_wcstol(const wchar_t *s, wchar_t **e, int b) { return wcstol(s, e, b); }
unsigned long __isoc23_wcstoul(const wchar_t *s, wchar_t **e, int b) { return wcstoul(s, e, b); }
long long __isoc23_wcstoll(const wchar_t *s, wchar_t **e, int b) { return wcstoll(s, e, b); }
unsigned long long __isoc23_wcstoull(const wchar_t *s, wchar_t **e, int b) { return wcstoull(s, e, b); }
int __isoc23_vsscanf(const char *s, const char *f, va_list a) { return vsscanf(s, f, a); }
int __isoc23_sscanf(const char *s, const char *f, ...) { va_list a; int r; va_start(a, f); r = vsscanf(s, f, a); va_end(a); return r; }
