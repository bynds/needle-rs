/* nd_sysinfo.c: see nd_sysinfo.h. */
#include "nd_sysinfo.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Read a small /proc or /sys file whole (they report a size of 0, so read until EOF). */
static char *read_small(const char *path) {
  FILE *f = fopen(path, "rb");
  char *b;
  size_t n = 0, cap = 4096;
  if (!f) return NULL;
  if (!(b = malloc(cap + 1))) {
    fclose(f);
    return NULL;
  }
  for (;;) {
    size_t r = fread(b + n, 1, cap - n, f);
    n += r;
    if (n < cap) break;
    if (cap >= (1u << 20)) break;
    {
      char *g = realloc(b, cap * 2 + 1);
      if (!g) break;
      b = g;
      cap *= 2;
    }
  }
  fclose(f);
  b[n] = 0;
  return b;
}

/* Rust's u64::from_str: optional '+', then one or more ASCII digits, nothing else. */
static int parse_u64(const char *s, size_t n, uint64_t *out) {
  uint64_t v = 0;
  size_t i = 0;
  if (n && s[0] == '+') i = 1;
  if (i == n) return 0;
  for (; i < n; i++) {
    unsigned d;
    if (s[i] < '0' || s[i] > '9') return 0;
    d = (unsigned)(s[i] - '0');
    if (v > (UINT64_MAX - d) / 10) return 0;
    v = v * 10 + d;
  }
  *out = v;
  return 1;
}

/* The second whitespace-separated field of the first line starting with `field`. */
static int field_u64(const char *path, const char *field, uint64_t *out) {
  char *s = read_small(path), *line, *p, *q;
  int ok = 0;
  if (!s) return 0;
  for (line = s; line && *line;) {
    char *nl = strchr(line, '\n');
    if (!strncmp(line, field, strlen(field))) {
      if (nl) *nl = 0;
      p = line;
      while (*p && !isspace((unsigned char)*p)) p++; /* field 1 */
      while (*p && isspace((unsigned char)*p)) p++;
      q = p;
      while (*q && !isspace((unsigned char)*q)) q++;
      ok = q > p && parse_u64(p, (size_t)(q - p), out);
      break;
    }
    line = nl ? nl + 1 : NULL;
  }
  free(s);
  return ok;
}

int nd_mem_available_kb(uint64_t *kb) { return field_u64("/proc/meminfo", "MemAvailable:", kb); }

int nd_cpu_seconds(double *out) {
  char *s = read_small("/proc/self/stat"), *p, *tok, *save = NULL;
  int i = 0, ok = 0;
  uint64_t ut = 0, st = 0;
  if (!s) return 0;
  p = strrchr(s, ')');
  if (p && p[1] && p[2]) {
    for (tok = strtok_r(p + 2, " \t\n", &save); tok; tok = strtok_r(NULL, " \t\n", &save), i++) {
      if (i == 11 && !parse_u64(tok, strlen(tok), &ut)) break;
      if (i == 12) {
        ok = parse_u64(tok, strlen(tok), &st);
        break;
      }
    }
  }
  free(s);
  /* Rust parses these as f64; a utime is an integer, so this is the same value. */
  if (ok) *out = ((double)ut + (double)st) / 100.0;
  return ok;
}

/* Rust's str::trim then f64::from_str, for the plain decimal text a thermal zone holds. */
static int read_millideg(const char *path, double *c) {
  char *s = read_small(path), *a, *end;
  size_t n;
  int ok = 0;
  if (!s) return 0;
  a = s;
  while (*a && isspace((unsigned char)*a)) a++;
  n = strlen(a);
  while (n && isspace((unsigned char)a[n - 1])) n--;
  a[n] = 0;
  /* strtod also takes hex, "inf" spellings and leading space; Rust's parser takes no hex. */
  if (n && !strpbrk(a, "xX \t")) {
    double v;
    errno = 0;
    v = strtod(a, &end);
    if (end == a + n) {
      *c = v / 1000.0;
      ok = 1;
    }
  }
  free(s);
  return ok;
}

/* Expand serde/zmij shortest text ("1e-7", "7e+20", "70.0") into Rust Display form. */
static void display_from_shortest(const char *t, char *buf, size_t cap) {
  char digits[64], out[400];
  const char *e = strchr(t, 'e');
  size_t nd = 0, o = 0, i;
  int neg = t[0] == '-', point = -1, exp10 = 0;
  const char *p = t + neg;
  if (!strcmp(t, "null")) {
    snprintf(buf, cap, "%s", t);
    return;
  }
  for (; *p && p != e; p++) {
    if (*p == '.')
      point = (int)nd;
    else if (nd < sizeof digits)
      digits[nd++] = *p;
  }
  if (point < 0) point = (int)nd;
  if (e) exp10 = atoi(e + 1);
  point += exp10;
  /* Drop trailing zeros after the point, and leading zeros. */
  while (nd > 0 && (int)nd > point && digits[nd - 1] == '0') nd--;
  if (neg) out[o++] = '-';
  if (point <= 0) {
    out[o++] = '0';
    if (nd) {
      out[o++] = '.';
      for (i = 0; i < (size_t)(-point) && o < sizeof out - 2; i++) out[o++] = '0';
      for (i = 0; i < nd && o < sizeof out - 1; i++) out[o++] = digits[i];
    }
  } else {
    for (i = 0; i < (size_t)point && o < sizeof out - 1; i++) out[o++] = i < nd ? digits[i] : '0';
    if (nd > (size_t)point) {
      out[o++] = '.';
      for (i = (size_t)point; i < nd && o < sizeof out - 1; i++) out[o++] = digits[i];
    }
  }
  out[o] = 0;
  /* Strip leading zeros of the integer part ("0070" cannot arise from zmij, kept for safety). */
  snprintf(buf, cap, "%s", out);
}

void nd_fmt_f64_display(double v, char *buf, size_t cap) {
  char t[ND_JSON_NUMBUF];
  if (v != v) {
    snprintf(buf, cap, "NaN");
    return;
  }
  if (v == 1.0 / 0.0 || v == -1.0 / 0.0) {
    snprintf(buf, cap, v > 0 ? "inf" : "-inf");
    return;
  }
  nd_json_write_f64(v, t);
  display_from_shortest(t, buf, cap);
}

void nd_fmt_f32_display(float v, char *buf, size_t cap) {
  char t[ND_JSON_NUMBUF];
  if (v != v) {
    snprintf(buf, cap, "NaN");
    return;
  }
  if (v == 1.0f / 0.0f || v == -1.0f / 0.0f) {
    snprintf(buf, cap, v > 0 ? "inf" : "-inf");
    return;
  }
  nd_json_write_f32(v, t);
  display_from_shortest(t, buf, cap);
}

int nd_gate_busy(const nd_gate *g, char *buf, size_t cap) {
  if (g->has_min_avail) {
    uint64_t kb;
    if (!nd_mem_available_kb(&kb)) {
      snprintf(buf, cap, "MemAvailable unreadable");
      return 1;
    }
    if (kb / 1024 < g->min_avail_mb) {
      snprintf(buf, cap, "MemAvailable %llu MB below %llu MB", (unsigned long long)(kb / 1024),
               (unsigned long long)g->min_avail_mb);
      return 1;
    }
  }
  if (g->thermal && g->has_max_temp) {
    double c;
    char mx[64];
    if (!read_millideg(g->thermal, &c)) {
      snprintf(buf, cap, "%s unreadable", g->thermal);
      return 1;
    }
    if (c > g->max_temp_c) {
      nd_fmt_f64_display(g->max_temp_c, mx, sizeof mx);
      snprintf(buf, cap, "%s at %.1f C above %s C", g->thermal, c, mx);
      return 1;
    }
  }
  return 0;
}

void nd_sysinfo_snapshot(const nd_gate *g, nd_obj *o) {
  uint64_t v;
  double d;
  if (field_u64("/proc/self/status", "VmHWM:", &v)) nd_obj_u64(o, "vm_hwm_kb", v); else nd_obj_null(o, "vm_hwm_kb");
  if (field_u64("/proc/self/status", "VmRSS:", &v)) nd_obj_u64(o, "vm_rss_kb", v); else nd_obj_null(o, "vm_rss_kb");
  if (field_u64("/proc/self/status", "Threads:", &v)) nd_obj_u64(o, "threads", v); else nd_obj_null(o, "threads");
  if (nd_mem_available_kb(&v)) nd_obj_u64(o, "mem_available_kb", v); else nd_obj_null(o, "mem_available_kb");
  if (nd_cpu_seconds(&d)) nd_obj_f64(o, "cpu_s", d); else nd_obj_null(o, "cpu_s");
  if (g->thermal && read_millideg(g->thermal, &d)) nd_obj_f64(o, "temp_c", d); else nd_obj_null(o, "temp_c");
}
