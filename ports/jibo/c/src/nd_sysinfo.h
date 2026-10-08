/* nd_sysinfo.h: read-only resource observations and the admission gate (runner/src/sysinfo.rs).
 * Nothing here changes a clock, a fan or a thermal limit. */
#ifndef ND_SYSINFO_H
#define ND_SYSINFO_H

#include "nd_obj.h"

typedef struct {
  int has_min_avail;
  uint64_t min_avail_mb;
  const char *thermal; /* NULL: none */
  int has_max_temp;
  double max_temp_c;
} nd_gate;

/* Why the runner should answer busy, into buf (NUL-terminated), or 0 when it may work. */
int nd_gate_busy(const nd_gate *g, char *buf, size_t cap);
/* MemAvailable in kB; 0 when unreadable. */
int nd_mem_available_kb(uint64_t *kb);
/* User + system CPU seconds of this process; 0 when unreadable. */
int nd_cpu_seconds(double *s);
/* The snapshot object: vm_hwm_kb, vm_rss_kb, threads, mem_available_kb, cpu_s, temp_c. */
void nd_sysinfo_snapshot(const nd_gate *g, nd_obj *out);
/* Rust's f64 Display (shortest round-trip digits, never an exponent) into buf. */
void nd_fmt_f64_display(double v, char *buf, size_t cap);
void nd_fmt_f32_display(float v, char *buf, size_t cap);

#endif
