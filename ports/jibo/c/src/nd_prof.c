/* nd_prof.c: see nd_prof.h. */
#include "nd_prof.h"

#include <string.h>

const char *const nd_prof_names[ND_P_N] = {"embed", "engram", "mhc", "qkv", "conv_rope",
                                           "attn", "gate_out", "mlp", "head"};

#ifdef ND_PROFILE
uint64_t (*nd_prof_clock)(void);
uint64_t nd_prof_total[ND_P_N], nd_prof_calls[ND_P_N];

void nd_prof_set_clock(uint64_t (*clock)(void)) { nd_prof_clock = clock; }

void nd_prof_take(uint64_t total[ND_P_N], uint64_t calls[ND_P_N]) {
  memcpy(total, nd_prof_total, sizeof nd_prof_total);
  memcpy(calls, nd_prof_calls, sizeof nd_prof_calls);
  memset(nd_prof_total, 0, sizeof nd_prof_total);
  memset(nd_prof_calls, 0, sizeof nd_prof_calls);
}
#else
void nd_prof_set_clock(uint64_t (*clock)(void)) { (void)clock; }

void nd_prof_take(uint64_t total[ND_P_N], uint64_t calls[ND_P_N]) {
  memset(total, 0, ND_P_N * sizeof *total);
  memset(calls, 0, ND_P_N * sizeof *calls);
}
#endif
