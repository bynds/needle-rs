/* nd_prof.h: opt-in per-operation cost accounting (-DND_PROFILE), the C side of needle-core's
 * prof.rs: the same operations at the same points, so the two engines' breakdowns compare line by
 * line. Each span adds the advance of a caller-supplied clock (nanoseconds on a host, instructions
 * retired in the perfvm guest) to its operation. Without ND_PROFILE the macros are empty blocks.
 *
 *   ND_PB(ND_P_QKV); ...work... ND_PE();      spans do not nest; a goto out of one skips it */
#ifndef ND_PROF_H
#define ND_PROF_H

#include <stdint.h>

enum {
  ND_P_EMBED = 0, /* token embedding rows */
  ND_P_ENGRAM,    /* hashing, table rows, key/value projections, value conv, the site gate */
  ND_P_MHC,       /* lane norms, phi projections, mix-down, scatter-up */
  ND_P_QKV,       /* norm_in and the Q/K/V projections */
  ND_P_CONV_ROPE, /* Q/K/V conv, Q/K norms, RoPE, int8 query quantisation */
  ND_P_ATTN,      /* attention, including writing the cache */
  ND_P_GATE_OUT,  /* gate and output projections, post_norm, the residual */
  ND_P_MLP,       /* pre_hada and the Hadamard MLP */
  ND_P_HEAD,      /* final norm and the logits head */
  ND_P_N
};

extern const char *const nd_prof_names[ND_P_N];

/* Set the clock (NULL: spans record nothing); read and reset the totals. */
void nd_prof_set_clock(uint64_t (*clock)(void));
void nd_prof_take(uint64_t total[ND_P_N], uint64_t calls[ND_P_N]);

#ifdef ND_PROFILE
extern uint64_t (*nd_prof_clock)(void);
extern uint64_t nd_prof_total[ND_P_N], nd_prof_calls[ND_P_N];
#define ND_PB(op)                                             \
  {                                                           \
    const int nd_p_op = (op);                                 \
    const uint64_t nd_p_t0 = nd_prof_clock ? nd_prof_clock() : 0;
#define ND_PE()                                               \
  if (nd_prof_clock) {                                        \
    nd_prof_total[nd_p_op] += nd_prof_clock() - nd_p_t0;      \
    nd_prof_calls[nd_p_op]++;                                 \
  }                                                           \
  }
#else
#define ND_PB(op) {
#define ND_PE() }
#endif

#endif
