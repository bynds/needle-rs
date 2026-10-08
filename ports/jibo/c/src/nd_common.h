/* nd_common.h: shared conventions of the C99 Needle 3 engine (ports/jibo/c).
 *
 * The engine is a transcription of needle-rs (crates/needle-core, crates/needle-infer at the
 * commits on this branch) and is held to bit-identical results against it. The rules that make
 * that possible, and that every file follows:
 *
 *  - Float arithmetic is single precision throughout: every literal carries an `f` suffix and the
 *    build uses -Wdouble-promotion, so nothing is silently computed in double.
 *  - No fused multiply-add: the build uses -ffp-contract=off (Rust never contracts a * b + c).
 *  - Sums keep the Rust order. A Rust `iter().sum::<f32>()` starts from -0.0, so its C
 *    transcription does too (`float s = -0.0f`); explicit accumulators start where Rust's do.
 *  - Transcendental functions are nd_expf, nd_logf, nd_sinf, nd_cosf, nd_powf, nd_tanhf
 *    (nd_math.h): transcriptions of the Rust `libm` crate 0.2.16 (itself musl's libm), which is
 *    what needle-core calls. sqrtf and roundf come from <math.h> (both are exactly specified).
 *  - Sizes are size_t; every size computed from file contents is checked for overflow before use
 *    (nd_mul_ok / nd_add_ok), and nothing is allocated from an unchecked header field.
 *  - Errors are returned, never aborted on: functions return 0 on success and a negative
 *    ND_E_* code on failure, with a message in an nd_err buffer when one is passed.
 */
#ifndef ND_COMMON_H
#define ND_COMMON_H

#include <stddef.h>
#include <stdint.h>

#define ND_OK 0
#define ND_E_NOMEM (-1)
#define ND_E_FORMAT (-2)  /* the input file or blob is malformed */
#define ND_E_BOUNDS (-3)  /* a size or index is out of range */
#define ND_E_IO (-4)
#define ND_E_ARG (-5)     /* a caller error */

typedef struct {
  char msg[256];
} nd_err;

void nd_seterr(nd_err *e, const char *fmt, ...);

/* a * b and a + b without wrapping; 0 if it would wrap. */
static inline int nd_mul_ok(size_t a, size_t b, size_t *out) {
  if (a != 0 && b > (size_t)-1 / a) return 0;
  *out = a * b;
  return 1;
}
static inline int nd_add_ok(size_t a, size_t b, size_t *out) {
  if (b > (size_t)-1 - a) return 0;
  *out = a + b;
  return 1;
}

/* calloc that refuses a wrapping count and treats a zero count as one element. */
void *nd_calloc(size_t n, size_t size);

#endif
