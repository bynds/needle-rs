/* nd_math.h: the transcendental functions of the C99 Needle 3 engine.
 *
 * Bit-identical transcriptions of the Rust `libm` crate 0.2.16 (MIT; derived from musl libm,
 * MIT, which derives these routines from FreeBSD msun / Sun fdlibm), as compiled for needle-core
 * (libm with default-features = false) on x86_64-unknown-linux-gnu and
 * armv7-unknown-linux-gnueabihf. Both targets compile the same generic code path for these
 * functions. See nd_math.c for the per-function sources and copyright notices.
 *
 * Results are bit-identical to the Rust functions for every input, NaN payloads excepted (any NaN
 * input or result may differ in payload bits, never in NaN-ness).
 */
#ifndef ND_MATH_H
#define ND_MATH_H

float nd_expf(float x);
float nd_logf(float x);
float nd_sinf(float x);
float nd_cosf(float x);
float nd_powf(float x, float y);
float nd_tanhf(float x);

#endif
