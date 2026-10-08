/* nd_math.c: expf, logf, sinf, cosf, powf, tanhf, transcribed from the Rust `libm` crate 0.2.16.
 *
 * Source: libm 0.2.16 (https://github.com/rust-lang/compiler-builtins, libm/), MIT license,
 * "libm in pure Rust" by Alex Crichton, Amanieu d'Antras, Jorge Aparicio, Trevor Gross and
 * contributors. It is a port of musl libm (MIT, Copyright (c) 2005-2020 Rich Felker et al.),
 * whose routines come from FreeBSD msun. The FreeBSD/Sun notices carried by the Rust sources are
 * reproduced below, next to the code they cover.
 *
 * Code path: needle-core depends on libm with default-features = false, so the `arch` feature is
 * off and every routine here is the generic Rust code on both x86_64-unknown-linux-gnu and
 * armv7-unknown-linux-gnueabihf (the only arch override, x87_expf, is for x86 without SSE). The
 * one place the targets could differ, sqrtf inside powf, is correctly rounded on both (generic
 * software sqrt in the crate; IEEE sqrtf from <math.h> here), so the results are the same.
 *
 * Types follow the Rust code exactly: where it computes in f64 (the sinf/cosf kernels and
 * rem_pio2f, rem_pio2_large, scalbn, floor) this file computes in double, with every float to
 * double conversion an explicit cast. force_eval! statements (which only raise FP exceptions) are
 * omitted; they do not affect results. Rust's wrapping integer arithmetic is done in uint32_t
 * where C signed arithmetic could overflow.
 */

/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 *
 * (k_rem_pio2.c: Developed at SunSoft, a Sun Microsystems, Inc. business; same notice.)
 *
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 * sinf/cosf/k_sinf/k_cosf/rem_pio2f debugged and optimized by Bruce D. Evans.
 */

#include "nd_math.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static uint32_t f2u(float x) {
  uint32_t u;
  memcpy(&u, &x, sizeof u);
  return u;
}
static float u2f(uint32_t u) {
  float x;
  memcpy(&x, &u, sizeof x);
  return x;
}
static uint64_t d2u(double x) {
  uint64_t u;
  memcpy(&u, &x, sizeof u);
  return u;
}
static double u2d(uint64_t u) {
  double x;
  memcpy(&x, &u, sizeof x);
  return x;
}

/* ---- generic/scalbn.rs (scalbnf, scalbn) -------------------------------------------------- */

static float scalbnf_(float x, int32_t n) {
  const float f_exp_max = u2f(0x7f000000u); /* 2^127 */
  const float mul = u2f(0x0c800000u);       /* 2^-126 * 2^24 = 2^-102 */
  if (n > 127) {
    x *= f_exp_max;
    n -= 127;
    if (n > 127) {
      x *= f_exp_max;
      n -= 127;
      if (n > 127) n = 127;
    }
  } else if (n < -126) {
    x *= mul;
    n += 102;
    if (n < -126) {
      x *= mul;
      n += 102;
      if (n < -126) n = -126;
    }
  }
  return x * u2f((uint32_t)(127 + n) << 23);
}

static double scalbn_(double x, int32_t n) {
  const double f_exp_max = u2d(0x7fe0000000000000ull); /* 2^1023 */
  const double mul = u2d(0x0360000000000000ull);       /* 2^-1022 * 2^53 = 2^-969 */
  if (n > 1023) {
    x *= f_exp_max;
    n -= 1023;
    if (n > 1023) {
      x *= f_exp_max;
      n -= 1023;
      if (n > 1023) n = 1023;
    }
  } else if (n < -1022) {
    x *= mul;
    n += 969;
    if (n < -1022) {
      x *= mul;
      n += 969;
      if (n < -1022) n = -1022;
    }
  }
  return x * u2d((uint64_t)(1023 + n) << 52);
}

/* ---- generic/floor.rs (f64) ---------------------------------------------------------------- */

static double floor_(double x) {
  uint64_t ix = d2u(x);
  int32_t e = (int32_t)((ix >> 52) & 0x7ff) - 1023;
  if (e >= 52) return x;
  if (e >= 0) {
    uint64_t m = 0x000fffffffffffffull >> e;
    if ((ix & m) == 0) return x;
    if (ix >> 63) ix += m;
    ix &= ~m;
    return u2d(ix);
  }
  if (!(ix >> 63)) return 0.0;
  if ((ix << 1) != 0) return -1.0;
  return x;
}

/* ---- expf.rs (origin: FreeBSD /usr/src/lib/msun/src/e_expf.c) ------------------------------ */

static const float EXPF_HALF[2] = {0.5f, -0.5f};
#define EXPF_LN2_HI 6.9314575195e-01f  /* 0x3f317200 */
#define EXPF_LN2_LO 1.4286067653e-06f  /* 0x35bfbe8e */
#define EXPF_INV_LN2 1.4426950216e+00f /* 0x3fb8aa3b */
#define EXPF_P1 1.6666625440e-1f       /*  0xaaaa8f.0p-26 */
#define EXPF_P2 (-2.7667332906e-3f)    /* -0xb55215.0p-32 */

float nd_expf(float x) {
  const float x1p127 = u2f(0x7f000000u);
  uint32_t hx = f2u(x);
  int32_t sign = (int32_t)(hx >> 31);
  int signb = sign != 0;
  int32_t k;
  float hi, lo, xx, c, y;
  hx &= 0x7fffffff;

  if (hx >= 0x42aeac50) {
    if (hx > 0x7f800000) return x; /* NaN */
    if (hx >= 0x42b17218 && !signb) {
      x *= x1p127; /* overflow */
      return x;
    }
    if (signb) {
      if (hx >= 0x42cff1b5) return 0.0f; /* underflow */
    }
  }

  if (hx > 0x3eb17218) {
    if (hx > 0x3f851592) {
      k = (int32_t)(EXPF_INV_LN2 * x + EXPF_HALF[sign]);
    } else {
      k = 1 - sign - sign;
    }
    {
      float kf = (float)k;
      hi = x - kf * EXPF_LN2_HI;
      lo = kf * EXPF_LN2_LO;
    }
    x = hi - lo;
  } else if (hx > 0x39000000) {
    k = 0;
    hi = x;
    lo = 0.0f;
  } else {
    return 1.0f + x;
  }

  xx = x * x;
  c = x - xx * (EXPF_P1 + xx * EXPF_P2);
  y = 1.0f + (x * c / (2.0f - c) - lo + hi);
  return k == 0 ? y : scalbnf_(y, k);
}

/* ---- logf.rs (origin: FreeBSD /usr/src/lib/msun/src/e_logf.c) ------------------------------ */

#define LOGF_LN2_HI 6.9313812256e-01f /* 0x3f317180 */
#define LOGF_LN2_LO 9.0580006145e-06f /* 0x3717f7d1 */
#define LOGF_LG1 0.66666662693f       /* 0xaaaaaa.0p-24 */
#define LOGF_LG2 0.40000972152f       /* 0xccce13.0p-25 */
#define LOGF_LG3 0.28498786688f       /* 0x91e9ee.0p-25 */
#define LOGF_LG4 0.24279078841f       /* 0xf89e26.0p-26 */

float nd_logf(float x) {
  const float x1p25 = u2f(0x4c000000u);
  uint32_t ix = f2u(x);
  int32_t k = 0;
  float f, s, z, w, t1, t2, r, hfsq, dk;

  if (ix < 0x00800000 || (ix >> 31) != 0) {
    if ((ix << 1) == 0) return -1.0f / (x * x); /* log(+-0) = -inf */
    if ((ix >> 31) != 0) return (x - x) / 0.0f;  /* log(-#) = NaN */
    k -= 25;                                     /* subnormal: scale up */
    x *= x1p25;
    ix = f2u(x);
  } else if (ix >= 0x7f800000) {
    return x;
  } else if (ix == 0x3f800000) {
    return 0.0f;
  }

  ix += 0x3f800000 - 0x3f3504f3;
  k += (int32_t)(ix >> 23) - 0x7f;
  ix = (ix & 0x007fffff) + 0x3f3504f3;
  x = u2f(ix);

  f = x - 1.0f;
  s = f / (2.0f + f);
  z = s * s;
  w = z * z;
  t1 = w * (LOGF_LG2 + w * LOGF_LG4);
  t2 = z * (LOGF_LG1 + w * LOGF_LG3);
  r = t2 + t1;
  hfsq = 0.5f * f * f;
  dk = (float)k;
  return s * (hfsq + r) + dk * LOGF_LN2_LO - hfsq + f + dk * LOGF_LN2_HI;
}

/* ---- k_sinf.rs, k_cosf.rs (origin: FreeBSD k_sinf.c, k_cosf.c); f64 throughout ------------- */

static const double KS1 = -0.166666666416265235595;     /* -0x15555554cbac77.0p-55 */
static const double KS2 = 0.0083333293858894631756;     /*  0x111110896efbb2.0p-59 */
static const double KS3 = -0.000198393348360966317347;  /* -0x1a00f9e2cae774.0p-65 */
static const double KS4 = 0.0000027183114939898219064;  /*  0x16cd878c3b46a7.0p-71 */

static float k_sinf(double x) {
  double z = x * x;
  double w = z * z;
  double r = KS3 + z * KS4;
  double s = z * x;
  return (float)((x + s * (KS1 + z * KS2)) + s * w * r);
}

static const double KC0 = -0.499999997251031003120;    /* -0x1ffffffd0c5e81.0p-54 */
static const double KC1 = 0.0416666233237390631894;    /*  0x155553e1053a42.0p-57 */
static const double KC2 = -0.00138867637746099294692;  /* -0x16c087e80f1e27.0p-62 */
static const double KC3 = 0.0000243904487962774090654; /*  0x199342e0ee5069.0p-68 */

static float k_cosf(double x) {
  double z = x * x;
  double w = z * z;
  double r = KC2 + z * KC3;
  return (float)(((1.0 + z * KC0) + w * KC1) + (w * z) * r);
}

/* ---- rem_pio2_large.rs (origin: FreeBSD /usr/src/lib/msun/src/k_rem_pio2.c) ---------------- */
/* Specialised to what rem_pio2f passes: one input chunk (nx = 1) and prec = 0 (jk = 3). The
 * table is the first 66 entries of ipio2 (the 32-bit-target table; the 64-bit table extends it
 * and float inputs never index past entry ~10). */

static const int32_t IPIO2[66] = {
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62, 0x95993C, 0x439041, 0xFE5163,
    0xABDEBB, 0xC561B7, 0x246E3A, 0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129,
    0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C, 0x7026B4, 0x5F7E41, 0x3991D6, 0x398353, 0x39F49C,
    0x845F8B, 0xBDF928, 0x3B1FF8, 0x97FFDE, 0x05980F, 0xEF2F11, 0x8B5A0A, 0x6D1F6D, 0x367ECF,
    0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5, 0xF17B3D, 0x0739F7, 0x8A5292,
    0xEA6BFB, 0x5FB11F, 0x8D5D08, 0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3,
    0x91615E, 0xE61B08, 0x659985, 0x5F14A0, 0x68408D, 0xFFD880, 0x4D7327, 0x310606, 0x1556CA,
    0x73A8C9, 0x60E27B, 0xC08C6B,
};

static const double PIO2[8] = {
    1.57079625129699707031e+00, /* 0x3FF921FB, 0x40000000 */
    7.54978941586159635335e-08, /* 0x3E74442D, 0x00000000 */
    5.39030252995776476554e-15, /* 0x3CF84698, 0x80000000 */
    3.28200341580791294123e-22, /* 0x3B78CC51, 0x60000000 */
    1.27065575308067607349e-29, /* 0x39F01B83, 0x80000000 */
    1.22933308981111328932e-36, /* 0x387A2520, 0x40000000 */
    2.73370053816464559624e-44, /* 0x36E38222, 0x80000000 */
    2.16741683877804819444e-51, /* 0x3569F31D, 0x00000000 */
};

static int32_t rem_pio2_large(const double *x, double *y, int32_t e0) {
  const double x1p24 = u2d(0x4170000000000000ull);
  const double x1p_24 = u2d(0x3e70000000000000ull);
  const int jk = 3; /* INIT_JK[prec = 0] */
  const int jp = jk;
  const int jx = 0; /* nx - 1 */
  double fw, z;
  int32_t n, ih, jv, q0, j;
  double f[20] = {0}, fq[20] = {0}, q[20] = {0};
  int32_t iq[20] = {0};
  int i, m, jz;

  jv = (e0 - 3) / 24;
  if (jv < 0) jv = 0;
  q0 = e0 - 24 * (jv + 1);

  j = jv - jx;
  m = jx + jk;
  for (i = 0; i <= m; i++) {
    f[i] = j < 0 ? 0.0 : (double)IPIO2[j];
    j++;
  }

  for (i = 0; i <= jk; i++) {
    int jj;
    fw = 0.0;
    for (jj = 0; jj <= jx; jj++) fw += x[jj] * f[jx + i - jj];
    q[i] = fw;
  }

  jz = jk;

  for (;;) { /* 'recompute */
    int32_t ii = 0;
    int carry;
    z = q[jz];
    for (j = jz; j >= 1; j--) {
      fw = (double)(int32_t)(x1p_24 * z);
      iq[ii] = (int32_t)(z - x1p24 * fw);
      z = q[j - 1] + fw;
      ii++;
    }

    z = scalbn_(z, q0);
    z -= 8.0 * floor_(z * 0.125);
    n = (int32_t)z;
    z -= (double)n;
    ih = 0;
    if (q0 > 0) {
      ii = iq[jz - 1] >> (24 - q0);
      n += ii;
      iq[jz - 1] -= ii << (24 - q0);
      ih = iq[jz - 1] >> (23 - q0);
    } else if (q0 == 0) {
      ih = iq[jz - 1] >> 23;
    } else if (z >= 0.5) {
      ih = 2;
    }

    if (ih > 0) {
      n += 1;
      carry = 0;
      for (i = 0; i < jz; i++) {
        int32_t jj = iq[i];
        if (carry == 0) {
          if (jj != 0) {
            carry = 1;
            iq[i] = 0x1000000 - jj;
          }
        } else {
          iq[i] = 0xffffff - jj;
        }
      }
      if (q0 > 0) {
        switch (q0) {
          case 1: iq[jz - 1] &= 0x7fffff; break;
          case 2: iq[jz - 1] &= 0x3fffff; break;
          default: break;
        }
      }
      if (ih == 2) {
        z = 1.0 - z;
        if (carry != 0) z -= scalbn_(1.0, q0);
      }
    }

    if (z == 0.0) {
      int32_t jj = 0;
      for (i = jz - 1; i >= jk; i--) jj |= iq[i];
      if (jj == 0) {
        int k = 1;
        while (iq[jk - k] == 0) k++;
        for (i = jz + 1; i <= jz + k; i++) {
          int t;
          f[jx + i] = (double)IPIO2[jv + i];
          fw = 0.0;
          for (t = 0; t <= jx; t++) fw += x[t] * f[jx + i - t];
          q[i] = fw;
        }
        jz += k;
        continue;
      }
    }
    break;
  }

  if (z == 0.0) {
    jz -= 1;
    q0 -= 24;
    while (iq[jz] == 0) {
      jz -= 1;
      q0 -= 24;
    }
  } else {
    z = scalbn_(z, -q0);
    if (z >= x1p24) {
      fw = (double)(int32_t)(x1p_24 * z);
      iq[jz] = (int32_t)(z - x1p24 * fw);
      jz += 1;
      q0 += 24;
      iq[jz] = (int32_t)fw;
    } else {
      iq[jz] = (int32_t)z;
    }
  }

  fw = scalbn_(1.0, q0);
  for (i = jz; i >= 0; i--) {
    q[i] = fw * (double)iq[i];
    fw *= x1p_24;
  }

  for (i = jz; i >= 0; i--) {
    int k = 0;
    fw = 0.0;
    while (k <= jp && k <= jz - i) {
      fw += PIO2[k] * q[i + k];
      k++;
    }
    fq[jz - i] = fw;
  }

  /* prec 0 */
  fw = 0.0;
  for (i = jz; i >= 0; i--) fw += fq[i];
  y[0] = ih == 0 ? fw : -fw;
  return n & 7;
}

/* ---- rem_pio2f.rs (origin: FreeBSD /usr/src/lib/msun/src/e_rem_pio2f.c) -------------------- */

static const double TOINT = 6755399441055744.0;               /* 1.5 / f64::EPSILON */
static const double INV_PIO2 = 6.36619772367581382433e-01;    /* 0x3FE45F30, 0x6DC9C883 */
static const double PIO2_1 = 1.57079631090164184570e+00;      /* 0x3FF921FB, 0x50000000 */
static const double PIO2_1T = 1.58932547735281966916e-08;     /* 0x3E5110b4, 0x611A6263 */

static int32_t rem_pio2f(float x, double *y) {
  double x64 = (double)x;
  double tx[1], ty[1];
  uint32_t ix = f2u(x) & 0x7fffffff;
  int sign;
  int32_t e0, n;

  if (ix < 0x4dc90fdb) { /* |x| ~< 2^28*(pi/2), medium size */
    double tmp = x64 * INV_PIO2 + TOINT;
    double f_n = tmp - TOINT;
    *y = x64 - f_n * PIO2_1 - f_n * PIO2_1T;
    return (int32_t)f_n;
  }
  if (ix >= 0x7f800000) { /* inf or NaN */
    *y = x64 - x64;
    return 0;
  }
  sign = (f2u(x) >> 31) != 0;
  e0 = (int32_t)((ix >> 23) - (0x7f + 23));
  tx[0] = (double)u2f(ix - ((uint32_t)e0 << 23));
  ty[0] = 0.0;
  n = rem_pio2_large(tx, ty, e0);
  if (sign) {
    *y = -ty[0];
    return -n;
  }
  *y = ty[0];
  return n;
}

/* ---- sinf.rs, cosf.rs (origin: FreeBSD s_sinf.c, s_cosf.c) --------------------------------- */

static const double FRAC_PI_2 = 1.57079632679489661923132169163975144;
#define S1_PIO2 (1.0 * FRAC_PI_2) /* 0x3FF921FB, 0x54442D18 */
#define S2_PIO2 (2.0 * FRAC_PI_2) /* 0x400921FB, 0x54442D18 */
#define S3_PIO2 (3.0 * FRAC_PI_2) /* 0x4012D97C, 0x7F3321D2 */
#define S4_PIO2 (4.0 * FRAC_PI_2) /* 0x401921FB, 0x54442D18 */

float nd_sinf(float x) {
  double x64 = (double)x;
  uint32_t ix = f2u(x);
  int sign = (ix >> 31) != 0;
  int32_t n;
  double y;
  ix &= 0x7fffffff;

  if (ix <= 0x3f490fda) { /* |x| ~<= pi/4 */
    if (ix < 0x39800000) return x;
    return k_sinf(x64);
  }
  if (ix <= 0x407b53d1) { /* |x| ~<= 5*pi/4 */
    if (ix <= 0x4016cbe3) {
      if (sign) return -k_cosf(x64 + S1_PIO2);
      return k_cosf(x64 - S1_PIO2);
    }
    return k_sinf(sign ? -(x64 + S2_PIO2) : -(x64 - S2_PIO2));
  }
  if (ix <= 0x40e231d5) { /* |x| ~<= 9*pi/4 */
    if (ix <= 0x40afeddf) {
      if (sign) return k_cosf(x64 + S3_PIO2);
      return -k_cosf(x64 - S3_PIO2);
    }
    return k_sinf(sign ? x64 + S4_PIO2 : x64 - S4_PIO2);
  }

  if (ix >= 0x7f800000) return x - x;

  n = rem_pio2f(x, &y);
  switch (n & 3) {
    case 0: return k_sinf(y);
    case 1: return k_cosf(y);
    case 2: return k_sinf(-y);
    default: return -k_cosf(y);
  }
}

float nd_cosf(float x) {
  double x64 = (double)x;
  uint32_t ix = f2u(x);
  int sign = (ix >> 31) != 0;
  int32_t n;
  double y;
  ix &= 0x7fffffff;

  if (ix <= 0x3f490fda) { /* |x| ~<= pi/4 */
    if (ix < 0x39800000) return 1.0f;
    return k_cosf(x64);
  }
  if (ix <= 0x407b53d1) { /* |x| ~<= 5*pi/4 */
    if (ix > 0x4016cbe3) return -k_cosf(sign ? x64 + S2_PIO2 : x64 - S2_PIO2);
    if (sign) return k_sinf(x64 + S1_PIO2);
    return k_sinf(S1_PIO2 - x64);
  }
  if (ix <= 0x40e231d5) { /* |x| ~<= 9*pi/4 */
    if (ix > 0x40afeddf) return k_cosf(sign ? x64 + S4_PIO2 : x64 - S4_PIO2);
    if (sign) return k_sinf(-x64 - S3_PIO2);
    return k_sinf(x64 - S3_PIO2);
  }

  if (ix >= 0x7f800000) return x - x;

  n = rem_pio2f(x, &y);
  switch (n & 3) {
    case 0: return k_cosf(y);
    case 1: return k_sinf(-y);
    case 2: return -k_cosf(y);
    default: return k_sinf(y);
  }
}

/* ---- powf.rs (origin: FreeBSD /usr/src/lib/msun/src/e_powf.c) ------------------------------ */

static const float POW_BP[2] = {1.0f, 1.5f};
static const float POW_DP_H[2] = {0.0f, 5.84960938e-01f}; /* 0x3f15c000 */
static const float POW_DP_L[2] = {0.0f, 1.56322085e-06f}; /* 0x35d1cfdc */
#define POW_TWO24 16777216.0f    /* 0x4b800000 */
#define POW_HUGE 1.0e30f
#define POW_TINY 1.0e-30f
#define POW_L1 6.0000002384e-01f /* 0x3f19999a */
#define POW_L2 4.2857143283e-01f /* 0x3edb6db7 */
#define POW_L3 3.3333334327e-01f /* 0x3eaaaaab */
#define POW_L4 2.7272811532e-01f /* 0x3e8ba305 */
#define POW_L5 2.3066075146e-01f /* 0x3e6c3255 */
#define POW_L6 2.0697501302e-01f /* 0x3e53f142 */
#define POW_P1 1.6666667163e-01f /* 0x3e2aaaab */
#define POW_P2 (-2.7777778450e-03f) /* 0xbb360b61 */
#define POW_P3 6.6137559770e-05f /* 0x388ab355 */
#define POW_P4 (-1.6533901999e-06f) /* 0xb5ddea0e */
#define POW_P5 4.1381369442e-08f /* 0x3331bb4c */
#define POW_LG2 6.9314718246e-01f   /* 0x3f317218 */
#define POW_LG2_H 6.93145752e-01f   /* 0x3f317200 */
#define POW_LG2_L 1.42860654e-06f   /* 0x35bfbe8c */
#define POW_OVT 4.2995665694e-08f   /* -(128-log2(ovfl+.5ulp)) */
#define POW_CP 9.6179670095e-01f    /* 0x3f76384f =2/(3ln2) */
#define POW_CP_H 9.6191406250e-01f  /* 0x3f764000 =12b cp */
#define POW_CP_L (-1.1736857402e-04f) /* 0xb8f623c6 =tail of cp_h */
#define POW_IVLN2 1.4426950216e+00f
#define POW_IVLN2_H 1.4426879883e+00f
#define POW_IVLN2_L 7.0526075433e-06f

float nd_powf(float x, float y) {
  float z, ax, z_h, z_l, p_h, p_l, y1, t1, t2, r, s, sn, t, u, v, w;
  int32_t i, j, k, yisint, n, hx, hy, ix, iy, is;

  hx = (int32_t)f2u(x);
  hy = (int32_t)f2u(y);
  ix = hx & 0x7fffffff;
  iy = hy & 0x7fffffff;

  if (iy == 0) return 1.0f;            /* x**0 = 1, even if x is NaN */
  if (hx == 0x3f800000) return 1.0f;   /* 1**y = 1, even if y is NaN */
  if (ix > 0x7f800000 || iy > 0x7f800000) return x + y;

  yisint = 0;
  if (hx < 0) {
    if (iy >= 0x4b800000) {
      yisint = 2;
    } else if (iy >= 0x3f800000) {
      k = (iy >> 23) - 0x7f;
      j = iy >> (23 - k);
      if ((j << (23 - k)) == iy) yisint = 2 - (j & 1);
    }
  }

  if (iy == 0x7f800000) { /* y is +-inf */
    if (ix == 0x3f800000) return 1.0f;
    if (ix > 0x3f800000) return hy >= 0 ? y : 0.0f;
    return hy >= 0 ? 0.0f : -y;
  }
  if (iy == 0x3f800000) return hy >= 0 ? x : 1.0f / x;
  if (hy == 0x40000000) return x * x;
  if (hy == 0x3f000000 && hx >= 0) return sqrtf(x);

  ax = fabsf(x);
  if (ix == 0x7f800000 || ix == 0 || ix == 0x3f800000) {
    z = ax;
    if (hy < 0) z = 1.0f / z;
    if (hx < 0) {
      if (((ix - 0x3f800000) | yisint) == 0) {
        z = (z - z) / (z - z);
      } else if (yisint == 1) {
        z = -z;
      }
    }
    return z;
  }

  sn = 1.0f;
  if (hx < 0) {
    if (yisint == 0) return (x - x) / (x - x);
    if (yisint == 1) sn = -1.0f;
  }

  if (iy > 0x4d000000) { /* |y| > 2**27 */
    if (ix < 0x3f7ffff8) return hy < 0 ? sn * POW_HUGE * POW_HUGE : sn * POW_TINY * POW_TINY;
    if (ix > 0x3f800007) return hy > 0 ? sn * POW_HUGE * POW_HUGE : sn * POW_TINY * POW_TINY;
    t = ax - 1.0f;
    w = (t * t) * (0.5f - t * (0.333333333333f - t * 0.25f));
    u = POW_IVLN2_H * t;
    v = t * POW_IVLN2_L - w * POW_IVLN2;
    t1 = u + v;
    is = (int32_t)f2u(t1);
    t1 = u2f((uint32_t)is & 0xfffff000u);
    t2 = v - (t1 - u);
  } else {
    float s2, s_h, s_l, t_h, t_l;
    n = 0;
    if (ix < 0x00800000) {
      ax *= POW_TWO24;
      n -= 24;
      ix = (int32_t)f2u(ax);
    }
    n += (ix >> 23) - 0x7f;
    j = ix & 0x007fffff;
    ix = j | 0x3f800000;
    if (j <= 0x1cc471) {
      k = 0;
    } else if (j < 0x5db3d7) {
      k = 1;
    } else {
      k = 0;
      n += 1;
      ix -= 0x00800000;
    }
    ax = u2f((uint32_t)ix);

    u = ax - POW_BP[k];
    v = 1.0f / (ax + POW_BP[k]);
    s = u * v;
    s_h = s;
    is = (int32_t)f2u(s_h);
    s_h = u2f((uint32_t)is & 0xfffff000u);
    is = (int32_t)((((uint32_t)ix >> 1) & 0xfffff000u) | 0x20000000u);
    t_h = u2f((uint32_t)is + 0x00400000u + ((uint32_t)k << 21));
    t_l = ax - (t_h - POW_BP[k]);
    s_l = v * ((u - s_h * t_h) - s_h * t_l);
    s2 = s * s;
    r = s2 * s2 * (POW_L1 + s2 * (POW_L2 + s2 * (POW_L3 + s2 * (POW_L4 + s2 * (POW_L5 + s2 * POW_L6)))));
    r += s_l * (s_h + s);
    s2 = s_h * s_h;
    t_h = 3.0f + s2 + r;
    is = (int32_t)f2u(t_h);
    t_h = u2f((uint32_t)is & 0xfffff000u);
    t_l = r - ((t_h - 3.0f) - s2);
    u = s_h * t_h;
    v = s_l * t_h + t_l * s;
    p_h = u + v;
    is = (int32_t)f2u(p_h);
    p_h = u2f((uint32_t)is & 0xfffff000u);
    p_l = v - (p_h - u);
    z_h = POW_CP_H * p_h;
    z_l = POW_CP_L * p_h + p_l * POW_CP + POW_DP_L[k];
    t = (float)n;
    t1 = ((z_h + z_l) + POW_DP_H[k]) + t;
    is = (int32_t)f2u(t1);
    t1 = u2f((uint32_t)is & 0xfffff000u);
    t2 = z_l - (((t1 - t) - POW_DP_H[k]) - z_h);
  }

  is = (int32_t)f2u(y);
  y1 = u2f((uint32_t)is & 0xfffff000u);
  p_l = (y - y1) * t1 + y * t2;
  p_h = y1 * t1;
  z = p_l + p_h;
  j = (int32_t)f2u(z);
  if (j > 0x43000000) {
    return sn * POW_HUGE * POW_HUGE; /* overflow */
  } else if (j == 0x43000000) {
    if (p_l + POW_OVT > z - p_h) return sn * POW_HUGE * POW_HUGE;
  } else if ((j & 0x7fffffff) > 0x43160000) {
    return sn * POW_TINY * POW_TINY; /* underflow */
  } else if ((uint32_t)j == 0xc3160000u && p_l <= z - p_h) {
    return sn * POW_TINY * POW_TINY;
  }

  i = j & 0x7fffffff;
  k = (i >> 23) - 0x7f;
  n = 0;
  if (i > 0x3f000000) {
    n = j + (0x00800000 >> (k + 1));
    k = ((n & 0x7fffffff) >> 23) - 0x7f;
    t = u2f((uint32_t)n & ~(0x007fffffu >> k));
    n = ((n & 0x007fffff) | 0x00800000) >> (23 - k);
    if (j < 0) n = -n;
    p_h -= t;
  }
  t = p_l + p_h;
  is = (int32_t)f2u(t);
  t = u2f((uint32_t)is & 0xffff8000u);
  u = t * POW_LG2_H;
  v = (p_l - (t - p_h)) * POW_LG2 + t * POW_LG2_L;
  z = u + v;
  w = v - (z - u);
  t = z * z;
  t1 = z - t * (POW_P1 + t * (POW_P2 + t * (POW_P3 + t * (POW_P4 + t * POW_P5))));
  r = (z * t1) / (t1 - 2.0f) - (w + z * w);
  z = 1.0f - (r - z);
  j = (int32_t)f2u(z);
  j = (int32_t)((uint32_t)j + ((uint32_t)n << 23));
  if ((j >> 23) <= 0) {
    z = scalbnf_(z, n); /* subnormal output */
  } else {
    z = u2f((uint32_t)j);
  }
  return sn * z;
}

/* ---- expm1f.rs (origin: FreeBSD /usr/src/lib/msun/src/s_expm1f.c) -------------------------- */

#define EXPM1_LN2_HI 6.9313812256e-01f  /* 0x3f317180 */
#define EXPM1_LN2_LO 9.0580006145e-06f  /* 0x3717f7d1 */
#define EXPM1_INV_LN2 1.4426950216e+00f /* 0x3fb8aa3b */
#define EXPM1_Q1 (-3.3333212137e-2f)    /* -0x888868.0p-28 */
#define EXPM1_Q2 1.5807170421e-3f       /*  0xcf3010.0p-33 */

static float expm1f_(float x) {
  const float x1p127 = u2f(0x7f000000u);
  uint32_t hx = f2u(x);
  int sign = (hx >> 31) != 0;
  int32_t k;
  float hi, lo, c = 0.0f, hfx, hxs, r1, t, e, twopk, uf;
  hx &= 0x7fffffff;

  if (hx >= 0x4195b844) { /* |x| >= 27*ln2 */
    if (hx > 0x7f800000) return x;
    if (sign) return -1.0f;
    if (hx > 0x42b17217) {
      x *= x1p127;
      return x;
    }
  }

  if (hx > 0x3eb17218) { /* |x| > 0.5 ln2 */
    if (hx < 0x3F851592) {
      if (!sign) {
        hi = x - EXPM1_LN2_HI;
        lo = EXPM1_LN2_LO;
        k = 1;
      } else {
        hi = x + EXPM1_LN2_HI;
        lo = -EXPM1_LN2_LO;
        k = -1;
      }
    } else {
      float tk;
      k = (int32_t)(EXPM1_INV_LN2 * x + (sign ? -0.5f : 0.5f));
      tk = (float)k;
      hi = x - tk * EXPM1_LN2_HI;
      lo = tk * EXPM1_LN2_LO;
    }
    x = hi - lo;
    c = (hi - x) - lo;
  } else if (hx < 0x33000000) { /* |x| < 2**-25 */
    return x;
  } else {
    k = 0;
  }

  hfx = 0.5f * x;
  hxs = x * hfx;
  r1 = 1.0f + hxs * (EXPM1_Q1 + hxs * EXPM1_Q2);
  t = 3.0f - r1 * hfx;
  e = hxs * ((r1 - t) / (6.0f - x * t));
  if (k == 0) return x - (x * e - hxs);
  e = x * (e - c) - c;
  e -= hxs;
  if (k == -1) return 0.5f * (x - e) - 0.5f;
  if (k == 1) {
    if (x < -0.25f) return -2.0f * (e - (x + 0.5f));
    return 1.0f + 2.0f * (x - e);
  }
  twopk = u2f((uint32_t)(0x7f + k) << 23);
  if (k < 0 || k > 56) {
    float yy = x - e + 1.0f;
    if (k == 128) {
      yy = yy * 2.0f * x1p127;
    } else {
      yy = yy * twopk;
    }
    return yy - 1.0f;
  }
  uf = u2f((uint32_t)(0x7f - k) << 23);
  if (k < 23) return (x - e + (1.0f - uf)) * twopk;
  return (x - (e + uf) + 1.0f) * twopk;
}

/* ---- tanhf.rs (musl src/math/tanhf.c) ------------------------------------------------------ */

float nd_tanhf(float x) {
  uint32_t ix = f2u(x);
  int sign = (ix >> 31) != 0;
  uint32_t w;
  float tt;
  ix &= 0x7fffffff;
  x = u2f(ix);
  w = ix;

  if (w > 0x3f0c9f54) { /* |x| > log(3)/2 or NaN */
    if (w > 0x41200000) {
      tt = 1.0f + 0.0f / x;
    } else {
      float t = expm1f_(2.0f * x);
      tt = 1.0f - 2.0f / (t + 2.0f);
    }
  } else if (w > 0x3e82c578) { /* |x| > log(5/3)/2 */
    float t = expm1f_(2.0f * x);
    tt = t / (t + 2.0f);
  } else if (w >= 0x00800000) {
    float t = expm1f_(-2.0f * x);
    tt = -t / (t + 2.0f);
  } else {
    tt = x; /* subnormal */
  }
  return sign ? -tt : tt;
}
