//! Primitives shared by more than one model generation.
//!
//! A function lives here only when every caller uses it with identical
//! semantics and it bakes in no geometry. Anything conditional on generation
//! stays with that generation.

extern crate alloc;

use crate::math;

/// Epsilon used by both `_rms_unit` and `_zcrms`, inside the square root.
pub const EPS: f32 = 1e-6;
/// Sinkhorn iterations upstream runs.
pub const SINKHORN_ITERS: usize = 20;

/// `y[i] += a * x[i]`: one product and one sum per element, in element order, so any split of
/// the work is the same bits. ARMv7 NEON builds (feature `neon`) do eight elements per step with
/// `cq_neon::axpy8` (the C port's `nd_axpy` is the same kernel), the rest as written here.
#[inline]
pub fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
    let n = y.len().min(x.len());
    #[allow(unused_mut)]
    let mut done = 0;
    #[cfg(all(target_arch = "arm", feature = "neon"))]
    if n >= 8 {
        // Safety: both slices hold at least `n` elements and are distinct borrows.
        unsafe { crate::cq_neon::axpy8(y.as_mut_ptr(), a, x.as_ptr(), n / 8) };
        done = n / 8 * 8;
    }
    // ARMv7 without NEON: unrolled by four, as the C port's nd_axpy (LLVM keeps this loop rolled
    // for 32-bit ARM, and the loop overhead was most of each step).
    #[cfg(all(target_arch = "arm", not(feature = "neon")))]
    {
        let m = (n - done) / 4 * 4;
        for (yy, xx) in y[done..done + m]
            .chunks_exact_mut(4)
            .zip(x[done..done + m].chunks_exact(4))
        {
            yy[0] += a * xx[0];
            yy[1] += a * xx[1];
            yy[2] += a * xx[2];
            yy[3] += a * xx[3];
        }
        done += m;
    }
    for (yi, &xi) in y[done..n].iter_mut().zip(&x[done..n]) {
        *yi += a * xi;
    }
}

/// `decode._rms_unit` — RMS normalisation with no learned scale.
///
/// Distinct from [`crate::norm::zc_rms_norm_vec`], which applies `(1 + γ)`.
/// The mHC router and the Engram gate both use the unscaled form, in every
/// generation.
#[inline]
pub fn rms_unit(x: &mut [f32]) {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / math::sqrt(mean_sq + EPS);
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// `rms_unit` into a separate destination.
#[inline]
pub fn rms_unit_to(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / math::sqrt(mean_sq + EPS);
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = v * inv;
    }
}

/// `architecture._sinkhorn` — alternating log-domain row/column normalisation of
/// an `n x n` matrix, returning `exp(log_K)`.
///
/// Row-major `m[i * n + j]`; `axis=-1` is a row (fixed `i`), `axis=-2` a column
/// (fixed `j`). Operates in place, leaving the exponentiated result.
pub fn sinkhorn(m: &mut [f32], n: usize) {
    debug_assert_eq!(m.len(), n * n);
    for _ in 0..SINKHORN_ITERS {
        for i in 0..n {
            let row = &mut m[i * n..(i + 1) * n];
            let lse = logsumexp(row);
            for v in row.iter_mut() {
                *v -= lse;
            }
        }
        for j in 0..n {
            let mut max = f32::NEG_INFINITY;
            for i in 0..n {
                max = max.max(m[i * n + j]);
            }
            let mut sum = 0.0f32;
            for i in 0..n {
                sum += math::exp(m[i * n + j] - max);
            }
            let lse = max + math::ln(sum);
            for i in 0..n {
                m[i * n + j] -= lse;
            }
        }
    }
    for v in m.iter_mut() {
        *v = math::exp(*v);
    }
}

/// Numerically stable `log(sum(exp(x)))`.
#[inline]
fn logsumexp(x: &[f32]) -> f32 {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return max;
    }
    let sum: f32 = x.iter().map(|&v| math::exp(v - max)).sum();
    max + math::ln(sum)
}
