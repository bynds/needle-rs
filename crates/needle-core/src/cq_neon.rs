//! ARMv7 NEON builds of the two CQ inner loops, in stable Rust inline assembly.
//!
//! Compiled only for `target_arch = "arm"` with this crate's `neon` feature, which must come with
//! `-C target-feature=+neon` (stable Rust 1.87 sets no `cfg(target_feature)` for ARM features, so
//! the cargo feature is the switch; without the flag the q-register operands fail to compile).
//! The `armv7-unknown-linux-gnueabihf`
//! target defaults to `-neon`, ARM32 NEON intrinsics are not stable, and LLVM does not
//! auto-vectorise f32 for ARMv7 NEON (it flushes denormals, so it is not IEEE): without this,
//! every CQ multiply on a 32-bit ARM runs scalar.
//!
//! Both kernels keep the scalar kernels' arithmetic: eight lanes (two q registers), lane `k`
//! accumulating elements `k, k + 8, ...` with `VMLA.F32`, which on ARMv7 rounds the product and
//! the sum separately, as the scalar build (VFPv3, no fused multiply-add) does. The caller sums
//! the eight lanes in order. The one difference: NEON flushes denormal inputs and results to
//! zero, where the scalar VFP path keeps them. Activations and weights of the shipped model are
//! far from that range; the parity tests run under emulation with this enabled.
//!
//! Registers: q8–q11 hold operands and q12–q13 the accumulators. They are d16–d27, caller-saved
//! in the AAPCS, so nothing callee-saved is touched.

use super::cq::ACC_LANES;

/// Lanes of `Σ_c u[8c + k] · x[8c + k]` over `chunks` chunks of eight.
///
/// # Safety
/// `u` and `x` must each be valid for `8 * chunks` reads, and `chunks >= 1`.
#[inline]
pub(crate) unsafe fn lanes8(u: *const f32, x: *const f32, chunks: usize) -> [f32; ACC_LANES] {
    let mut out = [0.0f32; ACC_LANES];
    core::arch::asm!(
        "vmov.i32 q12, #0",
        "vmov.i32 q13, #0",
        "2:",
        "vld1.32 {{d16-d19}}, [{u}]!",
        "vld1.32 {{d20-d23}}, [{x}]!",
        "vmla.f32 q12, q8, q10",
        "vmla.f32 q13, q9, q11",
        "subs {n}, {n}, #1",
        "bne 2b",
        "vst1.32 {{d24-d27}}, [{o}]",
        u = inout(reg) u => _,
        x = inout(reg) x => _,
        n = inout(reg) chunks => _,
        o = in(reg) out.as_mut_ptr(),
        out("q8") _, out("q9") _, out("q10") _, out("q11") _, out("q12") _, out("q13") _,
        options(nostack),
    );
    out
}

/// [`lanes8`] for two inputs against the same `u` (two positions of a batch against one decoded
/// weight group): `u` is loaded once and multiplied into both inputs' accumulators, each input's
/// lanes and order exactly as `lanes8` gives them. 9 instructions per 16 products against 12.
/// The second input goes through q0-q1 (d0-d3), caller-saved.
///
/// # Safety
/// `u`, `x0` and `x1` must each be valid for `8 * chunks` reads, and `chunks >= 1`.
#[inline]
pub(crate) unsafe fn lanes8x2(
    u: *const f32,
    x0: *const f32,
    x1: *const f32,
    chunks: usize,
) -> ([f32; ACC_LANES], [f32; ACC_LANES]) {
    let mut out0 = [0.0f32; ACC_LANES];
    let mut out1 = [0.0f32; ACC_LANES];
    core::arch::asm!(
        "vmov.i32 q12, #0",
        "vmov.i32 q13, #0",
        "vmov.i32 q14, #0",
        "vmov.i32 q15, #0",
        "2:",
        "vld1.32 {{d16-d19}}, [{u}]!",
        "vld1.32 {{d20-d23}}, [{x0}]!",
        "vld1.32 {{d0-d3}}, [{x1}]!",
        "vmla.f32 q12, q8, q10",
        "vmla.f32 q13, q9, q11",
        "vmla.f32 q14, q8, q0",
        "vmla.f32 q15, q9, q1",
        "subs {n}, {n}, #1",
        "bne 2b",
        "vst1.32 {{d24-d27}}, [{o0}]",
        "vst1.32 {{d28-d31}}, [{o1}]",
        u = inout(reg) u => _,
        x0 = inout(reg) x0 => _,
        x1 = inout(reg) x1 => _,
        n = inout(reg) chunks => _,
        o0 = in(reg) out0.as_mut_ptr(),
        o1 = in(reg) out1.as_mut_ptr(),
        out("q0") _, out("q1") _,
        out("q8") _, out("q9") _, out("q10") _, out("q11") _,
        out("q12") _, out("q13") _, out("q14") _, out("q15") _,
        options(nostack),
    );
    (out0, out1)
}

/// [`lanes8`] for four inputs against the same `u`: `u` is loaded once for four positions, each
/// position's lanes and order exactly as [`lanes8`] gives them. The inputs take q2-q3 in turn;
/// the result is four positions' lanes, position-major.
///
/// # Safety
/// `u` and each of `x[0..4]` must be valid for `8 * chunks` reads, and `chunks >= 1`.
#[inline]
pub(crate) unsafe fn lanes8x4(
    u: *const f32,
    x: [*const f32; 4],
    chunks: usize,
) -> [f32; 4 * ACC_LANES] {
    let mut out = [0.0f32; 4 * ACC_LANES];
    core::arch::asm!(
        "vmov.i32 q8, #0",
        "vmov.i32 q9, #0",
        "vmov.i32 q10, #0",
        "vmov.i32 q11, #0",
        "vmov.i32 q12, #0",
        "vmov.i32 q13, #0",
        "vmov.i32 q14, #0",
        "vmov.i32 q15, #0",
        "2:",
        "vld1.32 {{d0-d3}}, [{u}]!",
        "vld1.32 {{d4-d7}}, [{x0}]!",
        "vmla.f32 q8, q0, q2",
        "vmla.f32 q9, q1, q3",
        "vld1.32 {{d4-d7}}, [{x1}]!",
        "vmla.f32 q10, q0, q2",
        "vmla.f32 q11, q1, q3",
        "vld1.32 {{d4-d7}}, [{x2}]!",
        "vmla.f32 q12, q0, q2",
        "vmla.f32 q13, q1, q3",
        "vld1.32 {{d4-d7}}, [{x3}]!",
        "vmla.f32 q14, q0, q2",
        "vmla.f32 q15, q1, q3",
        "subs {n}, {n}, #1",
        "bne 2b",
        "vst1.32 {{d16-d19}}, [{o}]!",
        "vst1.32 {{d20-d23}}, [{o}]!",
        "vst1.32 {{d24-d27}}, [{o}]!",
        "vst1.32 {{d28-d31}}, [{o}]",
        u = inout(reg) u => _,
        x0 = inout(reg) x[0] => _,
        x1 = inout(reg) x[1] => _,
        x2 = inout(reg) x[2] => _,
        x3 = inout(reg) x[3] => _,
        n = inout(reg) chunks => _,
        o = inout(reg) out.as_mut_ptr() => _,
        out("q0") _, out("q1") _, out("q2") _, out("q3") _,
        out("q8") _, out("q9") _, out("q10") _, out("q11") _,
        out("q12") _, out("q13") _, out("q14") _, out("q15") _,
        options(nostack),
    );
    out
}

/// Lanes of the LUT-decoded group dot, four levels per byte (2-bit and ternary records): byte
/// `b` contributes `lut[4b .. 4b + 4]` to four consecutive lanes; two bytes fill the eight.
///
/// # Safety
/// `lut` must hold `256 * 4` floats, `g` `2 * pairs` bytes and `x` `8 * pairs` floats;
/// `pairs >= 1`.
#[inline]
pub(crate) unsafe fn lut4_lanes(
    lut: *const f32,
    g: *const u8,
    x: *const f32,
    pairs: usize,
) -> [f32; ACC_LANES] {
    let mut out = [0.0f32; ACC_LANES];
    core::arch::asm!(
        "vmov.i32 q12, #0",
        "vmov.i32 q13, #0",
        "2:",
        "ldrb {t0}, [{g}], #1",
        "ldrb {t1}, [{g}], #1",
        "add {t0}, {lut}, {t0}, lsl #4",
        "add {t1}, {lut}, {t1}, lsl #4",
        "vld1.32 {{d16-d17}}, [{t0}]",
        "vld1.32 {{d18-d19}}, [{t1}]",
        "vld1.32 {{d20-d23}}, [{x}]!",
        "vmla.f32 q12, q8, q10",
        "vmla.f32 q13, q9, q11",
        "subs {n}, {n}, #1",
        "bne 2b",
        "vst1.32 {{d24-d27}}, [{o}]",
        lut = in(reg) lut,
        g = inout(reg) g => _,
        x = inout(reg) x => _,
        n = inout(reg) pairs => _,
        o = in(reg) out.as_mut_ptr(),
        t0 = out(reg) _,
        t1 = out(reg) _,
        out("q8") _, out("q9") _, out("q10") _, out("q11") _, out("q12") _, out("q13") _,
        options(nostack),
    );
    out
}

/// [`lut4_lanes`] for two rows at once: each row keeps its own two accumulators and its own lane
/// order, so each row's lanes are bit for bit what `lut4_lanes` gives it. The rows share every
/// load of `x` and the loop control (19 instructions per 16 products against 22). The second
/// row's levels go through q0-q1 (d0-d3), caller-saved in the AAPCS.
///
/// # Safety
/// As [`lut4_lanes`], for both `g0` and `g1`.
#[inline]
pub(crate) unsafe fn lut4_lanes2(
    lut: *const f32,
    g0: *const u8,
    g1: *const u8,
    x: *const f32,
    pairs: usize,
) -> ([f32; ACC_LANES], [f32; ACC_LANES]) {
    let mut out0 = [0.0f32; ACC_LANES];
    let mut out1 = [0.0f32; ACC_LANES];
    core::arch::asm!(
        "vmov.i32 q12, #0",
        "vmov.i32 q13, #0",
        "vmov.i32 q14, #0",
        "vmov.i32 q15, #0",
        "2:",
        "ldrb {t0}, [{g0}], #1",
        "ldrb {t1}, [{g0}], #1",
        "add {t0}, {lut}, {t0}, lsl #4",
        "add {t1}, {lut}, {t1}, lsl #4",
        "vld1.32 {{d16-d17}}, [{t0}]",
        "vld1.32 {{d18-d19}}, [{t1}]",
        "ldrb {t0}, [{g1}], #1",
        "ldrb {t1}, [{g1}], #1",
        "add {t0}, {lut}, {t0}, lsl #4",
        "add {t1}, {lut}, {t1}, lsl #4",
        "vld1.32 {{d0-d1}}, [{t0}]",
        "vld1.32 {{d2-d3}}, [{t1}]",
        "vld1.32 {{d20-d23}}, [{x}]!",
        "vmla.f32 q12, q8, q10",
        "vmla.f32 q13, q9, q11",
        "vmla.f32 q14, q0, q10",
        "vmla.f32 q15, q1, q11",
        "subs {n}, {n}, #1",
        "bne 2b",
        "vst1.32 {{d24-d27}}, [{o0}]",
        "vst1.32 {{d28-d31}}, [{o1}]",
        lut = in(reg) lut,
        g0 = inout(reg) g0 => _,
        g1 = inout(reg) g1 => _,
        x = inout(reg) x => _,
        n = inout(reg) pairs => _,
        o0 = in(reg) out0.as_mut_ptr(),
        o1 = in(reg) out1.as_mut_ptr(),
        t0 = out(reg) _,
        t1 = out(reg) _,
        out("q0") _, out("q1") _,
        out("q8") _, out("q9") _, out("q10") _, out("q11") _,
        out("q12") _, out("q13") _, out("q14") _, out("q15") _,
        options(nostack),
    );
    (out0, out1)
}

/// As [`lut4_lanes`] for two levels per byte (4-bit records): four bytes fill the eight lanes.
///
/// # Safety
/// `lut` must hold `256 * 2` floats, `g` `4 * quads` bytes and `x` `8 * quads` floats;
/// `quads >= 1`.
#[inline]
pub(crate) unsafe fn lut2_lanes(
    lut: *const f32,
    g: *const u8,
    x: *const f32,
    quads: usize,
) -> [f32; ACC_LANES] {
    let mut out = [0.0f32; ACC_LANES];
    core::arch::asm!(
        "vmov.i32 q12, #0",
        "vmov.i32 q13, #0",
        "2:",
        "ldrb {t0}, [{g}], #1",
        "ldrb {t1}, [{g}], #1",
        "add {t0}, {lut}, {t0}, lsl #3",
        "add {t1}, {lut}, {t1}, lsl #3",
        "vld1.32 {{d16}}, [{t0}]",
        "vld1.32 {{d17}}, [{t1}]",
        "ldrb {t0}, [{g}], #1",
        "ldrb {t1}, [{g}], #1",
        "add {t0}, {lut}, {t0}, lsl #3",
        "add {t1}, {lut}, {t1}, lsl #3",
        "vld1.32 {{d18}}, [{t0}]",
        "vld1.32 {{d19}}, [{t1}]",
        "vld1.32 {{d20-d23}}, [{x}]!",
        "vmla.f32 q12, q8, q10",
        "vmla.f32 q13, q9, q11",
        "subs {n}, {n}, #1",
        "bne 2b",
        "vst1.32 {{d24-d27}}, [{o}]",
        lut = in(reg) lut,
        g = inout(reg) g => _,
        x = inout(reg) x => _,
        n = inout(reg) quads => _,
        o = in(reg) out.as_mut_ptr(),
        t0 = out(reg) _,
        t1 = out(reg) _,
        out("q8") _, out("q9") _, out("q10") _, out("q11") _, out("q12") _, out("q13") _,
        options(nostack),
    );
    out
}

/// `y[i] += a * x[i]` for `8 * chunks` elements, two q registers at a time.
///
/// Each element gets its own product, rounded, and its own sum, as the scalar loop computes it
/// (VMLA is not fused; the product `x * a` equals `a * x` exactly), so this is the scalar result
/// but for NEON's flush of subnormals to zero.
///
/// # Safety
/// `y` and `x` must each be valid for `8 * chunks` elements and must not overlap; `chunks >= 1`.
#[inline]
pub(crate) unsafe fn axpy8(y: *mut f32, a: f32, x: *const f32, chunks: usize) {
    core::arch::asm!(
        "vdup.32 q14, {a}",
        "2:",
        "vld1.32 {{d16-d19}}, [{x}]!",
        "vld1.32 {{d20-d23}}, [{y}]",
        "vmla.f32 q10, q8, q14",
        "vmla.f32 q11, q9, q14",
        "vst1.32 {{d20-d23}}, [{y}]!",
        "subs {n}, {n}, #1",
        "bne 2b",
        a = in(reg) a.to_bits(),
        y = inout(reg) y => _,
        x = inout(reg) x => _,
        n = inout(reg) chunks => _,
        out("q8") _, out("q9") _, out("q10") _, out("q11") _, out("q14") _,
        options(nostack),
    );
}

/// [`lut2_lanes`] for two rows at once, as [`lut4_lanes2`] is for `lut4_lanes`.
///
/// # Safety
/// As [`lut2_lanes`], for both `g0` and `g1`.
#[inline]
pub(crate) unsafe fn lut2_lanes2(
    lut: *const f32,
    g0: *const u8,
    g1: *const u8,
    x: *const f32,
    quads: usize,
) -> ([f32; ACC_LANES], [f32; ACC_LANES]) {
    let mut out0 = [0.0f32; ACC_LANES];
    let mut out1 = [0.0f32; ACC_LANES];
    core::arch::asm!(
        "vmov.i32 q12, #0",
        "vmov.i32 q13, #0",
        "vmov.i32 q14, #0",
        "vmov.i32 q15, #0",
        "2:",
        "ldrb {t0}, [{g0}], #1",
        "ldrb {t1}, [{g0}], #1",
        "add {t0}, {lut}, {t0}, lsl #3",
        "add {t1}, {lut}, {t1}, lsl #3",
        "vld1.32 {{d16}}, [{t0}]",
        "vld1.32 {{d17}}, [{t1}]",
        "ldrb {t0}, [{g0}], #1",
        "ldrb {t1}, [{g0}], #1",
        "add {t0}, {lut}, {t0}, lsl #3",
        "add {t1}, {lut}, {t1}, lsl #3",
        "vld1.32 {{d18}}, [{t0}]",
        "vld1.32 {{d19}}, [{t1}]",
        "ldrb {t0}, [{g1}], #1",
        "ldrb {t1}, [{g1}], #1",
        "add {t0}, {lut}, {t0}, lsl #3",
        "add {t1}, {lut}, {t1}, lsl #3",
        "vld1.32 {{d0}}, [{t0}]",
        "vld1.32 {{d1}}, [{t1}]",
        "ldrb {t0}, [{g1}], #1",
        "ldrb {t1}, [{g1}], #1",
        "add {t0}, {lut}, {t0}, lsl #3",
        "add {t1}, {lut}, {t1}, lsl #3",
        "vld1.32 {{d2}}, [{t0}]",
        "vld1.32 {{d3}}, [{t1}]",
        "vld1.32 {{d20-d23}}, [{x}]!",
        "vmla.f32 q12, q8, q10",
        "vmla.f32 q13, q9, q11",
        "vmla.f32 q14, q0, q10",
        "vmla.f32 q15, q1, q11",
        "subs {n}, {n}, #1",
        "bne 2b",
        "vst1.32 {{d24-d27}}, [{o0}]",
        "vst1.32 {{d28-d31}}, [{o1}]",
        lut = in(reg) lut,
        g0 = inout(reg) g0 => _,
        g1 = inout(reg) g1 => _,
        x = inout(reg) x => _,
        n = inout(reg) quads => _,
        o0 = in(reg) out0.as_mut_ptr(),
        o1 = in(reg) out1.as_mut_ptr(),
        t0 = out(reg) _,
        t1 = out(reg) _,
        out("q0") _, out("q1") _,
        out("q8") _, out("q9") _, out("q10") _, out("q11") _,
        out("q12") _, out("q13") _, out("q14") _, out("q15") _,
        options(nostack),
    );
    (out0, out1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand(seed: &mut u32, n: usize) -> alloc::vec::Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((*seed >> 8) as f32 / (1u32 << 24) as f32) * 4.0 - 2.0
            })
            .collect()
    }

    fn scalar_lanes(u: &[f32], x: &[f32]) -> [f32; ACC_LANES] {
        let mut l = [0.0f32; ACC_LANES];
        for c in 0..u.len() / ACC_LANES {
            for k in 0..ACC_LANES {
                l[k] += u[c * ACC_LANES + k] * x[c * ACC_LANES + k];
            }
        }
        l
    }

    #[test]
    fn neon_lanes_are_bit_identical_to_the_scalar_order() {
        let mut s = 7u32;
        for n in [8usize, 64, 128, 1024] {
            let (u, x) = (rand(&mut s, n), rand(&mut s, n));
            let got = unsafe { lanes8(u.as_ptr(), x.as_ptr(), n / 8) };
            let want = scalar_lanes(&u, &x);
            assert_eq!(
                got.map(f32::to_bits),
                want.map(f32::to_bits),
                "lanes8 n={n}"
            );
        }
        let lut4 = rand(&mut s, 1024);
        let lut2 = rand(&mut s, 512);
        let g: alloc::vec::Vec<u8> = (0..64u32).map(|i| (i * 37 + 11) as u8).collect();
        let x = rand(&mut s, 256);
        // Four levels per byte: 32 bytes are 128 values.
        let dec4: alloc::vec::Vec<f32> = g[..32]
            .iter()
            .flat_map(|&b| lut4[b as usize * 4..b as usize * 4 + 4].to_vec())
            .collect();
        let got = unsafe { lut4_lanes(lut4.as_ptr(), g.as_ptr(), x.as_ptr(), 16) };
        assert_eq!(
            got.map(f32::to_bits),
            scalar_lanes(&dec4, &x[..128]).map(f32::to_bits),
            "lut4"
        );
        // Two levels per byte: 64 bytes are 128 values.
        let dec2: alloc::vec::Vec<f32> = g
            .iter()
            .flat_map(|&b| lut2[b as usize * 2..b as usize * 2 + 2].to_vec())
            .collect();
        let got = unsafe { lut2_lanes(lut2.as_ptr(), g.as_ptr(), x.as_ptr(), 16) };
        assert_eq!(
            got.map(f32::to_bits),
            scalar_lanes(&dec2, &x[..128]).map(f32::to_bits),
            "lut2"
        );
    }
}
