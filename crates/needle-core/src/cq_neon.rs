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
