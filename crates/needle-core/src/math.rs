//! Thin wrappers around libm for no_std float math.

#[inline(always)]
pub fn exp(x: f32) -> f32 {
    libm::expf(x)
}

#[inline(always)]
pub fn ln(x: f32) -> f32 {
    libm::logf(x)
}

#[inline(always)]
pub fn sin(x: f32) -> f32 {
    libm::sinf(x)
}

#[inline(always)]
pub fn cos(x: f32) -> f32 {
    libm::cosf(x)
}

/// Square root, correctly rounded (IEEE 754), so every implementation gives the same bits.
///
/// libm has no 32-bit ARM path and computes it in software there, about a hundred integer
/// instructions per call; hard-float ARM has `vsqrt.f32`, which the C port's `sqrtf` compiles to.
#[inline(always)]
pub fn sqrt(x: f32) -> f32 {
    #[cfg(all(target_arch = "arm", target_abi = "eabihf"))]
    {
        let r: f32;
        // Safety: one VFP instruction on registers; VFP is part of the hard-float ABI.
        unsafe {
            core::arch::asm!(
                "vsqrt.f32 {r}, {x}",
                r = lateout(sreg) r,
                x = in(sreg) x,
                options(pure, nomem, nostack, preserves_flags),
            )
        };
        r
    }
    #[cfg(not(all(target_arch = "arm", target_abi = "eabihf")))]
    libm::sqrtf(x)
}

#[inline(always)]
pub fn tanh(x: f32) -> f32 {
    libm::tanhf(x)
}

#[inline(always)]
pub fn powf(base: f32, exp: f32) -> f32 {
    libm::powf(base, exp)
}

#[inline(always)]
pub fn round(x: f32) -> f32 {
    libm::roundf(x)
}

#[inline(always)]
pub fn abs(x: f32) -> f32 {
    libm::fabsf(x)
}

/// Convert IEEE 754 half-precision (binary16) bits to `f32`.
///
/// Exact for every input, including subnormals, infinities and NaN payloads.
/// Lives here so the `no_std` v2 loader and `cq` can share it.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        // Subnormal (or zero): value = mant * 2^-24.
        let val = mant as f32 / (1u32 << 24) as f32;
        return if sign != 0 { -val } else { val };
    }
    if exp == 31 {
        // Inf or NaN — preserve the mantissa payload.
        return f32::from_bits(sign | 0x7F80_0000 | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
}

#[cfg(test)]
mod f16_tests {
    use super::f16_to_f32;

    #[test]
    fn known_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xBC00), -1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0x3555), 0.33325195); // nearest half to 1/3
        assert_eq!(f16_to_f32(0x7BFF), 65504.0); // largest finite half
        assert_eq!(f16_to_f32(0x0001), 5.9604645e-8); // smallest subnormal
        assert_eq!(f16_to_f32(0x03FF), 6.0975552e-5); // largest subnormal
        assert_eq!(f16_to_f32(0x0400), 6.1035156e-5); // smallest normal
        assert!(f16_to_f32(0x7C00).is_infinite() && f16_to_f32(0x7C00) > 0.0);
        assert!(f16_to_f32(0xFC00).is_infinite() && f16_to_f32(0xFC00) < 0.0);
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    /// Round-trip against the f32 -> f16 direction for every representable half.
    #[test]
    fn every_finite_half_round_trips() {
        for bits in 0u32..=0xFFFF {
            let b = bits as u16;
            let v = f16_to_f32(b);
            if v.is_nan() {
                continue;
            }
            // f16 -> f32 is lossless, so re-narrowing must return the same bits.
            let back = f32_to_f16_bits(v);
            assert_eq!(back, b, "bits={b:#06x} value={v}");
        }
    }

    /// Round-to-nearest-even narrowing, test-only reference.
    fn f32_to_f16_bits(v: f32) -> u16 {
        let x = v.to_bits();
        let sign = ((x >> 31) as u16) << 15;
        let exp = ((x >> 23) & 0xFF) as i32;
        let mant = x & 0x7F_FFFF;
        if exp == 0xFF {
            return sign | 0x7C00 | ((mant >> 13) as u16);
        }
        let unbiased = exp - 127;
        if unbiased < -24 {
            return sign;
        }
        if unbiased < -14 {
            // Subnormal half: shift the implicit 1 back in.
            let shift = (-14 - unbiased) as u32;
            let m = (mant | 0x80_0000) >> (shift + 13);
            return sign | m as u16;
        }
        if unbiased > 15 {
            return sign | 0x7C00;
        }
        sign | (((unbiased + 15) as u16) << 10) | ((mant >> 13) as u16)
    }
}
