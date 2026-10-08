//! math_ref.rs: reference generator for tests/test_math.c.
//!
//! Computes the Rust `libm` 0.2.16 results that nd_math.c must reproduce bit for bit, with the
//! same libm configuration as needle-core (default features off; this crate's Cargo.toml pins
//! it). Built as part of the nd-c-ref crate (ports/jibo/c/ref); scripts/c-check.sh runs it,
//! for the target under test (x86_64 natively; armv7-unknown-linux-gnueabihf under
//! qemu-arm). Modes:
//!
//!   math_ref cases OUT [small]   input/output records for every function (see below)
//!   math_ref exh OUT FUNC        all 2^32 inputs of a unary function, as 65536 block hashes
//!
//! Byte formats (little endian):
//!   cases: "NDMC", then records { u32 func, u32 a_bits, u32 b_bits, u32 out_bits }
//!   exh:   "NDMX", u32 func, then 65536 x u64: FNV-1a-style hash of the 65536 outputs of block
//!          k (inputs k<<16 .. k<<16|0xffff, in order), each NaN output canonicalised to 0x7fc00000
//!   func:  0 expf, 1 logf, 2 sinf, 3 cosf, 4 powf, 5 tanhf
//!
//! A cases record is written for every input; test_math.c recomputes and compares (NaNs equal).

use std::fs::File;
use std::io::{BufWriter, Write};

fn unary(f: u32) -> fn(f32) -> f32 {
    match f {
        0 => libm::expf,
        1 => libm::logf,
        2 => libm::sinf,
        3 => libm::cosf,
        5 => libm::tanhf,
        _ => panic!("not a unary function: {f}"),
    }
}

fn canon(v: f32) -> u32 {
    if v.is_nan() {
        0x7fc0_0000
    } else {
        v.to_bits()
    }
}

fn block_hash(f: fn(f32) -> f32, block: u32) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for lo in 0..=0xffffu32 {
        let x = f32::from_bits(block << 16 | lo);
        h ^= canon(f(x)) as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn u32(&mut self) -> u32 {
        (self.next() >> 32) as u32
    }
    /// uniform f32 in [lo, hi]
    fn range(&mut self, lo: f64, hi: f64) -> f32 {
        let u = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        (lo + (hi - lo) * u) as f32
    }
}

fn specials() -> Vec<f32> {
    let mut v = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        -2.0,
        3.0,
        -3.0,
        10.0,
        -10.0,
        100.0,
        1e6,
        -1e6,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7f80_0001),
        f32::from_bits(0xffc1_2345),
        f32::MAX,
        -f32::MAX,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::from_bits(0x007f_ffff),
        f32::from_bits(0x807f_ffff),
        f32::from_bits(0x0040_0000),
        f32::EPSILON,
        std::f32::consts::PI,
        -std::f32::consts::PI,
        std::f32::consts::FRAC_PI_2,
        std::f32::consts::FRAC_PI_4,
        std::f32::consts::E,
        std::f32::consts::LN_2,
        88.72284,
        88.72283,
        -87.33655,
        -103.97208,
        -103.97209,
        0.34657359,
        1.03972077,
        27.0 * 0.6931472,
        9.0,
        10.0001,
        0.5493,
        0.2554,
        1e-30,
        -1e-30,
        1.0e30,
        16777216.0,
        8388608.0,
        2.5,
        -2.5,
        1.0000001,
        0.99999994,
        105414350.0,
        421657420.0,
        1e38,
        -1e38,
        100000.0,
    ];
    // every branch threshold of the six functions, and its neighbours
    for t in [
        0x42aeac50u32,
        0x42b17218,
        0x42cff1b5,
        0x3eb17218,
        0x3f851592,
        0x39000000,
        0x3f3504f3,
        0x3f490fda,
        0x39800000,
        0x407b53d1,
        0x4016cbe3,
        0x40e231d5,
        0x40afeddf,
        0x4dc90fdb,
        0x3f0c9f54,
        0x41200000,
        0x3e82c578,
        0x4195b844,
        0x42b17217,
        0x33000000,
        0x3f7ffff8,
        0x3f800007,
        0x4d000000,
        0x4b800000,
        0x43000000,
        0x43160000,
        0x00800000,
    ] {
        for d in [-2i32, -1, 0, 1, 2] {
            let b = (t as i32 + d) as u32;
            v.push(f32::from_bits(b));
            v.push(-f32::from_bits(b));
        }
    }
    v
}

fn cases(out: &str, small: bool) {
    let mut w = BufWriter::new(File::create(out).unwrap());
    w.write_all(b"NDMC").unwrap();
    let mut count = [0u64; 6];
    let mut put = |w: &mut BufWriter<File>, f: u32, a: f32, b: f32| {
        let r = if f == 4 {
            libm::powf(a, b)
        } else {
            unary(f)(a)
        };
        for v in [f, a.to_bits(), b.to_bits(), r.to_bits()] {
            w.write_all(&v.to_le_bytes()).unwrap();
        }
        count[f as usize] += 1;
    };
    let scale = if small { 1 } else { 4 };
    let nrand = 500_000 * scale;
    let nsweep = 250_000 * scale;
    let mut rng = Rng(0x6e64_5f6d_6174_6821);
    let sp = specials();

    for f in [0u32, 1, 2, 3, 5] {
        for &x in &sp {
            put(&mut w, f, x, 0.0);
        }
        for _ in 0..nrand {
            let x = f32::from_bits(rng.u32());
            put(&mut w, f, x, 0.0);
        }
    }
    let lin = |lo: f64, hi: f64, i: usize, n: usize| (lo + (hi - lo) * i as f64 / n as f64) as f32;
    // expf on [-100, 100]
    for i in 0..=nsweep {
        put(&mut w, 0, lin(-100.0, 100.0, i, nsweep), 0.0);
    }
    for _ in 0..nsweep {
        let x = rng.range(-100.0, 100.0);
        put(&mut w, 0, x, 0.0);
    }
    // logf on (0, 1e6]: linear, log-spaced down to the smallest subnormal, random
    for i in 1..=nsweep {
        put(&mut w, 1, lin(0.0, 1e6, i, nsweep), 0.0);
    }
    for i in 0..=nsweep {
        let e = -149.0 + (19.94 + 149.0) * i as f64 / nsweep as f64;
        put(&mut w, 1, (2f64.powf(e)) as f32, 0.0);
    }
    for _ in 0..nsweep {
        let x = rng.range(0.0, 1e6);
        put(&mut w, 1, x, 0.0);
    }
    // sinf, cosf on [-10000, 10000], plus large arguments (rem_pio2_large)
    for f in [2u32, 3] {
        for i in 0..=2 * nsweep {
            put(&mut w, f, lin(-10000.0, 10000.0, i, 2 * nsweep), 0.0);
        }
        for _ in 0..nsweep {
            let x = rng.range(-10000.0, 10000.0);
            put(&mut w, f, x, 0.0);
        }
        for _ in 0..nsweep {
            // |x| >= 0x4dc90fdb (about 4.2e8) up to FLT_MAX
            let b = 0x4dc9_0fdb + rng.u32() % (0x7f80_0000 - 0x4dc9_0fdb);
            let s = rng.u32() & 0x8000_0000;
            put(&mut w, f, f32::from_bits(b | s), 0.0);
        }
    }
    // tanhf on [-20, 20]
    for i in 0..=nsweep {
        put(&mut w, 5, lin(-20.0, 20.0, i, nsweep), 0.0);
    }
    for _ in 0..nsweep {
        let x = rng.range(-20.0, 20.0);
        put(&mut w, 5, x, 0.0);
    }
    // powf: specials crossed, RoPE (base 1e5 / 1e4 / 5e5 / 1e6, exponents k/d and dense [0, 1]),
    // random bit pairs, random positive bases with moderate exponents
    for &a in &sp {
        for &b in &sp {
            put(&mut w, 4, a, b);
        }
    }
    for base in [100000.0f32, 10000.0, 500000.0, 1000000.0, 1.0e5 + 1.0] {
        for d in [16usize, 32, 48, 64, 80, 96, 128, 160, 192, 256, 512] {
            for k in 0..=d {
                put(&mut w, 4, base, k as f32 / d as f32);
                put(&mut w, 4, base, -(k as f32 / d as f32));
                put(&mut w, 4, base, (2 * k) as f32 / d as f32);
            }
        }
        for i in 0..=nsweep {
            put(&mut w, 4, base, lin(0.0, 1.0, i, nsweep));
        }
    }
    for _ in 0..nsweep {
        let y = rng.range(-1.0, 1.0);
        put(&mut w, 4, 100000.0, y);
    }
    for _ in 0..2 * nrand {
        let (a, b) = (f32::from_bits(rng.u32()), f32::from_bits(rng.u32()));
        put(&mut w, 4, a, b);
    }
    for _ in 0..nrand {
        let a = rng.range(0.0, 1e6);
        let b = rng.range(-10.0, 10.0);
        put(&mut w, 4, a, b);
    }
    for _ in 0..nrand {
        // near-1 bases, huge exponents (|y| > 2^27 branch) and integer exponents of negatives
        let a = f32::from_bits(0x3f7f_fff0 + rng.u32() % 0x20)
            * if rng.u32() & 1 == 0 { 1.0 } else { -1.0 };
        let b = if rng.u32() & 1 == 0 {
            f32::from_bits(0x4d00_0000 + rng.u32() % 0x0100_0000)
        } else {
            (rng.u32() % 401) as f32 - 200.0
        };
        put(&mut w, 4, a, b);
    }
    w.flush().unwrap();
    eprintln!(
        "cases {out}: exp {} log {} sin {} cos {} pow {} tanh {}",
        count[0], count[1], count[2], count[3], count[4], count[5]
    );
}

fn exh(out: &str, f: u32) {
    let func = unary(f);
    let nthreads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1) as u32;
    let mut hashes = vec![0u64; 65536];
    std::thread::scope(|s| {
        let chunks: Vec<_> = hashes.chunks_mut((65536 / nthreads) as usize + 1).collect();
        let mut start = 0u32;
        for c in chunks {
            let len = c.len() as u32;
            let st = start;
            s.spawn(move || {
                for (i, h) in c.iter_mut().enumerate() {
                    *h = block_hash(func, st + i as u32);
                }
            });
            start += len;
        }
    });
    let mut w = BufWriter::new(File::create(out).unwrap());
    w.write_all(b"NDMX").unwrap();
    w.write_all(&f.to_le_bytes()).unwrap();
    for h in hashes {
        w.write_all(&h.to_le_bytes()).unwrap();
    }
    w.flush().unwrap();
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(|s| s.as_str()) {
        Some("cases") => cases(&a[2], a.get(3).map(|s| s == "small").unwrap_or(false)),
        Some("exh") => exh(&a[2], a[3].parse().unwrap()),
        _ => {
            eprintln!("usage: math_ref cases OUT [small] | math_ref exh OUT FUNC");
            std::process::exit(2);
        }
    }
}
