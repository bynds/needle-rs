//! Opt-in per-operation cost accounting (feature `profile`).
//!
//! With the feature off, `span` returns a zero-sized guard and compiles to nothing. With it on,
//! each guard adds the clock's advance between its creation and drop to its operation's total.
//! The clock is supplied by the caller (`set_clock`): wall-clock nanoseconds on a host, or
//! instructions retired from the PMU in the perfvm guest, where counts are exact. The C99 port
//! (ports/jibo/c, `-DND_PROFILE`) uses the same operations at the same points, so the two engines'
//! breakdowns compare line by line.

/// Where the cost goes. Spans do not nest; together they cover the forward pass and decode step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Op {
    /// Token embedding rows.
    Embed,
    /// Engram: hashing, table rows, key/value projections, value convolution.
    Engram,
    /// mHC: lane norms, the phi projections, mix-down and scatter-up.
    Mhc,
    /// norm_in and the Q/K/V projections.
    Qkv,
    /// The Q/K/V causal convolution, Q/K norms and RoPE (and the int8 query quantisation).
    ConvRope,
    /// Attention, including writing the cache.
    Attn,
    /// The gate and output projections, post_norm and the residual.
    GateOut,
    /// pre_hada and the Hadamard MLP.
    Mlp,
    /// The final norm and the tied-embedding logits head.
    Head,
}

pub const OPS: usize = 9;
pub const NAMES: [&str; OPS] = [
    "embed",
    "engram",
    "mhc",
    "qkv",
    "conv_rope",
    "attn",
    "gate_out",
    "mlp",
    "head",
];

#[cfg(feature = "profile")]
mod imp {
    use super::{Op, OPS};
    use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    static CLOCK: AtomicUsize = AtomicUsize::new(0);
    static TOTALS: [AtomicU64; OPS] = [const { AtomicU64::new(0) }; OPS];
    static CALLS: [AtomicU64; OPS] = [const { AtomicU64::new(0) }; OPS];

    /// The clock spans read. Until it is set, spans record nothing.
    pub fn set_clock(f: fn() -> u64) {
        CLOCK.store(f as usize, Ordering::Relaxed);
    }

    fn now() -> Option<u64> {
        let p = CLOCK.load(Ordering::Relaxed);
        // Safety: only `set_clock` stores here, and it stores a `fn() -> u64`.
        (p != 0).then(|| unsafe { core::mem::transmute::<usize, fn() -> u64>(p) }())
    }

    pub struct Span {
        op: Op,
        start: Option<u64>,
    }

    #[inline]
    pub fn span(op: Op) -> Span {
        Span { op, start: now() }
    }

    impl Drop for Span {
        fn drop(&mut self) {
            if let (Some(s), Some(e)) = (self.start, now()) {
                TOTALS[self.op as usize].fetch_add(e.wrapping_sub(s), Ordering::Relaxed);
                CALLS[self.op as usize].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Totals and call counts per operation since the last reset, and reset them.
    pub fn take() -> [(u64, u64); OPS] {
        core::array::from_fn(|i| {
            (
                TOTALS[i].swap(0, Ordering::Relaxed),
                CALLS[i].swap(0, Ordering::Relaxed),
            )
        })
    }
}

#[cfg(not(feature = "profile"))]
mod imp {
    use super::{Op, OPS};

    pub struct Span;

    // Empty, so ending a span with `drop` reads the same with the feature on or off.
    impl Drop for Span {
        #[inline(always)]
        fn drop(&mut self) {}
    }

    #[inline(always)]
    pub fn span(_: Op) -> Span {
        Span
    }

    pub fn set_clock(_: fn() -> u64) {}

    pub fn take() -> [(u64, u64); OPS] {
        [(0, 0); OPS]
    }
}

pub use imp::{set_clock, span, take, Span};
