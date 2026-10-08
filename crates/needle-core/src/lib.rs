#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::excessive_precision)]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod attn;
pub mod config;
pub mod cq;
#[cfg(all(target_arch = "arm", feature = "neon"))]
mod cq_neon;
pub mod ffn;
pub mod hadamard;
pub mod kernels;
pub mod layers;
pub mod math;
pub mod model;
pub mod norm;
pub mod ops;
pub mod quant;
pub mod rope;
pub mod v2;
pub mod v3;

pub use config::TransformerConfig;
