//! Needle 3 on the original Jibo: the runner's library half (catalogue, validation, service),
//! kept separate from `main.rs` so it can be tested on the host and under ARMv7 emulation.

pub mod catalog;
pub mod service;
pub mod sha256;
pub mod sysinfo;
pub mod validate;
