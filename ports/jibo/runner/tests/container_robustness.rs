//! A damaged or hostile container must be refused, not crash the process.
//!
//! Release builds abort on panic, so on the robot a panic in the loader is a dead service, not an
//! error message. This corrupts one field at a time of the real `needle3.cact` (header words,
//! directory records, the tokenizer blob, truncations) and requires every load to come back as
//! `Ok` or `Err`. Run it under ARMv7 emulation too: 32-bit `usize` is where an unchecked product
//! of two header fields wraps instead of overflowing visibly.
//!
//! Needs `weights/needle3.cact` (or `NEEDLE_JIBO_CACT`). Without it the test skips loudly, and
//! fails if `NEEDLE_JIBO_REQUIRE_FIXTURES=1`, which the release gate sets.

use needle_infer::v3_engine::V3Engine;
use std::panic::{catch_unwind, AssertUnwindSafe};

const HEADER_WORDS: usize = 49;
const CODEBOOK_LEN: usize = 28;
const REC: usize = 44;

fn container() -> Option<Vec<u8>> {
    let path = std::env::var("NEEDLE_JIBO_CACT").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../../weights/needle3.cact").to_string()
    });
    match std::fs::read(&path) {
        Ok(b) => Some(b),
        Err(e) => {
            assert!(
                std::env::var("NEEDLE_JIBO_REQUIRE_FIXTURES").as_deref() != Ok("1"),
                "required fixture missing: {path}: {e}"
            );
            eprintln!("SKIPPING container robustness: {path}: {e}");
            None
        }
    }
}

fn put32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn put64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

/// Load `bytes`; a panic is the failure. Returns whether the load succeeded.
fn survives(what: &str, bytes: Vec<u8>, panics: &mut Vec<String>) -> bool {
    // An allocation failure aborts rather than unwinds; this line names the case that did it.
    if std::env::var_os("NEEDLE_JIBO_TRACE").is_some() {
        eprintln!("case: {what}");
    }
    match catch_unwind(AssertUnwindSafe(|| V3Engine::from_bytes(bytes).is_ok())) {
        Ok(ok) => ok,
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            panics.push(format!("{what}: {msg}"));
            false
        }
    }
}

const EXTREMES: [u32; 6] = [0, 1, 0x0000_FFFF, 0x4000_0000, 0x7FFF_FFFF, 0xFFFF_FFFF];

#[test]
fn every_single_field_corruption_is_refused_or_loaded_without_panic() {
    let Some(base) = container() else { return };
    // Quiet the default hook: each caught panic would otherwise print a backtrace.
    std::panic::set_hook(Box::new(|_| {}));
    let mut panics = Vec::new();
    let num_tensors = u32::from_le_bytes(base[4..8].try_into().unwrap()) as usize;
    let dir = HEADER_WORDS * 4 + CODEBOOK_LEN * 4;

    // Header geometry: every word but the tag and rope_theta.
    for w in 1..HEADER_WORDS - 1 {
        for &v in &EXTREMES {
            let mut b = base.clone();
            put32(&mut b, w * 4, v);
            survives(&format!("header word {w} = {v:#x}"), b, &mut panics);
        }
    }

    // Directory: the first records (embedding, layer 0), the mHC/engram region and the last
    // (tokenizer). Fields: shape[0..4], offset, nbytes, group, bits.
    let mut recs: Vec<usize> = (0..30).collect();
    recs.extend([
        num_tensors / 2,
        num_tensors - 12,
        num_tensors - 2,
        num_tensors - 1,
    ]);
    for &r in &recs {
        let at = dir + r * REC;
        for (field, off) in [("shape0", 4), ("shape1", 8), ("group", 36), ("bits", 40)] {
            for &v in &EXTREMES {
                let mut b = base.clone();
                put32(&mut b, at + off, v);
                survives(&format!("record {r} {field} = {v:#x}"), b, &mut panics);
            }
        }
        for (field, off) in [("offset", 20), ("nbytes", 28)] {
            for v in [0u64, 1, base.len() as u64 - 1, u32::MAX as u64, u64::MAX] {
                let mut b = base.clone();
                put64(&mut b, at + off, v);
                survives(&format!("record {r} {field} = {v:#x}"), b, &mut panics);
            }
        }
        let mut b = base.clone();
        b[at] = 0xEE; // dtype
        survives(&format!("record {r} dtype = 0xEE"), b, &mut panics);
    }

    // The tokenizer blob's own header: n_pieces, the special ids, the first piece length.
    let tok = dir + (num_tensors - 1) * REC;
    let tok_off = u64::from_le_bytes(base[tok + 20..tok + 28].try_into().unwrap()) as usize;
    for (field, off) in [
        ("n_pieces", 0),
        ("pad", 4),
        ("eos", 8),
        ("bos", 12),
        ("unk", 16),
    ] {
        for &v in &EXTREMES {
            let mut b = base.clone();
            put32(&mut b, tok_off + off, v);
            survives(&format!("tokenizer {field} = {v:#x}"), b, &mut panics);
        }
    }
    for v in [0u16, 0xFFFF] {
        let mut b = base.clone();
        b[tok_off + 24 + 5..tok_off + 24 + 7].copy_from_slice(&v.to_le_bytes());
        survives(&format!("tokenizer piece 0 len = {v:#x}"), b, &mut panics);
    }

    // Truncations, including inside the header, codebook, directory and last blob.
    for cut in [
        0,
        3,
        100,
        dir - 1,
        dir + REC * 3 + 5,
        base.len() / 2,
        base.len() - 1,
    ] {
        survives(
            &format!("truncated to {cut}"),
            base[..cut].to_vec(),
            &mut panics,
        );
    }

    let _ = std::panic::take_hook();
    assert!(
        panics.is_empty(),
        "{} corruptions panicked instead of being refused:\n{}",
        panics.len(),
        panics.join("\n")
    );
    // The untouched container still loads.
    assert!(V3Engine::from_bytes(base).is_ok());
}
