//! The single-field corruptions of `tests/container_robustness.rs`, one line per case:
//! `<case>\t<ok|err>`, for the C99 port's `tests/test_corrupt.c` to replay and match.
//!
//!   cargo run --release -p needle-jibo --example corrupt_cases -- weights/needle3.cact > verdicts.tsv
//!
//! With `--forward`, each accepted container also runs a 4-token prefill, and a panic there is
//! reported as `ok-panics`: a container the loader accepts but the first request would abort on.
//! An allocation failure aborts even an unwinding build; `--from N` resumes after case N - 1.

use needle_infer::v3_engine::V3Engine;

const HEADER_WORDS: usize = 49;
const CODEBOOK_LEN: usize = 28;
const REC: usize = 44;
const EXTREMES: [u32; 6] = [0, 1, 0x0000_FFFF, 0x4000_0000, 0x7FFF_FFFF, 0xFFFF_FFFF];

fn put32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn put64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

static SEEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn case(what: String, bytes: Vec<u8>) {
    let i = SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let from = std::env::args()
        .skip_while(|a| a != "--from")
        .nth(1)
        .map_or(0, |n| n.parse().expect("--from N"));
    if i < from {
        return;
    }
    let forward = std::env::args().any(|a| a == "--forward");
    let verdict = match V3Engine::from_bytes(bytes) {
        Err(_) => "err",
        Ok(_) if !forward => "ok",
        Ok(e) => {
            let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut cache = needle_core::v3::V3Cache::new(&e.model.cfg, 8);
                e.model.prefill(&[2, 10, 20, 30], &mut cache).len()
            }));
            if run.is_ok() {
                "ok"
            } else {
                "ok-panics"
            }
        }
    };
    println!("{what}\t{verdict}");
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let path = std::env::args().nth(1).expect("MODEL");
    let base = std::fs::read(&path).expect("read");
    let num_tensors = u32::from_le_bytes(base[4..8].try_into().unwrap()) as usize;
    let dir = HEADER_WORDS * 4 + CODEBOOK_LEN * 4;
    for w in 1..HEADER_WORDS - 1 {
        for &v in &EXTREMES {
            let mut b = base.clone();
            put32(&mut b, w * 4, v);
            case(format!("header word {w} = {v:#x}"), b);
        }
    }
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
                case(format!("record {r} {field} = {v:#x}"), b);
            }
        }
        for (field, off) in [("offset", 20), ("nbytes", 28)] {
            for v in [0u64, 1, base.len() as u64 - 1, u32::MAX as u64, u64::MAX] {
                let mut b = base.clone();
                put64(&mut b, at + off, v);
                case(format!("record {r} {field} = {v:#x}"), b);
            }
        }
        let mut b = base.clone();
        b[at] = 0xEE;
        case(format!("record {r} dtype = 0xEE"), b);
    }
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
            case(format!("tokenizer {field} = {v:#x}"), b);
        }
    }
    for v in [0u16, 0xFFFF] {
        let mut b = base.clone();
        b[tok_off + 24 + 5..tok_off + 24 + 7].copy_from_slice(&v.to_le_bytes());
        case(format!("tokenizer piece 0 len = {v:#x}"), b);
    }
    for cut in [
        0,
        3,
        100,
        dir - 1,
        dir + REC * 3 + 5,
        base.len() / 2,
        base.len() - 1,
    ] {
        case(format!("truncated to {cut}"), base[..cut].to_vec());
    }
    case("untouched".into(), base);
}
