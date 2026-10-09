//! Per-operation cost of the v3 engine on three workloads, for the Rust/C99 performance work.
//!
//!   cargo run --release -p needle-infer --features profile --example v3_profile -- \
//!       MODEL TOOLS.json QUERY [--decode N] [--clock instructions|ns]
//!
//! Workloads, the same in `ports/jibo/c/tests/bench_engine.c`:
//!   prefill     one batched prefill of the prompt into a fresh cache
//!   decode      N greedy decode steps after it
//!   prefix_hit  what a tool-prefix cache hit costs: copy the cache after `</tools>`, then step
//!               the rest of the prompt
//! One JSON line each: the total, and per operation (needle_core::prof) its cost and span count.
//! `instructions` is user-space instructions retired, from the PMU (exact under perfvm's
//! `-icount`); `ns` is wall-clock nanoseconds. The default is instructions when the PMU answers.

use needle_core::prof;
use needle_core::v3::{KvPrecision, V3Cache};
use needle_infer::v3_engine::V3Engine;
use std::sync::OnceLock;

static PMU: OnceLock<i32> = OnceLock::new();

fn open_pmu() -> i32 {
    // struct perf_event_attr, PERF_TYPE_HARDWARE / PERF_COUNT_HW_INSTRUCTIONS, user space only.
    let mut attr = [0u8; 128];
    attr[4..8].copy_from_slice(&128u32.to_ne_bytes()); // size
    attr[8..16].copy_from_slice(&1u64.to_ne_bytes()); // config: instructions
    let flags: u64 = (1 << 5) | (1 << 6); // exclude_kernel, exclude_hv
    attr[40..48].copy_from_slice(&flags.to_ne_bytes());
    // Safety: a valid attr buffer of its declared size; pid 0, cpu -1, no group, no flags.
    unsafe { libc::syscall(libc::SYS_perf_event_open, attr.as_ptr(), 0, -1, -1, 0) as i32 }
}

fn instructions() -> u64 {
    let fd = *PMU.get().expect("pmu");
    let mut v = 0u64;
    // Safety: reads 8 bytes into a u64.
    let n = unsafe { libc::read(fd, (&mut v as *mut u64).cast(), 8) };
    if n == 8 {
        v
    } else {
        0
    }
}

fn nanos() -> u64 {
    static T0: OnceLock<std::time::Instant> = OnceLock::new();
    T0.get_or_init(std::time::Instant::now).elapsed().as_nanos() as u64
}

fn report(workload: &str, clock: &str, total: u64, extra: &str) {
    let ops = prof::take();
    let mut fields = Vec::new();
    for (i, (t, c)) in ops.iter().enumerate() {
        fields.push(format!("\"{}\":[{t},{c}]", prof::NAMES[i]));
    }
    println!(
        "{{\"engine\":\"rust\",\"workload\":\"{workload}\",\"clock\":\"{clock}\",\"total\":{total},{extra}\"ops\":{{{}}}}}",
        fields.join(",")
    );
}

fn argmax(v: &[f32]) -> u32 {
    let mut b = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[b] {
            b = i;
        }
    }
    b as u32
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (model, tools, query) = (&a[1], &a[2], &a[3]);
    let mut steps = 32;
    let mut want = "auto".to_string();
    let mut i = 4;
    while i + 1 < a.len() {
        match a[i].as_str() {
            "--decode" => steps = a[i + 1].parse().expect("--decode N"),
            "--clock" => want = a[i + 1].clone(),
            k => panic!("unknown {k}"),
        }
        i += 2;
    }
    let fd = if want == "ns" { -1 } else { open_pmu() };
    PMU.set(fd).unwrap();
    let (clock, now): (&str, fn() -> u64) = if fd >= 0 {
        ("instructions", instructions)
    } else {
        assert!(want != "instructions", "no PMU: perf_event_open failed");
        ("ns", nanos)
    };
    prof::set_clock(now);

    let tools_json = std::fs::read_to_string(tools).expect("tools");
    let e = V3Engine::load(model).expect("load");
    let m = &e.model;
    let ids = e.prompt_ids(query, &tools_json, None);
    prof::take();

    let mut cache = V3Cache::with_precision(&m.cfg, ids.len() + steps, KvPrecision::F32);
    let t0 = now();
    let mut logits = m.prefill(&ids, &mut cache);
    report(
        "prefill",
        clock,
        now() - t0,
        &format!("\"tokens\":{},", ids.len()),
    );

    let t0 = now();
    for _ in 0..steps {
        logits = m.decode_step(&mut cache, argmax(&logits));
    }
    report("decode", clock, now() - t0, &format!("\"tokens\":{steps},"));

    let end = e.tokenizer.id_of("</tools>").expect("marker");
    let split = ids.iter().position(|&t| t == end).expect("prefix") + 1;
    let mut stored = V3Cache::with_precision(&m.cfg, split, KvPrecision::F32);
    m.prefill(&ids[..split], &mut stored);
    prof::take();
    let t0 = now();
    let mut c = stored.clone();
    for &t in &ids[split..] {
        logits = m.decode_step(&mut c, t);
    }
    report(
        "prefix_hit",
        clock,
        now() - t0,
        &format!("\"tokens\":{},\"reused\":{split},", ids.len() - split),
    );
    // Keep the result live, and give a run-to-run identity check.
    eprintln!("argmax {}", argmax(&logits));
}
