//! Reference traces for the C99 engine (`ports/jibo/c`): every stage the C port must reproduce
//! bit for bit, computed by needle-core itself.
//!
//!   cargo run --release -p needle-jibo --example trace -- \
//!       --model weights/needle3.cact --tools T.json --query "..." --out DIR [--steps N] [--depth D]
//!
//! Writes into DIR (all little-endian):
//!   ids.bin        u32 prompt token ids (BOS included), as V3Engine::prompt_ids
//!   cells.bin      f32 [seq, layers + 1, d_model], forward_cells
//!   seq.bin        f32 [seq, rows], forward_sequence
//!   prefill.bin    f32 [rows], prefill into an f32 cache
//!   decode.bin     f32 [steps, rows], greedy decode steps after that prefill (f32 cache)
//!   decode_i8.bin  f32 [steps + 1, rows], prefill and the same steps with an int8 cache
//!   steps.bin      u32 [steps], the greedy tokens fed to the decode steps
//!   head.bin       f32 [out_dim], forward_head with the confidence head (absent without one)
//!   trace.txt      key=value summary

use needle_core::v3::{KvPrecision, V3Cache};
use needle_infer::cact::CactV3;
use needle_infer::v3_engine::V3Engine;
use std::io::Write;
use std::path::{Path, PathBuf};

fn write_f32(path: &Path, v: &[f32]) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("create"));
    for x in v {
        f.write_all(&x.to_le_bytes()).expect("write");
    }
}

fn write_u32(path: &Path, v: &[u32]) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("create"));
    for x in v {
        f.write_all(&x.to_le_bytes()).expect("write");
    }
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

fn main() {
    let mut model = PathBuf::from("weights/needle3.cact");
    let (mut tools, mut query, mut out) = (None, String::new(), None);
    let (mut steps, mut depth) = (8usize, 0usize);
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let v = it.next().unwrap_or_else(|| panic!("{k} needs a value"));
        match k.as_str() {
            "--model" => model = v.into(),
            "--tools" => tools = Some(PathBuf::from(v)),
            "--query" => query = v,
            "--out" => out = Some(PathBuf::from(v)),
            "--steps" => steps = v.parse().expect("--steps"),
            "--depth" => depth = v.parse().expect("--depth"),
            _ => panic!("unknown flag {k}"),
        }
    }
    let out = out.expect("--out DIR");
    let tools_json = std::fs::read_to_string(tools.expect("--tools FILE")).expect("tools");
    std::fs::create_dir_all(&out).expect("mkdir");

    let cact = CactV3::load(&model).expect("load");
    let eng = if depth == 0 {
        V3Engine::from_cact(&cact)
    } else {
        V3Engine::from_cact_at_depth(&cact, depth)
    }
    .expect("engine");
    let m = &eng.model;
    let ids = eng.prompt_ids(&query, &tools_json, None);
    let rows = m.cfg.logit_rows();
    write_u32(&out.join("ids.bin"), &ids);
    write_f32(&out.join("cells.bin"), &m.forward_cells(&ids));
    write_f32(&out.join("seq.bin"), &m.forward_sequence(&ids));

    let mut cache = V3Cache::with_precision(&m.cfg, ids.len() + steps, KvPrecision::F32);
    let pre = m.prefill(&ids, &mut cache);
    write_f32(&out.join("prefill.bin"), &pre);
    let (mut fed, mut dec) = (Vec::new(), Vec::new());
    let mut tok = argmax(&pre);
    for _ in 0..steps {
        fed.push(tok);
        let l = m.decode_step(&mut cache, tok);
        tok = argmax(&l);
        dec.extend_from_slice(&l);
    }
    write_u32(&out.join("steps.bin"), &fed);
    write_f32(&out.join("decode.bin"), &dec);

    let mut c8 = V3Cache::with_precision(&m.cfg, ids.len() + steps, KvPrecision::Int8);
    let mut d8 = m.prefill(&ids, &mut c8);
    for &t in &fed {
        d8.extend_from_slice(&m.decode_step(&mut c8, t));
    }
    write_f32(&out.join("decode_i8.bin"), &d8);

    let mut head_dim = 0;
    if let Some(h) = &eng.confidence {
        let o = m.forward_head(&ids, h);
        head_dim = o.len();
        write_f32(&out.join("head.bin"), &o);
    }
    let text = format!(
        "seq={}\nrows={}\nlayers={}\nd_model={}\nsteps={}\ndepth={}\nhead_out={}\n",
        ids.len(),
        rows,
        m.cfg.num_layers,
        m.cfg.d_model,
        steps,
        depth,
        head_dim
    );
    std::fs::write(out.join("trace.txt"), &text).expect("trace.txt");
    print!("{text}");
}
