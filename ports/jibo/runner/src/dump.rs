//! Reference vectors for one operation of the real model, for `backend/jibo-cq-bench` to check
//! its CPU (scalar, NEON) and GPU kernels against.
//!
//! The weights are the container's own bytes; the activations are seeded random values (a
//! kernel's correctness and cost do not depend on where its input came from). The expected
//! outputs are computed by the Rust engine's own kernels, so the C kernels are compared with the
//! arithmetic the runner actually executes.
//!
//! Writes into DIR:
//!   op.txt         key=value lines describing the operation
//!   w.bin          CQ: the record's bytes (packed indices, then FP16 group norms)
//!   levels.bin     CQ: the codebook levels for this width, f32
//!   x.bin          CQ: prepared (rotated, zero-padded) activations, f32 [tokens, in_padded]
//!   y.bin          CQ: W · x, f32 [tokens, out], token-major
//!   a.bin, b.bin   Kronecker: the stage's learned factors, f32 [ba, ba] and [bb, bb]
//!   z.bin          Kronecker: inputs, f32 [tokens, ba * bb]
//!   k.bin          Kronecker: aᵀ · Z · b per token, f32 [tokens, ba * bb]
//!   q.bin, kk.bin, v.bin, o.bin   attention (`attn.global`, `attn.local<W>`): queries
//!                  [tokens, heads, qk], keys [tokens, kv_heads, qk], values [tokens, kv_heads, v]
//!                  and the engine's causal grouped-query attention over them [tokens, heads, v]

use needle_infer::cact::CactV3;
use needle_infer::v3::{config_from_geometry, V3Layout};
use std::io::Write;
use std::path::Path;

/// SplitMix64 normals by Box–Muller: reproducible across hosts and targets.
fn normals(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (u1, u2) = (next().max(1e-12), next());
            ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
        })
        .collect()
}

fn write_f32(path: &Path, v: &[f32]) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    for x in v {
        f.write_all(&x.to_le_bytes())?;
    }
    f.flush()
}

/// `tensor` is `embedding`, `l<N>.<q_proj|k_proj|v_proj|gate_proj|out_proj>`,
/// `l<N>.mlp.w<1|2|3>`, or `attn.global` / `attn.local<W>` (the model's head geometry, seeded
/// inputs, window W).
pub fn dump(model: &Path, tensor: &str, tokens: usize, out: &Path) -> Result<String, String> {
    let bytes = std::fs::read(model).map_err(|e| format!("{}: {e}", model.display()))?;
    let model_sha256 = crate::sha256::hex(&bytes);
    let cact = CactV3::from_bytes(bytes.clone()).map_err(|e| e.to_string())?;
    let cfg = config_from_geometry(&cact.geom).map_err(|e| e.to_string())?;
    let layout = V3Layout::derive(&cfg, cact.records()).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let io = |e: std::io::Error| e.to_string();
    let tokens = tokens.max(1);

    let layer_of = |s: &str| -> Result<usize, String> {
        let l: usize = s
            .strip_prefix('l')
            .and_then(|n| n.parse().ok())
            .ok_or(format!("bad layer in {tensor:?}"))?;
        if l >= layout.layers.len() {
            return Err(format!("layer {l} >= {}", layout.layers.len()));
        }
        Ok(l)
    };
    let parts: Vec<&str> = tensor.split('.').collect();
    let mut meta = vec![
        format!("model_sha256={model_sha256}"),
        format!("tensor={tensor}"),
        format!("tokens={tokens}"),
        "activations=seeded normal (splitmix64 seed 7, Box-Muller)".into(),
    ];

    if parts.first() == Some(&"attn") && parts.len() == 2 {
        use needle_core::v3::attention::{attend, AttnDims, KvStore};
        let window = match parts[1] {
            "global" => None,
            w => Some(
                w.strip_prefix("local")
                    .and_then(|n| n.parse::<usize>().ok())
                    .filter(|n| *n > 0)
                    .ok_or(format!(
                        "attention window in {tensor:?}: global or local<W>"
                    ))?,
            ),
        };
        let d = AttnDims {
            seq: tokens,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            qk_head_dim: cfg.qk_head_dim,
            v_head_dim: cfg.v_head_dim,
        };
        let q = normals(7, tokens * d.num_heads * d.qk_head_dim);
        let k = normals(8, tokens * d.num_kv_heads * d.qk_head_dim);
        let v = normals(9, tokens * d.num_kv_heads * d.v_head_dim);
        let mut o = vec![0.0f32; tokens * d.num_heads * d.v_head_dim];
        attend(&q, KvStore::F32 { k: &k, v: &v }, d, window, &mut o);
        write_f32(&out.join("q.bin"), &q).map_err(io)?;
        write_f32(&out.join("kk.bin"), &k).map_err(io)?;
        write_f32(&out.join("v.bin"), &v).map_err(io)?;
        write_f32(&out.join("o.bin"), &o).map_err(io)?;
        meta.extend([
            "op=attn".into(),
            format!("heads={}", d.num_heads),
            format!("kv_heads={}", d.num_kv_heads),
            format!("qk={}", d.qk_head_dim),
            format!("vd={}", d.v_head_dim),
            format!("window={}", window.unwrap_or(0)),
        ]);
    } else if parts.len() == 3 && parts[1] == "mlp" {
        let l = layer_of(parts[0])?;
        let m = layout.layers[l].mlp;
        let (ai, bi) = match parts[2] {
            "w1" => (m[5], m[6]),
            "w2" => (m[7], m[8]),
            "w3" => (m[9], m[10]),
            s => return Err(format!("unknown stage {s}")),
        };
        let a = cact.floats(ai).map_err(|e| e.to_string())?;
        let b = cact.floats(bi).map_err(|e| e.to_string())?;
        let (ba, bb) = needle_core::v3::kernels::hada_blocks(cfg.hada_n);
        if a.len() != ba * ba || b.len() != bb * bb {
            return Err("Kronecker factors do not match hada_blocks(hada_n)".into());
        }
        let n = ba * bb;
        let z = normals(7, tokens * n);
        let mut k = vec![0.0f32; tokens * n];
        for t in 0..tokens {
            needle_core::v3::kernels::kron_apply(
                &z[t * n..(t + 1) * n],
                &a,
                &b,
                ba,
                bb,
                &mut k[t * n..(t + 1) * n],
            );
        }
        write_f32(&out.join("a.bin"), &a).map_err(io)?;
        write_f32(&out.join("b.bin"), &b).map_err(io)?;
        write_f32(&out.join("z.bin"), &z).map_err(io)?;
        write_f32(&out.join("k.bin"), &k).map_err(io)?;
        meta.extend(["op=kron".into(), format!("ba={ba}"), format!("bb={bb}")]);
    } else {
        let idx = match parts.as_slice() {
            ["embedding"] => layout.embedding,
            [l, which] => {
                let li = &layout.layers[layer_of(l)?];
                match *which {
                    "q_proj" => li.q_proj,
                    "k_proj" => li.k_proj,
                    "v_proj" => li.v_proj,
                    "gate_proj" => li.gate_proj,
                    "out_proj" => li.out_proj,
                    s => return Err(format!("unknown projection {s}")),
                }
            }
            _ => return Err(format!("unknown tensor {tensor:?}")),
        };
        let r = *cact.record(idx);
        let w = cact.cq(idx).map_err(|e| e.to_string())?;
        let levels = needle_core::cq::codebook_slice(&cact.codebook, w.bits)
            .map_err(|e| format!("{e:?}"))?;
        let blob = &bytes[r.offset as usize..(r.offset + r.nbytes) as usize];
        let x = normals(7, tokens * w.in_feat);
        let mut xh = vec![0.0f32; tokens * w.in_padded];
        for t in 0..tokens {
            w.prepare_input(
                &x[t * w.in_feat..(t + 1) * w.in_feat],
                &mut xh[t * w.in_padded..(t + 1) * w.in_padded],
            );
        }
        let mut y = vec![0.0f32; tokens * w.out_feat];
        let mut acc = vec![0.0f32; tokens];
        w.matmul_rows_prepared(&xh, tokens, 0, w.out_feat, &mut y, &mut acc);
        std::fs::write(out.join("w.bin"), blob).map_err(io)?;
        write_f32(&out.join("levels.bin"), levels).map_err(io)?;
        write_f32(&out.join("x.bin"), &xh).map_err(io)?;
        write_f32(&out.join("y.bin"), &y).map_err(io)?;
        meta.extend([
            "op=cq".into(),
            format!("record={idx}"),
            format!("out={}", w.out_feat),
            format!("in={}", w.in_feat),
            format!("in_padded={}", w.in_padded),
            format!("group={}", w.group),
            format!("bits={}", w.bits),
        ]);
    }
    let text = meta.join("\n") + "\n";
    std::fs::write(out.join("op.txt"), &text).map_err(io)?;
    Ok(text)
}
