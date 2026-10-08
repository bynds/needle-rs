#!/usr/bin/env python3
"""A synthetic Needle 3 container at the shipped geometry, from upstream's own code.

Every model-dependent test in this repository skips without `weights/needle3.cact`, and a host
that cannot reach Hugging Face cannot fetch it. This builds the next best thing: a container
with the published geometry (config.json of Cactus-Compute/needle3), the shipped quantisation
scheme (`embedding=4,mhc=4,default=2`, group 128, probe heads at HEAD_BITS) and a real
SentencePiece tokenizer with upstream's special ids and chat markers, but **seeded random
weights**. It is written by upstream's exporter (`needle.model.export.write_export`) from
parameters initialised by upstream's Flax model, so the canon, packing and header are the
publisher's, not a re-implementation.

What it is for: running the whole graph (prefill, decode, cache, Engram, mHC, heads, grammar)
on the host and under ARMv7 emulation; host-versus-target parity on identical bytes; timing and
memory at the real shapes. What it is not: a model. Its answers are noise, and nothing about
task quality can be read from it.

    PYTHONPATH=/path/to/needle JAX_PLATFORMS=cpu python3 make_synthetic_v3.py OUT_DIR \
        [--seed 0] [--layers 20] [--corpus FILE ...]

Writes OUT_DIR/synthetic-needle3.cact, OUT_DIR/synthetic-tokenizer.model and
OUT_DIR/synthetic-manifest.json (seed, geometry, upstream commit, sizes, sha256).
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--kv-window", type=int, default=256)
    ap.add_argument("--corpus", nargs="*", default=None,
                    help="text files to train the tokenizer on (default: this repo's sources)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import jax
    import jax.numpy as jnp
    import sentencepiece as spm
    import needle
    from needle.model import export
    from needle.model.architecture import SimpleAttentionNetwork, TransformerConfig
    from needle.model.tokenizer import CHAT_MARKERS, SANTokenizer

    # ---- tokenizer: real SentencePiece, upstream ids (pad 0, eos 1, bos 2, unk 3, markers 4..)
    corpus = a.corpus
    if not corpus:
        root = pathlib.Path(__file__).resolve().parents[3]
        corpus = [str(p) for p in sorted(root.rglob("*"))
                  if p.suffix in (".rs", ".md", ".py", ".json") and p.is_file()
                  and "target" not in p.parts and p.stat().st_size < 2_000_000]
    text_path = a.out / "synthetic-corpus.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        for p in corpus:
            f.write(open(p, encoding="utf-8", errors="ignore").read())
            f.write("\n")
    prefix = str(a.out / "synthetic-tokenizer")
    spm.SentencePieceTrainer.train(
        input=str(text_path), model_prefix=prefix, vocab_size=a.vocab, model_type="unigram",
        byte_fallback=True, user_defined_symbols=CHAT_MARKERS, pad_id=0, eos_id=1, bos_id=2,
        unk_id=3, hard_vocab_limit=False, character_coverage=1.0, num_threads=4,
        minloglevel=2, max_sentence_length=1 << 16, input_sentence_size=2_000_000,
        shuffle_input_sentence=True, seed_sentencepiece_size=200_000)
    os.remove(text_path)
    tok = SANTokenizer(prefix + ".model")
    vocab = tok.sp.GetPieceSize()

    # ---- model: published geometry (Cactus-Compute/needle3 config.json)
    cfg = TransformerConfig(
        vocab_size=vocab, d_model=768, num_heads=12, num_kv_heads=2, num_layers=a.layers,
        qk_head_dim=48, v_head_dim=64, max_seq_len=a.max_seq_len, rope_theta=100000.0,
        engram_orders=(2, 3), engram_slots=18432,
        engram_layers=tuple(l for l in (3, 7, 11, 15, 19) if l < a.layers),
        global_layers=tuple(l for l in (4, 9, 14, 19) if l < a.layers),
        sliding_window=1024, mhc_lanes=4, qkv_conv_taps=3, kv_bits=8, dtype="float32")
    model = SimpleAttentionNetwork(cfg)
    tokens = jnp.zeros((1, 8), jnp.int32).at[0, 0].set(2)
    variables = model.init(jax.random.PRNGKey(a.seed), tokens,
                           method=lambda m, t: (m(t), m.forward_confidence(t)))
    params = jax.tree_util.tree_map(np.asarray, variables["params"])
    # Upstream initialises gates, biases and conditioning to zero or identity; a model that is
    # mostly identity leaves whole paths unexercised. Perturb every float leaf so each one
    # carries signal, deterministically from the seed.
    rng = np.random.default_rng(a.seed)

    def perturb(x):
        if not np.issubdtype(x.dtype, np.floating):
            return x
        s = max(float(np.std(x)), 0.02) * 0.25
        return (x + rng.normal(0.0, s, x.shape)).astype(x.dtype)

    params = jax.tree_util.tree_map(perturb, params)

    # ---- the shipped scheme: embedding=4, mhc=4, default=2 (probe heads keep HEAD_BITS)
    # The public exporter only packs 4-bit (`_cq_pack` checks the module's WEIGHT_BITS); its
    # LSB-first packing is width-generic, so the guard is pointed at each tensor's width.
    plain_q, plain_bits = export._q, export.WEIGHT_BITS

    def scheme_q(name, mat, bits, group):
        if bits == 2:  # the default width this script passes; heads pass HEAD_BITS
            if name == "embedding" or name.startswith("mhc_"):
                bits = 4
        export.WEIGHT_BITS = bits
        try:
            return plain_q(name, mat, bits, group)
        finally:
            export.WEIGHT_BITS = plain_bits

    export._q = scheme_q
    out = a.out / "synthetic-needle3.cact"
    info = export.write_export(params, cfg, str(out), bits=2, group=128, tokenizer=tok,
                               kv_window=a.kv_window)
    export._q, export.WEIGHT_BITS = plain_q, plain_bits

    raw = out.read_bytes()
    try:
        commit = subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.dirname(needle.__file__)), "rev-parse", "HEAD"],
            text=True).strip()
    except Exception:
        commit = None
    manifest = {
        "kind": "synthetic (seeded random weights; not a model)",
        "seed": a.seed,
        "file": out.name,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "tensors": info["tensors"],
        "tokenizer_pieces": vocab,
        "tokenizer_model_sha256": hashlib.sha256(
            open(prefix + ".model", "rb").read()).hexdigest(),
        "scheme": "embedding=4,mhc=4,default=2,group=128",
        "geometry": {"d_model": 768, "num_layers": a.layers, "num_heads": 12, "num_kv_heads": 2,
                     "qk_head_dim": 48, "v_head_dim": 64, "vocab": vocab,
                     "max_seq_len": a.max_seq_len, "kv_window": a.kv_window,
                     "engram_sites": list(cfg.engram_layers),
                     "global_layers": list(cfg.global_layers)},
        "upstream": {"repo": "cactus-compute/needle", "commit": commit,
                     "jax": jax.__version__, "numpy": np.__version__,
                     "sentencepiece": spm.__version__},
    }
    (a.out / "synthetic-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    sys.exit(main())
