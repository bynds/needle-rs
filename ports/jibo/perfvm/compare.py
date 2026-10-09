#!/usr/bin/env python3
"""compare.py RESULTS.jsonl [BASELINE.jsonl]: per-operation instruction counts (millions) for each
variant and workload of a bench.sh run, and with a baseline, the change against it."""
import json, sys

OPS = ["embed", "engram", "mhc", "qkv", "conv_rope", "attn", "gate_out", "mlp", "head"]

def load(p):
    return {(r["variant"], r["workload"]): r for r in map(json.loads, open(p))}

cur = load(sys.argv[1])
base = load(sys.argv[2]) if len(sys.argv) > 2 else {}
for wl in ["prefill", "decode", "prefix_hit"]:
    print(f"\n{wl} (M instructions{', change vs baseline' if base else ''})")
    print(f"{'variant':11}" + "".join(f"{o:>10}" for o in OPS + ["total"]))
    for var in ["rust-plain", "c-plain", "rust-neon", "c-neon"]:
        r = cur.get((var, wl))
        if not r:
            continue
        vals = [r["ops"][o][0] for o in OPS] + [r["total"]]
        print(f"{var:11}" + "".join(f"{v/1e6:10.0f}" for v in vals))
        b = base.get((var, wl))
        if b:
            bv = [b["ops"][o][0] for o in OPS] + [b["total"]]
            print(f"{'':11}" + "".join(f"{(v-w)/w*100 if w else 0:+9.1f}%" for v, w in zip(vals, bv)))
