#!/usr/bin/env bash
# prepare-robot.sh OUT_DIR: everything a robot session needs, built and checked on this host, in
# one directory to copy as is. Nothing here touches the robot.
#
#   JIBO_CC or JIBO_SYSROOT   as for build-jibo.sh
#   MODEL                     default weights/needle3.cact; must match manifests/needle-model-sha256.txt
#
# OUT_DIR gets: needle-jibo, needle-jibo-neon, needle-rs, jibo-cq-bench, the model, fixtures/, client/, ops/ (dump-op reference vectors: four CQ/MLP operations at 16
# tokens for prefill and 1 for decode, and attention at 64 and 256 positions), preflight.sh, build-manifest.json, MANIFEST.sha256
# ("sha256 bytes path" per file). robot-run.sh deploys it with BUNDLE=OUT_DIR.
set -euo pipefail
OUT=${1:?usage: prepare-robot.sh OUT_DIR}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
MODEL=${MODEL:-$ROOT/weights/needle3.cact}
want=$(awk '{print $1}' "$HERE/manifests/needle-model-sha256.txt")
got=$(sha256sum "$MODEL" | cut -d' ' -f1)
[ "$got" = "$want" ] || { echo "$MODEL is not the pinned model ($got)" >&2; exit 1; }

rm -rf "$OUT"
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
"$HERE/scripts/build-jibo.sh" "$OUT"
rm -f "$OUT"/*.readelf.txt "$OUT"/*.feature-tree.txt
cp "$MODEL" "$OUT/needle3.cact"
mkdir -p "$OUT/fixtures" "$OUT/client" "$OUT/ops"
cp "$HERE"/fixtures/*.json "$HERE"/fixtures/*.jsonl "$HERE"/fixtures/query.txt "$OUT/fixtures/"
cp "$HERE"/client/needle-client.js "$HERE"/client/needle-client.test.js "$OUT/client/"
cp "$HERE/scripts/preflight.sh" "$OUT/"
chmod +x "$OUT/preflight.sh"

# Reference vectors, made by the host build (the dump is the Rust kernels' own output).
cargo +"${RUST_TOOLCHAIN:-1.87.0}" build --locked --release -p needle-jibo --manifest-path "$ROOT/Cargo.toml" >/dev/null
for t in l0.q_proj l0.out_proj embedding l0.mlp.w1; do
  for n in 16 1; do
    "$ROOT/target/release/needle-jibo" dump-op "$MODEL" --tensor "$t" --tokens "$n" --out "$OUT/ops/$t.t$n" >/dev/null
  done
done
# Attention over a whole prompt: a global layer at 64 and 256 positions, a windowed one at 64.
for spec in attn.global:64 attn.global:256 attn.local16:64; do
  t=${spec%%:*}; n=${spec##*:}
  "$ROOT/target/release/needle-jibo" dump-op "$MODEL" --tensor "$t" --tokens "$n" --out "$OUT/ops/$t.t$n" >/dev/null
done

(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 | sed 's|^\./||' | sort \
  | while read -r f; do printf '%s %s %s\n' "$(sha256sum "$f" | cut -d' ' -f1)" "$(stat -c %s "$f")" "$f"; done) \
  > "$OUT/MANIFEST.sha256"
(cd "$OUT" && sh ./preflight.sh --quick | grep -E '^(ok|FAIL) +sizes' || true)
du -sh "$OUT"
echo "prepared: $OUT ($(wc -l < "$OUT/MANIFEST.sha256") files). Deploy with: BUNDLE=$OUT scripts/robot-run.sh deploy"
