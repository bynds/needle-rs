#!/usr/bin/env bash
# gate.sh: the port's release gate on a development host. Unlike a plain `cargo test`, which skips
# every model-dependent comparison when the checkpoint is absent, this fails when a required
# fixture is missing, so a green gate means the comparisons ran.
#
#   JIBO_SYSROOT=/path/to/sysroot ports/jibo/scripts/gate.sh [--qemu]
#
# Requires: weights/needle3.cact with the pinned sha256; tests/v3_forward_vectors.{json,f32},
# tests/v3_component_vectors.{json,f32} and tests/v3_ladder_vectors.json (generate with the
# tools/gen_v3_*.py oracles at the pinned upstream commit, see the README). --qemu also runs the
# runner's tests and the fixture suite as ARMv7 binaries under qemu-arm (slow: about an hour).
#
# Known failures: none. (v3_forward_parity::cells_and_confidence_match_the_reference failed at
# upstream 4de5049 on a global-RMS metric; this branch judges each cell against its own RMS. The
# list below stays as the place to record one honestly instead of skipping it.)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
TC=${RUST_TOOLCHAIN:-1.87.0}
KNOWN_FAILURES=""
cd "$ROOT"

fail() { echo "GATE FAIL: $*" >&2; exit 1; }
want_sha=$(awk '{print $1}' "$HERE/manifests/needle-model-sha256.txt")
[ -f weights/needle3.cact ] || fail "weights/needle3.cact missing (manifests/needle-model-revision.txt)"
[ "$(sha256sum weights/needle3.cact | cut -d' ' -f1)" = "$want_sha" ] || fail "needle3.cact is not the pinned model"
for f in tests/v3_forward_vectors.json tests/v3_forward_vectors.f32 tests/v3_component_vectors.json \
         tests/v3_component_vectors.f32 tests/v3_ladder_vectors.json; do
  [ -f "$f" ] || fail "$f missing (generate with the JAX oracle, README: Oracles)"
done
cargo "+$TC" fmt --all -- --check || fail "rustfmt"

log=$(mktemp)
NEEDLE_JIBO_REQUIRE_FIXTURES=1 cargo "+$TC" test --locked --release --no-fail-fast \
  -p needle-core -p needle-infer -p needle-c -p needle-rs-cli -p needle-jibo -- --nocapture >"$log" 2>&1 || true
grep -E '^test result' "$log" | awk '{p+=$4; f+=$6} END {print "host tests: passed", p, "failed", f}'
failed=$(grep -E '^test .* FAILED$' "$log" | awk '{print $2}' | sed 's/.*:://' | sort -u || true)
unexpected=$(comm -23 <(echo "$failed" | grep -v '^$' | sort -u) <(tr ' ' '\n' <<<"$KNOWN_FAILURES" | sort -u) || true)
[ -z "$unexpected" ] || { grep -E 'FAILED|panicked' "$log" | head -20; fail "unexpected test failures: $unexpected"; }
if grep -qiE 'skipping v3 (forward|e2e|cact|tokenizer|component)|SKIPPING container' "$log"; then
  fail "a Needle 3 comparison skipped: $(grep -iE 'skipping v3|SKIPPING' "$log" | head -3)"
fi
echo "known failures (recorded, not skipped): $(echo "$failed" | tr '\n' ' ')"

: "${JIBO_SYSROOT:?set JIBO_SYSROOT (scripts/fetch-sysroot.sh) for the ARMv7 build}"
"$HERE/scripts/build-jibo.sh" target/jibo-gate >/dev/null
cat target/jibo-gate/abi-check.txt

if [ "${1:-}" = "--qemu" ]; then
  export CARGO_TARGET_ARMV7_UNKNOWN_LINUX_GNUEABIHF_LINKER="$HERE/scripts/jibo-cc-standin.sh"
  export CARGO_TARGET_ARMV7_UNKNOWN_LINUX_GNUEABIHF_RUNNER="qemu-arm -L $JIBO_SYSROOT"
  NEEDLE_JIBO_REQUIRE_FIXTURES=1 cargo "+$TC" test --locked --release \
    --target armv7-unknown-linux-gnueabihf -p needle-jibo || fail "armv7 runner tests"
  qemu-arm -L "$JIBO_SYSROOT" target/jibo-gate/needle-jibo bench weights/needle3.cact \
    --tools "$HERE/fixtures/tools-extended.json" --requests "$HERE/fixtures/requests.jsonl" \
    --reps 1 --warmup 0 --debug-text > target/jibo-gate/qemu-suite.jsonl
  python3 "$HERE/scripts/score.py" "$HERE/fixtures/expected.jsonl" target/jibo-gate/qemu-suite.jsonl \
    --label "emulated ARM"
fi
echo "GATE PASS"
