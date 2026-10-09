#!/usr/bin/env bash
# bench.sh LABEL [DECODE_STEPS]: build the profiling benches of both engines for ARMv7 (plain and
# NEON) from this tree, run them in perfvm, and write $PERFVM_OUT/LABEL.jsonl: one line per engine,
# variant and workload (prefill, decode, prefix_hit), with exact user-space instruction counts in
# total and per operation. Rust: needle-infer/examples/v3_profile.rs (feature profile). C:
# ports/jibo/c/tests/bench_engine.c (-DND_PROFILE).
#   JIBO_SYSROOT   the stand-in sysroot      PERFVM_OUT   default $TMPDIR/perfvm/results
set -euo pipefail
LABEL=${1:?usage: bench.sh LABEL [DECODE_STEPS]}
STEPS=${2:-8}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
: "${JIBO_SYSROOT:?set JIBO_SYSROOT}"
W=${TMPDIR:-/tmp}/perfvm
OUT=${PERFVM_OUT:-$W/results}
B=$W/bin-$LABEL
mkdir -p "$OUT" "$B"
T=armv7-unknown-linux-gnueabihf
STANDIN=$ROOT/ports/jibo/scripts/jibo-cc-standin.sh
export CARGO_TARGET_ARMV7_UNKNOWN_LINUX_GNUEABIHF_LINKER=$STANDIN
(cd "$ROOT" && cargo +1.87.0 build -q --release --target $T -p needle-infer --features profile --example v3_profile)
cp "$ROOT/target/$T/release/examples/v3_profile" "$B/rust-plain"
(cd "$ROOT" && CARGO_TARGET_ARMV7_UNKNOWN_LINUX_GNUEABIHF_RUSTFLAGS="-C target-feature=+neon" \
  CARGO_TARGET_DIR="$ROOT/target/jibo-neon" \
  cargo +1.87.0 build -q --release --target $T -p needle-infer --features profile,neon --example v3_profile 2>/dev/null)
cp "$ROOT/target/jibo-neon/$T/release/examples/v3_profile" "$B/rust-neon"
for v in plain neon; do
  if [ $v = plain ]; then fpu=vfpv3-d16; x=""; else fpu=neon; x=-DND_NEON; fi
  make -s -C "$ROOT/ports/jibo/c" CC="$STANDIN" OUT="$B/c-$v-build" \
    ARCH="-march=armv7-a -mfpu=$fpu -mfloat-abi=hard" CFLAGS="-O2 -DND_PROFILE $x" "$B/c-$v-build/bench_engine"
  cp "$B/c-$v-build/bench_engine" "$B/c-$v"
done
Q=$(cat "$ROOT/ports/jibo/fixtures/query.txt")
J=$B/jobs
: > "$J"
for b in rust-plain c-plain rust-neon c-neon; do
  printf '/bin/%s\t/work/needle3.cact\t/work/tools.json\t%s\t--decode\t%s\n' "$b" "$Q" "$STEPS" >> "$J"
done
"$HERE/run.sh" "$J" bin/rust-plain="$B/rust-plain" bin/rust-neon="$B/rust-neon" bin/c-plain="$B/c-plain" \
  bin/c-neon="$B/c-neon" work/needle3.cact="$ROOT/weights/needle3.cact" \
  work/tools.json="$ROOT/ports/jibo/fixtures/tools.json" > "$B/console.txt"
python3 - "$B/console.txt" "$OUT/$LABEL.jsonl" "$LABEL" <<'PY'
import json, sys
variant, rows = None, []
for line in open(sys.argv[1]):
    if line.startswith('perfvm: job /bin/'):
        variant = line.split('/bin/')[1].split()[0]
    elif line.startswith('{'):
        v = json.loads(line)
        v['variant'], v['label'] = variant, sys.argv[3]
        rows.append(v)
open(sys.argv[2], 'w').write(''.join(json.dumps(r) + '\n' for r in rows))
print(f'{len(rows)} results -> {sys.argv[2]}')
for r in rows:
    print(f"{r['variant']:11} {r['workload']:10} {r['total']/1e9:8.3f} G instructions")
PY
