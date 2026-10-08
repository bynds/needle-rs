#!/usr/bin/env bash
# c-check.sh: every parity gate of the C99 engine (ports/jibo/c), from references the Rust code
# computes on the spot. Nothing here runs on the robot.
#
#   ports/jibo/scripts/c-check.sh [--arm] [--san] [--full] [--work DIR]
#
#   (default)  x86-64 build of the C tests, run against fresh references
#   --arm      also cross-build with jibo-cc-standin.sh (plain and -DND_NEON) and run under
#              qemu-arm (JIBO_SYSROOT, default /tmp/claude-0/sysroot); slow
#   --san      also an ASan+UBSan build (-fno-sanitize-recover=all) of every test
#   --full     the large corpora (30M/10M float formats, 500k JSON documents, every f32 bit
#              pattern for the JSON writer, exhaustive 2^32 sweeps for nd_math); hours under --arm
#
# Gates, in the order a port has to pass them:
#   1 math       nd_math vs Rust libm 0.2.16 (exp, log, sin, cos, pow, tanh): bit equality
#   2 tokenizer  nd_tok / nd_prompt vs sp_tokenizer.rs / prompt.rs: ids, bytes, prompts
#   3 json       nd_json vs serde_json 1.0.149: accept/reject, error text, DOM, output bytes
#   4 grammar    nd_grammar vs constrained.rs: byte table, tool parsing, every mask
#   5 runner     nd_catalog / nd_validate / nd_grounding vs the runner's modules
#   6 cq         CQ kernels vs `needle-jibo dump-op` (q_proj, out_proj, embedding)
#   7 model      cells, all-position logits, prefill, decode (f32 and int8 KV), confidence
#                head vs runner/examples/trace.rs, at full depth and ladder depths 4 and 13
#   8 corrupt    1,512 single-field corruptions: the C and Rust loaders accept and refuse the
#                same containers, and none crashes
#   9 runner     needle-jibo-c serve on fixtures/c-parity-edge.jsonl (malformed lines, unknown
#                keys, limits, NUL, Unicode, health, truncation) byte-identical to the Rust runner
# End-to-end runner parity (the request suites) is scripts/c-parity.py; see ports/jibo/c/README.md.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)        # ports/jibo
ROOT=$(cd "$HERE/../.." && pwd)
C="$HERE/c"
ARM=0 SAN=0 FULL=0 WORK=${TMPDIR:-/tmp}/nd-c-check
while [ $# -gt 0 ]; do
  case "$1" in
    --arm) ARM=1 ;; --san) SAN=1 ;; --full) FULL=1 ;;
    --work) WORK=$2; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
MODEL=${NEEDLE_JIBO_CACT:-$ROOT/weights/needle3.cact}
SYSROOT=${JIBO_SYSROOT:-/tmp/claude-0/sysroot}
D=$WORK/ref; mkdir -p "$D"
REFT=$WORK/ref-target
fail=0
say() { printf '\n== %s\n' "$*"; }
gate() { # name, command...: run, record pass/fail
  local name=$1; shift
  if "$@"; then echo "PASS $name"; else echo "FAIL $name"; fail=1; fi
}

say "references (Rust)"
(cd "$C/ref" && CARGO_TARGET_DIR=$REFT cargo +1.87.0 build --offline --release -q)
(cd "$ROOT" && cargo build --release -q -p needle-jibo --bin needle-jibo --example trace --example corrupt_cases)
R=$REFT/release
J="$ROOT/target/release"
$R/math_ref cases "$D/math.bin" $([ $FULL = 1 ] || echo small)
$R/tok_ref "$MODEL" "$ROOT" "$D/tok.bin"
$R/json_ref docs "$D/json.bin" "$D/json.txt" 12345 200000
$R/json_ref floats64 "$D/f64.bin" 777 4000000
$R/json_ref floats32 "$D/f32.bin" 999 2000000
mkdir -p "$D/grammar"; $R/grammar_ref "$D/grammar" 7 500
$R/runner_ref "$D/runner.jsonl"
for t in l0.q_proj l0.out_proj embedding; do
  "$J/needle-jibo" dump-op "$MODEL" --tensor $t --tokens 16 --out "$D/op-$t" > /dev/null
done
Q=$(cat "$HERE/fixtures/query.txt")
"$J/examples/trace" --model "$MODEL" --tools "$HERE/fixtures/tools.json" --query "$Q" --out "$D/trace" --steps 12 > /dev/null
for d in 4 13; do
  "$J/examples/trace" --model "$MODEL" --tools "$HERE/fixtures/tools.json" \
    --query "Turn the volume down to three" --out "$D/trace-d$d" --steps 6 --depth $d > /dev/null
done
"$J/examples/corrupt_cases" "$MODEL" > "$D/verdicts.tsv"
if [ $FULL = 1 ]; then
  $R/json_ref docs "$D/json2.bin" "$D/json2.txt" 98765 500000
  $R/json_ref floats64 "$D/f64big.bin" 31337 30000000
  $R/json_ref floats32 "$D/f32big.bin" 4242 10000000
  $R/json_ref f32all "$D/f32all.txt"
  for f in 0 1 2 3 5; do $R/math_ref exh "$D/exh_$f.bin" $f; done
fi

run_suite() { # label, runner prefix (empty or qemu), build dir, reference suffix
  local label=$1 pre=$2 O=$3
  say "gates: $label"
  gate "$label math" $pre "$O/test_math" cases "$D/math$4.bin"
  gate "$label tokenizer" $pre "$O/test_tok" "$MODEL" "$D/tok.bin"
  gate "$label json" $pre "$O/test_json" --docs "$D/json.bin" "$D/json.txt" --floats64 "$D/f64.bin" --floats32 "$D/f32.bin"
  gate "$label grammar" $pre "$O/test_grammar" "$D/grammar"
  gate "$label runner" $pre "$O/test_runner_mods" "$D/runner$4.jsonl"
  gate "$label cq" $pre "$O/test_cq" "$MODEL" "$D/op-l0.q_proj" "$D/op-l0.out_proj" "$D/op-embedding"
  gate "$label model" $pre "$O/test_model" "$MODEL" "$D/trace"
  gate "$label model-d4" $pre "$O/test_model" "$MODEL" "$D/trace-d4" 4
  gate "$label model-d13" $pre "$O/test_model" "$MODEL" "$D/trace-d13" 13
  gate "$label corrupt" $pre "$O/test_corrupt" "$MODEL" "$D/verdicts.tsv"
  gate "$label serve-edge" python3 "$HERE/scripts/c-parity.py" --rust "$J/needle-jibo" --c "$O/needle-jibo-c" \
    --runner-prefix "$pre" --model "$MODEL" --tools "$HERE/fixtures/tools-extended.json" \
    --requests "$HERE/fixtures/c-parity-edge.jsonl" --out "$WORK/serve-$label" --label edge \
    -- --debug-text --max-total-tokens 600
  if [ $FULL = 1 ]; then
    gate "$label json-full" $pre "$O/test_json" --docs "$D/json2.bin" "$D/json2.txt" --floats64 "$D/f64big.bin" --floats32 "$D/f32big.bin"
    gate "$label json-f32all" $pre "$O/test_json" --f32all "$D/f32all.txt"
    for f in 0 1 2 3 5; do gate "$label math-exh-$f" $pre "$O/test_math" exh "$D/exh$4_$f.bin"; done
  fi
}

TESTS="test_math test_tok test_json test_grammar test_runner_mods test_cq test_model test_corrupt"
build() { # out dir, extra make args...
  local O=$1; shift
  make -s -C "$C" OUT="$O" "$@" $(for t in $TESTS; do echo "$O/$t"; done) "$O/needle-jibo-c"
}

say "build: x86-64"
build "$WORK/x86"
run_suite x86 "" "$WORK/x86" ""

if [ $SAN = 1 ]; then
  say "build: x86-64 ASan+UBSan"
  build "$WORK/san" CFLAGS="-O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all"
  run_suite san "" "$WORK/san" ""
fi

if [ $ARM = 1 ]; then
  export JIBO_SYSROOT=$SYSROOT
  STANDIN=$HERE/scripts/jibo-cc-standin.sh
  # armv7 references where a result depends on the target's usize (runner) or NaN encoding (math).
  (cd "$C/ref" && CARGO_TARGET_DIR=$REFT CARGO_TARGET_ARMV7_UNKNOWN_LINUX_GNUEABIHF_LINKER=$STANDIN \
     cargo +1.87.0 build --offline --release -q --target armv7-unknown-linux-gnueabihf --bin runner_ref --bin math_ref)
  RA=$REFT/armv7-unknown-linux-gnueabihf/release
  qemu-arm -L "$SYSROOT" "$RA/runner_ref" "$D/runner-arm.jsonl"
  qemu-arm -L "$SYSROOT" "$RA/math_ref" cases "$D/math-arm.bin" $([ $FULL = 1 ] || echo small)
  if [ $FULL = 1 ]; then for f in 0 1 2 3 5; do qemu-arm -L "$SYSROOT" "$RA/math_ref" exh "$D/exh-arm_$f.bin" $f; done; fi
  for v in plain neon; do
    say "build: armv7 $v"
    build "$WORK/arm-$v" CC="$STANDIN" CFLAGS="-O2 $([ $v = neon ] && echo -DND_NEON)"
    gate "arm-$v abi" "$HERE/scripts/check-jibo-abi.sh" "$WORK/arm-$v/needle-jibo-c"
    run_suite "arm-$v" "qemu-arm -L $SYSROOT" "$WORK/arm-$v" "-arm"
  done
fi

say "result"
[ $fail = 0 ] && echo "all gates passed" || echo "SOME GATES FAILED"
exit $fail
