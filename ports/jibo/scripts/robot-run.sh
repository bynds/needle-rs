#!/usr/bin/env bash
# robot-run.sh STEP: the robot-side steps of the Needle port, for the owner to run.
#
# Adapted from bynds/strands-decider ports/jibo/scripts/robot-run.sh (Apache-2.0, commit 29e78ff).
# Every step is a deployment step: run it only within the agreed scope (which robot, which
# directory, which user). Nothing here flashes, overwrites system files, stops services, adds
# swap, changes clocks, fans or thermal limits, or commands motors. It copies into one directory
# and `cleanup` removes it. Results land in ports/jibo/results/robot-<step>-<time>.*, labelled
# "physical Jibo".
#
# Environment (all required for the steps that use them; nothing is guessed):
#   JIBO_SSH    SSH destination; prefer the jibo-skill user (uid 2000, group video)
#   JIBO_DIR    an isolated directory on a filesystem with ~60 MB free whose backing is known.
#               /tmp may be RAM-backed: `env` prints `df -T` for the directory; check it first.
#   JIBO_ENV    the shell prefix reproducing the game host's run.sh display access and library
#               shim, e.g. 'DISPLAY=:0 LD_LIBRARY_PATH=/path/to/shim' (for `gl` only)
#   BUILD       scripts/build-jibo.sh output, plus jibo-cq-bench (see README)
#   MODEL       the pinned needle3.cact
#   OPS         a directory of dump-op outputs (one subdirectory per operation)
#
# Steps, in the order to run them:
#   env       uname, glibc and libgcc_s present, df -T of JIBO_DIR, MemAvailable (no copy)
#   deploy    copy binaries, model, fixtures and OPS to JIBO_DIR; sha256 of everything
#   smoke     needle-jibo info, then one request (the handoff's fixture), with /proc peak RSS
#   suite     the 21 fixture requests, 1 warmup + 3 measured, full depth, f32 KV, serial
#   ops       jibo-cq-bench --backend ref,neon on every OPS operation (CPU only, no display)
#   gl        jibo-cq-bench --backend gl, sweeping --tt and --rows-per-dispatch (needs JIBO_ENV)
#   status    thermal zones, readable clocks, MemAvailable, load (read-only)
#   cleanup   remove JIBO_DIR
set -euo pipefail
STEP=${1:?usage: robot-run.sh env|deploy|smoke|suite|ops|gl|status|cleanup}
: "${JIBO_SSH:?set JIBO_SSH}" "${JIBO_DIR:?set JIBO_DIR}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RESULTS="$HERE/results"
mkdir -p "$RESULTS"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$RESULTS/robot-$STEP-$STAMP"
remote() { ssh -o BatchMode=yes "$JIBO_SSH" "$@"; }
# Peak resident set of a remote command, sampled from /proc while it runs.
with_hwm() {
  remote "cd '$JIBO_DIR' && ( $1 ) & pid=\$!; peak=0; \
    while kill -0 \$pid 2>/dev/null; do r=\$(awk '/VmHWM/{print \$2}' /proc/\$pid/status 2>/dev/null); \
    [ -n \"\$r\" ] && [ \"\$r\" -gt \"\$peak\" ] && peak=\$r; sleep 0.2; done; wait \$pid; rc=\$?; \
    echo \"{\\\"label\\\":\\\"physical Jibo\\\",\\\"vm_hwm_kb_sampled\\\":\$peak,\\\"exit\\\":\$rc}\""
}

case "$STEP" in
  env)
    remote "uname -a; id; (/lib/libc.so.6 2>/dev/null || /lib/arm-linux-gnueabihf/libc.so.6) | head -1; \
            ls -l /lib/ld-linux-armhf.so.3 /lib/libgcc_s.so.1 /usr/lib/libgcc_s.so.1 2>&1; \
            mkdir -p '$JIBO_DIR' && df -T '$JIBO_DIR'; grep -E 'MemTotal|MemAvailable|SwapTotal' /proc/meminfo; \
            grep -m1 -E 'model name|Processor' /proc/cpuinfo; grep -m1 Features /proc/cpuinfo; nproc" \
      | tee "$OUT.txt"
    ;;
  deploy)
    : "${BUILD:?set BUILD}" "${MODEL:?set MODEL}"
    remote "mkdir -p '$JIBO_DIR/fixtures' '$JIBO_DIR/ops'"
    scp -q "$BUILD/needle-jibo" "$BUILD/needle-rs" "$BUILD/jibo-cq-bench" "$MODEL" "$JIBO_SSH:$JIBO_DIR/"
    scp -q "$HERE"/fixtures/*.json "$HERE"/fixtures/*.jsonl "$HERE"/fixtures/query.txt "$JIBO_SSH:$JIBO_DIR/fixtures/"
    if [ -n "${OPS:-}" ]; then scp -qr "$OPS"/* "$JIBO_SSH:$JIBO_DIR/ops/"; fi
    remote "cd '$JIBO_DIR' && ls -la && find . -type f | sort | xargs sha256sum" | tee "$OUT.txt"
    ;;
  smoke)
    remote "cd '$JIBO_DIR' && ./needle-jibo info needle3.cact --tools fixtures/tools.json" | tee "$OUT.jsonl"
    with_hwm "./needle-jibo run needle3.cact --tools fixtures/tools.json --query \"\$(cat fixtures/query.txt)\" --debug-text" \
      | tee -a "$OUT.jsonl"
    ;;
  suite)
    with_hwm "./needle-jibo bench needle3.cact --tools fixtures/tools-extended.json --requests fixtures/requests.jsonl \
              --reps 3 --warmup 1 --debug-text" | tee "$OUT.jsonl"
    python3 "$HERE/scripts/score.py" "$HERE/fixtures/expected.jsonl" "$OUT.jsonl" --label "physical Jibo CPU" \
      | tee "$OUT.score.txt" || true
    ;;
  ops)
    remote "cd '$JIBO_DIR' && for d in ops/*/; do ./jibo-cq-bench \"\$d\" --backend ref --reps 5; \
            ./jibo-cq-bench \"\$d\" --backend neon --reps 5; done" | tee "$OUT.jsonl"
    ;;
  gl)
    : "${JIBO_ENV:?set JIBO_ENV}"
    for tt in 1 4 8 16; do
      for rows in 64 256 1024 100000; do
        remote "cd '$JIBO_DIR' && for d in ops/*/; do $JIBO_ENV ./jibo-cq-bench \"\$d\" --backend gl --reps 5 \
                --tt $tt --rows-per-dispatch $rows; done" | tee -a "$OUT.jsonl"
      done
    done
    ;;
  status)
    remote 'for z in /sys/class/thermal/thermal_zone*; do echo "$(cat $z/type 2>/dev/null) $(cat $z/temp 2>/dev/null)"; done;
            for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq /sys/kernel/debug/clock/gbus/rate; do
              [ -r "$f" ] && echo "$f $(cat $f)"; done;
            grep -E "MemTotal|MemAvailable" /proc/meminfo; cat /proc/loadavg' | tee "$OUT.txt"
    ;;
  cleanup)
    remote "rm -rf '$JIBO_DIR'" && echo "removed $JIBO_DIR on $JIBO_SSH"
    ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
