#!/usr/bin/env bash
# robot-run.sh STEP: the robot-side steps of the Needle port, for the owner to run.
#
# Adapted from bynds/strands-decider ports/jibo/scripts/robot-run.sh (Apache-2.0, commit 29e78ff).
# Every step is a deployment step: run it only within the agreed scope (which robot, which
# directory, which user). Nothing here flashes, overwrites system files, stops services, adds
# swap, changes clocks, fans or thermal limits, or commands motors. It copies one prepared bundle
# into one directory and `cleanup` removes it. Results land in ports/jibo/results/robot-*,
# labelled "physical Jibo".
#
# Environment (nothing is guessed):
#   JIBO_SSH    SSH destination; prefer the jibo-skill user (uid 2000, group video)
#   JIBO_DIR    an isolated directory with ~60 MB free on a filesystem whose backing is known.
#               /tmp may be RAM-backed: `env` and `preflight` report the filesystem type.
#   BUNDLE      the output of scripts/prepare-robot.sh (deploy only)
#   JIBO_ENV    the shell prefix reproducing the game host's run.sh display access and library
#               shim, e.g. 'DISPLAY=:0 LD_LIBRARY_PATH=/path/to/shim' (gl only)
#   BIN         suite/smoke: needle-jibo (default), needle-jibo-neon, needle-jibo-c or
#               needle-jibo-c-neon (the C99 translation; same options and output)
#
# Steps, in order (each later step runs `preflight.sh --quick` first and stops if it fails):
#   env        uname, glibc, libgcc_s, df, MemAvailable, CPU features (before copying anything)
#   deploy     copy BUNDLE to JIBO_DIR, then the full preflight (every sha256)
#   preflight  the full preflight again (e.g. after a reboot)
#   smoke      needle-jibo info, then the handoff's fixture, with peak RSS
#   suite      the 21 original requests, 1 warmup + 3 measured
#   suite-big  the 195-request dev + held-out suite, 1 pass (about 10 minutes per 100 on the host
#              at x86 speed; expect far longer on the A15)
#   ops        jibo-cq-bench scalar and NEON on every ops/ operation (CPU only, no display)
#   gl         jibo-cq-bench GL compute, sweeping --tt and --rows-per-dispatch (needs JIBO_ENV)
#   status     thermal zones, readable clocks, MemAvailable, load (read-only)
#   cleanup    remove JIBO_DIR
set -euo pipefail
STEP=${1:?usage: robot-run.sh env|deploy|preflight|smoke|suite|suite-big|ops|gl|status|cleanup}
: "${JIBO_SSH:?set JIBO_SSH}" "${JIBO_DIR:?set JIBO_DIR}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RESULTS="$HERE/results"
mkdir -p "$RESULTS"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BIN=${BIN:-needle-jibo}
OUT="$RESULTS/robot-$STEP-$BIN-$STAMP"
remote() { ssh -o BatchMode=yes "$JIBO_SSH" "$@"; }
pre() { remote "cd '$JIBO_DIR' && sh ./preflight.sh --quick" | tee "$OUT.preflight.txt" | grep -q 'PREFLIGHT PASS' \
          || { cat "$OUT.preflight.txt"; echo "preflight failed; not running $STEP" >&2; exit 1; }; }
# Peak resident set of a remote command, sampled from /proc while it runs.
with_hwm() {
  remote "cd '$JIBO_DIR' && ( $1 ) & pid=\$!; peak=0; \
    while kill -0 \$pid 2>/dev/null; do r=\$(awk '/VmHWM/{print \$2}' /proc/\$pid/status 2>/dev/null); \
    [ -n \"\$r\" ] && [ \"\$r\" -gt \"\$peak\" ] && peak=\$r; sleep 0.2; done; wait \$pid; rc=\$?; \
    echo \"{\\\"label\\\":\\\"physical Jibo\\\",\\\"bin\\\":\\\"$BIN\\\",\\\"vm_hwm_kb_sampled\\\":\$peak,\\\"exit\\\":\$rc}\""
}

case "$STEP" in
  env)
    remote "uname -a; id; (/lib/libc.so.6 2>/dev/null || /lib/arm-linux-gnueabihf/libc.so.6) | head -1; \
            ls -l /lib/ld-linux-armhf.so.3 /lib/libgcc_s.so.1 /usr/lib/libgcc_s.so.1 2>&1; \
            mkdir -p '$JIBO_DIR' && df -k '$JIBO_DIR'; grep \" \$(df '$JIBO_DIR' | awk 'NR==2{print \$6}') \" /proc/mounts; \
            grep -E 'MemTotal|MemAvailable|SwapTotal' /proc/meminfo; \
            grep -m1 -E 'model name|Processor' /proc/cpuinfo; grep -m1 Features /proc/cpuinfo; nproc" \
      | tee "$OUT.txt"
    ;;
  deploy)
    : "${BUNDLE:?set BUNDLE (scripts/prepare-robot.sh output)}"
    [ -f "$BUNDLE/MANIFEST.sha256" ] || { echo "$BUNDLE is not a prepared bundle" >&2; exit 2; }
    remote "mkdir -p '$JIBO_DIR'"
    scp -qr "$BUNDLE"/. "$JIBO_SSH:$JIBO_DIR/"
    remote "cd '$JIBO_DIR' && sh ./preflight.sh" | tee "$OUT.txt"
    ;;
  preflight)
    remote "cd '$JIBO_DIR' && sh ./preflight.sh" | tee "$OUT.txt"
    ;;
  smoke)
    pre
    remote "cd '$JIBO_DIR' && ./$BIN info needle3.cact --tools fixtures/tools.json" | tee "$OUT.jsonl"
    with_hwm "./$BIN run needle3.cact --tools fixtures/tools.json --query \"\$(cat fixtures/query.txt)\" --debug-text" \
      | tee -a "$OUT.jsonl"
    ;;
  suite)
    pre
    with_hwm "./$BIN bench needle3.cact --tools fixtures/tools-extended.json --requests fixtures/requests.jsonl \
              --reps 3 --warmup 1 --debug-text" | tee "$OUT.jsonl"
    python3 "$HERE/scripts/score.py" "$HERE/fixtures/expected.jsonl" "$OUT.jsonl" --label "physical Jibo CPU $BIN" \
      | tee "$OUT.score.txt" || true
    ;;
  suite-big)
    pre
    for split in dev heldout; do
      with_hwm "./$BIN bench needle3.cact --tools fixtures/suite-tools.json --requests fixtures/suite-$split.jsonl \
                --reps 1 --warmup 0 --debug-text --confidence" | tee "$OUT.$split.jsonl"
    done
    ;;
  ops)
    pre
    remote "cd '$JIBO_DIR' && for d in ops/*/; do ./jibo-cq-bench \"\$d\" --backend ref --reps 5; \
            case \"\$d\" in *attn*) ;; *) ./jibo-cq-bench \"\$d\" --backend neon --reps 5 ;; esac; done" | tee "$OUT.jsonl"
    ;;
  gl)
    : "${JIBO_ENV:?set JIBO_ENV}"
    pre
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
