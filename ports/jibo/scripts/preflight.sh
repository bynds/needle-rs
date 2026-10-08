#!/bin/sh
# preflight.sh [--quick]: run ON the robot, from inside the deployed bundle, before any step.
#
# POSIX sh and BusyBox tools only (Jibo's firmware is Buildroot; there is no bash guarantee).
# It reads and checks; it changes nothing. Exit status 0 only if every hard check passes.
#
#   hard:  every file matches MANIFEST.sha256 (skipped with --quick, which checks sizes only);
#          the binaries start (usage text, exit 2); the glibc is 2.21 or newer; libgcc_s.so.1 and
#          the armhf loader exist; MemAvailable is at least MIN_AVAIL_MB (default 300)
#   warn:  the bundle sits on tmpfs/ramfs (the 35 MB model then costs RAM twice); less than
#          FREE_MB (default 20) free here; the CPU reports no NEON (needle-jibo-neon cannot run)
#
# For testing on a development host only: PREFLIGHT_ROOT prefixes the system paths checked and
# PREFLIGHT_RUN prefixes the binaries run, e.g. PREFLIGHT_ROOT=/opt/jibo-sysroot
# PREFLIGHT_RUN="qemu-arm -L /opt/jibo-sysroot". Leave both unset on the robot.
set -u
cd "$(dirname "$0")" || exit 2
R=${PREFLIGHT_ROOT:-}
RUN=${PREFLIGHT_RUN:-}
QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1
MIN_AVAIL_MB=${MIN_AVAIL_MB:-300}
FREE_MB=${FREE_MB:-20}
fail=0
ok() { echo "ok    $*"; }
bad() { echo "FAIL  $*"; fail=1; }
warn() { echo "warn  $*"; }

# 1. Bundle integrity
if [ ! -f MANIFEST.sha256 ]; then
  bad "MANIFEST.sha256 missing: not a prepared bundle"
elif [ $QUICK = 1 ]; then
  n=0
  while read -r sum size file; do
    [ -f "$file" ] || { bad "missing $file"; continue; }
    s=$(wc -c < "$file" | tr -d ' ')
    [ "$s" = "$size" ] || bad "$file is $s bytes, want $size"
    n=$((n + 1))
  done < MANIFEST.sha256
  ok "sizes of $n files"
elif command -v sha256sum >/dev/null 2>&1; then
  if awk '{print $1 "  " $3}' MANIFEST.sha256 | sha256sum -c >/dev/null 2>&1; then
    ok "sha256 of $(wc -l < MANIFEST.sha256 | tr -d ' ') files"
  else
    bad "sha256 mismatch (run: awk '{print \$1 \"  \" \$3}' MANIFEST.sha256 | sha256sum -c)"
  fi
else
  warn "no sha256sum on this system; sizes only"
  QUICK=1
fi

# 2. Loader, C library, unwinder
[ -e "$R/lib/ld-linux-armhf.so.3" ] && ok "/lib/ld-linux-armhf.so.3" || bad "no /lib/ld-linux-armhf.so.3"
libc=$(ls "$R/lib/libc.so.6" "$R/lib/arm-linux-gnueabihf/libc.so.6" 2>/dev/null | head -n 1)
if [ -n "$libc" ]; then
  v=$($RUN "$libc" 2>/dev/null | head -n 1 | sed -n 's/.*version \([0-9][0-9.]*\).*/\1/p')
  case "$v" in
    2.2[1-9]*|2.[3-9]*|[3-9].*) ok "glibc $v" ;;
    "") warn "glibc version unreadable from $libc" ;;
    *) bad "glibc $v is older than 2.21" ;;
  esac
else
  bad "no libc.so.6"
fi
gcc_s=""
for p in "$R/lib/libgcc_s.so.1" "$R/usr/lib/libgcc_s.so.1" "$R/lib/arm-linux-gnueabihf/libgcc_s.so.1"; do
  [ -e "$p" ] && gcc_s=$p
done
[ -n "$gcc_s" ] && ok "libgcc_s.so.1" || bad "no libgcc_s.so.1 (Rust's unwinder library)"

# 3. The binaries start
for b in needle-jibo needle-jibo-neon needle-jibo-c needle-jibo-c-neon jibo-cq-bench; do
  [ -x "$b" ] || { case "$b" in needle-jibo-neon|needle-jibo-c|needle-jibo-c-neon) continue ;; esac; bad "$b missing or not executable"; continue; }
  $RUN ./"$b" >/dev/null 2>&1
  rc=$?
  [ $rc = 2 ] && ok "$b starts" || bad "$b exited $rc instead of printing usage (loader or ABI problem)"
done

# 4. Machine
grep -q neon /proc/cpuinfo && ok "CPU reports NEON" || warn "CPU reports no NEON: do not run needle-jibo-neon or needle-jibo-c-neon"
avail=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
if [ -z "$avail" ]; then
  warn "no MemAvailable in /proc/meminfo"
elif [ "$avail" -ge "$MIN_AVAIL_MB" ]; then
  ok "MemAvailable ${avail} MB (>= $MIN_AVAIL_MB)"
else
  bad "MemAvailable ${avail} MB below $MIN_AVAIL_MB: do not start inference now"
fi
here=$(pwd -P)
fstype=$(awk -v d="$here" '$2 != "" && index(d, $2) == 1 { if (length($2) > best) { best = length($2); t = $3 } } END { print t }' /proc/mounts)
case "$fstype" in
  tmpfs|ramfs) warn "bundle is on $fstype: the model occupies RAM as a file and again when loaded" ;;
  "") warn "filesystem type of $here unknown" ;;
  *) ok "filesystem $fstype" ;;
esac
free=$(df -k . 2>/dev/null | awk 'NR == 2 {print int($4 / 1024)}')
[ -n "$free" ] && [ "$free" -lt "$FREE_MB" ] && warn "only ${free} MB free here"
for z in /sys/class/thermal/thermal_zone*; do
  [ -r "$z/temp" ] && echo "info  $(cat "$z/type" 2>/dev/null) $(cat "$z/temp") mC"
done
[ $fail = 0 ] && echo "PREFLIGHT PASS" || echo "PREFLIGHT FAIL"
exit $fail
