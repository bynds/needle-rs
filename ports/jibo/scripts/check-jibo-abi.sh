#!/usr/bin/env bash
# check-jibo-abi.sh BINARY...: will these ELF files load on Jibo?
#
# Adapted from bynds/strands-decider, ports/jibo/scripts/check-jibo-abi.sh (branch
# claude/inference-engine-jibo-port-rucz8e, commit 29e78ff, Apache-2.0). Changes for a Rust
# executable: libgcc_s.so.1 is allowed (Rust std's unwinder library; present on any GCC-built
# firmware, and named in the owner-run check below), its GCC_* symbol versions are capped at
# what Linaro GCC 4.8.4 ships, and the ARM build attributes are read too.
#
# Jibo: 32-bit ARMv7, hard float, glibc 2.21, libstdc++ up to GLIBCXX_3.4.20, no development
# files. For each binary: ELF32 ARM; the hard-float ABI flag and Tag_ABI_VFP_args; Tag_CPU_arch
# v7 (nothing newer); the armhf loader; NEEDED limited to libc, libm, libdl, libpthread, librt,
# libgcc_s and the loader; no GLIBC_ version above 2.21; no GCC_ version above 4.8.0; no
# GLIBCXX/CXXABI at all (the port links no C++). It reads files only and runs nothing.
#
# With JIBO_FS set to the owner's copy of the robot filesystem (work/fs/p2), every NEEDED library
# must also exist there, and the closure of those libraries is checked against the same ceiling.
set -euo pipefail
READELF=${READELF:-arm-linux-gnueabihf-readelf}
MAX_GLIBC=2.21
MAX_GCC=4.8.0
[ $# -gt 0 ] || { echo "usage: check-jibo-abi.sh BINARY..." >&2; exit 2; }
command -v "$READELF" >/dev/null || { echo "no $READELF (set READELF)" >&2; exit 2; }

ver_gt() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" = "$1" ] && [ "$1" != "$2" ]; }

check_versions() { # file -> problems on stdout
  local vers v
  vers=$("$READELF" -V "$1" | grep -o "Name: [A-Z_]*[0-9.]*" | awk '{print $2}' | sort -u)
  for v in $vers; do
    case "$v" in
      GLIBC_PRIVATE) echo "needs GLIBC_PRIVATE" ;;
      GLIBC_[0-9]*) ver_gt "${v#GLIBC_}" "$MAX_GLIBC" && echo "needs $v (Jibo has $MAX_GLIBC)" ;;
      GCC_[0-9]*) ver_gt "${v#GCC_}" "$MAX_GCC" && echo "needs $v (Linaro 4.8.4 libgcc_s)" ;;
      GLIBCXX_*|CXXABI_*) echo "needs $v (the port links no libstdc++)" ;;
    esac
  done
  return 0
}

status=0
for bin in "$@"; do
  problems=()
  hdr=$("$READELF" -h "$bin")
  attrs=$("$READELF" -A "$bin")
  grep -q "Class:.*ELF32" <<<"$hdr" || problems+=("not ELF32")
  grep -q "Machine:.*ARM" <<<"$hdr" || problems+=("not ARM")
  grep -q "hard-float ABI" <<<"$hdr" || problems+=("not the hard-float ABI")
  grep -q "Tag_ABI_VFP_args: VFP registers" <<<"$attrs" || problems+=("Tag_ABI_VFP_args is not VFP registers")
  if grep -q "Tag_CPU_arch:" <<<"$attrs"; then
    grep -q "Tag_CPU_arch: v7" <<<"$attrs" || problems+=("$(grep 'Tag_CPU_arch:' <<<"$attrs" | head -1 | xargs) (want v7)")
  fi
  "$READELF" -l "$bin" | grep -q "/lib/ld-linux-armhf.so.3" || problems+=("loader is not /lib/ld-linux-armhf.so.3")
  needed=$("$READELF" -d "$bin" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p')
  for lib in $needed; do
    case "$lib" in
      libc.so.6|libm.so.6|libdl.so.2|libpthread.so.0|librt.so.1|libgcc_s.so.1|ld-linux-armhf.so.3) ;;
      *) problems+=("needs $lib") ;;
    esac
    if [ -n "${JIBO_FS:-}" ]; then
      found=$(find "$JIBO_FS/lib" "$JIBO_FS/usr/lib" -maxdepth 2 -name "$lib" 2>/dev/null | head -1)
      if [ -z "$found" ]; then
        problems+=("$lib not present in JIBO_FS")
      else
        while read -r p; do [ -n "$p" ] && problems+=("$lib (robot copy): $p"); done < <(check_versions "$(readlink -f "$found")")
      fi
    fi
  done
  while read -r p; do [ -n "$p" ] && problems+=("$p"); done < <(check_versions "$bin")
  vers=$("$READELF" -V "$bin" | grep -o "Name: [A-Z_]*[0-9.]*" | awk '{print $2}' | sort -uV | tr '\n' ' ')
  if [ ${#problems[@]} -eq 0 ]; then
    echo "ok   $bin  NEEDED: $(tr '\n' ' ' <<<"$needed")| versions: $vers"
  else
    status=1
    for p in "${problems[@]}"; do echo "FAIL $bin: $p"; done
  fi
done
if [ -n "${JIBO_ABI_CHECKER:-}" ]; then
  "$JIBO_ABI_CHECKER" "$@" || status=1
fi
exit $status
