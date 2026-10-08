#!/usr/bin/env bash
# jibo-cc-standin.sh: a C compiler/linker driver with the same contract as the owner's jibo-armcc
# wrapper, built from this host's arm-linux-gnueabihf-gcc and a glibc 2.21 sysroot
# (fetch-sysroot.sh). Cargo uses it as the armv7 linker; build-jibo.sh uses it for C helpers.
#
#   JIBO_SYSROOT   the sysroot directory (required)
#
# The host cross gcc searches its own (glibc 2.39) library directories before any --sysroot, so
# the sysroot's start files, libraries and headers are named explicitly, as in the Decider port's
# build-jibo.sh. libgcc is linked statically into C objects; Rust's std still asks for
# libgcc_s.so.1, which the sysroot provides from GCC 4.9.
set -euo pipefail
: "${JIBO_SYSROOT:?set JIBO_SYSROOT (scripts/fetch-sysroot.sh output)}"
R=$(cd "$JIBO_SYSROOT" && pwd)
M=arm-linux-gnueabihf
CC=${JIBO_HOST_CROSS_CC:-arm-linux-gnueabihf-gcc}
GCCINC=$("$CC" -print-file-name=include)
link=1
for a in "$@"; do case "$a" in -c|-S|-E) link=0 ;; esac; done
ARGS=(-march=armv7-a -mfpu=neon -mfloat-abi=hard
      -nostdinc -isystem "$GCCINC" -isystem "$R/usr/include/$M" -isystem "$R/usr/include")
if [ $link = 1 ]; then
  ARGS+=(-B"$R/usr/lib/$M/" -L"$R/usr/lib/$M" -L"$R/lib/$M"
         -Wl,--sysroot="$R" -Wl,-rpath-link,"$R/lib/$M" -Wl,-rpath-link,"$R/usr/lib/$M"
         -Wl,--dynamic-linker=/lib/ld-linux-armhf.so.3 -Wl,--hash-style=both)
fi
exec "$CC" "${ARGS[@]}" "$@"
