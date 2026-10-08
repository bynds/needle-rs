#!/usr/bin/env bash
# build-jibo.sh OUT_DIR: cross-build needle-jibo, the upstream CLI (a reference) and jibo-cq-bench,
# check the ABI, and write a build manifest. Nothing here runs on, or copies to, the robot.
#
#   JIBO_CC=<the owner's jibo-armcc wrapper>   the intended linker: Linaro GCC 4.8.4 with the
#                                             robot's target libraries
#   JIBO_SYSROOT=<dir>                        a stand-in: this host's arm-linux-gnueabihf-gcc
#                                             against a glibc 2.21 sysroot (fetch-sysroot.sh),
#                                             through scripts/jibo-cc-standin.sh
#   RUST_TOOLCHAIN                            default 1.87.0 (the workspace's rust-version)
#   JIBO_RUSTFLAGS                            extra target flags, recorded in the manifest. The
#                                             baseline uses none: the target's defaults are
#                                             ARMv7-A, VFPv3-D16, Thumb-2, no NEON.
#
# Also builds needle-jibo-neon: the runner with the `neon` feature and `-C target-feature=+neon`
# (needle-core/src/cq_neon.rs), in its own target directory, so the robot can run the baseline and
# the NEON build side by side. Set JIBO_NO_NEON=1 to skip it. Rust 1.87 warns that `neon` is an
# unstable -Ctarget-feature; it is applied all the same, and the build fails if it is not.
#
# Also builds needle-jibo-c and needle-jibo-c-neon, the C99 translation (ports/jibo/c), with the same
# compiler: the first for the baseline float ABI (VFPv3-D16, no NEON), the second with -DND_NEON.
#
# Each package is built in its own cargo invocation. Building needle-jibo together with
# needle-rs-cli would unify features and switch on the CLI's rayon `parallel` path in the runner.
set -euo pipefail
OUT=${1:?usage: build-jibo.sh OUT_DIR}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
TC=${RUST_TOOLCHAIN:-1.87.0}
T=armv7-unknown-linux-gnueabihf
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

if [ -n "${JIBO_CC:-}" ]; then
  LINKER="$JIBO_CC"
  route="JIBO_CC=$JIBO_CC"
elif [ -n "${JIBO_SYSROOT:-}" ]; then
  export JIBO_SYSROOT
  LINKER="$HERE/scripts/jibo-cc-standin.sh"
  route="stand-in sysroot $(cat "$JIBO_SYSROOT/PACKAGES.sha256" 2>/dev/null | awk '{print $1}' | tr '\n' ' ')"
else
  echo "set JIBO_CC (the jibo-armcc wrapper) or JIBO_SYSROOT (scripts/fetch-sysroot.sh)" >&2
  exit 2
fi

# Host-native flags must not leak into the target build.
if [ -n "${RUSTFLAGS:-}" ] || [ -n "${CARGO_ENCODED_RUSTFLAGS:-}" ]; then
  echo "RUSTFLAGS is set in the environment; unset it (target flags go in JIBO_RUSTFLAGS)" >&2
  exit 2
fi
export CARGO_TARGET_ARMV7_UNKNOWN_LINUX_GNUEABIHF_LINKER="$LINKER"
export CARGO_TARGET_ARMV7_UNKNOWN_LINUX_GNUEABIHF_RUSTFLAGS="${JIBO_RUSTFLAGS:-}"

cd "$ROOT"
for pkg in needle-jibo needle-rs-cli; do
  cargo "+$TC" build --locked --release --target "$T" -p "$pkg"
done
cp "target/$T/release/needle-jibo" "target/$T/release/needle-rs" "$OUT/"
if [ -z "${JIBO_NO_NEON:-}" ]; then
  CARGO_TARGET_ARMV7_UNKNOWN_LINUX_GNUEABIHF_RUSTFLAGS="${JIBO_RUSTFLAGS:-} -C target-feature=+neon" \
    CARGO_TARGET_DIR="$ROOT/target/jibo-neon" \
    cargo "+$TC" build --locked --release --target "$T" -p needle-jibo --features neon
  cp "target/jibo-neon/$T/release/needle-jibo" "$OUT/needle-jibo-neon"
  n=$(${OBJDUMP:-arm-linux-gnueabihf-objdump} -d "$OUT/needle-jibo-neon" | grep -cE 'vmla\.f32[[:space:]]+q1[23]' || true)
  [ "$n" -ge 6 ] || { echo "needle-jibo-neon: the NEON CQ kernels are missing ($n VMLA)" >&2; exit 1; }
fi

# The C operation benchmark (scalar, NEON, desktop-GL compute). C99 for Linaro GCC 4.8.4; libGL
# and libX11 are opened with dlopen, so it links libc, libm and libdl only.
"$LINKER" -O2 -std=c99 -march=armv7-a -mfpu=neon -mfloat-abi=hard -static-libgcc \
  -o "$OUT/jibo-cq-bench" "$HERE/backend/jibo-cq-bench.c" -ldl -lm

# The C99 translation of the engine and runner (ports/jibo/c): libc, libm and libpthread only.
for v in plain neon; do
  if [ $v = plain ]; then arch="-march=armv7-a -mfpu=vfpv3-d16 -mfloat-abi=hard"; extra=""; name=needle-jibo-c
  else arch="-march=armv7-a -mfpu=neon -mfloat-abi=hard"; extra="-DND_NEON"; name=needle-jibo-c-neon; fi
  make -s -C "$HERE/c" CC="$LINKER" OUT="$OUT/c-$v" ARCH="$arch" CFLAGS="-O2 $extra" "$OUT/c-$v/needle-jibo-c"
  cp "$OUT/c-$v/needle-jibo-c" "$OUT/$name"
  rm -rf "$OUT/c-$v"
done

READELF=${READELF:-arm-linux-gnueabihf-readelf}
"$READELF" -h -A -l -d --version-info "$OUT/needle-jibo" > "$OUT/needle-jibo.readelf.txt"
bins=("$OUT/needle-jibo" "$OUT/needle-rs" "$OUT/jibo-cq-bench" "$OUT/needle-jibo-c" "$OUT/needle-jibo-c-neon")
[ -f "$OUT/needle-jibo-neon" ] && bins+=("$OUT/needle-jibo-neon")
"$HERE/scripts/check-jibo-abi.sh" "${bins[@]}" | tee "$OUT/abi-check.txt"

{
  echo "{"
  echo "  \"label\": \"armv7 cross-build\","
  echo "  \"source_commit\": \"$(git rev-parse HEAD)\","
  echo "  \"source_dirty\": $( [ -z "$(git status --porcelain -- crates ports Cargo.toml Cargo.lock)" ] && echo false || echo true ),"
  echo "  \"cargo_lock_sha256\": \"$(sha256sum Cargo.lock | cut -d' ' -f1)\","
  echo "  \"rustc\": \"$(rustc "+$TC" -V)\","
  echo "  \"target\": \"$T\","
  echo "  \"target_rustflags\": \"${JIBO_RUSTFLAGS:-}\","
  echo "  \"linker\": \"$route\","
  echo "  \"cc\": \"$(${JIBO_CC:-arm-linux-gnueabihf-gcc} --version | head -1)\","
  echo "  \"needle_jibo_sha256\": \"$(sha256sum "$OUT/needle-jibo" | cut -d' ' -f1)\","
  echo "  \"needle_rs_sha256\": \"$(sha256sum "$OUT/needle-rs" | cut -d' ' -f1)\","
  if [ -f "$OUT/needle-jibo-neon" ]; then echo "  \"needle_jibo_neon_sha256\": \"$(sha256sum "$OUT/needle-jibo-neon" | cut -d' ' -f1)\","; fi
  echo "  \"needle_jibo_c_sha256\": \"$(sha256sum "$OUT/needle-jibo-c" | cut -d' ' -f1)\","
  echo "  \"needle_jibo_c_neon_sha256\": \"$(sha256sum "$OUT/needle-jibo-c-neon" | cut -d' ' -f1)\","
  echo "  \"jibo_cq_bench_sha256\": \"$(sha256sum "$OUT/jibo-cq-bench" | cut -d' ' -f1)\""
  echo "}"
} > "$OUT/build-manifest.json"
cargo "+$TC" tree --locked --target "$T" -p needle-jibo -e features > "$OUT/needle-jibo.feature-tree.txt"
if grep -q rayon "$OUT/needle-jibo.feature-tree.txt"; then
  echo "needle-jibo pulled in rayon; the runner must be serial" >&2
  exit 1
fi
echo "built: $OUT"
