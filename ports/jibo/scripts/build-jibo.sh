#!/usr/bin/env bash
# build-jibo.sh OUT_DIR: cross-build needle-jibo (and the upstream CLI as a reference) for Jibo,
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

READELF=${READELF:-arm-linux-gnueabihf-readelf}
"$READELF" -h -A -l -d --version-info "$OUT/needle-jibo" > "$OUT/needle-jibo.readelf.txt"
"$HERE/scripts/check-jibo-abi.sh" "$OUT/needle-jibo" "$OUT/needle-rs" | tee "$OUT/abi-check.txt"

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
  echo "  \"needle_rs_sha256\": \"$(sha256sum "$OUT/needle-rs" | cut -d' ' -f1)\""
  echo "}"
} > "$OUT/build-manifest.json"
cargo "+$TC" tree --locked --target "$T" -p needle-jibo -e features > "$OUT/needle-jibo.feature-tree.txt"
if grep -q rayon "$OUT/needle-jibo.feature-tree.txt"; then
  echo "needle-jibo pulled in rayon; the runner must be serial" >&2
  exit 1
fi
echo "built: $OUT"
