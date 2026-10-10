#!/usr/bin/env bash
# fetch-upstream.sh DIR: Cactus Compute's published Whistle weights and prebuilt engine library
# (Apache-2.0, Hugging Face), pinned by SHA-256. Used only through the public C API in needle.h.
#   DIR/whistle.cact            Cactus-Compute/whistle
#   DIR/linux-armv7/{libneedle.a,needle.h}, DIR/linux-x86_64/{libneedle.a,needle.h}
set -euo pipefail
D=${1:?usage: fetch-upstream.sh DIR}
H=https://huggingface.co/Cactus-Compute
get() { # url dest sha256
  mkdir -p "$(dirname "$2")"
  [ -f "$2" ] || curl -fsSL -o "$2" "$1"
  echo "$3  $2" | sha256sum -c --quiet
}
get $H/whistle/resolve/main/whistle.cact "$D/whistle.cact" b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb
get $H/needle3/resolve/main/linux-armv7/libneedle.a "$D/linux-armv7/libneedle.a" f05b3b1f75a8030a1e54167bf0ed38f8f4c1592e4f903ce7fbd262784246fc0e
get $H/needle3/resolve/main/linux-armv7/needle.h "$D/linux-armv7/needle.h" abbfcaa7b82208e9063008a95c27e5f0d9d7ac98f1b23db3a1cf07d271331593
get $H/needle3/resolve/main/linux-x86_64/libneedle.a "$D/linux-x86_64/libneedle.a" 802d96bf6e8f8fdab53edecb8034bce1f23d38bc6aadc1757f152ed04abbaad2
get $H/needle3/resolve/main/linux-x86_64/needle.h "$D/linux-x86_64/needle.h" abbfcaa7b82208e9063008a95c27e5f0d9d7ac98f1b23db3a1cf07d271331593
echo "ok: $D"
