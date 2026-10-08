# Results

One file per run. The label in each file name is the evidence level:

| Prefix | Meaning |
| --- | --- |
| `host-reference-` | x86_64, this source, pinned model, 1 thread. Timings order configurations; they are not Jibo numbers. |
| `emulated-arm-` | The ARMv7 binaries under qemu-arm 8.2.2 against the glibc 2.21 stand-in sysroot. Correctness only: no speed claims. |
| `software-rendered-gl-` | Mesa llvmpipe (GL 4.5 core) under Xvfb. Correctness only. |
| `official-native-emulated-` | Cactus-Compute's `linux-armv7/needle` under qemu with a modern armhf glibc (it cannot load on Jibo). Its own serving policy. |
| `robot-` | Physical Jibo, written by `scripts/robot-run.sh`. None yet. |

Suite files are `needle-jibo bench` output: one response per line (with `bench.warmup`), then a
summary line. Score them with `scripts/score.py fixtures/expected.jsonl FILE`. The completions in
them are answers to the mock fixture requests, nothing else.
