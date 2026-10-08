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

## Notes on individual runs

- `emulated-arm-suite-f32-full.jsonl`: `needle-jibo` sha256
  `9581c95af8a89678632964b97bcfd9eb56ca3e491ccdcf75c0f750f040b9d79b`, built from commit 0bab4f5
  (the streaming fix, before the loader bounds checks, which add refusals only and change no
  arithmetic). 21/21 responses identical to `host-reference-suite-f32-full.jsonl` (first measured
  repetition) in text, status, calls and token counts. Its `vm_hwm_kb` is qemu's own process and
  `cpu_s` reads 0 under qemu-user: neither is a robot number.
- `official-native-emulated-suite.jsonl`: `linux-armv7/needle` sha256
  `0d7fd5a896ebb6db25e32e2a3f571602cb8f33ffd89f3286faeb54c7fe874d4c`, `--threads 1`, one `--prompt`
  per request against `fixtures/tools-extended.json`, qemu-arm 8.2.2 with Ubuntu's armhf glibc 2.39.
  Its own JSON per line, plus `request_id` and `exit`. Its `prefill_tps`, `decode_tps` and
  `peak_ram_mb` are emulator numbers.
- `host-reference-suite195-{dev,heldout}-f32.jsonl`: the 195-request suite, full depth, f32 KV,
  `--debug-text --confidence`, run by a build without the grounding check (its `grounded` is
  null); every grounding figure in the README is `needle-jibo regrade` of these files. Their
  timings were taken while emulated ARM runs shared the host: not timing evidence.
