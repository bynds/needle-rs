# Needle 3 on the original Jibo

Native, local text-to-tool inference for the original Jibo robot: 32-bit ARMv7 (Tegra K1,
Cortex-A15, NEON, hard float), glibc 2.21, the robot's own firmware. This directory is the port's
first milestone: a pinned CPU reference, an ABI-clean ARMv7 executable, a bounded runner that says
exactly what happened to each request, and a validated operation-level CPU/GPU benchmark.

**Evidence levels.** Every result below is labelled **host reference** (x86_64, this source),
**emulated ARM** (the ARMv7 binary under `qemu-arm`), **software-rendered GL** (Mesa llvmpipe) or
**official native under emulation**. Nothing here has run on a Jibo yet: there is no robot access
from the build host. The robot steps are written and listed under [On the robot](#on-the-robot);
**physical Jibo CPU** and **physical Jibo GPU** results are pending. Emulator and llvmpipe timings
are not speed claims.

- [What is here](#what-is-here)
- [Pins](#pins)
- [Build](#build)
- [Run](#run)
- [Results](#results)
- [Findings that change how Needle is used](#findings-that-change-how-needle-is-used)
- [Changes to the runtime](#changes-to-the-runtime)
- [On the robot](#on-the-robot)
- [Not done yet](#not-done-yet)

## What is here

| Path | What it is |
| --- | --- |
| `runner/` (`needle-jibo`) | One loaded model, bounded requests on stdin/stdout or a 0660 Unix socket with a bounded queue. Authorized tool catalogue, admission on the real prompt length, deadlines, memory/thermal `busy` gates, schema validation, explicit outcomes, phase timings. Serial (no rayon). Also `bench` and `dump-op`. |
| `runner/tests/container_robustness.rs` | 1,511 single-field corruptions of the real container; every one must be refused or loaded, never panic. Runs under ARMv7 emulation. |
| `backend/jibo-cq-bench.c` | One real operation (a CQ projection or a learned Kronecker MLP stage) on scalar C, NEON and desktop-GL 4.3 compute, checked against the Rust engine's output and timed with uploads, fences and readback counted. |
| `scripts/fetch-sysroot.sh` | A stand-in glibc 2.21 armhf sysroot (Ubuntu 15.04 packages, sha256-pinned). |
| `scripts/jibo-cc-standin.sh` | A linker/compiler driver with the same contract as the owner's `jibo-armcc`. |
| `scripts/build-jibo.sh` | The ARMv7 build: each package in its own cargo invocation, ABI check, feature tree, manifest. |
| `scripts/check-jibo-abi.sh` | Will it load on Jibo: ELF32 ARM, hard-float ABI, `v7`, the armhf loader, allowed `NEEDED`, nothing above `GLIBC_2.21` or `GCC_4.8.0`, no libstdc++. |
| `scripts/gate.sh` | The release gate: fails, not skips, when the model or oracle vectors are absent. |
| `scripts/robot-run.sh` | The owner-run robot steps. |
| `scripts/score.py` | Scores runner output against `fixtures/expected.jsonl`. |
| `tools/make_synthetic_v3.py` | A seeded-random container at the shipped geometry and bit scheme, written by upstream's own exporter, for when the model cannot be fetched. |
| `fixtures/` | The handoff's `tools.json` and `query.txt` (byte-stable), a four-tool catalogue and 21 requests with expectations. Mock tools only: none is a verified Jibo API. |
| `manifests/` | Commits, hashes, toolchains, licences. |
| `results/` | Machine-readable outcomes, one file per run, labelled. |

## Pins

| What | Value |
| --- | --- |
| Runtime | `Geekgineer/needle-rs` @ `4de50494fd60f417b24c37e4d972f95d128f8a0f` (0.3.1, MIT), plus the commits on this branch |
| Rust | 1.87.0 (`manifests/rustc.txt`), `Cargo.lock` locked; the only lock change is the new local `needle-jibo` package |
| Model | `Cactus-Compute/needle3` @ `2ae11323dc000f5e70c49f7403efa6af12ba9e67`, `needle3.cact`, 35,335,380 bytes, sha256 `c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38` (Apache-2.0) |
| Model geometry (from the container) | 768 wide, 20 blocks, 12 heads / 2 KV, qk 48, v 64, vocab 8192, 4 mHC lanes, Engram at 3/7/11/15/19, global layers 4/9/14/19, CQ `embedding=4,mhc=4,default=2`, group 128 |
| Publisher oracle | `cactus-compute/needle` @ `dd857748ef4d8a91028fce9ea01bf793cf2c9caa` (the commit `tools/cact_params_v3.py` pins), JAX 0.11.2 / flax 0.12.10 on CPU (`manifests/python-oracle-env.txt`) |
| Official native binary | `linux-armv7/needle` at the model revision (`manifests/official-armv7-sha256.txt`), black box only |
| Stand-in sysroot | Ubuntu 15.04 armhf `libc6`/`libc6-dev` 2.21-0ubuntu4.3, `linux-libc-dev` 3.19.0-15.15, `libgcc1` 4.9.1 (`manifests/sysroot-packages.txt`) |
| Stand-in cross compiler | Ubuntu `gcc-arm-linux-gnueabihf` 13.3.0, `qemu-arm` 8.2.2 |
| Reused infrastructure | `bynds/strands-decider` @ `29e78ff` (Apache-2.0): ABI checker, robot script structure, GL context setup; each adapted file says so |

The owner's `jibo-armcc` wrapper (Linaro GCC 4.8.4) and `work/fs/p2` were not available on the
build host. The Decider port did not have them either: it used a stand-in sysroot, as this port
does. With the wrapper, `JIBO_CC=... scripts/build-jibo.sh` uses it, and
`JIBO_FS=work/fs/p2 scripts/check-jibo-abi.sh ...` also checks the robot's own copies of every
`NEEDED` library.

## Build

```bash
# Host reference (Rust installs and dependency fetches happen here, never on Jibo)
rustup toolchain install 1.87.0 --profile minimal
rustup target add --toolchain 1.87.0 armv7-unknown-linux-gnueabihf
cargo +1.87.0 build --locked --release -p needle-jibo        # alone: see "feature unification" below

# The model, by revision, checked
REV=$(cat ports/jibo/manifests/needle-model-revision.txt)
curl -fL -o weights/needle3.cact https://huggingface.co/Cactus-Compute/needle3/resolve/$REV/needle3.cact
sha256sum -c ports/jibo/manifests/needle-model-sha256.txt

# ARMv7: the owner's wrapper, or the stand-in sysroot
JIBO_CC=/path/to/jibo-armcc ports/jibo/scripts/build-jibo.sh out/jibo
ports/jibo/scripts/fetch-sysroot.sh /opt/jibo-sysroot && \
  JIBO_SYSROOT=/opt/jibo-sysroot ports/jibo/scripts/build-jibo.sh out/jibo

# The release gate (add --qemu for the emulated-ARM run; needs qemu-user)
JIBO_SYSROOT=/opt/jibo-sysroot ports/jibo/scripts/gate.sh
```

**Oracles.** The JAX comparisons need upstream's code at the pinned commit:
`git -C needle worktree add ../needle-oracle dd857748ef4d8a91028fce9ea01bf793cf2c9caa`, a venv
with `jax jaxlib flax optax numpy sentencepiece huggingface_hub`, then
`JAX_PLATFORMS=cpu PYTHONPATH=../needle-oracle:tools python tools/gen_v3_{forward,component,ladder}_parity.py`.

**Feature unification.** Building `needle-jibo` in the same cargo invocation as `needle-rs-cli`
switches on the CLI's `needle-infer/parallel` for the runner too (resolver 2 unifies features
across the packages of one build). Measured: 5 threads and 1.3 cores of CPU for what should be a
serial run. `build-jibo.sh` builds each package separately and fails if rayon appears in the
runner's feature tree.

## Run

```bash
needle-jibo serve needle3.cact --tools fixtures/tools-extended.json            # JSONL on stdin/stdout
needle-jibo serve needle3.cact --tools catalogue.json --socket /run/needle.sock --queue 2 \
    --deadline-ms 20000 --min-avail-mb 300 --thermal /sys/class/thermal/thermal_zone0/temp --max-temp-c 70
echo '{"request_id":"u1","query":"Start a 90 second timer.","tools":["start_timer"]}' | needle-jibo serve ...
```

A response (real output, host reference, full depth, f32 KV):

```json
{"request_id":"num-digits","status":"candidate",
 "calls":[{"name":"start_timer","arguments":{"seconds":90}}],
 "model_sha256":"c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38",
 "depth":20,"kv_precision":"f32","constrained":false,
 "prompt_truncated":false,"stop_reason":"im_end","schema_valid":true,"grounded":null,
 "confidence_raw":null,
 "tokens":{"prompt":201,"generated":29,"budget":256,"positions":230},
 "timing":{"queue_ms":0.0,"wall_ms":2846.094,"tokenize_ms":4.26,"prefill_ms":2328.113,
           "decode_ms":508.621,"first_token_ms":2337.454,"confidence_ms":null}}
```

`status` is one of:

| Status | Meaning |
| --- | --- |
| `candidate` | The turn ended on its own (`im_end`/`eos`), both markers closed, every call passes the schema. Still only a proposal: grounding and policy are the application's (`grounded` is always `null` here). |
| `no_call` | The model answered `[]`: a considered abstention, distinct from every failure below. |
| `needs_clarification` | A call omitted a required argument. Not invented. |
| `unsupported` | A call named a tool outside the request's offered set. |
| `invalid_output` | No marker, an unterminated marker, malformed JSON, or an argument of the wrong type, out of range, outside its enum, or undeclared. `detail` says which. |
| `incomplete` | The token budget ran out. The engine's own extractor would hand back the half payload; the runner does not. |
| `truncated` | Refused before compute: the prompt does not fit `--max-total-tokens` with `--min-new-tokens` to spare. |
| `timeout` | The deadline passed while queued or between tokens (a prefill in flight completes first). |
| `busy` | Queue full, `MemAvailable` below `--min-avail-mb`, or the thermal zone above `--max-temp-c`. |
| `invalid_request` | The request line is malformed, names an unknown key or tool, or exceeds a bound. |

The catalogue may use `string` (with `enum`, `maxLength`), `integer` and `number` (with
`minimum`, `maximum`) and `boolean`, and `required`. Anything else (`pattern`, arrays, nested
objects, `anyOf`, `additionalProperties`, ...) is refused at startup rather than ignored. The
catalogue's own bytes, key order included, go into the prompt. Nothing is logged;
`--debug-text` adds the completion to responses.

## Results

All with the pinned model, greedy, full depth (20), f32 KV, unconstrained unless stated,
1 thread. Requests are the 21 in `fixtures/requests.jsonl` against `fixtures/tools-extended.json`.

### Correctness (gates G1, G2, G3)

| Check | Result | Evidence |
| --- | --- | --- |
| Model file | 35,335,380 bytes, tag `0x05E12A84`, 581 tensors, embedded tokenizer 8192 pieces | host |
| Upstream tests with the real model | 382 passed, 1 failed (below) | host reference, `results/host-tests.txt` |
| Same suites as ARMv7 binaries | 319 passed, the same 1 failed | emulated ARM |
| Logits vs the JAX oracle, 57 positions | max abs 1.98e-4 vs RMS 28.9 (6.9e-6), **0 argmax mismatches** | host reference and emulated ARM, identical |
| Prefill-then-decode vs full sequence | 0.0 deviation, 0 argmax mismatches | host and emulated ARM |
| Engram, attention, HadamardMLP components vs JAX | ≤ 1.1e-5 relative | host |
| `cells_and_confidence_match_the_reference` | **fails at upstream 4de5049**: 1.67e-4 vs a 1e-4 tolerance on one global RMS. Per cell ≤ 4.6e-5 of the cell's RMS; worst element 24,350,496 vs 24,350,512 (6.6e-7 of itself, f32 rounding at 2.4e7) | host and emulated ARM; recorded in `gate.sh`, not skipped |
| Corrupted containers (1,511 single-field cases, x86_64 and armv7) | all refused or loaded, none panics, none aborts, **after** the fixes below | host, emulated ARM |
| Fixture suite, ARMv7 binary vs host | identical text, status, calls and token counts on every request | emulated ARM, `results/qemu-armv7-suite.jsonl` |
| ARMv7 ABI | `needle-jibo`, `needle-rs`, `jibo-cq-bench`: ELF32 ARM, hard float, `v7`; highest `GLIBC_2.18`, `GCC_4.3.0`; NEEDED libc, libm, libdl, libpthread, librt, libgcc_s | `results/abi-check.txt` |

### Task behaviour (G6, first pass)

`exact`: status and calls exactly as expected; `unsafe`: a `candidate` that differs from the
expectation. 20 scored requests (one is recorded, not scored).

| Configuration | exact | unsafe | decisions changed vs baseline | p50 wall | prefill | decode | peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (20 layers, f32 KV) | 12 | 6 | — | 3.35 s | 13.2 ms/tok | 18.1 ms/tok | 82 MB |
| `--constrain` | 12 | 6 | 0 | | | | |
| `--kv-int8` | 12 | 6 | **2** | 3.53 s | 14.1 | 19.9 | 77 MB |
| `--layers 16` | 12 | 6 | 6 | 2.91 s | 11.0 | 16.1 | 70 MB |
| `--layers 12` | 12 | 5 | 9 | 2.34 s | 9.4 | 13.1 | 63 MB |
| `--layers 8` | 4 | 5 | 20 | 1.70 s | 6.5 | 9.3 | 56 MB |

Host reference timings: one x86_64 core running the engine's AVX2 kernels. They order the
configurations; they say nothing about the A15. Against the handoff's 150 MiB budget the host
peak is 82 MB (64-bit, 1 thread); the robot's number, and the combined total with the eye, vision,
Oído and the Decider, are to be measured.

Baseline failures, all schema-valid and so beyond what validation can catch: "seven and a half
minutes" gives 950 s (the reasoning says "2.5 * 60"); "one hour" gives 60 s; "2 minutes and 15
seconds" gives 120 s; "how are you today?" and "do a backflip" give a 30-minute timer; "Set a
timer." invents 300 s. `--kv-int8` turns "Mute yourself" into a 60-second timer. Same aggregate
score, different decisions: compare decisions, not scores.

### Operations: CPU and GPU (G5, first pass)

`jibo-cq-bench` on real tensors of the pinned model, 16 tokens of seeded activations, against the
Rust kernels' output (`needle-jibo dump-op`):

| Operation | ref (C, Rust order) | NEON | GL compute |
| --- | --- | --- | --- |
| `l0.q_proj` 576x768 CQ2 | bit-identical (host and emulated ARM) | bit-identical (emulated ARM) | 1.3e-6 of RMS, cosine 1.0 (llvmpipe) |
| `l0.out_proj` 768x768 CQ2 | bit-identical | bit-identical | 1.5e-6 at every `--tt` 1/8/16 x rows/dispatch 64/256/all (llvmpipe) |
| `embedding` 8192x768 CQ4 (the tied LM head) | bit-identical | bit-identical | 2.3e-6 (llvmpipe) |
| `l0.mlp.w1` learned 32x32 Kronecker stage | bit-identical | bit-identical | bit-identical (llvmpipe) |

So the NEON kernels reproduce the engine's ARMv7 arithmetic exactly (non-fused VMLA in the
engine's lane order) and the GL kernels agree to f32 rounding. Their speed on the Tegra is the
open question; `robot-run.sh ops` and `gl` measure it.

### Where the CPU time goes (scalar path, the one ARMv7 runs)

callgrind, one real request (113 prompt + 34 generated tokens), host build with `simd` off so the
same kernels run as on ARMv7: prefill 65%, decode 27%, model sha256 at load 7%. By kernel: CQ
batched matmul 37%, CQ matvec 30% (decode, including the 8192-row tied LM head every token),
activation Walsh–Hadamard 9%, learned Kronecker MLP 6%. Heap allocation: ~40,000 calls per
request but 0.04% of instructions, so scratch reuse is not where the time is.

On ARMv7 the engine has no SIMD at all in these kernels: its explicit intrinsics are gated on
`x86_64` and `aarch64`, the `armv7-unknown-linux-gnueabihf` target defaults to `-neon`, LLVM does not
auto-vectorise f32 for ARMv7 NEON (it is not IEEE: denormals flush), and stable Rust 1.87 has no
ARM32 NEON intrinsics. The CQ kernels, about two thirds of the instructions, are the first CPU
target; `backend/` has bit-identical NEON versions ready to time.

## Findings that change how Needle is used

1. **A query ending in an emoji killed the process** (upstream engine, fixed here). Byte-fallback
   decoding made the streaming code slice inside a character; release builds abort on panic. The
   upstream CLI dies on `"Pon un temporizador de cinco minutos, por favor — gracias 🙂"`.
2. **Hostile containers could abort the loader on 32-bit** (upstream, fixed here): `num_layers`
   of 2^30 allocated 240 GB, a tokenizer record offset of 1 sized vectors from garbage, zero
   Engram tables divided by zero, CQ size products could wrap a 32-bit `usize`, and `MAX_GROUP`
   was documented but not enforced.
3. **`--constrain` constrains names and argument keys only.** Values, types, enums, ranges and
   required arguments are not constrained; it changed no decision on these fixtures. Validation
   is the runner's job and it does it.
4. **The engine returns unterminated payloads** when the budget runs out; `--json` prints them as
   if complete. The runner reports `incomplete`.
5. **Three references disagree, as the handoff warned.** For "Set a timer for seven and a half
   minutes." the pinned Rust engine (f32 KV, 1 thread) emits `seconds: 7.5` (rejected: not an
   integer); the official `linux-armv7` binary (its own serving policy; config.json declares 8-bit
   KV and activations) emits `seconds: 750` with confidence 0.47 and flags it `ungrounded` itself.
   Neither is right (450). See `results/official-armv7-qemu-suite.jsonl`.
6. **The official `linux-armv7` binary cannot run on Jibo**: it needs `GLIBC_2.34`. It is a
   reference under emulation only.
7. **The published config ships 8-bit KV** (`kv_cache_bits: 8`, `kv_window: 256`); the Rust
   runtime defaults to f32. The runner keeps f32 as the reference and reports `kv_precision` on every
   response; int8 changed 2 of 21 decisions here.

## Changes to the runtime

On this branch, each its own commit with a regression test:

- `needle-infer`: streaming no longer slices inside a character (`stream_delta`); `V3Result` gains
  `prompt_tokens` and `V3Timing`; `prompt_ids()` for admission; `generate_controlled()` with a
  per-step `keep_going` and `StopReason::Cancelled`. The clock reads zero on
  `wasm32-unknown-unknown`, where `Instant::now` panics.
- `needle-infer` loader: `CactV3Geometry::check_bounds` (fields and u64 products, before anything
  is sized), tokenizer piece count bounded by its blob, zero Engram tables rejected.
- `needle-core`: `CqWeight::from_blob` checks its size arithmetic and enforces `MAX_GROUP`.

No numerics changed: the oracle comparisons above are identical before and after.

## On the robot

Nothing has been copied to or run on a Jibo. Running anything there, even from `/tmp`, is a
deployment step inside the owner's agreed scope. With `JIBO_SSH` (prefer `jibo-skill`),
`JIBO_DIR` (check `df -T` first: `/tmp` may be RAM), `BUILD`, `MODEL`, `OPS` and, for the GPU,
`JIBO_ENV` (the game host's `run.sh` display access and shim):

```bash
ports/jibo/scripts/robot-run.sh env      # glibc, libgcc_s, df -T, MemAvailable: confirms the ABI assumptions
ports/jibo/scripts/robot-run.sh deploy
ports/jibo/scripts/robot-run.sh smoke    # info + the handoff's fixture, with peak RSS
ports/jibo/scripts/robot-run.sh status   # before, and again after each step below
ports/jibo/scripts/robot-run.sh suite    # 21 requests, 1 warmup + 3 measured: physical Jibo CPU
ports/jibo/scripts/robot-run.sh ops      # scalar vs NEON on the real tensors
ports/jibo/scripts/robot-run.sh gl       # GL compute, --tt x --rows-per-dispatch sweep: physical Jibo GPU
ports/jibo/scripts/robot-run.sh cleanup
```

Produce `OPS` with `needle-jibo dump-op weights/needle3.cact --tensor NAME --tokens 16 --out ops/NAME`
for `l0.q_proj`, `l0.out_proj`, `embedding` and `l0.mlp.w1` (and `--tokens 1` for decode shapes).
Stage the steps: first the fixture alone, then the suite while Jibo is idle, then with normal
listening, eye and face tracking. Stop on service degradation, unbounded queues, abnormal
temperature or memory pressure.

## Not done yet

- **Every physical Jibo measurement**: latency, CPU, peak and combined memory, thermals, effect on
  the eye/vision/audio services, GL dispatch on the GK20A. Blocked on robot access.
- The NEON kernels inside the engine. They are bit-identical in the benchmark; wiring them into
  `needle-core` (a C helper behind an audited ABI, since stable Rust has no ARM32 NEON intrinsics)
  waits for `robot-run.sh ops` to show they are worth it.
- The GPU beyond one CQ projection and one Kronecker stage (attention, convolution and Engram
  state, a complete prefill), which waits for the measured dispatch costs.
- Task quality: 12 of 20 exact on the fixtures at full depth is not good enough to act on unchecked.
  Grounding checks, a confidence gate calibrated on held-out robot tasks, and the
  Decider as a second opinion are application work; none is done here.
- Integration with Oído (ASR) and the Decider: the runner's Unix socket and outcomes are the
  interface; no client is written yet.
