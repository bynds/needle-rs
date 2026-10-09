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
| `runner/` (`needle-jibo`) | One loaded model, bounded requests on stdin/stdout or a 0660 Unix socket with a bounded queue. Authorized tool catalogue, admission on the real prompt length, deadlines, memory/thermal `busy` gates, schema validation, explicit outcomes, phase timings. Grounding check (`grounding.rs`). Serial (no rayon). Also `bench`, `regrade` and `dump-op`. |
| `runner/tests/container_robustness.rs` | 1,511 single-field corruptions of the real container; every one must be refused or loaded, never panic, and every one that loads must survive a prefill. Runs under ARMv7 emulation. |
| `c/` (`needle-jibo-c`) | A C99 translation of the engine and the runner, bit-identical to needle-core and byte-identical to `needle-jibo` (see `c/README.md`). Needs only libc, libm and libpthread. |
| `runner/examples/trace.rs`, `corrupt_cases.rs` | Reference stages (cells, logits, prefill, decode, head) and loader verdicts for the C port's gates. |
| `scripts/c-check.sh`, `scripts/c-parity.py` | The C port's parity gates (x86-64, ASan+UBSan, ARMv7 under qemu) and the runner-against-runner request comparison. |
| `client/needle-client.js` | ES5 client for the runner's socket (built-in `net` only): only a `candidate` reaches a handler, all or nothing. `needle-client.test.js` tests it. |
| `../../crates/needle-core/src/cq_neon.rs` | The engine's ARMv7 NEON CQ kernels (feature `neon`, with `-C target-feature=+neon`); `needle-jibo-neon` is built with them. |
| `backend/jibo-cq-bench.c` | One real operation (a CQ projection, a learned Kronecker MLP stage, or causal grouped-query attention) on scalar C, NEON and desktop-GL 4.3 compute, checked against the Rust engine's output and timed with uploads, fences and readback counted. |
| `scripts/fetch-sysroot.sh` | A stand-in glibc 2.21 armhf sysroot (Ubuntu 15.04 packages, sha256-pinned). |
| `scripts/jibo-cc-standin.sh` | A linker/compiler driver with the same contract as the owner's `jibo-armcc`. |
| `scripts/build-jibo.sh` | The ARMv7 build: each package in its own cargo invocation, ABI check, feature tree, manifest. |
| `scripts/check-jibo-abi.sh` | Will it load on Jibo: ELF32 ARM, hard-float ABI, `v7`, the armhf loader, allowed `NEEDED`, nothing above `GLIBC_2.21` or `GCC_4.8.0`, no libstdc++. |
| `scripts/gate.sh` | The release gate: fails, not skips, when the model or oracle vectors are absent. |
| `scripts/prepare-robot.sh` | One checksummed bundle with everything a robot session needs. |
| `scripts/preflight.sh` | Runs on the robot (POSIX sh): checksums, loader, glibc, libgcc_s, that each binary starts, memory, tmpfs. |
| `scripts/robot-run.sh` | The owner-run robot steps; preflight before each. |
| `scripts/score.py` | Scores runner output against `fixtures/expected.jsonl`. |
| `tools/make_synthetic_v3.py` | A seeded-random container at the shipped geometry and bit scheme, written by upstream's own exporter, for when the model cannot be fetched. |
| `fixtures/` | The handoff's `tools.json` and `query.txt` (byte-stable), a four-tool catalogue and 21 requests with expectations, and `make_suite.py`'s 195 requests over six tools, split into dev (tune here) and held-out (report here). Mock tools only: none is a verified Jibo API. |
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

A response (real output, host reference, full depth, f32 KV; `grounded` as the current runner
reports it):

```json
{"request_id":"num-digits","status":"candidate",
 "calls":[{"name":"start_timer","arguments":{"seconds":90}}],
 "model_sha256":"c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38",
 "depth":20,"kv_precision":"f32","constrained":false,
 "prompt_truncated":false,"stop_reason":"im_end","schema_valid":true,"grounded":true,
 "confidence_raw":null,
 "tokens":{"prompt":201,"generated":29,"budget":256,"positions":230,"prefix_reused":0},
 "timing":{"queue_ms":0.0,"wall_ms":2846.094,"tokenize_ms":4.26,"prefill_ms":2328.113,
           "decode_ms":508.621,"first_token_ms":2337.454,"confidence_ms":null}}
```

`status` is one of:

| Status | Meaning |
| --- | --- |
| `candidate` | The turn ended on its own (`im_end`/`eos`), both markers closed, every call passes the schema, and (by default) every number and free-text argument is something the query says. Still only a proposal: intent and policy are the application's. |
| `no_call` | The model answered `[]`: a considered abstention, distinct from every failure below. |
| `needs_clarification` | A call omitted a required argument, or (grounding) an argument the query does not state: `ungrounded` names it and `rejected_calls` holds the proposal, so the application can ask. |
| `low_confidence` | Only with `--min-confidence`: the confidence head scored the candidate below it. Off by default (it did not pay on the dev split). |
| `unsupported` | A call named a tool outside the request's offered set. |
| `invalid_output` | No marker, an unterminated marker, malformed JSON, or an argument of the wrong type, out of range, outside its enum, or undeclared. `detail` says which. |
| `incomplete` | The token budget ran out. The engine's own extractor would hand back the half payload; the runner does not. |
| `truncated` | Refused before compute: the prompt does not fit `--max-total-tokens` with `--min-new-tokens` to spare. |
| `timeout` | The deadline passed while queued or between tokens (a prefill in flight completes first). |
| `busy` | Queue full, `MemAvailable` below `--min-avail-mb`, or the thermal zone above `--max-temp-c`. |
| `invalid_request` | The request line is malformed, names an unknown key or tool, or exceeds a bound. |

**Tool-prefix cache** (on by default; `--no-prefix-cache` turns it off). Everything up to and
including `</tools>` (BOS, the system turn, the catalogue) is the same for every request with the
same tool set, and on a small catalogue it is most of the prompt. The runner keeps the KV cache
state after that prefix. A request with the same prefix and cache precision starts from a copy
and steps only its own tokens; any other request computes its prefix and replaces the stored one.
One entry is kept, and `tokens.prefix_reused` reports how many prompt tokens came from it.

The results are bit-identical with and without the cache. One prefill of the prefix followed by
decode steps over the rest leaves the same cache and logits as one prefill of the whole prompt.
`crates/needle-infer/tests/v3_prefix_cache.rs` and the C port's `test_model` hold it to that for
both cache precisions. **Measured (x86-64):** on the handoff fixture after a first request with the
same catalogue, 97 of 113 prompt tokens are reused and prefill falls from 994 ms to 177 ms.
**Measured (qemu), relative only:** for the same pair on the ARMv7 C build, prefill falls from
37.7 s to 6.3 s and the request's wall time from 51.0 s to 20.0 s. On the 21-request suite, 20 of
21 requests are served from the cache; on the 195-request suite, 194 of 195. Every response is
identical with the cache on and off, in both runners, and the two runners agree byte for byte with
it on (`results/host-reference-c-parity.jsonl`).

The catalogue may use `string` (with `enum`, `maxLength`), `integer` and `number` (with
`minimum`, `maximum`) and `boolean`, and `required`. Anything else (`pattern`, arrays, nested
objects, `anyOf`, `additionalProperties`, ...) is refused at startup rather than ignored. The
catalogue's own bytes, key order included, go into the prompt. Nothing is logged;
`--debug-text` adds the completion to responses.

**Grounding** (`--grounding enforce`, the default). A schema cannot tell "set a timer for seven
and a half minutes" → 950 s from 450 s. The runner reads the query for what it states and requires
each argument to be one of those: a `*second*` parameter a stated duration ("2 minutes and 15
seconds" = 135, "an hour and a half" = 5400), `hour`/`minute` a stated clock time ("7:30 pm",
"half past seven", "noon"), any other number a stated number or an alias (mute/off = 0, max = the
schema maximum), free text a substring. `strict` also requires enum values to be said ("wave" for
"Wave hello!"). `report` annotates only; `off` skips it. English number words only: a Spanish or
French request asks for clarification instead. `needle-jibo regrade` re-applies a policy to saved
responses without the model.

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
| Fixture suite, ARMv7 binary vs host | identical text, status, calls and token counts on every request | emulated ARM, `results/emulated-arm-suite-f32-full.jsonl` |
| NEON build (`needle-jibo-neon`), fixture suite | identical text, token counts, status and calls on all 21 requests | emulated ARM, `results/emulated-arm-suite-neon.jsonl` |
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

### The 195-request suite and grounding (G6)

`fixtures/make_suite.py`, six mock tools (timer, cancel, alarm with hour and minute, volume,
animation, weather by city), 98 dev and 97 held-out requests, about a quarter of them negatives
(small talk, numbers that are not requests, absent tools, out-of-range values, missing values).
The policy was chosen on dev (grounding on, no confidence gate) before the held-out split was
scored. Full depth, f32 KV, host reference.

| | dev exact | dev unsafe | **held-out exact** | **held-out unsafe** |
| --- | --- | --- | --- | --- |
| schema validation only (`--grounding off`) | 62 / 97 | 33 | 47 / 97 | **42** |
| grounding (default) | 65 | 5 | 51 | **5** |
| grounding `strict` | 66 | 4 | 52 | 3 |
| grounding + `--min-confidence 0.3` (dev only) | 63 | 4 | | |

Unsafe means the runner would have handed the application a `candidate` that differs from the
expectation. Without grounding that is 43% of held-out requests; with it, 5%. On both splits
grounding refused no correct call except the two non-English ones ("Pon un temporizador de cinco
minutos", "Mets un minuteur de dix minutes"), which become `needs_clarification`. What still gets
through is intent, not values: "My dog is called Max" → weather for Max, "Five minutes ago I ate
lunch" → a five-minute timer, "I said I don't want to dance" → dance, "What's the capital of
France?" → weather for France. Those need a policy above the runner (an addressed-to-the-robot
check, the Decider as a second opinion), measured on real robot requests. The confidence head
did not separate right from wrong well enough to pay on dev: at 0.3 it removed one more unsafe
answer and two correct ones. Even with grounding, 51 of 97 held-out requests are exactly right;
the rest are mostly refusals the application must handle (`needs_clarification`, `no_call`).

`results/host-reference-suite195-{dev,heldout}-f32.jsonl`; re-grade them with any policy:
`needle-jibo regrade FILE --tools fixtures/suite-tools.json --requests fixtures/suite-heldout.jsonl --grounding strict`.

### Operations: CPU and GPU (G5, first pass)

`jibo-cq-bench` on real tensors of the pinned model, 16 tokens of seeded activations, against the
Rust kernels' output (`needle-jibo dump-op`):

| Operation | ref (C, Rust order) | NEON | GL compute |
| --- | --- | --- | --- |
| `l0.q_proj` 576x768 CQ2 | bit-identical (host and emulated ARM) | bit-identical (emulated ARM) | 1.3e-6 of RMS, cosine 1.0 (llvmpipe) |
| `l0.out_proj` 768x768 CQ2 | bit-identical | bit-identical | 1.5e-6 at every `--tt` 1/8/16 x rows/dispatch 64/256/all (llvmpipe) |
| `embedding` 8192x768 CQ4 (the tied LM head) | bit-identical | bit-identical | 2.3e-6 (llvmpipe) |
| `l0.mlp.w1` learned 32x32 Kronecker stage | bit-identical | bit-identical | bit-identical (llvmpipe) |
| causal GQA attention, 64 positions, 12 heads / 2 KV, global and 16-wide window | 1.0e-6 (glibc `expf` vs the engine's `exp`) | — | 1.6–2.0e-6, online softmax, 1 to 10 dispatches (llvmpipe) |

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

## Performance work: both engines, exact ARMv7 instruction counts

`perfvm/` measures both engines on Jibo's instruction set without a robot. It runs the ARMv7
binaries in full-system qemu with `-icount`, so the emulated PMU's instruction count is exact and
repeatable. Each engine carries matching per-operation spans (needle-core `prof.rs`, C
`nd_prof.h`; opt-in, compiled out otherwise). `perfvm/bench.sh` runs the same three workloads on
both engines, plain (VFPv3-D16, as shipped) and NEON: a 113-token prefill, 8 decode steps, and a
tool-prefix cache hit. Each change below was kept only if every bit-parity gate still held: the C
and Rust traces on x86 and ARMv7, plain and NEON, and the C99 gates.

**Measured (perfvm), instructions:**

| Build | Prefill: before → now | Decode, 8 steps: before → now |
|---|---|---|
| Rust plain | 35.3 G → 26.5 G (−25%) | 3.01 G → 2.74 G (−9%) |
| C plain | 47.5 G → 25.7 G (−46%) | 2.49 G → 2.35 G (−6%) |
| Rust NEON | 15.6 G → 9.7 G (−38%)\* | 1.51 G → 1.02 G (−33%)\* |
| C NEON | 17.9 G → 9.4 G (−48%)\* | 1.65 G → 0.97 G (−41%)\* |

\* The first NEON measurement already includes the first three changes below.
`results/emulated-arm-perf-*.jsonl` has every operation.

Each engine taught the other something:

- **C → Rust: compute only the head rows that are used.** Rust's prefill ran the 8192 × 768
  logits head for every prompt position and kept the last; `forward_head` (confidence scoring)
  and `forward_cells` ran it for every position and kept none. They now compute the last row or
  none. Head cost in prefill fell 98%.
- **C → Rust: register lanes on 32-bit ARM.** The C kernels' eight named accumulators beat
  LLVM's code for the staged lane arrays in the plain build: −5.6% on the 2-bit projections.
  On the 4-bit head they cost 3.9%, so that width keeps the staged form.
- **Rust → C: the same, the other way.** GCC did not keep lane arrays in registers anywhere.
  The C batched matmul used 1.9× Rust's instructions until it got named locals and an unrolled,
  per-width LUT decode: C prefill −42% in one step.
- **Rust → C: Rust's NEON assembly.** GCC compiles the arm_neon.h intrinsics for the eight-lane
  loops to 11 instructions per step against the asm's 7. C now uses Rust's exact kernels as
  `__asm__`: C NEON prefill −23%.
- **Both: one lane norm for three gates.** The mHC post and residual gates read the same
  normalised lanes as the pre gate, but each recomputed the norm and the Hadamard preparation.
  One now serves all three: mHC −29 to −42%.
- **Both: loop interchange and a shared NEON `axpy`.** mHC mixing, the Kronecker MLP stages and
  attention's accumulate were all "add a scaled row" in disguise. Interchanged to whole-row
  loops (each element keeps its additions in the same order), they share one `axpy` kernel,
  NEON in both languages and unrolled in plain C: MLP −65% and attention −45 to −55% on NEON.
- **Both: two rows per pass in the NEON matvec.** Decode is mostly LUT matvecs. A kernel that
  takes two weight rows at once, each with its own accumulators and lane order, shares every
  activation load and the loop control: NEON decode −8 to −11% for the 2-bit projections, and
  the logits head −10 to −12% for the 4-bit ones.
- **Both: no division in attention.** The 32-bit ARM baseline has no divide instruction, so the
  ring-slot `%` in each attention step was a library call per cached position and head. The slot
  is now stepped: attention −4 to −7% in decode.
- **Both: batched Engram projections.** The Engram key and value projections ran a matvec per
  position, each preparing the same input. They now use the batched matmul (bit-identical by
  construction) with one shared preparation: Engram −11 to −34%.

Every change keeps the bits. The batched and per-position paths sum in the same lane order;
interchanged loops keep each element's additions in order; NEON's VMLA rounds the product and
the sum separately, as the scalar code does. The one exception is the existing NEON caveat:
NEON flushes subnormals to zero.

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
5. **Three references, and the serving policy matters more than the arithmetic.** On the 21
   fixtures with the four-tool catalogue, the official `linux-armv7` binary (under emulation) emits
   the same calls as the pinned Rust engine wherever it emits any; the difference is its serving
   layer, which suppresses calls it judges ungrounded or unconfident. That turns three wrong Rust
   candidates (small talk and "do a backflip" as 30-minute timers, an invented 300 s for "Set a
   timer.") into `[]`, and also drops one correct call (the Spanish five-minute timer). Its
   `validation.ungrounded` flag marks all six schema-valid but wrong Rust answers, and four right
   ones; its raw `confidence` ranges 0.008 to 0.95 and is not calibrated. A grounding check of that
   kind belongs in the application, measured on robot tasks. The prompt matters too: with the
   two-tool `tools.json`, "seven and a half minutes" gives `7.5` (Rust) and `750` (official, flagged
   ungrounded, confidence 0.47); with four tools both give 950. The right answer is 450.
   `results/official-native-emulated-suite.jsonl`.
6. **The official `linux-armv7` binary cannot run on Jibo**: it needs `GLIBC_2.34`. It is a
   reference under emulation only.
7. **Schema-valid is not safe.** Without grounding, 42 of 97 held-out requests produce an
   executable wrong call; grounding the arguments in the query brings that to 5. See the table above.
8. **The published config ships 8-bit KV** (`kv_cache_bits: 8`, `kv_window: 256`); the Rust
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
- `needle-infer`: `V3Engine::enable_prefix_cache` keeps the cache state after the prompt's tool
  prefix and starts later requests with the same prefix from a copy (`V3Cache` is now `Clone`);
  `V3Result::prefix_reused`. Bit-identical results (`tests/v3_prefix_cache.rs`). The runner turns
  it on unless `--no-prefix-cache`.
- `needle-core`: `V3Model::new` checks every decoded vector against the length the forward pass
  indexes it at, and the geometry against what it can run. Before this, 24 of the robustness
  test's corruptions loaded and then aborted at the first request (one through a 17 GB
  allocation), and 19 ran on wrong-sized tensors. The C port's loader found this; both loaders now
  agree on all 1,511 cases.
- `needle-core`: ARMv7 NEON CQ kernels (`cq_neon.rs`) behind the `neon` feature, in stable inline
  assembly. Default builds are unchanged. Stable Rust 1.87 sets no `cfg(target_feature)` for ARM
  features, so the feature is the switch and must come with `-C target-feature=+neon` (rustc warns
  that `neon` is an unstable `-Ctarget-feature`; it is applied, and the build fails without it).
  Same lane order and non-fused `VMLA` as the scalar kernels: bit-identical in a unit test under
  emulation; NEON flushes denormals, the one possible difference.
- `needle-infer` test: `cells_and_confidence_match_the_reference` judges each residual cell
  against its own RMS (it failed at upstream on one global RMS; the worst cell is 4.6e-5).

No numerics changed in the default build: the oracle comparisons above are identical before and
after.

## On the robot

Nothing has been copied to or run on a Jibo. Running anything there, even from `/tmp`, is a
deployment step inside the owner's agreed scope. First, on the development host:

```bash
JIBO_SYSROOT=/opt/jibo-sysroot ports/jibo/scripts/prepare-robot.sh out/bundle   # or JIBO_CC=...
```

That builds `needle-jibo`, `needle-jibo-neon`, `needle-rs` and `jibo-cq-bench`, checks their ABI,
adds the model, fixtures, client, reference vectors for the measured operations and
`preflight.sh`, and writes `MANIFEST.sha256` (about 50 MB, mostly the model). Then, with `JIBO_SSH` (prefer
`jibo-skill`), `JIBO_DIR` (the preflight reports the filesystem: `/tmp` may be RAM), `BUNDLE` and,
for the GPU, `JIBO_ENV` (the game host's `run.sh` display access and shim):

```bash
ports/jibo/scripts/robot-run.sh env        # before copying: glibc, libgcc_s, mounts, MemAvailable, CPU
ports/jibo/scripts/robot-run.sh deploy     # copies the bundle, then the full preflight (every sha256)
ports/jibo/scripts/robot-run.sh smoke      # info + the handoff's fixture, with peak RSS
ports/jibo/scripts/robot-run.sh status     # before, and again after each step below
ports/jibo/scripts/robot-run.sh suite                       # 21 requests: physical Jibo CPU
BIN=needle-jibo-neon ports/jibo/scripts/robot-run.sh suite  # the same with the NEON kernels
ports/jibo/scripts/robot-run.sh ops        # scalar vs NEON on the real tensors
ports/jibo/scripts/robot-run.sh gl         # GL compute sweep: physical Jibo GPU
ports/jibo/scripts/robot-run.sh suite-big  # the 195 requests, when there is time
ports/jibo/scripts/robot-run.sh cleanup
```

Every step after `deploy` runs `preflight.sh --quick` first and stops if it fails. Stage the
steps: first the fixture alone, then the suite while Jibo is idle, then with normal listening,
eye and face tracking. Stop on service degradation, unbounded queues, abnormal temperature or
memory pressure.

## Not done yet

- **Every physical Jibo measurement**, for `needle-jibo-c` as for the Rust runner: latency, CPU, peak and combined memory, thermals, effect on
  the eye/vision/audio services, GL dispatch on the GK20A, and whether `needle-jibo-neon` is faster.
  Blocked on robot access; the bundle and steps are ready.
- The GPU beyond single operations (one CQ projection, one Kronecker stage, the attention core):
  projections, convolution, RoPE, Engram and the cache on the device, and a complete prefill. That
  waits for measured dispatch costs on the robot.
- Intent errors that grounding cannot see (statements with numbers, names taken as cities,
  negation). An addressed-to-the-robot check or the Decider as a second opinion would sit above
  the runner; neither is built, and both need real robot requests to measure.
- Grounding for languages other than English.
- Integration with Oído (ASR) and the Decider beyond the socket and the client.
