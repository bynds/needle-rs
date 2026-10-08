# needle-jibo-c: a C99 translation of the Needle 3 engine and runner

This is a C99 port of the Needle 3 inference path (`crates/needle-core`, `crates/needle-infer`) and
of the Jibo runner (`ports/jibo/runner`). Its target is the original Jibo: `armv7-unknown-linux-gnueabihf`,
glibc 2.21, a Linaro-era GCC. It needs no Rust toolchain on the build machine and no runtime
beyond libc, libm and libpthread.

**It is the same engine, not a lookalike.** Every number it computes is bit-identical to
needle-core, on x86-64 and on ARMv7, with and without NEON:

- cells and logits;
- the KV cache, in f32 and int8;
- the confidence head.

Every response the runner writes is byte-identical to the Rust runner's, except for measured times
and resource readings. Each claim below names the check that establishes it. Nothing has been run
on a robot.

| Evidence level | Meaning |
|---|---|
| **measured (x86-64)** | run on the build host |
| **measured (qemu)** | ARMv7 binary, glibc 2.21 stand-in sysroot, `qemu-arm` user-mode emulation |
| **not measured** | needs the robot |

## Layout

| File | Transcribes | Notes |
|---|---|---|
| `src/nd_cact.[ch]` | `needle-infer/src/cact.rs` | header, codebook and directory parsed with `check_bounds`; nothing is allocated from an unchecked field |
| `src/nd_cq.[ch]` | `needle-core/src/cq.rs` | CQ2/3/4 and ternary; LUT matvec, batched matmul, FWHT; NEON under `ND_NEON` |
| `src/nd_load.c`, `src/nd_model.[ch]` | `needle-infer/src/v3.rs`, `needle-core/src/v3/*` | canonical layout, ladder rungs, mHC, Engram, GQA with Q/K/V conv, RoPE, Kronecker MLP, probe heads, KV cache |
| `src/nd_math.[ch]` | Rust `libm` 0.2.16 (musl/FreeBSD) | `expf logf sinf cosf powf tanhf`, as needle-core links them |
| `src/nd_tok.[ch]`, `src/nd_prompt.[ch]` | `sp_tokenizer.rs`, `prompt.rs`, `tokenizer.rs` | SentencePiece BPE with byte fallback; chat template; `to_snake_case` with Rust 1.87 Unicode tables |
| `src/nd_grammar.[ch]` | `constrained.rs` | the tool-call grammar inside `<tool_call>` markers |
| `src/nd_engine.[ch]` | `v3_engine.rs` | prompt ids, prefill, greedy or SplitMix64 sampling, stop rules, streamed deltas, `confidence_for` |
| `src/nd_json.[ch]` | serde_json 1.0.149 | DOM parser and writer; zmij float formatting; serde's error text |
| `src/nd_catalog.[ch]`, `src/nd_validate.[ch]`, `src/nd_grounding.[ch]` | `runner/src/{catalog,validate,grounding}.rs` | catalogue, schema validation, grounding |
| `src/nd_service.[ch]`, `src/main.c` | `runner/src/{service,main}.rs` | admission, classification, confidence gate, `info run serve bench regrade` |
| `src/nd_obj.[ch]`, `src/nd_sha256.[ch]`, `src/nd_sysinfo.[ch]` | serde `Value` maps, `sha256.rs`, `sysinfo.rs` | sorted-key objects, the model hash and its sidecar, `/proc` readings and the busy gate |
| `tests/test_*.c` | | one parity test per module, against Rust-made references |
| `ref/` | | `nd-c-ref`, the Rust reference generators (standalone crate, never shipped) |

The port is about 15,200 lines of C. The tests are 5,800 lines of C and Rust.

## Build

```sh
make                                    # host build: build/needle-jibo-c, build/libnd.a
make build/test_model                   # any tests/test_*.c

# ARMv7 for Jibo, with the owner's toolchain wrapper (see ../README.md):
make CC=/path/to/jibo-armcc ARCH="-march=armv7-a -mfpu=neon -mfloat-abi=hard" OUT=build-arm
make ... CFLAGS="-O2 -DND_NEON"         # NEON CQ kernels (same bits)

# or with the stand-in toolchain used for the measurements below:
JIBO_SYSROOT=/path/to/sysroot make CC=$PWD/../scripts/jibo-cc-standin.sh OUT=build-arm
../scripts/check-jibo-abi.sh build-arm/needle-jibo-c
```

The C is plain C99 with `-D_POSIX_C_SOURCE=200809L`, and it compiles warning-free under
`-Wall -Wextra -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wmissing-prototypes -pedantic`. It
avoids anything a 2014 Linaro GCC 4.9 lacks: no C11 atomics, no `_Generic`, no VLAs. The pthread
use is one worker thread for `serve --socket`.

`-ffp-contract=off` is part of the contract and is always passed. A fused multiply-add rounds
once, where the Rust rounds twice, so a compiler allowed to contract `a * b + c` changes bits. NEON
builds use `vmlaq_f32`, which is non-fused, in Rust's lane order.

**Measured (qemu):** the stand-in ARMv7 binary needs at most `GLIBC_2.17`, so it is compatible
with Jibo's 2.21. Its shared libraries are `libm.so.6`, `libpthread.so.0`, `libc.so.6` and
`ld-linux-armhf.so.3`.

## Run

`needle-jibo-c` takes the Rust runner's options and writes its JSON. See `../README.md` for the
protocol and the client. `dump-op` stays in the Rust runner.

```sh
build/needle-jibo-c run   weights/needle3.cact --tools fixtures/tools.json --query "set a timer for ten minutes"
build/needle-jibo-c serve weights/needle3.cact --tools fixtures/tools.json --socket /tmp/needle.sock
```

## How bit-identity is kept

These rules hold in every file, and each test exists to catch a break in one of them.

- **f32 arithmetic in Rust's order.**
  - Literals carry an `f` suffix, and `-Wdouble-promotion` flags any silent widening.
  - Dot products use 8 lanes: lane *k* accumulates elements *k*, *k*+8, and so on. The lanes are
    then summed in order, starting from `-0.0`, as Rust's `.sum()` does. Group totals are
    `total += norm * group_sum`.
  - `f32::max` / `min` / `clamp` and the saturating `as u8` / `as u32` casts are reproduced,
    including NaN and out-of-range behaviour.
- **Rust's libm, not the platform's.** needle-core is `no_std` and links the `libm` crate, so
  `nd_math.c` is that code, transcribed. glibc's `expf` differs in the last bit for some inputs.
  `sqrtf` and `roundf` are correctly rounded everywhere, so `<math.h>` is used for those.
- **Where Rust uses std, C uses the platform.**
  - The engine's sampler and the confidence sigmoid call `f32::exp` in std, which is the platform
    `expf`; so does the C.
  - `{:.1}` / `{:.3}` formatting and glibc `printf` both round the exact binary value, half to even.
    The check that found this is in the history of `nd_service.c`.
- **serde_json's observable behaviour.** It decides key order (a `Value` map is a BTreeMap, so keys
  come out sorted), float text (zmij, not ryu), and the error messages the runner echoes
  (`not JSON: expected value at line 1 column 1`).

## Parity gates

`../scripts/c-check.sh` regenerates every reference with the Rust code itself, builds the tests and
runs them. `--arm` repeats every gate on ARMv7 under qemu, for both a plain build and an `ND_NEON`
build. `--san` repeats them under ASan+UBSan (`-fno-sanitize-recover=all`), and `--full` adds the
exhaustive corpora.

| # | Gate | Reference | Size | x86-64 | ARMv7 qemu |
|---|---|---|---|---|---|
| 1 | math | `ref/math_ref` (libm 0.2.16) | 39.1M cases; every 2^32 input of exp, log, sin, cos, tanh | 0 mismatches | 0 on the cases; exhaustive for exp, log and tanh; sin and cos only partial (see *Not identical*) |
| 2 | tokenizer and prompt | `ref/tok_ref` | 67k encodes, 1.19M decodes, 1,377 prompts, 64k snake-case names, 361 damaged blobs | 0 | 0 |
| 3 | JSON | `ref/json_ref` (serde_json 1.0.149) | 700k documents with error text; 40M floats; every f32 | 0 | 0 on 200k documents and 6M floats |
| 4 | grammar | `ref/grammar_ref` | 257,846 masks over 6,494 scenarios | 0 | 0 |
| 5 | runner modules | `ref/runner_ref` | 54,172 records (catalogues, validation, grounding, serde text) | 0 | 0, against an armv7 reference |
| 6 | CQ kernels | `needle-jibo dump-op` | q_proj, out_proj, embedding (8192×768), 16 tokens | 0 differing floats | 0 |
| 7 | model | `runner/examples/trace.rs` | a 113-token prompt: cells, all-position logits, prefill, 12 decode steps with f32 and int8 caches, confidence head; ladder depths 4 and 13 | 0 differing floats | 0, both plain and NEON |
| 8 | corruption | `runner/examples/corrupt_cases.rs` | 1,512 single-field corruptions | same verdict on every case, no crash | — |

The "0" entries are counts of differing bits. They are not tolerances, and gate 7 compares float
bit patterns: the cosine is exactly 1 and every argmax agrees. This is stricter than a "cosine >
0.999, same top token" gate. needle-core in turn matches the JAX reference at 6.853e-6
max |Δlogit| with 0 argmax mismatches, on both x86 and emulated ARMv7 (`../README.md`). So the C
inherits that numerical parity exactly, without a second tolerance.

### End-to-end: the request suites

`../scripts/c-parity.py` runs the Rust runner and the C runner on the same requests. It compares
every response line as bytes, with only the measured `timing` object removed (plus `load_ms`,
`resources` and `model_sha256_source` from health lines). It then re-grades the saved completions
with both runners under other policies.

| Run | Requests | Options | Differing lines | Regrade policies (differing) |
|---|---|---|---|---|
| `requests.jsonl` + `tools-extended.json` | 21 | `--debug-text --confidence` | **0** | strict, off, report with `--min-confidence 0.5`: **0, 0, 0** |
| `requests.jsonl` + `tools-extended.json` | 21 | `--debug-text --constrain --kv-int8` | **0** | |
| `c-parity-edge.jsonl` + `tools-extended.json` | 28 lines (27 answered, 1 blank) | `--debug-text --max-total-tokens 600` | **0** | |
| `suite-dev` + `suite-heldout` + `suite-tools.json` | 195 | `--debug-text --confidence` | **0** | strict, off with `--min-confidence 0.6`: **0, 0** |

**Measured (x86-64).** The edge file covers:
- a health line;
- non-JSON lines and serde error text;
- a non-object request;
- unknown keys, reported in sorted order with Rust's `{:?}` quoting;
- a bad `request_id`;
- an empty query, a query containing NUL, and an oversized query;
- an unknown tool and an empty tool list;
- out-of-range `max_new_tokens`, and `max_new_tokens: 1.0`;
- duplicate keys, and trailing garbage;
- a lone surrogate, and Cyrillic and emoji text;
- a 20,000-byte line;
- a prompt that does not fit.

### What the corruption gate found in the Rust runner

The C loader checks every FP16 vector against the length the forward pass indexes it at, and the
geometry against what the forward pass can run. Porting the Rust robustness test showed that the
Rust loader did neither. Of the 1,512 corruptions, 24 loaded and then aborted the process at the
first request. One of them aborted with a 17 GB allocation; with `panic = "abort"` the others
abort too. Another 19 loaded and ran on wrong-sized tensors.

`V3Model::new` now makes the same checks. `runner/tests/container_robustness.rs` now runs a
prefill on every container that loads, so a deferred abort fails the test. After the fix, the Rust
and C loaders agree on all 1,512 cases.

## Not identical, by design or by limit

- **Error text from the loader.** Loader errors are refused at the same points, but the wording is
  the C's (for example `tensor 26 holds 0 values, the geometry needs 6144`), not
  `V3LoadError`'s Display. These messages appear only on stderr at startup, never in a response.
- **Inputs Rust cannot receive.**
  - Invalid UTF-8 reaching the tokenizer, the catalogue or the grammar is refused with `ND_E_ARG`;
    a Rust `&str` cannot hold it.
  - An empty user-defined piece in a corrupt tokenizer is skipped. Rust's encoder loops forever
    on one.
- **Allocation failure** returns an error where Rust aborts.
- **`maxLength` above 2^32** is refused on 32-bit targets, as Rust's `usize::try_from` refuses it
  there.
- **Exhaustive ARM sweeps of `sinf` / `cosf`** were cut short under qemu, at about an hour per
  function. The x86 sweeps cover every input. On ARM the 39M-case set, which includes 12M sin/cos
  cases, matched.
- **`dump-op`** (kernel reference vectors) is a Rust-runner tool and was not ported.
- **Speed.**
  - **Measured (qemu), relative only:** the fixture query (113-token prompt, 34 generated tokens)
    on the ARMv7 builds under `qemu-arm`. Emulated time tracks instruction count loosely, and
    says nothing about Jibo's absolute speed.

    | Build | Prefill | Decode |
    |---|---|---|
    | Rust `needle-jibo` (plain) | 63.4 s | 20.2 s |
    | C `needle-jibo-c` (plain) | 63.0 s | 22.0 s |
    | Rust `needle-jibo-neon` | 63.4 s | 19.7 s |
    | C `needle-jibo-c -DND_NEON` | 55.1 s | 18.9 s |

    The first C build took 2.3× Rust's decode time. The LUT matvec was then instantiated per
    indices-per-byte, as Rust's const generic is, with the same operations in the same order.
  - **Measured (x86-64):** the host is not the target. Rust's x86 build dispatches its CQ kernels
    to AVX2-compiled copies by CPUID, while the C host build is baseline x86-64.
  - **Not measured:** on Jibo, timings need the robot.
- **NEON and subnormals.** ARMv7 NEON flushes subnormal floats to zero, while VFP and x86 do not.
  The `ND_NEON` kernels therefore equal the scalar ones only when no product or partial sum is
  subnormal. That holds on every gate input; Rust's `neon` feature has the same property, and the
  plain build is exact by construction. For a guarantee over every input, use the plain build.

## Brief cross-check

These items were checked against the implementation brief:

| Brief item | Status |
|---|---|
| Tag `0x05E12A84`, other tags refused; 196-byte `<48If` header; 44-byte records | `nd_cact.c`; tag, header and record sizes are constants checked at parse |
| Validate shapes, dtypes, bit widths, sizes and offsets before access | `nd_cact_parse` (bounds) and `nd_load.c` (shapes, lengths, geometry); gate 8 |
| Layout derived from the header, not a hard-coded 581 | the canonical walk derives the slots from the geometry. `needle3.cact` has 581 records: 115 CQ2 and 7 CQ4 (group 128), 456 FP16, 2 FP32 permutations, 1 RAW tokenizer |
| CQ2 main weights, CQ4 embedding / lane maps / probe matrices, FP16 elementwise | as read from the container, above |
| Activation fake-quant (int8) and KV-cache quantisation | int8 activation fake-quant runs on every path, as in needle-core. The container declares `kv_bits = 8`. The runner defaults to an f32 cache, exact to the reference; `--kv-int8` gives the int8 cache. Both are bit-identical to Rust (gate 7) |
| Context 8192, local window 1024 | read from the header (`max_seq_len`, `sliding_window`), never assumed |
| Depth ladder: keep endpoints, bisect the widest gap, ties to the left, run in ascending order | `ladder_order`; depth 4 runs `[0, 9, 14, 19]`; gates 7 at depths 4 and 13 |
| Retrieval detected from `heads.manifest` | only the confidence head is loaded; there is no embedding head, so no retrieval |
| Tokenisation exact | gate 2, plus gate 7's prompt ids |
| Numerical parity | bit-identical (gate 7) |
| Behavioural parity | identical JSON to the Rust runner (suites above). Comparing with upstream `needle_complete` would need the Python sandbox, which is **not measured** here |
| `<think>` first, grammar inside tool calls, grounding and confidence gates | as the Rust runner (the engine does not force `<think>`; the model emits it) |
