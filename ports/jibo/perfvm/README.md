# perfvm: exact ARMv7 instruction counts

Both engines' cost on Jibo's instruction set, measured without a robot. `qemu-user`, which the
rest of the port uses, emulates fine but has no PMU: a guest's `perf_event_open` would count the
emulator's own x86 instructions. Full-system qemu does have one. With `-icount shift=0`, TCG
retires instructions deterministically, and the emulated PMU's instructions-retired event becomes
exact. The guest is an arm64 Ubuntu kernel (`fetch-kernel.sh`, pinned by sha256). It runs the
unmodified armhf binaries in AArch32 EL0 against the glibc 2.21 stand-in sysroot, from an
initramfs that `mkinitramfs.py` writes with no root access.

```sh
JIBO_SYSROOT=/path/to/sysroot perfvm/bench.sh LABEL [DECODE_STEPS]   # build, run, LABEL.jsonl
perfvm/compare.py RESULTS.jsonl [BASELINE.jsonl]                     # per-operation table
perfvm/run.sh JOBS DEST=SRC...                                        # any ARMv7 programs
```

`bench.sh` builds four variants from the tree: Rust and C, each plain (VFPv3-D16, as shipped)
and NEON. It runs three workloads on the handoff's 113-token prompt:
- **prefill:** one batched prefill;
- **decode:** N greedy decode steps;
- **prefix_hit:** the cost of a tool-prefix cache hit, which copies the stored state and steps
  the 16 suffix tokens.

Each line gives the total and, per operation, the instructions and span count. The engines'
spans are placed at the same points in the same taxonomy (needle-core `prof.rs`, C
`nd_prof.h`): embed, engram, mhc, qkv, conv_rope, attn, gate_out, mlp, head.
`selftest.c` checks the counter itself: a 6-instruction loop costs 6 per iteration, the same on
every run.

What the counts are not: time. Instructions on a Cortex-A15 differ in cost (a NEON VMLA, a cache
miss), and qemu models none of that. They are the right measure for comparing two kernels that
do the same work, which is what this is for. Absolute latency needs the robot.

Boot takes 3 s; one `bench.sh` run (four binaries) takes about 6 minutes on the build host.
