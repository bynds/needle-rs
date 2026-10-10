# Whistle on Jibo: closed-set voice commands (prototype, in progress)

Jev-style recognition: instead of free-form transcription, Whistle's speech embedding (public
`needle_embed`, one 512-wide row per 80 ms) is pooled and scored by a small classifier over a fixed
set of command templates plus "none", with a confidence threshold. Only Needle and Whistle are
used; upstream's engine is used as published, through `needle.h`, with no reverse engineering.

## Status

| Step | Result |
|---|---|
| Upstream engine on Jibo's ARMv7 + glibc 2.21 | `libneedle.a` (linux-armv7) links into our program against the 2.21 sysroot; the binary needs nothing above GLIBC_2.18. Embeddings on emulated ARMv7 match x86 (per-frame cosine ≥ 0.99993). |
| Cost on ARMv7 (perfvm, exact instructions) | load 0.13 G; embed ≈ 0.27 G per second of audio (1.2 s: 0.32 G, 3.7 s: 1.01 G); full transcription 1.0–2.0 G for the same clips |
| Embedding quality (upper bound) | logistic regression on mean/max/std-pooled embeddings, trained on 3,000 real Fluent Speech Commands clips: 98.7% intent accuracy on 3,793 clips from 10 unseen speakers |
| Synthetic-only training, real test | pending |

## Layout

- `wi_intent.[ch]`: the classifier in C99 (pooling, standardisation, linear softmax), model format `WIM1`.
- `wi_export.py`: converts a trained model (`experiment/train_eval.py --out`) to `WIM1`.
- `wi_run.c`: end-to-end driver: `needle_load` + `needle_embed` + `wi_classify`, with perfvm
  instruction counts per stage.
- `fetch-upstream.sh`: upstream weights and engine library, pinned by SHA-256.
- `shims/`: link shims for running upstream's library against glibc 2.21 with Ubuntu's armhf
  libc++ 18 (static): libc++ 19's byte hash, glibc 2.38's C23 number parsers, glibc 2.34's 64-bit
  time calls. None touches Whistle's arithmetic.
- `experiment/`: the host prototype. Piper TTS (LibriTTS-R, 904 speakers) with augmentation for
  training data, Fluent Speech Commands and LibriSpeech for real-speech tests (research licences:
  evaluation only, never shipped), the classifier trainer, and a transcribe-then-match baseline.
