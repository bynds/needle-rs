# Whistle on Jibo: closed-set voice commands (prototype)

Jev-style recognition: instead of free-form transcription, Whistle's speech embedding (public
`needle_embed`, one 512-wide row per 80 ms) is pooled and scored by a small classifier over a fixed
set of command templates plus "none", with a confidence threshold. Only Needle and Whistle are
used; upstream's engine is used as published, through `needle.h`, with no reverse engineering.

## Verdict

The approach works, within the limits below. Trained on synthetic speech only, the classifier
recognises real speakers it never heard better than Whistle's own transcription does, at about half
the ARM compute, and it runs on Jibo's ARMv7 and glibc 2.21 through upstream's public API.

| Measure | Result |
|---|---|
| Real test (FSC: 3,793 clips, 10 unseen speakers, 31 intents) | **97.65%** trained on synthetic speech only; 98.7% trained on 3,000 real clips (upper bound) |
| Same 500 real clips: classifier vs. Whistle transcription + nearest phrasing | 97.8% vs. 96.2%. At about 96% of commands accepted (threshold 0.7): **0.6% vs. 2.3% wrong actions** |
| Real off-topic speech accepted as a command (threshold 0.7) | 1.0% of 498 LibriSpeech clips (transcription + matching: 0 of 200) |
| Upstream engine on ARMv7 + glibc 2.21 | links and runs; binary needs nothing above GLIBC_2.18; embeddings match x86 (per-frame cosine ≥ 0.99993) |
| ARMv7 cost (perfvm, exact instructions) | embed ≈ 0.27 G per second of audio (1.2 s: 0.32 G; 3.7 s: 1.01 G); classify 0.4–0.6 M; transcription for comparison 1.0–2.0 G; load 0.13 G once |
| End to end on emulated ARMv7 (upstream ARM engine + `wi_intent.c`) | 199 of 200 clips get the host's label; 99.4% on the 180 real commands; no false accepts on 20 off-topic clips |

All numbers are in `results.json`.

## Limits and next steps

- **Microphones.** FSC is near-field phone and laptop audio. Jibo listens far-field through its
  own array, so a few dozen real commands recorded on the robot are the next test. The training
  augmentation (reverb, band limits, noise) is meant for that but is not proven there yet.
- **Time on the robot.** perfvm gives instruction counts, not seconds. At an assumed 1–2 G
  instructions per second on the Tegra K1, a 3–4 s command embeds in roughly 0.5–1 s. That has to
  be timed on the robot, owner-run.
- **Off-topic speech.** It was tested only on read audiobook sentences, which are long. Short
  near-misses ("turn the page", "lights were nice") are the harder case. Add them as negatives and
  test them; when the classifier says "none" or is unsure, fall back to transcription + Needle.
- **Slots.** Each FSC phrasing is a whole template. "Kitchen lights on" was once taken as
  plain "lights on". Templates with many values (timer durations) should get per-slot heads,
  Jev-style, rather than one class per expansion.
- **Templates and licences.** FSC's phrasings and audio are research-only and serve here only as a
  stand-in test. Jibo's templates should come from its own tool catalogue; the trained model is not
  committed.

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
