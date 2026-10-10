"""Synthetic training data for the template classifier: Piper TTS (LibriTTS-R, 904 speakers) plus
acoustic augmentation, written as 16 kHz mono float32 PCM with a JSONL manifest.

usage: gen_tts.py OUTDIR SET_NAME JOBS.jsonl WORKERS
JOBS.jsonl lines: {"id": ..., "text": ..., "label": ..., "speaker": int, "seed": int, "augment": bool}
"""
import json
import os
import sys
import multiprocessing as mp

import numpy as np
from scipy.signal import resample_poly, fftconvolve, butter, sosfilt

VOICE = "/tmp/claude-0/w/data/libritts_r.onnx"
SR = 16000
_voice = None


def colored_noise(n, rng, kind):
    w = rng.standard_normal(n)
    if kind == "white":
        return w
    f = np.fft.rfft(w)
    k = np.arange(len(f)) + 1.0
    f /= np.sqrt(k) if kind == "pink" else k
    x = np.fft.irfft(f, n)
    return x / (np.std(x) + 1e-9)


def augment(x, rng):
    # silence padding
    pre, post = rng.uniform(0.05, 0.5, 2)
    x = np.concatenate([np.zeros(int(pre * SR)), x, np.zeros(int(post * SR))])
    # room reverb: exponentially decaying noise tail, random RT60 and wet level
    if rng.random() < 0.6:
        rt60 = rng.uniform(0.15, 0.8)
        n = int(rt60 * SR)
        t = np.arange(n) / SR
        ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
        ir[0] = 1.0 / rng.uniform(0.3, 1.0)  # direct path relative to the tail
        y = fftconvolve(x, ir)[: len(x)]
        x = y / (np.max(np.abs(y)) + 1e-9) * np.max(np.abs(x))
    # channel: telephone band or a random low-pass, plus a DC/rumble high-pass
    r = rng.random()
    if r < 0.25:
        x = sosfilt(butter(4, [300, 3400], "bandpass", fs=SR, output="sos"), x)
    elif r < 0.55:
        x = sosfilt(butter(4, rng.uniform(3500, 7500), "lowpass", fs=SR, output="sos"), x)
    x = sosfilt(butter(2, 80, "highpass", fs=SR, output="sos"), x)
    # additive coloured noise at a random SNR
    snr = rng.uniform(5, 35)
    noise = colored_noise(len(x), rng, rng.choice(["white", "pink", "brown"]))
    p_sig = np.mean(x ** 2) + 1e-12
    x = x + noise * np.sqrt(p_sig / (10 ** (snr / 10)))
    # level
    peak = 10 ** (rng.uniform(-24, -1) / 20)
    return x / (np.max(np.abs(x)) + 1e-9) * peak


def synth(job):
    global _voice
    from piper import PiperVoice, SynthesisConfig
    if _voice is None:
        _voice = PiperVoice.load(VOICE)
    rng = np.random.default_rng(job["seed"])
    if job["text"] is None:  # noise/silence only
        x = np.zeros(int(rng.uniform(0.8, 3.0) * SR))
        x[0] = 1e-6
    else:
        cfg = SynthesisConfig(
            speaker_id=job["speaker"],
            length_scale=float(rng.uniform(0.8, 1.25)),
            noise_scale=float(rng.uniform(0.4, 0.9)),
            noise_w_scale=float(rng.uniform(0.5, 1.0)),
        )
        audio = np.concatenate([c.audio_float_array for c in _voice.synthesize(job["text"], cfg)])
        x = resample_poly(audio.astype(np.float64), 320, 441)  # 22050 -> 16000
    if job["augment"]:
        if job["text"] is None:
            x = np.zeros_like(x)
            snr_floor = colored_noise(len(x), rng, rng.choice(["white", "pink", "brown"]))
            x = snr_floor / (np.max(np.abs(snr_floor)) + 1e-9) * 10 ** (rng.uniform(-60, -20) / 20)
        else:
            x = augment(x, rng)
    path = os.path.join(OUT, f"{job['id']}.f32")
    x.astype(np.float32).tofile(path)
    return {**job, "path": path, "seconds": len(x) / SR}


def init(out):
    global OUT
    OUT = out
    os.environ.setdefault("OMP_NUM_THREADS", "1")


if __name__ == "__main__":
    out, name, jobs_path, workers = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    jobs = [json.loads(l) for l in open(jobs_path)]
    with mp.Pool(workers, initializer=init, initargs=(d,)) as pool, open(os.path.join(out, f"{name}.jsonl"), "w") as f:
        for i, r in enumerate(pool.imap_unordered(synth, jobs, chunksize=8)):
            f.write(json.dumps(r) + "\n")
            if (i + 1) % 1000 == 0:
                print(name, i + 1, flush=True)
