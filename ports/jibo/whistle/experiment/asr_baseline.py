"""Baseline: upstream Whistle's own transcription (public CLI), then the nearest of the 248 FSC
phrasings by string similarity, then that phrasing's intent. Also the same on real non-command
speech: how often a transcript lands close enough to a command to be (wrongly) accepted.

usage: asr_baseline.py SET N OUT.jsonl
"""
import csv
import difflib
import json
import os
import random
import re
import subprocess
import sys
import tempfile

W = "/tmp/claude-0/w"
NEEDLE = "/tmp/claude-0/upstream-x86/needle"
MODEL = "/tmp/claude-0/whistle-dl/whistle.cact"
FSC = f"{W}/data/fluent_speech_commands_dataset/data"


def norm(s):
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def phrasings():
    m = {}
    for s in ("train", "valid", "test"):
        for r in csv.DictReader(open(f"{FSC}/{s}_data.csv")):
            m[norm(r["transcription"])] = f"{r['action']}|{r['object']}|{r['location']}"
    return m


def transcribe(f32):
    with tempfile.NamedTemporaryFile(suffix=".wav") as t:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", "16000", "-ac", "1", "-i", f32, t.name], check=True)
        p = subprocess.run(["unshare", "-n", NEEDLE, "--model", MODEL, "--audio", t.name], capture_output=True, text=True)
        try:
            return json.loads(p.stdout.strip().splitlines()[-1])["text"]
        except Exception:
            return ""


def main():
    name, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    rows = [json.loads(l) for l in open(f"{W}/real/{name}.jsonl")]
    random.Random(5).shuffle(rows)
    ph = phrasings()
    keys = list(ph)
    with open(out, "w") as f:
        for r in rows[:n]:
            text = transcribe(r["path"])
            t = norm(text)
            scores = [(difflib.SequenceMatcher(None, t, k).ratio(), k) for k in keys]
            score, best = max(scores)
            f.write(json.dumps({"label": r["label"], "text": text, "match": best, "intent": ph[best], "score": score}) + "\n")


if __name__ == "__main__":
    main()
