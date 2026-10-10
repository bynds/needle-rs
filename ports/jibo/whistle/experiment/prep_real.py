import csv, io, json, os
import numpy as np, soundfile as sf
import pyarrow.parquet as pq
from scipy.signal import resample_poly
FSC = '/tmp/claude-0/w/data/fluent_speech_commands_dataset'
OUT = '/tmp/claude-0/w/real'
def put(name, items):
    d = f'{OUT}/{name}'; os.makedirs(d, exist_ok=True)
    with open(f'{OUT}/{name}.jsonl', 'w') as f:
        for i, (x, sr, meta) in enumerate(items):
            if x.ndim > 1: x = x.mean(1)
            if sr != 16000: x = resample_poly(x, 16000, sr)
            p = f'{d}/{i}.f32'; x.astype(np.float32).tofile(p)
            f.write(json.dumps({**meta, 'path': p, 'seconds': len(x) / 16000}) + '\n')
    print(name, i + 1, flush=True)
for s in ['test', 'valid', 'train']:
    rows = list(csv.DictReader(open(f'{FSC}/data/{s}_data.csv')))
    def gen():
        for r in rows:
            x, sr = sf.read(f"{FSC}/{r['path']}", dtype='float32')
            yield x, sr, dict(text=r['transcription'], label=f"{r['action']}|{r['object']}|{r['location']}", speaker=r['speakerId'])
    put(f'fsc_{s}', gen())
t = pq.read_table('/tmp/claude-0/w/data/ls500.parquet').to_pylist()
def gl():
    for r in t:
        x, sr = sf.read(io.BytesIO(r['audio']['bytes']), dtype='float32')
        yield x, sr, dict(text=r['text'], label='none', speaker=r['speaker_id'])
put('libri500', gl())
