import csv, json, random, re
import pyarrow.parquet as pq
FSC = '/tmp/claude-0/w/data/fluent_speech_commands_dataset'
rows = [r for s in ['train', 'valid', 'test'] for r in csv.DictReader(open(f'{FSC}/data/{s}_data.csv'))]
phr = sorted({(r['transcription'].strip(), f"{r['action']}|{r['object']}|{r['location']}") for r in rows})
assert len({p for p, _ in phr}) == 248, len(phr)
rnd = random.Random(7)
train_spk, test_spk = list(range(0, 800)), list(range(800, 904))
jobs = {'train_pos': [], 'train_neg': [], 'synth_test': []}
seed = 1000
for text, lab in phr:
    for s in rnd.sample(train_spk, 40):
        seed += 1; jobs['train_pos'].append(dict(id=f'p{seed}', text=text, label=lab, speaker=s, seed=seed, augment=True))
    for s in rnd.sample(test_spk, 4):
        seed += 1; jobs['synth_test'].append(dict(id=f't{seed}', text=text, label=lab, speaker=s, seed=seed, augment=True))
# negatives: wikitext sentences (4-18 words), one random training speaker each, plus noise-only clips
lines = pq.read_table('/tmp/claude-0/w/data/wikitext_train.parquet').column('text').to_pylist()
sents = []
for l in lines:
    l = l.strip()
    if not l or l.startswith('='): continue
    l = l.replace(' @-@ ', '-').replace(' @,@ ', ',').replace(' @.@ ', '.')
    l = re.sub(r'\s+([,.;:!?\)])', r'\1', l).replace('( ', '(')
    for s in re.split(r'(?<=[.!?])\s+', l):
        w = s.split()
        if 4 <= len(w) <= 18 and re.fullmatch(r"[A-Za-z0-9 ,.;:'\"!?()-]+", s) and '<unk>' not in s:
            sents.append(s)
rnd.shuffle(sents)
for i, s in enumerate(sents[:1800]):
    seed += 1; jobs['train_neg'].append(dict(id=f'n{seed}', text=s, label='none', speaker=rnd.choice(train_spk), seed=seed, augment=True))
for i in range(300):
    seed += 1; jobs['train_neg'].append(dict(id=f'z{seed}', text=None, label='none', speaker=0, seed=seed, augment=True))
for i, s in enumerate(sents[1800:2100]):
    seed += 1; jobs['synth_test'].append(dict(id=f'tn{seed}', text=s, label='none', speaker=rnd.choice(test_spk), seed=seed, augment=True))
for k, v in jobs.items():
    with open(f'/tmp/claude-0/w/gen/{k}_jobs.jsonl', 'w') as f:
        for j in v: f.write(json.dumps(j) + '\n')
    print(k, len(v))
print('labels', len({l for _, l in phr}), 'sample neg:', sents[:3])
