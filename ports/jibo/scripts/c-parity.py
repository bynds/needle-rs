#!/usr/bin/env python3
"""Byte-for-byte comparison of the Rust runner and the C99 runner (ports/jibo/c).

Runs `serve` (stdin) on both with the same model, catalogue, requests and options, and compares
every response line as bytes after removing only what is measured rather than computed: the
`timing` object, and `load_ms` / `resources` / `model_sha256_source` in health lines. Then
re-grades the Rust responses with both runners' `regrade` under other policies and compares those.

  c-parity.py --rust target/release/needle-jibo --c build/needle-jibo-c --model weights/needle3.cact \
      --tools ports/jibo/fixtures/tools-extended.json --requests ports/jibo/fixtures/requests.jsonl \
      --out target/c-parity [--label NAME] [-- extra runner options]

Exit 0 when every line is identical. Writes OUT/<label>.{rust,c}.jsonl and a summary JSON line.
`--runner-prefix "qemu-arm -L SYSROOT"` runs the C binary under emulation.
"""
import argparse, json, os, re, shlex, subprocess, sys, time

MEASURED = [re.compile(r',?"timing":\{[^{}]*\}'), re.compile(r',?"load_ms":[0-9.eE+-]+'),
            re.compile(r',?"resources":\{[^{}]*\}'), re.compile(r',?"model_sha256_source":"[a-z]+"')]


def strip(line):
    for r in MEASURED:
        line = r.sub('', line)
    return line.replace('{,', '{')


def run(cmd, stdin_path, out_path):
    t = time.time()
    with open(stdin_path, 'rb') as i, open(out_path, 'wb') as o:
        p = subprocess.run(cmd, stdin=i, stdout=o, stderr=subprocess.PIPE)
    if p.returncode != 0:
        sys.exit(f"{cmd[0]} failed ({p.returncode}): {p.stderr.decode(errors='replace')}")
    return time.time() - t


def compare(a_path, b_path):
    a = open(a_path, encoding='utf-8').read().splitlines()
    b = open(b_path, encoding='utf-8').read().splitlines()
    diffs = [i for i in range(max(len(a), len(b)))
             if i >= len(a) or i >= len(b) or strip(a[i]) != strip(b[i])]
    return len(a), len(b), diffs


def main():
    ap = argparse.ArgumentParser()
    for k in ['rust', 'c', 'model', 'tools', 'requests', 'out']:
        ap.add_argument('--' + k, required=True)
    ap.add_argument('--label', default='parity')
    ap.add_argument('--runner-prefix', default='')
    ap.add_argument('--regrade', action='append', default=[],
                    help='extra policy for regrade, e.g. "--grounding strict --min-confidence 0.5"')
    ap.add_argument('extra', nargs='*')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = [a.model, '--tools', a.tools] + a.extra
    rp, cp = (os.path.join(a.out, f'{a.label}.{w}.jsonl') for w in ('rust', 'c'))
    tr = run([a.rust, 'serve'] + base, a.requests, rp)
    tc = run(shlex.split(a.runner_prefix) + [a.c, 'serve'] + base, a.requests, cp)
    nr, nc, diffs = compare(rp, cp)
    summary = {'label': a.label, 'options': a.extra, 'lines_rust': nr, 'lines_c': nc,
               'differing': len(diffs), 'rust_s': round(tr, 1), 'c_s': round(tc, 1)}
    for i in diffs[:5]:
        print(f'line {i + 1} differs', file=sys.stderr)
    regrades = []
    for pol in a.regrade:
        opts = shlex.split(pol)
        outs = []
        for w, b in (('rust', a.rust), ('c', a.c)):
            pre = shlex.split(a.runner_prefix) if w == 'c' else []
            o = os.path.join(a.out, f'{a.label}.regrade{len(regrades)}.{w}.jsonl')
            with open(o, 'wb') as f:
                p = subprocess.run(pre + [b, 'regrade', rp, '--tools', a.tools, '--requests', a.requests] + opts,
                                   stdout=f, stderr=subprocess.PIPE)
            if p.returncode != 0:
                sys.exit(f'{w} regrade failed: {p.stderr.decode(errors="replace")}')
            outs.append(o)
        n1, n2, d = compare(*outs)
        regrades.append({'policy': pol, 'lines': [n1, n2], 'differing': len(d)})
        for i in d[:5]:
            print(f'regrade "{pol}" line {i + 1} differs', file=sys.stderr)
    summary['regrade'] = regrades
    print(json.dumps(summary))
    sys.exit(0 if not diffs and nr == nc and all(r['differing'] == 0 for r in regrades) else 1)


main()
