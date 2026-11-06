#!/usr/bin/env python3
"""Run authored Raster/native tokenizer fixtures and compare every checkpoint.

Build the staged-infer examples and the two Raster guest programs first.
This intentionally calls the real chain runner, including identity batches.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--raster', type=Path, required=True, help='cargo-raster executable')
parser.add_argument('--output', type=Path, default=root / 'target/tokenizer-validation')
parser.add_argument('--tokenizer', type=Path, default=root / 'tests/tokenizer/ranked-tokenizer.json')
parser.add_argument('--corpus', type=Path, default=root / 'tests/tokenizer/ranked-reference.json')
parser.add_argument('--case', action='append', dest='cases')
parser.add_argument('--authenticated', action='store_true')
parser.add_argument('--window', type=int, default=2, help='fraud-proof window; empty finalization has fewer than 32 steps')
args = parser.parse_args()
tool = args.raster.resolve()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
corpus = json.loads(args.corpus.read_text())
cases = args.cases or [c['name'] for c in corpus['cases']]
target = Path(os.environ.get('CARGO_TARGET_DIR', root / 'target')).resolve()
reports = []

def run(command, log):
    with log.open('w') as f:
        subprocess.run([str(s) for s in command], cwd=root, stdout=f, stderr=subprocess.STDOUT, check=True)

def storage(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())

for case in cases:
    fixture = out / case
    fixture.mkdir(exist_ok=True)
    run([target / 'release/examples/tokenizer_fixture', fixture, args.tokenizer.resolve(), args.corpus.resolve(), case], fixture / 'native.log')
    raster = fixture / 'raster'
    raster.mkdir(exist_ok=True)
    assert not any(raster.iterdir()), f'Use a fresh output directory: {raster}'
    started = time.monotonic()
    mode = ['--fraud-proof-window-size', args.window] if args.authenticated else ['--no-auth']
    run([tool, 'raster', 'chain', 'run', fixture / 'Raster.toml', '--run', raster, *mode], fixture / 'raster.log')
    wall = time.monotonic() - started
    run([target / 'release/examples/tokenizer_compare', fixture, raster], fixture / 'compare.log')
    if args.authenticated:
        run([tool, 'raster', 'chain', 'audit', fixture / 'Raster.toml', raster / 'chain-commitment', '--execution'], fixture / 'audit.log')
    report = json.loads((fixture / 'fixture.json').read_text())
    report.update(stage_count=report['repeat_count'] + 2, raster_seconds=wall,
                  raster_bytes=storage(raster), native_bytes=storage(Path(report['native_chain'])),
                  authenticated=args.authenticated, checkpoint_parity='passed')
    reports.append(report)
    (out / 'report.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(f'PASS {case}: {report["stage_count"]} checkpoints, Raster {wall:.2f}s, {report["raster_bytes"]} bytes', flush=True)

# A range needing more merge rounds than its batches allow. With the one batch
# that follows its seed taken away, it must still finish, through Raster and
# natively, with the same output recording the fault.
insufficient = out / 'insufficient-merges'
insufficient.mkdir(exist_ok=True)
extended = dict(corpus, cases=corpus['cases'] + [
    {'name': 'past-one-batch', 'text': 'ABCDEFGHIJKLMNOPQR' * 5, 'token_ids': [529] * 5}])
extended_path = insufficient / 'corpus.json'
extended_path.write_text(json.dumps(extended, ensure_ascii=False))
fixture = insufficient / 'fixture'
fixture.mkdir(exist_ok=True)
run([target / 'release/examples/tokenizer_fixture', fixture, args.tokenizer.resolve(), extended_path, 'past-one-batch'], insufficient / 'fixture.log')
manifest = insufficient / 'Raster.toml'
text = (fixture / 'Raster.toml').read_text()
assert text.count('count = 1\n') == 1, 'the fixture should need exactly one batch after its seed'
manifest.write_text(text.replace('count = 1\n', 'count = 0\n'))
raster = insufficient / 'raster'
raster.mkdir(exist_ok=True)
assert not any(raster.iterdir()), f'Use a fresh output directory: {raster}'
run([tool, 'raster', 'chain', 'run', manifest, '--run', raster, '--no-auth'], insufficient / 'raster.log')
with (insufficient / 'native.log').open('w') as log:
    subprocess.run([str(target / 'release/staged-infer'), 'chain', 'run'], cwd=insufficient, stdout=log, stderr=subprocess.STDOUT, check=True)
native = max((insufficient / 'target/staged-infer/runs').iterdir(), key=lambda p: p.name)
output = raster / 'prompt_prepare/output.bin'
assert output.read_bytes() == (native / 'prompt_prepare/output.bin').read_bytes(), 'Raster and native differ'
shown = subprocess.run([str(tool), 'raster', 'show', str(output)], cwd=root, capture_output=True, text=True, check=True).stdout
assert 'fault: 6' in shown, 'the stage did not record unfinished merges'
print('PASS insufficient merges: both finished with the same output, fault 6', flush=True)
