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
    run([tool, 'raster', 'chain', 'run', fixture / 'Raster.toml', '--run', raster, '--fraud-proof-window-size', args.window, *([] if args.authenticated else ['--no-auth'])], fixture / 'raster.log')
    wall = time.monotonic() - started
    run([target / 'release/examples/tokenizer_compare', fixture, raster], fixture / 'compare.log')
    if args.authenticated:
        run([tool, 'raster', 'chain', 'audit', fixture / 'Raster.toml', raster / 'chain-commitment', '--execution'], fixture / 'audit.log')
    report = json.loads((fixture / 'fixture.json').read_text())
    report.update(stage_count=report['repeat_count'] + 2, raster_seconds=wall,
                  raster_bytes=storage(raster), native_bytes=storage(Path(report['native_chain'])),
                  authenticated=args.authenticated, checkpoint_parity='passed')
    if case == 'after-eight':
        # Ten pieces require nine successful merges: one batch must fail.
        insufficient = fixture / 'insufficient'
        insufficient.mkdir()
        text = (fixture / 'Raster.toml').read_text()
        assert text.count('count = 1\n') == 1
        manifest = insufficient / 'Raster.toml'
        manifest.write_text(text.replace('count = 1\n', 'count = 0\n'))
        replay = insufficient / 'raster'
        replay.mkdir()
        with (insufficient / 'run.log').open('w') as log:
            rejected = subprocess.run([str(tool), 'raster', 'chain', 'run', str(manifest), '--run', str(replay), *([] if args.authenticated else ['--no-auth'])], cwd=root, stdout=log, stderr=subprocess.STDOUT)
        assert rejected.returncode != 0
        assert 'tokenization budget exhausted' in (insufficient / 'run.log').read_text()
        assert not (replay / 'prompt_prepare/output.bin').exists()
        report['insufficient_budget_rejected'] = True
    reports.append(report)
    (out / 'report.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(f'PASS {case}: {report["stage_count"]} checkpoints, Raster {wall:.2f}s, {report["raster_bytes"]} bytes', flush=True)
