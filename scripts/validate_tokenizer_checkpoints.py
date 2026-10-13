#!/usr/bin/env python3
"""Execute every long-prompt tokenizer checkpoint through authored Raster programs.

Independent jobs use native predecessor outputs, as challenge preparation does.
Every predecessor is itself checked against its executed Raster result. Exact
input/output checkpoint equality therefore establishes the whole dataflow by
induction. No identity batches are skipped. The assembled comparison directory
is test output, not an authenticated chain claim. Use validate_tokenizer.py for
sequential chain execution and execution audits. The default uses selected-stage
CLI replay; --program-dir launches prebuilt stages with the CLI's no-auth setup.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import time

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('fixture', type=Path)
parser.add_argument('--raster', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--jobs', type=int, default=4)
parser.add_argument('--program-dir', type=Path, help='use prebuilt Raster stage executables, avoiding a Cargo build per checkpoint')
args = parser.parse_args()
assert args.jobs > 0
fixture, out, tool = args.fixture.resolve(), args.output.resolve(), args.raster.resolve()
meta = json.loads((fixture / 'fixture.json').read_text())
native = Path(meta['native_chain'])
names = ['prompt_merge_seed'] + [f'prompt_merge_b{b}' for b in range(meta['repeat_count'])] + ['prompt_prepare']
assert [s['name'] for s in json.loads((native / 'execution-times.json').read_text())['stages']] == names
out.mkdir(parents=True, exist_ok=True)
assert not any(out.iterdir()), 'Use a fresh output directory'
combined = out / 'checkpoints'
combined.mkdir()
started = time.monotonic()

def replay(index):
    name = names[index]
    work = out / 'replays' / name
    work.mkdir(parents=True)
    if args.program_dir:
        stage = work / name
        stage.mkdir()
        for file in ['input.json', 'input_manifest.json']:
            shutil.copy2(native / name / file, stage / file)
        project = 'prompt-prepare' if name == 'prompt_prepare' else 'prompt-merge'
        program = args.program_dir.resolve() / project
        # Match Raster CLI's unauthenticated launch: explicit mode, output
        # directory, and the recorded committed input argument files.
        env = dict(os.environ, RASTER_AUTH='0', RASTER_OUTPUT_DIR=str(stage))
        began = time.monotonic_ns()
        with (work / 'run.log').open('w') as log:
            subprocess.run([str(program), '--input', str(stage / 'input.json'), '--input-manifest', str(stage / 'input_manifest.json')], cwd=root / 'raster-stages' / project, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        timing = {'name':name, 'exec_duration_ns':time.monotonic_ns()-began}
        shutil.copytree(stage, combined / name)
        return index, timing
    if index:
        previous = names[index-1]
        producer = work / previous
        producer.mkdir()
        for file in ['output.bin', 'output.rindex', 'output_manifest.json']:
            shutil.copy2(native / previous / file, producer / file)
    with (work / 'run.log').open('w') as log:
        subprocess.run([str(tool), 'raster', 'chain', 'run', str(fixture / 'Raster.toml'), '--no-auth', '--run', str(work), '--stage', name], cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    timing = json.loads((work / 'execution-times.json').read_text())
    stage = next(s for s in timing['stages'] if s['name'] == name)
    shutil.copytree(work / name, combined / name)
    return index, stage

stages = {}
pool = ThreadPoolExecutor(max_workers=args.jobs)
try:
    pending = [pool.submit(replay, i) for i in range(len(names))]
    for future in as_completed(pending):
        index, timing = future.result()
        stages[index] = timing
        if len(stages) % 25 == 0 or len(stages) == len(names):
            print(f'Executed {len(stages)}/{len(names)} Raster checkpoints', flush=True)
finally:
    pool.shutdown(wait=True, cancel_futures=True)
(combined / 'execution-times.json').write_text(json.dumps({'stages': [stages[i] for i in range(len(names))]}, indent=2))
with (out / 'compare.log').open('w') as log:
    subprocess.run([str(root / 'target/release/examples/tokenizer_compare'), str(fixture), str(combined)], cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
report = dict(meta, validation_mode='independent prebuilt Raster stage executions' if args.program_dir else 'independent selected-stage CLI replays',
              stage_count=len(names), parallel_jobs=args.jobs, raster_validation_wall_seconds=time.monotonic()-started,
              raster_stage_timings=[stages[i] for i in range(len(names))],
              raster_checkpoint_bytes=sum(p.stat().st_size for p in combined.rglob('*') if p.is_file()),
              native_bytes=sum(p.stat().st_size for p in native.rglob('*') if p.is_file()), checkpoint_parity='passed')
if args.program_dir:
    report['stage_binary_sha256'] = {name: hashlib.sha256((args.program_dir / name).read_bytes()).hexdigest() for name in ['prompt-merge', 'prompt-prepare']}
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print((out / 'compare.log').read_text(), flush=True)
