"""Compare two built iris binaries on one unchanged run spec and checkpoint trace.

Use an actual pre-change binary for --baseline, not --staged-subprocess: changing
the main-process backend would confound the comparison. No tracked files change.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def run(label, binary, spec, cwd, output):
    log = output / f"{label}.log"
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [str(binary), "claim", "build", "--run", str(spec)],
            cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=False,
        )
    elapsed = time.perf_counter() - started
    if result.returncode:
        raise RuntimeError(f"{label} failed ({result.returncode}); see {log}")
    trace_paths = [line.removeprefix("checkpoint trace: ").strip()
                   for line in log.read_text(encoding="utf-8").splitlines()
                   if line.startswith("checkpoint trace: ")]
    if len(trace_paths) != 1:
        raise RuntimeError(f"{label} did not report exactly one checkpoint trace")
    trace = Path(trace_paths[0])
    if not trace.is_absolute():
        trace = cwd / trace
    chain = trace.parent
    performance = chain / "staged-performance.json"
    return {
        "binary": str(binary), "wall_seconds": elapsed, "chain_dir": str(chain),
        "trace": json.loads(trace.read_text()),
        "checkpoint_hashes": (chain / "checkpoints.txt").read_text().splitlines(),
        "execution_times": json.loads((chain / "execution-times.json").read_text()),
        "performance": json.loads(performance.read_text()) if performance.exists() else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cwd, spec, output = args.cwd.resolve(), args.run.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    spec_hash = hashlib.sha256(spec.read_bytes()).hexdigest()
    results = {}
    for label, binary in [("baseline", args.baseline), ("candidate", args.candidate)]:
        if hashlib.sha256(spec.read_bytes()).hexdigest() != spec_hash:
            raise RuntimeError("run spec changed during benchmark")
        results[label] = run(label, binary.resolve(), spec, cwd, output)
        (output / f"{label}.json").write_text(json.dumps(results[label], indent=2))
    matched = (results["baseline"]["trace"] == results["candidate"]["trace"]
               and results["baseline"]["checkpoint_hashes"] == results["candidate"]["checkpoint_hashes"])
    report = {
        "run_spec": str(spec), "run_spec_sha256": spec_hash,
        "cache_budget_override": os.environ.get("STAGED_INFER_WEIGHT_CACHE_BYTES"),
        "checkpoint_parity": matched,
        "baseline_seconds": results["baseline"]["wall_seconds"],
        "candidate_seconds": results["candidate"]["wall_seconds"],
        "speedup": results["baseline"]["wall_seconds"] / results["candidate"]["wall_seconds"],
        "note": "One baseline-first pair; repeat with reversed binary order to assess OS-cache effects.",
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not matched:
        raise SystemExit("checkpoint parity failed")


if __name__ == "__main__":
    main()
