"""Compare two built iris binaries on one unchanged specification, case and checkpoint list.

Use an actual pre-change binary for --baseline, not --native-stage-subprocess: changing
the main-process backend would confound the comparison. No tracked files change.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def run(label, binary, specification, case, cwd, output):
    log = output / f"{label}.log"
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [str(binary), "claim", "build", "--specification", str(specification), "--case", str(case)],
            cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=False,
        )
    elapsed = time.perf_counter() - started
    if result.returncode:
        raise RuntimeError(f"{label} failed ({result.returncode}); see {log}")
    trace_paths = [line.removeprefix("checkpoint list: ").strip()
                   for line in log.read_text(encoding="utf-8").splitlines()
                   if line.startswith("checkpoint list: ")]
    if len(trace_paths) != 1:
        raise RuntimeError(f"{label} did not report exactly one checkpoint list")
    trace = Path(trace_paths[0])
    if not trace.is_absolute():
        trace = cwd / trace
    pipeline = trace.parent
    performance = pipeline / "pipeline-performance.json"
    return {
        "binary": str(binary), "wall_seconds": elapsed, "pipeline_dir": str(pipeline),
        "trace": json.loads(trace.read_text()),
        "checkpoint_hashes": (pipeline / "checkpoints.txt").read_text().splitlines(),
        "execution_times": json.loads((pipeline / "execution-times.json").read_text()),
        "performance": json.loads(performance.read_text()) if performance.exists() else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cwd, specification, output = args.cwd.resolve(), args.specification.resolve(), args.output.resolve()
    case = args.case.resolve()
    output.mkdir(parents=True, exist_ok=False)
    spec_hash = hashlib.sha256(specification.read_bytes() + case.read_bytes()).hexdigest()
    results = {}
    for label, binary in [("baseline", args.baseline), ("candidate", args.candidate)]:
        if hashlib.sha256(specification.read_bytes() + case.read_bytes()).hexdigest() != spec_hash:
            raise RuntimeError("specification or case changed during benchmark")
        results[label] = run(label, binary.resolve(), specification, case, cwd, output)
        (output / f"{label}.json").write_text(json.dumps(results[label], indent=2))
    matched = (results["baseline"]["trace"] == results["candidate"]["trace"]
               and results["baseline"]["checkpoint_hashes"] == results["candidate"]["checkpoint_hashes"])
    report = {
        "specification": str(specification), "case": str(case), "specification_and_case_sha256": spec_hash,
        "cache_budget_override": os.environ.get("IWEBS_PIPELINE_WEIGHT_CACHE_BYTES"),
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
