#!/usr/bin/env python3
"""Exercise sparse bucket recursion through Raster's real transition guest.

Requires cargo-raster, RISC Zero, and a retained authenticated stage fixture
(input.json, input_manifest.json, output.bin), e.g. from `iris test parity`.
Injects a deterministic trace fault; estimates its window without proving.
All generated artifacts stay under IRIS target/. Raster source is untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prompt-bind", "prompt-check"])
    parser.add_argument("fixture", type=Path)
    parser.add_argument("bucket", choices=["vocab", "merge"])
    parser.add_argument("--window", type=int, default=8)
    args = parser.parse_args()
    fixture = args.fixture.resolve(strict=True)
    project = ROOT / "raster-stages" / args.stage
    target = ROOT / "target" / "bucket-transition"
    target.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"{args.stage}-{args.bucket}-", dir=target))
    print(f"Regression artifacts: {output}", flush=True)

    def run(name, *arguments):
        log = output / f"{name}.log"
        with log.open("w") as stream:
            result = subprocess.run(["cargo", "raster", *map(str, arguments)],
                                    cwd=project, stdout=stream, stderr=subprocess.STDOUT)
        require(result.returncode == 0, f"{name} failed: {log}\n{log.read_text()[-6000:]}")
        return log.read_text()

    run("build", "build", "--backend", "risc0")
    run("verify", "program", "--verify")
    cfs = json.loads((project / "target/raster/cfs.json").read_text())
    site = f"identity_{args.bucket}bucket_entry"
    sequence = next(s for s in cfs["sequences"] if s["id"] == f"identity_{args.bucket}bucket_contents")
    recur = next(item["RecurTile"] for item in sequence["items"] if "RecurTile" in item)
    binding = recur["sources"][0]["Indexed"]
    require(binding["value"]["SequenceScope"]["input_index"] == 0, "bucket list must be a sequence input")
    require(len(binding["indexes"]) == 1 and binding["indexes"][0]["SequenceScope"]["input_index"] == 1,
            "bucket index must be declared in the recursion's CFS binding")
    inputs = ["--input", fixture / "input.json", "--input-manifest", fixture / "input_manifest.json"]
    commit = output / "disputed.bin"
    recorded = run("record", "run", *inputs, "--commit", commit,
                   "--fraud-proof-window-size", args.window, "--fraud-step", f"tile:{site}")
    # Fault injection changes the trace, never the actual computed result.
    artifacts = re.findall(r"Run artifacts dir: (.+)", recorded)
    require(len(artifacts) == 1, "missing recorded run directory")
    actual = Path(artifacts[0].strip()) / "output.bin"
    require(hashlib.sha256(actual.read_bytes()).digest() == hashlib.sha256((fixture / "output.bin").read_bytes()).digest(),
            "sequence restructuring changed the stage output")
    report_path = output / "estimate.json"
    run("estimate", "run", *inputs, "--audit", commit, "--estimate", "--report", report_path)
    report = json.loads(report_path.read_text())
    require(report["outcome"] == "estimated" and report["complete"] and report["error"] is None,
            "transition estimation did not complete")
    require(report["actual_step_count"] == report["measured_step_count"] == args.window,
            "the full requested window must execute")
    steps = report["steps"]
    # A named fault targets the first entry, or RecurEnd for an empty bucket.
    end = steps[-1]
    require(end["tile_id"] == site or end["step_kind"] == "RecurEnd", "wrong fault target")
    coordinates = end["coordinates"][:-1] if end["step_kind"] == "Exec" else end["coordinates"]
    require(any(s["step_kind"] == "RecurStart" and s["coordinates"] == coordinates and s["transition"]
                for s in steps), "window missed the bucket RecurStart transition")
    require(all(s["transition"] for s in steps), "a transition was not executed")
    require(report["final_journal"]["current_state"] == "Finished", "missing final transition verdict")
    print(f"PASS {args.stage}/{args.bucket}: {len(steps)} guest transitions; unchanged stage output", flush=True)


if __name__ == "__main__":
    main()
