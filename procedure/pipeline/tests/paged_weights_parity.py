"""Compare preserved baseline, embedding-only, and current builds with authenticated Raster.

The three bin directories must contain iwebs and procedure-pipeline. The baseline and
current directories must also contain the unchanged Raster stage binaries.
No production eager switch is needed or introduced.
"""
import argparse
import json
import os
from pathlib import Path
import time

from selective_ple_parity import STAGES, read, sha, invoke, compare, expected_names, project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-bin", required=True, type=Path)
    parser.add_argument("--embedding-bin", required=True, type=Path)
    parser.add_argument("--bin-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    baseline, embedding, current = (p.resolve() for p in (args.baseline_bin, args.embedding_bin, args.bin_dir))
    root = Path(__file__).resolve().parents[4]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    suffix = ".exe" if os.name == "nt" else ""
    bundle = root / "model-bundles/parity-gemma"
    env = os.environ.copy()
    for key in ["RASTER_AUTH", "RASTER_TRACE_PATH", "RASTER_TRACE_FORMAT", "RASTER_OUTPUT_DIR",
                "RASTER_PARALLELISM"]:
        env.pop(key, None)
    env.update(RISC0_SKIP_BUILD="1", RAYON_NUM_THREADS="4", IWEBS_PIPELINE_PLE_LOADING="selective")
    artifacts = output / "model-artifacts"
    invoke([current / f"iwebs{suffix}", "model", "import", "--model", bundle, "--artifact-root", artifacts,
            "--model-id", "parity-gemma", "--no-stage-fixtures"], root, env, output / "import.log")
    model = artifacts / "parity-gemma/manifest.json"
    report = {"cases": [], "unchanged_raster_binaries": {}, "build_sha256": {},
              "timing_note": "Single small-model runs; use repeated multi-page benchmarks for loading comparisons."}
    for name in STAGES:
        a, b = sha((baseline / f"{name}{suffix}").read_bytes()), sha((current / f"{name}{suffix}").read_bytes())
        assert a == b, f"Raster binary changed: {name}"
        report["unchanged_raster_binaries"][name] = a
    for label, binaries in [("baseline", baseline), ("embedding-only", embedding), ("updated", current)]:
        report["build_sha256"][label] = {name: sha((binaries / f"{name}{suffix}").read_bytes())
                                        for name in ["iwebs", "procedure-pipeline"]}
    for case in read(bundle / "cases.json"):
        print(f"Running {case['name']}", flush=True)
        directory = output / case["name"]
        directory.mkdir()
        spec = directory / "specification.toml"
        spec.write_text(f"raw_prompt = {str(case['raw_prompt']).lower()}\nmax_new_tokens = {case['tokens']}\n\n"
                        f"[[model]]\nmanifest = {json.dumps(str(model))}\n", encoding="utf-8")
        case_file = directory / "case.toml"
        case_file.write_text(f"prompt = {json.dumps(case['prompt'], ensure_ascii=False)}\n", encoding="utf-8")
        record = {"name": case["name"], "runs": {}}
        pipelines = {}
        for label, binaries, budget, workers, parallel in [
            ("baseline", baseline, 64 << 20, 2, "on"),
            ("embedding-only", embedding, 64 << 20, 2, "on"),
            ("updated", current, 64 << 20, 2, "on"),
            ("updated-zero", current, 0, 1, "on"),
            ("updated-small", current, 32768, 2, "on"),
            ("updated-serial", current, 64 << 20, 1, "off"),
        ]:
            child_env = env | {"IWEBS_PIPELINE_WEIGHT_CACHE_BYTES": str(budget),
                               "IWEBS_PIPELINE_AUX_PARALLELISM": str(workers), "RASTER_PARALLELISM": parallel}
            seconds = invoke([binaries / f"iwebs{suffix}", "parity-worker", "--specification", spec, "--case", case_file,
                              "--output", directory / label], root, child_env, directory / f"{label}.log")
            pipeline = Path(read(directory / label / "result.json")["pipeline_dir"])
            pipelines[label] = pipeline
            perf = read(pipeline / "pipeline-performance.json")
            record["runs"][label] = {"seconds": seconds, "pipeline_dir": str(pipeline),
                                     "sampled_peak_rss_bytes": perf["sampled_peak_rss_bytes"],
                                     "budget_bytes": budget, "workers": workers, "parallel": parallel}
        source_pipeline = pipelines["baseline"]
        trace = read(source_pipeline / "checkpoint_list.json")
        names = [entry["stage"] for entry in trace]
        expected_names(names)
        for pipeline in pipelines.values():
            assert read(pipeline / "checkpoint_list.json") == trace
            hashes = compare(source_pipeline, pipeline, names)
            assert (pipeline / "checkpoints.txt").read_text().splitlines() == hashes
        replay = directory / "authenticated-raster"
        replay.mkdir()
        start = time.perf_counter()
        for name in names:
            source, destination = source_pipeline / name, replay / name
            destination.mkdir()
            inputs = read(source / "input.json")
            for entry in inputs.values():
                for field in ("path", "index_path"):
                    if field in entry:
                        try:
                            relative = Path(entry[field]).relative_to(source_pipeline)
                        except ValueError:
                            continue
                        entry[field] = str(replay / relative)
            (destination / "input.json").write_text(json.dumps(inputs), encoding="utf-8")
            (destination / "input_manifest.json").write_bytes((source / "input_manifest.json").read_bytes())
            replay_env = env | {"RASTER_AUTH": "1", "RASTER_OUTPUT_DIR": str(destination),
                                "RASTER_TRACE_PATH": str(destination / "trace.json"), "RASTER_TRACE_FORMAT": "json"}
            invoke([current / f"{project(name)}{suffix}", "--input", destination / "input.json",
                    "--input-manifest", destination / "input_manifest.json"], root, replay_env, destination / "run.log")
            compare(source_pipeline, replay, [name])
        record["authenticated_raster_seconds"] = time.perf_counter() - start
        record["checkpoint_hashes"] = compare(source_pipeline, replay, names)
        record["checkpoints_compared"] = len(names)
        record["passed"] = True
        report["cases"].append(record)
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"{case['name']}: {len(names)} checkpoints match all builds and Raster", flush=True)


if __name__ == "__main__":
    main()
