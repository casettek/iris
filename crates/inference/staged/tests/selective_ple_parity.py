"""Release parity gate against unchanged authenticated Raster stage binaries.

Requires a built iris, staged-infer, and all ten stage binaries in --bin-dir.
Writes only to a new --output directory. Uses the pinned parity-gemma bundle.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checkpoint(directory, name):
    return {
        "stage": name,
        "input_commitment": sha((directory / "input_manifest.json").read_bytes()),
        "output_commitment": read(directory / "output_manifest.json")["output"]["commitment"],
        "output_sha256": sha((directory / "output.bin").read_bytes()),
    }


def checkpoint_hash(value):
    # Same field order and compact JSON as serde_json::to_vec(Checkpoint).
    return sha(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode())


def compare(left, right, names):
    hashes = []
    for name in names:
        for filename in ("output.bin", "output.rindex", "output_manifest.json", "input_manifest.json"):
            assert (left / name / filename).read_bytes() == (right / name / filename).read_bytes(), f"{name}/{filename} differs: {left} vs {right}"
        a, b = checkpoint(left / name, name), checkpoint(right / name, name)
        assert a == b, f"{name}: checkpoint fields differ"
        assert checkpoint_hash(a) == checkpoint_hash(b), f"{name}: checkpoint hashes differ"
        hashes.append(checkpoint_hash(a))
    return hashes


def invoke(command, cwd, env, log):
    start = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run([str(x) for x in command], cwd=cwd, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"exit {result.returncode}: {command}; see {log}")
    return time.perf_counter() - start


def project(name):
    if name.startswith("prompt_merge_"):
        return "prompt-merge"
    if name.startswith(("prefill_prepare_aux_l", "decode_aux_t")):
        return "prefill-prepare-aux"
    if name.startswith(("prefill_range_l", "decode_range_t")):
        return "prefill-range"
    if name.startswith("decode_select_t"):
        return "decode-select-token"
    if name.startswith("decode_embed_t"):
        return "decode-embed"
    if name.startswith("decode_finalize_t"):
        return "prefill-finalize"
    return name.replace("_", "-")


def expected_names(names, tokens):
    # Independent inventory for the four-layer bundle, including the final
    # decode pass after the last selection. Tokenizer merge count is dynamic.
    merges = [n for n in names if re.fullmatch(r"prompt_merge_b\d+", n)]
    assert merges == [f"prompt_merge_b{i}" for i in range(len(merges))]
    expected = ["prompt_merge_seed", *merges, "prompt_prepare", "input_embedding"]
    expected += [f"prefill_prepare_aux_l{i}" for i in range(4)]
    expected += [f"prefill_range_l{i}" for i in range(4)]
    expected += ["prefill_finalize", "decode_init"]
    for token in range(tokens):
        expected += [f"decode_select_t{token}", f"decode_embed_t{token}"]
        expected += [f"decode_aux_t{token}_l{i}" for i in range(4)]
        expected += [f"decode_range_t{token}_l{i}" for i in range(4)]
        expected += [f"decode_finalize_t{token}"]
    expected += ["output_finalize"]
    assert names == expected, "missing, reordered, or unexpected stage"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    binaries, output = args.bin_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    suffix = ".exe" if os.name == "nt" else ""
    iris = binaries / f"iris{suffix}"
    bundle = root / "model-bundles/parity-gemma"
    env = os.environ.copy()
    for key in ["RASTER_AUTH", "RASTER_TRACE_PATH", "RASTER_TRACE_FORMAT", "RASTER_OUTPUT_DIR",
                "DIRECT_INFER_COMPARE_TRACE", "STAGED_INFER_PLE_LOADING"]:
        env.pop(key, None)
    env.update(RISC0_SKIP_BUILD="1", RAYON_NUM_THREADS="4")
    artifacts = output / "model-artifacts"
    invoke([iris, "model", "import", "--model", bundle, "--artifact-root", artifacts,
            "--model-id", "parity-gemma", "--no-stage-fixtures"], root, env, output / "import.log")
    model = artifacts / "parity-gemma/manifest.json"
    report = {"authenticated_raster": True, "cases": [], "binary_sha256": {},
              "timing_note": "Informational single runs, eager first; eager/selective use the same cache budget and worker count."}
    for binary in ["iris", "staged-infer", "prompt-prepare", "prompt-merge", "input-embedding",
                   "prefill-prepare-aux", "prefill-range", "prefill-finalize", "decode-init",
                   "decode-select-token", "decode-embed", "output-finalize"]:
        report["binary_sha256"][binary] = sha((binaries / f"{binary}{suffix}").read_bytes())
    for case in read(bundle / "cases.json"):
        print(f"Running {case['name']}", flush=True)
        directory = output / case["name"]
        directory.mkdir()
        spec = directory / "run.toml"
        spec.write_text(f"model_manifest = {json.dumps(str(model))}\nprompt = {json.dumps(case['prompt'], ensure_ascii=False)}\nraw_prompt = {str(case['raw_prompt']).lower()}\ntokens = {case['tokens']}\n", encoding="utf-8")
        record = {"name": case["name"], "runs": {}}
        chains = {}
        for label, mode, budget, workers in [
            ("eager", "eager", 64 << 20, 2),
            ("selective", "selective", 64 << 20, 2),
            ("selective-zero", "selective", 0, 1),
            ("selective-small", "selective", 32768, 2),
        ]:
            child_env = env | {"STAGED_INFER_PLE_LOADING": mode, "STAGED_INFER_WEIGHT_CACHE_BYTES": str(budget), "STAGED_INFER_AUX_PARALLELISM": str(workers)}
            seconds = invoke([iris, "parity-worker", "--run", spec, "--output", directory / label], root, child_env, directory / f"{label}.log")
            chain = Path(read(directory / label / "result.json")["chain_dir"])
            chains[label] = chain
            perf = read(chain / "staged-performance.json")
            assert perf["ple_loading"] == mode
            record["runs"][label] = {"seconds": seconds, "chain_dir": str(chain), "sampled_peak_rss_bytes": perf["sampled_peak_rss_bytes"], "mode": mode, "budget_bytes": budget, "workers": workers}
        baseline = chains["eager"]
        trace = read(baseline / "checkpoint_trace.json")
        names = [entry["stage"] for entry in trace]
        expected_names(names, case["tokens"])
        for chain in chains.values():
            assert read(chain / "checkpoint_trace.json") == trace
            hashes = compare(baseline, chain, names)
            assert (chain / "checkpoints.txt").read_text().splitlines() == hashes
        replay = directory / "authenticated-raster"
        replay.mkdir()
        start = time.perf_counter()
        for name in names:
            source, destination = baseline / name, replay / name
            destination.mkdir()
            inputs = read(source / "input.json")
            for entry in inputs.values():
                for field in ("path", "index_path"):
                    if field in entry:
                        path = Path(entry[field])
                        try:
                            relative = path.relative_to(baseline)
                        except ValueError:
                            continue
                        entry[field] = str(replay / relative)
            (destination / "input.json").write_text(json.dumps(inputs), encoding="utf-8")
            (destination / "input_manifest.json").write_bytes((source / "input_manifest.json").read_bytes())
            replay_env = env | {"RASTER_AUTH": "1", "RASTER_OUTPUT_DIR": str(destination),
                                "RASTER_TRACE_PATH": str(destination / "trace.json"), "RASTER_TRACE_FORMAT": "json"}
            invoke([binaries / f"{project(name)}{suffix}", "--input", destination / "input.json", "--input-manifest", destination / "input_manifest.json"], root, replay_env, destination / "run.log")
            # Stop at the first mismatch before consuming the next checkpoint.
            compare(baseline, replay, [name])
        record["authenticated_raster_seconds"] = time.perf_counter() - start
        record["checkpoint_hashes"] = compare(baseline, replay, names)
        record["checkpoints_compared"] = len(names)
        record["passed"] = True
        report["cases"].append(record)
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"{case['name']}: {len(names)} checkpoints match", flush=True)
    print(f"Report: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
