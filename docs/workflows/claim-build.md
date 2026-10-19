# Claim Build Workflow

`claim build` reruns inference through the checkpointed staged path and writes the phase-two claim artifacts.

```bash
cargo run --release -p iris-cli -- claim build --run inference.toml
# or
just claim inference.toml
```

Flow:

1. The CLI loads the run spec and its model manifest, then verifies the weight, config, tokenizer, and model-specific template hashes once before preparing the run. Missing template provenance is an error; claims never fall back to the root `Raster.toml`.
2. Prompt pieces and `prepared_run.json` are written under `target/iris/runs/<run>/`.
3. A run-specific `Raster.toml` is generated in the same run directory with the requested prompt artifact and token count.
4. `staged-infer` reads the run manifest and expands the chain stages.
5. Each stage synthesizes `input.json` and `input_manifest.json`, runs the host implementation, and writes Raster-compatible output artifacts.
6. `inference-artifacts` scans the completed chain directory and writes the claim files.

Per-stage checkpoint files live under `target/staged-infer/runs/<run>/<stage>/`:

- `input.json`
- `input_manifest.json`
- `output.bin`
- `output.rindex`
- `output_manifest.json`

Claim files live beside the generated checkpoint tree:

- `checkpoint_trace.json`: ordered checkpoint records.
- `checkpoints.txt`: one SHA-256 per checkpoint record.
- `claim_bundle.json`: whole-claim input/output commitments plus the checkpoint trace and prepared-run metadata references.

`checkpoints.txt` is the compact public checkpoint reference used by the hash-based challenge flow. For manual challenge testing, copy and corrupt one hash:

```bash
cargo run --release -p iris-cli -- claim corrupt \
  --checkpoints target/staged-infer/runs/.../checkpoints.txt \
  --random
# or
just corrupt target/staged-infer/runs/.../checkpoints.txt
```

The helper writes a corrupted hash list plus a local `*.corruption.json` manifest that records the selected checkpoint for debugging. Stage-based selection is also available when `checkpoint_trace.json` is beside the hash list:

```bash
cargo run --release -p iris-cli -- claim corrupt \
  --checkpoints target/staged-infer/runs/.../checkpoints.txt \
  --stage decode_select_t42
```

The claim path must preserve per-stage artifacts even if the host execution becomes faster internally. Those files are the surface the challenge path verifies.

## Weight reuse and auxiliary workers

Normal checkpointed execution retains immutable materialized weights and prepared
transformer matrices across stages. Auxiliary stages use a run-scoped process
pool; layer `l` always uses worker `l % worker_count`. Each worker executes jobs
serially with a fresh Raster input resolver and stage scope. Auxiliary-wave
barriers, deterministic kernels, output encoding, and post-write checkpoint
verification remain unchanged. Model artifacts must remain immutable during a run.

The total retained-weight budget defaults to 25% of available RAM at startup,
with a reported 1 GiB fallback if memory detection is unavailable. Override it
with `STAGED_INFER_WEIGHT_CACHE_BYTES`; zero disables retention. Half goes to the
main process and half is divided among workers; runs without auxiliary waves
use the whole budget in the main process. `STAGED_INFER_AUX_PARALLELISM` sets the
worker count (default up to four, capped by available parallelism and wave size).

The budget covers owned weight allocations, including spare capacities and
duplicate prepared representations, with a conservative per-entry metadata
allowance. It does **not** bound total RSS, active-stage scratch, output caches,
or OS file caches. Admission is stable first-fit: full caches bypass new entries
instead of evicting every layer during a cyclic scan. A small budget therefore
retains only a subset of weights. Existing entry-count cache settings belong to
the legacy uncheckpointed helper and do not override this run-owned byte budget.
The unique prefill embedding stage bypasses retention: its embedding type differs
from decode's, so retaining that one-use value cannot produce a cache hit.

`staged-performance.json` records the allocation, auxiliary process starts,
per-stage load/kernel/encode-write timings, cumulative cache counters, and a
100 ms sampled peak RSS for the runner and its descendants. Weight preparation
time is a subset of kernel time, not an additional phase to sum. Cache snapshots
are cumulative per process: use the final main snapshot and the final snapshot
for each worker instead of summing stage snapshots. Existing
`execution-times.json` readers remain compatible. Diagnostic single-stage
subprocesses also write `staged-timings.json`.

Worker errors fail the run, terminate peers, and reap the pool; jobs are never
silently retried. The hidden `--staged-subprocess` diagnostic backend retains its
per-stage process behavior. Authenticated Raster reference runs remain separate
from unauthenticated native checkpoint generation.

## Validation and benchmarking

`cargo test -p staged-infer -p inference-kernels -p iris-cli` includes a two-layer,
three-decode-iteration fixture that compares every checkpoint across cache
budgets, worker counts, and the diagnostic backend. To compare the same fixture
against actual authenticated Raster stage execution:

```bash
cargo build -p input-embedding -p prefill-prepare-aux -p prefill-range \
  -p prefill-finalize -p decode-init -p decode-select-token -p decode-embed \
  -p output-finalize --bins
STAGED_INFER_REPLAY_BIN_DIR="$PWD/target/debug" cargo test -p staged-infer \
  --test worker_parity repeated_decode_matches_authenticated_raster_stage_execution -- --ignored
```

Use an actual pre-change release binary for performance comparisons; comparing
against `--staged-subprocess` would also change the main-stage execution backend.
Keep the model artifacts and run spec identical:

```bash
python scripts/benchmark-staged-reuse.py --baseline /path/to/old/iris \
  --candidate target/release/iris --run /path/to/300-in-128-out.toml \
  --output target/staged-benchmark
```

The harness saves both logs, timings, and trace/hash comparisons. It reports
measured speedup only after both runs complete; fixture parity alone is not a
production performance result. Repeat with reversed binary order to check for
OS-cache effects. Native Windows builds use `scripts/build-windows.ps1`.
