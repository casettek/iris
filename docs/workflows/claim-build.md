# Claim Build Workflow

`claim build` reruns inference through the checkpointed staged path and writes the phase-two claim artifacts.

```bash
cargo run --release -p iris-cli -- claim build --run inference.toml
# or
just claim inference.toml
```

Flow:

1. The CLI loads the run spec and its model manifest, then verifies the weight, config, tokenizer, and model-specific template hashes once before preparing the run. Missing template provenance is an error.
2. Direct inference discovers the output ids. They must fill `tokens` unless the last one is EOS.
3. The prompt's pieces are tokenized natively and the tokens cut into ranges of at most the model's range size, and the ids into ranges of the same size. A prompt range may end between any two tokens.
4. The original text (`text`), the run's configuration (`config`), each prompt range's token marks (`marks_{j}`: one number per piece, 1 where a token starts) and each claimed range (`claimed_{k}`) are written as externals under `target/iris/runs/<run>/prompt/`, and a run-specific `Raster.toml` beside it commits to them. The configuration holds the model id, `raw_prompt`, `tokens` and the range size. The model id comes from the model manifest, which recomputes it from its commitments by role when it is loaded.
5. `staged-infer` reads the run manifest and expands the [chain](../internals/raster-chain.md).
6. Each stage synthesizes `input.json` and `input_manifest.json`, runs the host implementation, and writes Raster-compatible output artifacts.
7. Claim build applies the rule a challenger will. If the final record's committed verdict counts any claimed id that is not the greedy selection, claim build fails: direct and teacher-forced inference are not in parity, and the claim would refute itself.
8. `inference-artifacts` scans the completed chain directory and writes the claim files.

`prepared_run.json` holds no copy of the ids: the claim's output is the chain's final `output_finalize` record, which also holds the input text's SHA-256, the model id and the output text's SHA-256.

When `iris infer` has already run this spec, pass the ids it printed with
`--direct-output-ids 1,2,3` and step 2 is skipped: the claim is built from them
without a second direct pass. Nothing else changes. The ids must still fill
`tokens` unless the last is EOS, the chain re-derives every token, and ids that
are not the run's greedy output fail the build at step 7.

The hidden debug flag `--claimed-ids 1,2,3` claims the given ids instead of
discovering them. A rejection is then a warning, and the claim is written anyway.
This builds deliberately false claims for challenge tests.

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
  --stage range_p0_l3
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
three-claimed-range fixture that compares every checkpoint across cache
budgets, worker counts, and the diagnostic backend. To compare the same fixture
against actual authenticated Raster stage execution:

```bash
cargo build -p range-origin -p decode-init -p input-embedding -p prefill-prepare-aux \
  -p prefill-range -p sequence-check -p output-finalize --bins
STAGED_INFER_REPLAY_BIN_DIR="$PWD/target/debug" cargo test -p staged-infer \
  --test worker_parity repeated_ranges_match_authenticated_raster_stage_execution -- --ignored
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

A final record with a nonzero model fault is rejected even when native and
Raster checkpoints agree. Every model-consuming stage binds its actual component
to the descriptor carried by the expected model context. Transformer stages check
layer progression and prior/donor context; finalization requires a completed
same-model decode edge. Decoder text, special-token flags, and EOS flags are
authenticated at each consumed index, including entries after the first EOS.
Missing verification inputs or model context cannot produce an accepted record.

Sparse tokenizer/embedding reads authenticate selected complete buckets or pages.
Their separate witness files are inputs to the authorized stage programs; cache
hits do not bypass current witness checks. The tokenizer witness is kept for
the run beside the tokenizer, keyed by its commitment. Within one stage a merge
bucket is checked against the witness the first time it is read, and later
reads of that bucket in the same stage reuse the answer. Dense projections and transformer
weights retain complete sweeps. See the
[validation report](../proposals/model-identity-validation.md).
