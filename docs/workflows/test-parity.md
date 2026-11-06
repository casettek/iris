# Three-path parity

Run the local regression gate from the repository root:

```sh
just test-parity
# Equivalent:
cargo run --release --locked -p iris-cli -- test-parity
```

The command runs the pinned synthetic Gemma model through the complete Raster
chain (`--no-auth`), the checkpointed native-staged chain, and native-direct
generation. Each execution owns its intermediate outputs and runtime state.
All three fixed cases run every time. There are no numerical tolerances.

Independent exact-ID tokenizer fixtures and authenticated tokenizer-chain checks
are described in [ranked BPE](../proposals/ranked-bpe-tokenizer.md).

The sibling `../raster` checkout and its normal Rust/RISC Zero build prerequisites
must be available, as required by this workspace's path dependencies. The gate
builds that checkout's Raster CLI into `target/parity-tools`; it does not use the
installed `cargo-raster`. It builds inference stage binaries in release mode
under `target/parity-host` and runs `det-num`'s independent arithmetic vectors.
These isolated host builds use RISC Zero's `RISC0_SKIP_BUILD=1` option to omit
unused proof guests and enable incremental compilation. Normal inference and
proving build settings are unaffected; the report records these build settings.

The first invocation includes compilation. A warmed invocation reuses compiled
binaries, but still imports the model bundle freshly and reruns every inference path.
Elapsed time is informational; there is no timing threshold or CI integration.

## What must agree

The gate imports with `--range-size 6` (six tokens per range), so every case's
prompt and output split into several ranges, including ranges inside and past the
16-token sliding window. Preparation discovers each output with direct decode, and
the chain then checks it.

- Raster/native-staged: every expected checkpoint's canonical output bytes and
  structural commitment, with recorded payload/index roots checked independently.
- Direct's own coverage: pinned input IDs, every complete fixed-point logit vector
  used for selection, argmax selection, selected-token history, active PLE, cache
  growth/eviction, and position advancement.
- The chain and direct: the prompt ranges' token ids concatenate to direct's
  tokenization; `output_finalize` is byte-identical to direct's record; the last
  range's check is byte-identical to direct's last `decode_select_t*`.
- Position by position, stitching each layer's ranges together: every
  `embed_*`, `aux_*_l*` and `range_*_l*` row at position `i` equals direct's
  prefill row (`i < prompt_len`) or decode step `i - prompt_len`'s single row, and
  every range starts where the previous one ended. Every K/V entry a range publishes,
  inherited or its own, equals direct's entry at the same position. Direct runs no
  pass for its last token, so the chain's row for it has no reference. Nothing is
  scored from that row, and it still matches between Raster and native-staged.
- Layers 2 and 3 of every range borrow layers 0 and 1 of the same range.
- The last prompt context: no fault, complete, and holding the prompt's SHA-256, the
  model id and the prompt's token count.
- The final record: its context is the last prompt context's, its
  `output_text_sha256` is the SHA-256 of its own text, its faults and verdict are
  clean, and the native result artifact and direct's result are exactly this record.
- Over wrong cuts, on the templated case: the prompt stages finish through both
  Raster and native-staged, agree byte for byte, and record the expected fault.
  This includes cuts out of order and a range left out.
- Over a 3 KiB text of single-byte atoms: `prompt_bind` finishes through both and
  agrees byte for byte. Raster limits what a tile may write per text chunk and
  native does not, so only this run shows that the chunk size fits.
- Over empty prompt ranges, non-greedy claimed ids, claimed ids outside the
  vocabulary and an output short of its budget, on the templated case: the whole chain finishes through both Raster and
  native-staged, agrees byte for byte, and the final record names the failure.
- Missing, duplicate, unexpected, malformed, or error-bearing outputs fail.

Raster's ordered stage inventory comes from its public `chain-commitment` file;
this metadata does not imply authenticated execution. Native-staged and older
Raster runs can supply the inventory through `execution-times.json`. Timing JSON
is not required from the current Raster CLI. A malformed commitment fails instead
of falling back to timing metadata, and the recorded stage order must still match
the independently expected chain.

`iris compare-teacher-forced --teacher-forced <chain-dir> --direct <parity-worker --direct output>`
(hidden) runs the same position-by-position comparison on any chain, reading one
layer at a time.

Batched prefill over `N` rows and single-row decode must be bit-identical, or an
honest claimer would convict itself.

## Challenge leg

On the templated case the gate then runs `challenge build`, the command a
challenger runs, under `<case>/challenges/`:

- An honest claim reports no divergence, and a claim over non-greedy ids reports
  no divergence and is rejected on its final record.
- A falsified checkpoint in each stage family long enough to challenge is found at
  that stage and replays through authenticated Raster.
- An honest claim from another model, the pinned bundle with one weight changed in
  every tensor and two vocabulary ids exchanged, is found at `prompt_bind` and
  replays. The challenger never reads that model.
- A claim of the honest run whose checkpoints came from elsewhere must first differ
  at the stage named: other text or configuration (`prompt_bind`), other cuts
  (`prompt_seed_p0`), other ids (`embed_c0`), that model's tokenizer at each prompt
  stage, its embedding, PLE, weights, head and decoder at the first stage reading
  each, PLE layers exchanged, and stages wired to the wrong predecessor.

A stage [too short for the replay window](../issues/short-stage-fraud-window.md)
cannot be challenged yet, so for those the leg checks only that recomputation finds
it. Each case's result in `report.json` says whether it `replayed`.

Two wirings stop a layer stage through both Raster and native-staged, because
positions no longer line up with the cache: restarting positions at a later range,
and dropping a range's cache. The claimer has no output to post, and only a manifest
the claimer rewired reaches this. The same holds for transformer layers exchanged,
whose caches differ by attention type and donor.

This leg needs `cargo raster` on the path, because `challenge build` calls it. Fault
proving is not part of the gate.

## Results and failures

Each invocation creates `target/parity/<run-id>/report.json` and prints its path.
The versioned report records model-bundle checksums, executable hashes, comparison
counts, results, timings, and any failure. Case directories contain their run
spec, preparation artifacts, independent execution outputs, and separate logs.
Build/import logs are at the run root. Files are retained on success and failure.

The command exits nonzero on any failed build, execution, coverage assertion, or
comparison. Numerical differences identify the execution pair, case, semantic
stage, first differing value where available, and byte offset. Execution failures
identify the corresponding log. Tests never rewrite manual stage fixtures, existing
model imports, or `Raster.lock` files.

Normal `infer` remains free of diagnostic serialization. Library users can opt
into `DirectInferenceExecutor::run_with_diagnostics` to receive canonical boundary
bytes, indexes, and commitments alongside the ordinary inference report.

## Model Bundle Maintenance

`model-bundles/parity-gemma` contains the committed weights, config, tokenizer, corpus,
checksums, and standard-library-only generator. To verify provenance without
changing any bundle files:

```sh
python3 model-bundles/parity-gemma/generate.py --check
```

To deliberately update the bundle, edit the generator and run it without
`--check`, review all changed inputs and checksums, then rerun the gate. The
generator supplies independently calculated tokenization expectations; the gate
does not bless current implementation outputs as new expected values.

The model uses four layers, hidden width 128, FFN width 2048, four query heads,
two KV heads, head width 32, PLE width 32, and vocabulary size 512. Sliding/full
attention alternates, layers 2/3 borrow from layers 0/1, and the sliding window is
16 tokens. The embedding tables, head, and attention and MLP matrices are dense
signed `Q8_0` blocks; each MLP matrix spans a full and a partial Raster page.

The short raw prompt has 5 input tokens and generates 1 token. The second raw
prompt has 15 input tokens and generates 8. The wrapped prompt has 33 input
tokens, including punctuation, repeated spaces, Unicode and byte fallback, and
generates 8. The corpus exercises competing ranked BPE merges and both prompt modes. Nonconstant
logits and multiple selected token IDs prevent a degenerate bundle from passing.

This gate checks host computational parity. Proof generation and native/zkVM
equivalence are separate from this local command.

## Selective PLE loading in staged inference

Staged `aux_*_l*` stages default to selective PLE
loading. They verify the parameters, projection, and embedding metadata against
the declared Raster input commitment, retain those small static values within
the existing weight-cache budget, and read each required embedding page once per
invocation. Pages are not retained across stages. The projection is still prepared
for each invocation. Artifact files must remain immutable during a run.

The implementation lives in `staged-infer`. It uses Raster's existing public
selection/witness APIs, including explicit proof and typed-payload checks when
the staged process runs with `RASTER_AUTH=0`. Raster stage implementations,
Raster libraries, shared kernels, stage identities, and checkpoint formats are
unchanged. Postcard inputs retain eager loading; malformed Raster inputs fail.

`STAGED_INFER_PLE_LOADING=selective|eager` selects the path for chain execution,
persistent workers, and standalone staged execution. An unset variable means
`selective`; other values are errors. The requested mode is recorded as
`ple_loading` in `staged-performance.json`. For a real-model comparison, run the
same release binary and run spec with `eager`, then `selective`, keeping the
weight-cache budget, worker count, and Rayon thread count identical. Compare
every entry in `checkpoint_trace.json` and `checkpoints.txt`; repeat in reversed
order to assess OS-cache effects. On PowerShell, set the mode with:

```powershell
$env:STAGED_INFER_PLE_LOADING = 'eager'
# Run your normal iris claim build --run <spec> command.
$env:STAGED_INFER_PLE_LOADING = 'selective'
# Repeat the same command and compare the two chain directories.
```

Rust callers constructing auxiliary inputs from owned weights use
`routines::prefill_prepare_aux::Inputs::eager(embedded, layer)`; file loaders
choose the configured representation internally.

The focused regression and loading benchmark are in the staged crate:

```sh
RASTER_AUTH=0 cargo test --release --locked -p staged-infer --lib ple::tests -- --test-threads=1
RASTER_AUTH=0 cargo test --release --locked -p staged-infer --lib ple::tests::selective_loading_benchmark -- --ignored --exact --nocapture
cargo test --release --locked -p staged-infer --test worker_parity -- --test-threads=1
```

With `STAGED_INFER_REPLAY_BIN_DIR` pointing at the built stage binaries, also run
`ple::tests::multi_page_matches_authenticated_raster_stage` with
`--lib -- --ignored --exact`, and the existing
`repeated_ranges_match_authenticated_raster_stage_execution` test with
`--test worker_parity -- --ignored --exact`. They cover authenticated multi-page
and saturation cases, and repeated-range worker execution, respectively.

For a full small-model gate, build `iris-cli`, `staged-infer`, and all Raster
stage binaries in release mode into the same directory, with `RISC0_SKIP_BUILD=1`
for this host-only test. Then run (use the configured Python executable on Windows):

```sh
python crates/inference/staged/tests/selective_ple_parity.py \
  --bin-dir target/windows/release --output target/selective-ple-parity
```

The output directory must be new. This imports the pinned `parity-gemma` bundle
and runs all three cases through eager PLE execution, selective PLE execution with
zero/constrained/sufficient cache budgets and one/two workers, and the unchanged
Raster binaries with **authentication enabled**. Each execution consumes its own
preceding outputs. The test compares every output/index/manifest byte and every
checkpoint hash. The report records
binary hashes, elapsed times, and sampled process-tree memory. Timing has no pass
threshold: this model's PLE table fits in one page, so the separate multi-page
benchmark measures the loading reduction. This is native authenticated execution,
not a zkVM proof-generation test.


## Selective main embeddings and paged output head

These are the standard staged implementations for Raster inputs, not modes to
opt into. `embed_*` stages cache verified dimensions, scale,
and page metadata, then gather each needed page once per invocation.
`check_*` stages authenticate the complete projection
page count and score every vocabulary row in batches of at most 32 pages. The
head never creates a complete owned matrix. Only parameters and metadata enter
the existing run-owned cache; selected/streamed pages are invocation-local.

The verifier is staged-local and uses public Raster APIs. Scalar/struct values
are checked against proven typed payloads. Main embedding/head pages are parsed
directly from the proven raw payload using Raster's canonical parser, with
explicit root, selector, and geometry checks. Head page counts are authenticated
list metadata, including empty lists. This applies with `RASTER_AUTH=0` too.
Artifact files remain immutable during a run. Postcard and mixed-encoding inputs
retain eager compatibility; invalid Raster selections never trigger fallback.

No new environment switch is provided. `STAGED_INFER_PLE_LOADING` still controls
only PLE. The diagnostic performance report records `embedding_loading=selective`
and `head_loading=paged`. Head page loading and verification occur inside its
kernel-duration interval; focused benchmarks report those costs separately.
Canonical outputs, indexes, manifests, post-write verification, and checkpoint
hashes are unchanged. Eager constructors on the affected routine `Inputs`
types support comparisons from already-materialized test values.

Preserve release builds before editing and after embedding-only implementation.
Then run the complete pinned small-model gate against those builds:

```powershell
python crates/inference/staged/tests/paged_weights_parity.py `
  --baseline-bin target/staged-pages-validation/baseline-bin `
  --embedding-bin target/staged-pages-validation/embedding-bin `
  --bin-dir target/windows/release `
  --output target/staged-pages-parity
```

The output directory must be new. All three cases run through the three builds,
plus updated zero/constrained cache budgets and serial execution. Each chain
consumes its own prior outputs. The unchanged authenticated Raster binaries then
execute the complete declared sequence. The
harness compares output/index/input-manifest/output-manifest bytes, commitments,
and checkpoint hashes, and checks that the Raster binaries themselves did not
change. Its report includes wall times and sampled memory; the small-model
numbers are not estimates of Gemma speedup.

Focused tests live in `weight_tests` in the staged crate. Run them both normally
and in a separate process with `RASTER_AUTH=0`. With
`STAGED_INFER_REPLAY_BIN_DIR` pointing to `target/windows/release`, explicitly run
the ignored authenticated multi-page and standalone-publication tests. Run the
ignored `embedding_loading_benchmark` and `head_loading_benchmark` in release
mode with `--nocapture --test-threads=1` and no competing build. The fixtures
cover multiple batches, fixed-point boundaries, corruption, forged list lengths,
postcard compatibility, and rejection before output publication. Existing worker
parity tests cover persistent workers and per-stage subprocess execution.

### Manual Gemma comparison

Use a short run spec first (for example four requested output tokens), with the
same prompt and model manifest for all builds. Invoke the CLI directly so the
backend request timeout does not obscure whether the chain completed. Keep
cache budget, worker count, Rayon threads, and PLE mode identical. For example,
from the Iris repository, using the retained budget from the earlier run:

```powershell
$env:STAGED_INFER_WEIGHT_CACHE_BYTES = '3584417792'
$env:STAGED_INFER_AUX_PARALLELISM = '4'
$env:RAYON_NUM_THREADS = '4'
$env:STAGED_INFER_PLE_LOADING = 'selective'
Remove-Item Env:RASTER_PARALLELISM -ErrorAction SilentlyContinue
& ./target/staged-pages-validation/baseline-bin/iris.exe parity-worker `
  --run ./gemma-short.toml --output ./target/gemma-pages-before
& ./target/staged-pages-validation/embedding-bin/iris.exe parity-worker `
  --run ./gemma-short.toml --output ./target/gemma-pages-embedding
& ./target/windows/release/iris.exe parity-worker `
  --run ./gemma-short.toml --output ./target/gemma-pages-after
```

Use fresh output directories on repeats and reverse build order to assess OS
cache effects. Each `result.json` identifies its `chain_dir`. Compare every
entry of `checkpoint_trace.json` and `checkpoints.txt`, not just final text or
EOS, and inspect `staged-performance.json` for embedding/check timings
and sampled process-tree memory. Then repeat the original full-length run.
