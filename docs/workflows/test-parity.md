# Three-path parity

Run the local regression gate from the repository root:

```sh
just test-parity
# Equivalent:
cargo run --release --locked -p iris-cli -- test-parity
```

The command runs the pinned synthetic Gemma model through the complete Raster
chain (`--no-auth`), checkpointed native-staged inference, and native-direct
inference. Each execution owns its intermediate outputs and runtime state.
All three fixed cases run every time. There are no numerical tolerances.

The staged inventories include `prompt_merge_seed` and every scheduled
`prompt_merge_b{b}` checkpoint. Direct execution compares final `prompt_prepare`
and downstream boundaries without inventing intermediate tokenizer checkpoints.
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

- Raster/native-staged: every expected checkpoint's canonical output bytes and
  structural commitment, with recorded payload/index roots checked independently.
- Native-staged/direct: tokenized input IDs, every complete fixed-point logit
  vector used for selection, selected-token history, and every final result field.
- Direct's corresponding activation, PLE, embedding, and KV boundaries also match.
- Missing, duplicate, unexpected, malformed, or error-bearing outputs fail.
  Pinned input IDs, argmax selection, generation counts, donor bindings, cache
  growth/eviction, and position advancement are checked separately from parity.

Raster's ordered stage inventory comes from its public `chain-commitment` file;
this metadata does not imply authenticated execution. Native-staged and older
Raster runs can supply the inventory through `execution-times.json`. Timing JSON
is not required from the current Raster CLI. A malformed commitment fails instead
of falling back to timing metadata, and the recorded stage order must still match
the independently expected chain.

Staged execution has an extra decode pass after selecting its last token. That
pass must exist and match between Raster and native-staged. Direct intentionally
stops before it. Its required boundary inventory is explicit and independent of
the order in which PLE work is scheduled.

## Results and failures

Each invocation creates `target/parity/<run-id>/report.json` and prints its path.
The versioned report records model-bundle checksums, executable hashes, comparison
counts, results, timings, and any failure. Case directories contain their run
spec, preparation artifacts, independent execution outputs, and separate logs.
Build/import logs are at the run root. Files are retained on success and failure.

The command exits nonzero on any failed build, execution, coverage assertion, or
comparison. Numerical differences identify the execution pair, case, semantic
stage, first differing value where available, and byte offset. Execution failures
identify the corresponding log. Tests never rewrite the root manifest, manual
stage fixtures, existing model imports, or `Raster.lock` files.

Normal `infer` remains free of diagnostic serialization. Library users can opt
into `DirectInferenceExecutor::run_with_diagnostics` to receive canonical boundary
bytes, indexes, and commitments alongside the ordinary inference report.
`DIRECT_INFER_COMPARE_TRACE=<checkpoint_trace.json>` now fails on mismatches,
duplicate/missing reference entries, and incomplete comparisons. It expects a
complete production staged trace for the same model and token count.

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

The model uses four layers, hidden width 128, FFN width 512, four query heads,
two KV heads, head width 32, PLE width 8, and vocabulary size 512. Sliding/full
attention alternates, layers 2/3 borrow from layers 0/1, and the sliding window is
16 tokens. Dense signed Q16.16 matrices span full and partial Raster pages.

The short raw prompt has 5 input tokens and generates 1 token. The second raw
prompt has 15 input tokens and generates 8. The wrapped prompt has 33 input
tokens, including punctuation, repeated spaces, Unicode and byte fallback, and
generates 8. The corpus exercises competing ranked BPE merges and both prompt modes. Nonconstant
logits and multiple selected token IDs prevent a degenerate bundle from passing.

This gate checks host computational parity. Proof generation and native/zkVM
equivalence are separate from this local command.

## Selective PLE loading in staged inference

Staged `prefill_prepare_aux_l*` and `decode_aux_t*_l*` default to selective PLE
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
`repeated_decode_matches_authenticated_raster_stage_execution` test with
`--test worker_parity -- --ignored --exact`. They cover authenticated multi-page
and saturation cases, and repeated-decode worker execution, respectively.

For a full small-model gate, build `iris-cli`, `staged-infer`, and all ten Raster
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
checkpoint hash, including the last declared decode pass. The report records
binary hashes, elapsed times, and sampled process-tree memory. Timing has no pass
threshold: this model's PLE table fits in one page, so the separate multi-page
benchmark measures the loading reduction. This is native authenticated execution,
not a zkVM proof-generation test.


## Selective main embeddings and paged output head

These are the standard staged implementations for Raster inputs, not modes to
opt into. `input_embedding` and `decode_embed` cache verified dimensions, scale,
and page metadata, then gather each needed page once per invocation.
`prefill_finalize` and `decode_finalize_*` authenticate the complete projection
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
hashes are unchanged. Eager constructors on the three affected routine `Inputs`
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
execute the complete declared sequence, including its final decode pass. The
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
EOS, and inspect `staged-performance.json` for embedding/finalization timings
and sampled process-tree memory. Then repeat the original full-length run.
