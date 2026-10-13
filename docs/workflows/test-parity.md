# Three-path parity

Run the local regression gate from the repository root:

```sh
just test-parity
# Equivalent:
cargo run --release --locked -p raster-inference-cli -- test-parity
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
