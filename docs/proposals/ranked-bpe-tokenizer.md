# Ranked BPE using existing Raster staging

The implementation uses ordinary fixed-count chain repeats and existing checkpoint records. It makes no changes to RasterCore or the adjacent Raster libraries. This replaces the earlier proposal for a stage-produced count and planner. See the [validation results](../validation/ranked-bpe-2026-09-10.md), including the unresolved authenticated decode-initialization failure.

## Execution contract

`prompt_merge_seed` and `prompt_merge_b{b}` run the same `prompt-merge` program. A batch contains eight explicit `call_seq!(merge_once, ...)` sites. Each operation:

1. Scans the immutable current piece list and eligible dictionary records.
2. Chooses the lowest merge rank, retaining the leftmost candidate on a tie.
3. Walks the same input and appends a fresh draft with that one merge.
4. Finalizes the draft before the next operation selects its pieces.

The search carries the previous piece, current position, and winning candidate. Dictionary lists remain authorized recur-sequence references; nested lookups iterate actual dictionary records. The application pass receives `Block<BpePiece>` windows of at most 256 pieces. Its absolute positions are computed inside its tile. A merge can cross an application block boundary. Blocks are execution units, never tokenization boundaries.

An operation without a candidate preserves every piece. All scheduled Raster batches still execute after convergence. Staged-native produces the same batch output and may stop its internal native loop once no candidate remains.

The final `prompt_prepare` independently searches for remaining candidates and rejects an insufficient budget. It resolves the converged pieces to the existing `PromptTokenization { token_ids: List<u32> }` interface consumed by embedding. Missing vocabulary entries fail instead of silently emitting token zero.

## Fixed scheduling

For the actual frozen initial piece count `P`:

```text
total_batches = max(1, ceil(max(P - 1, 0) / 8))
repeat_count = total_batches - 1
```

Every successful operation removes one piece, so this is a sufficient upper bound. Empty input still receives one seed batch and finalization. Ceiling arithmetic avoids addition overflow, and conversion to the existing `u32` repeat count is checked.

Run preparation writes the count into the ordinary repeat named `tokenize`. Its first iteration binds to the seed; later iterations bind to the previous batch. The named `tokenize.pieces` export selects the last batch, with the seed as its zero-repeat fallback. Finalization consumes this export.

The manifest renderer replaces counts by repeat name: `tokenize` receives the piece-derived count and `decode` receives the requested generation count. No delayed expansion, dynamic planner, synthetic round list, new checkpoint type, or Raster library extension is involved.

## Prepared input and compatibility

A piece is `{ text: String, segment: u32 }`. Adjacent pieces can merge only within the same segment. Added special tokens occupy their own segments. The synthetic `</w>` terminator is gone; literal `</w>` text receives ordinary tokenization, and the last actual element is processed normally.

Shared host preparation preserves existing chat rendering and supports the imported Gemma BPE profile: literal space-to-`▁` normalization, the declared literal-space pre-tokenizer, atomic special tokens, character pieces, byte fallback, and fused unknowns. BPE without an unknown token drops otherwise unrepresentable characters, matching the pinned reference. Unsupported normalization, pre-tokenization, post-processing, padding, truncation, dropout, and added-token behaviors are rejected explicitly rather than approximated.

Raster's committed entry is the prepared piece list and tokenizer table. Raw-text rendering and preparation remain host work; independent reference fixtures check that boundary. They are not represented as guest execution over raw text.

The native kernel applies the same rank/position rule. Staged-native dispatches, serializes, and caches each seed/repeated batch. Direct inference runs to convergence without publishing intermediate merge checkpoints. Its final tokenization and downstream boundaries participate in the three-path parity gate.

## Artifacts and replay

Model import generates the seed/repeat/final topology and uses the same tokenizer-table preparation as inference. Table schemas and hash buckets remain compatible; piece schemas and tokenizer program identities change. Full imports regenerate model-specific templates and their recorded provenance. Run-specific pieces are generated from the selected prompt.

Frozen prepared runs now use version 2. Replay rejects older prepared-run formats explicitly. Existing claims and their hashes are not rewritten; replaying a legacy claim requires its original implementation and artifacts. Old single-stage templates are also rejected by the run renderer.

Checkpoint records, ordered expansion, first-divergence detection, and selected-stage replay are unchanged. Tokenizer batches are ordinary addressable stages, including `prompt_merge_seed` and `prompt_merge_b1`. Challenge preparation copies the necessary preceding outputs and replays the first divergent stage through the existing Raster command.

## Validation tooling

The independent oracle is `huggingface/tokenizers` version **0.22.2**. Checked-in fixtures pin the exact tokenizer file SHA-256 and expected IDs:

- `tests/tokenizer/synthetic-reference.json`: the parity model, including 3,000 reference tokens with competing ranked merges.
- `tests/tokenizer/ranked-reference.json`: overlapping equal-rank candidates, newly exposed merges, seven/eight/nine and sixteen/seventeen merge boundaries, identity batches, special boundaries, Unicode, literal markers, and a merge crossing position 256.
- `tests/tokenizer/gemma-reference.json`: the production tokenizer, including approximately 3,000 reference tokens. Its source tokenizer is supplied locally and verified against the pinned checksum.

Run host checks with:

```bash
cargo test --workspace
cargo check -p prompt-merge -p prompt-prepare --no-default-features
BPE_REFERENCE_TOKENIZER=/path/to/tokenizer.json \
  cargo test --release -p staged-infer --test tokenizer production_reference_ids -- --ignored
just test-parity
```

The three-path gate compares all Raster/staged-native checkpoints, final prompt token IDs, every available direct boundary, complete logit vectors, selected tokens, and final results. The parity bundle includes a competing-rank case that the previous greedy algorithm tokenized differently.

Follow the repository [Raster authoring skill](../../.claude/skills/raster/SKILL.md), including source review. In each tokenizer project, generate CFS, build RISC0 guests, run committed fixtures, audit them, and verify the program identity. Then inspect both generated CFS files:

```bash
python3 scripts/check_tokenizer_cfs.py
```

The CFS check requires eight explicit operations, authorized prior outputs, real piece/rule iteration, the bounded application tile, and a final convergence assertion. It permits inline slots only for sanctioned new drafts.

Generate and validate authenticated tokenizer chains:

```bash
cargo build --release -p staged-infer --examples
python3 scripts/validate_tokenizer.py \
  --raster /path/to/cargo-raster --authenticated \
  --output target/tokenizer-validation
```

This runs actual sequential chains, compares every input/output checkpoint and payload root, and performs execution audits. A two-step fraud-proof window supports empty finalization; larger fixtures can use `--window 32`. The `after-eight` case also runs with a deliberately insufficient budget and requires rejection without accepted final output.

Regenerate the standalone stage inputs through the fixture tool:

```bash
cargo run --release -p staged-infer --example tokenizer_fixture -- \
  runtime/prompt-fixtures/ranked-bpe \
  tests/tokenizer/ranked-tokenizer.json tests/tokenizer/ranked-reference.json \
  after-eight --install-stage-inputs
```

For larger checkpoint comparisons, `scripts/validate_tokenizer_checkpoints.py` runs every selected-stage replay in parallel. Its optional `--program-dir` runs the prebuilt Raster stage executables directly with the CLI's no-auth arguments and environment, avoiding a Cargo build per checkpoint. Native predecessor artifacts seed each job; every predecessor is independently compared with its actual Raster result, establishing equality throughout the linked chain. No scheduled identity batch is skipped. This comparison directory is test evidence, not an authenticated full-chain claim. Sequential authenticated chains are tested separately.

```bash
target/release/examples/tokenizer_fixture target/tokenizer-long \
  model-bundles/parity-gemma/tokenizer.json \
  tests/tokenizer/synthetic-reference.json long-3000
python3 scripts/validate_tokenizer_checkpoints.py target/tokenizer-long \
  --raster /path/to/cargo-raster --program-dir target/release \
  --output target/tokenizer-long-checkpoints --jobs 4
```

Build both Raster programs from the current source before using `--program-dir`. The report records their executable hashes; guest program identities are verified separately.

Oracle regeneration requires the pinned Python library and uses `scripts/tokenizer_reference.py` and `tests/tokenizer/generate_ranked.py`. Expected IDs are never generated by our inference implementation.

## Cost model

Eight merges per stage and 256-piece application tiles bound operations and materialization, not total stage duration. Global scans and fresh drafts still make worst-case total work quadratic in the initial piece count. The conservative budget can schedule many identity batches.

Validation reports record stage count, native time, Raster execution/validation time, storage, and replay measurements. Parallel validation wall time must not be presented as sequential inference latency. Authenticated trace/commit storage is reported separately from output checkpoint storage. Full three-path host parity complements, but does not replace, authenticated full-chain execution and audit.
