# Raster Chain Internals

The root `Raster.toml` is a chain manifest: it has a `[chain]` table and no `[program]` table. The member directories under `raster-stages/` are ordinary Raster programs with their own `Cargo.toml`, `Raster.lock`, no-std tile library, and sequence entry point.

```text
model import
  -> tokenizer and prompt pieces
  -> embedding table
  -> PLE layer externals
  -> transformer layer externals
  -> output head and decoder
  -> Raster.toml chain manifest
```

Stage outputs are linked by structural commitment, not Rust type name. Boundary structs that cross stage edges therefore need to stay field-for-field compatible between producer and consumer crates.

The central activation boundary is:

```text
ActivationSequence {
  rows,
  errors,
  kv,
  start_position,
}
```

High-level stage order:

1. `prompt-prepare`: tokenize prompt pieces into token ids.
2. `input-embedding`: gather embedding rows for token ids.
3. `prefill-prepare-aux`: compute per-layer PLE inputs.
4. `prefill-range`: run one transformer layer and publish activation/KV state.
5. `prefill-finalize`: score the output head.
6. Decode repeat: select token, embed token, run per-layer aux/range stages, finalize logits.
7. `output-finalize`: decode generated ids into text and publish `InferenceResult` fields.

This is the default autoregressive chain (`Raster.toml`).

## Teacher-forced chain

A full import also writes `Raster.teacher-forced.toml` beside the model's
`Raster.toml`, and records its hash in the model manifest's provenance. Run specs
select it with `chain = "teacher-forced"`. It verifies a *known* output with one
pass per layer over `prompt ‖ claimed_output` instead of replaying the decode loop:

1. Tokenization, `prompt_prepare`: unchanged.
2. `input-embedding`: embeds the chain-tokenized prompt ids, then the claim's
   output ids (a run-specific `claimed` external), into one `N = prompt_len +
   tokens` row sequence. The autoregressive chain binds an empty committed claim
   (`no_claim`) to the same program.
3. `prefill-prepare-aux`, `prefill-range`: the unchanged programs over all `N` rows.
4. `sequence-check`: for each claimed token `k`, scores row `prompt_len - 1 + k`
   against the head and takes the first-maximum argmax, exactly as
   `prefill-finalize` followed by `decode-select-token` would.
5. `output-finalize`: unchanged, reading the checked edge.

The join lives inside `input-embedding` instead of a stage of its own: a stage
that only concatenates ids has a trace no longer than the 128-item fraud window,
which Raster cannot commit, so a lie there could never be disputed (see
[`short-stage-fraud-window`](../issues/short-stage-fraud-window.md)). Stages are
sized so every one can be challenged.

Causal attention means row `i` of the pass sees exactly the tokens the decode
step at position `i` saw, so for an honest claim every row and KV entry equals the
autoregressive chain's, the check stage's `DecodeEdge` equals its final
`decode.edge`, and the final checkpoint hash is identical.

`sequence-check` re-derives the greedy selections rather than asserting the
claimed ones: an aborting stage has no output to hash or replay. A claim is
therefore well-formed only if `output_finalize`'s ids equal its posted ids.
False claims are caught one of three ways: a doctored layer output diverges at that
layer, a doctored check output diverges at `sequence_check`, and honest
checkpoints over non-greedy tokens publish the greedy ids instead of the posted
ones (`challenge build` reports this as a rejected claimed output).

For Gemma 4 E2B the chain has 75 stages plus one per tokenizer merge round,
independent of the token count. A single `prefill_range` stage now processes `N` rows, and
`sequence_check` sweeps the head once per claimed token.

The layer loop lives in the chain manifest rather than in one Raster program. A generated model expands one program over many stage instances, preserving bounded replay units and per-stage auditability.

Weight-matrix and logit sweeps use one page per tile: each `call_recur!`
walks the existing `List<BytesPage>` without a `chunk`. The auxiliary
projection, all transformer matrix projections, and the output-head projection
replay one page per iteration. The same programs serve prefill and decode.
Attention and cache-copy sweeps keep their existing chunk size of 64.

Native-staged and native-direct retain their whole-matrix kernels and must
produce exactly the same stage outputs as Raster. To group pages, add
`chunk = N` to the corresponding `call_recur!` sites, take
`RecurInput<Block<BytesPage>>` in the step tile, and rebuild and verify the
three affected program identities as described in [development](development.md).
