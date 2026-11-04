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
`Raster.toml`, and records its hash and range size in the model manifest's
provenance. Run specs select it with `chain = "teacher-forced"`. It verifies a
*known* output with one pass over `prompt ‖ claimed_output`, cut into ranges of
at most `--range-size` positions (default 512), instead of replaying the decode
loop.

### Ranges

Run preparation cuts the prompt's initial pieces into ranges of at most
`range_size` tokens, each ending at a boundary no merge can cross: two pieces
whose adjacent characters appear together in no vocabulary token. Every merge
produces a vocabulary token, so pieces on either side of such a boundary tokenize
independently to exactly the ids of the whole prompt. On Gemma 4 E2B only one
token (`>▁</`) spans a space, so almost every space is a boundary. Run
preparation merges each stretch between boundaries natively to count its
tokens, then packs whole stretches into ranges. It fails if one stretch alone is
longer than `range_size` tokens. The claimed ids are cut into ranges of the same
size.

Each prompt range `p{j}`:

1. `prompt_merge_seed_p{j}`, `prompt_merge_p{j}_b{b}`, `prompt_prepare_p{j}`:
   the range's pieces, 64 merges per stage. Each merge scans only the range, so
   total tokenizer work is linear in the prompt. Every range runs the same
   number of batches, enough for the range needing the most merges. Run
   preparation fills it in, and `challenge build` recomputes it. Too few
   cannot pass: `prompt_prepare` rejects pieces that could still merge.
2. `embed_p{j}`: `input-embedding` over the range's ids, starting where the
   previous range's embedding ended (`range_origin` for the first).
3. `aux_p{j}_l{l}`, `range_p{j}_l{l}`: the unchanged `prefill-prepare-aux` and
   `prefill-range` programs. A layer's range takes its rows from the layer below
   and its inherited K/V cache from its own previous range. That is exactly a
   decode step with more than one row, so the layer program needs no change.
   Sliding-window layers carry only the window; global layers carry every
   earlier key.

Each claimed range `c{k}` runs `embed_c{k}` (an empty prompt, this range's
claimed ids), `aux_c{k}_l{l}`, `range_c{k}_l{l}`, and `check_c{k}`.

The chain runs every tokenizer and embedding stage first, then each layer across
all ranges in order (`aux_p*_l{l}`, `aux_c*_l{l}`, `range_p*_l{l}`,
`range_c*_l{l}`) before the next layer, then the checks. Range `j` of layer `l`
needs only layer `l − 1`'s range `j` and layer `l`'s range `j − 1`, so this
order is valid. It loads each layer's weights once per claim instead of once
per range, and a challenger recomputing up to a dispute at layer `l` runs only
layers `0..=l`. Measured on Gemma 4 E2B at 568 prompt tokens, range-by-range
order made the layer stages 3× slower than one stage per layer.

`check_c{k}`:
`sequence-check` scores the previous range's last row followed by this range's
rows against the head, takes the first-maximum argmax at each position exactly as
`prefill-finalize` followed by `decode-select-token` would, and appends the
selections to the previous check's transcript. The last check's `DecodeEdge`
feeds the unchanged `output-finalize`.

The repeat counts, the last prompt range's index, and the per-range externals
(`pieces_{j}`, `claimed_{k}`) are filled in by run preparation into the frozen run
manifest. `range-origin` is the empty sequence at position 0 that the first
range of each binding starts from, because a repeat's `first` fallback must name a
stage. The autoregressive chain binds its single `input_embedding` to it too.

### Soundness

Causal attention means row `i` sees exactly the tokens the decode step at
position `i` saw. For an honest claim every row and K/V entry equals the
autoregressive chain's, the last check's `DecodeEdge` equals its final
`decode.edge`, and `output_finalize` is identical.

`sequence-check` re-derives the greedy selections rather than asserting the
claimed ones: an aborting stage has no output to hash or replay. A claim is
therefore well-formed only if `output_finalize`'s ids equal its posted ids.
False claims are caught one of three ways: a doctored layer output diverges at
that (layer, range) stage, a doctored check output diverges at that check, and
honest checkpoints over non-greedy tokens publish the greedy ids instead of the
posted ones (`challenge build` reports this as a rejected claimed output).

### Cost

Every per-position stage covers at most `range_size` positions, so its replay
cost is bounded independently of the prompt length, except that a global-attention
layer's late ranges still read every earlier key. The stage count grows linearly
with the prompt: per prompt range one seed, at least one more merge batch (a
range needs about one per 64 merges), prepare, embed and two stages per layer.

Carrying K/V forward copies it, because Raster can extend only an object created
inside the same program, and a stage's input is not one. Sliding-window layers
copy at most the window; global layers copy all earlier positions, so their
total K/V data grows quadratically with the number of ranges.

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
