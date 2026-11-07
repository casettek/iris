# Raster Chain Internals

A chain manifest has a `[chain]` table and no `[program]` table. `model import` writes a template for the model, and `claim build` fills it in for one run. The member directories under `raster-stages/` are ordinary Raster programs with their own `Cargo.toml`, `Raster.lock`, no-std tile library, and sequence entry point.

```text
model import
  -> tokenizer and decoder externals
  -> embedding table
  -> PLE layer externals
  -> transformer layer externals
  -> output head
  -> Raster.toml chain template
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

The chain verifies a *known* output with one pass over `prompt ‖ claimed_output`, cut into ranges of at most `--range-size` positions (default 512). Its programs:

| Program | Stages | Work |
| --- | --- | --- |
| `range-origin`, `decode-init` | `range_origin`, `decode_init` | The empty sequence and the empty edge the first range starts from |
| `prompt-bind` | `prompt_bind` | Hash the original text and split it into pieces |
| `prompt-seed`, `prompt-merge`, `prompt-prepare` | `prompt_seed_p{j}`, `prompt_merge_p{j}_b{b}`, `prompt_prepare_p{j}` | Check a range's cut and tokenize it |
| `input-embedding` | `embed_p{j}`, `embed_c{k}` | Gather embedding rows |
| `prefill-prepare-aux` | `aux_p{j}_l{l}`, `aux_c{k}_l{l}` | Per-layer PLE inputs |
| `prefill-range` | `range_p{j}_l{l}`, `range_c{k}_l{l}` | Run one transformer layer over one range |
| `sequence-check` | `check_c{k}` | Re-derive the greedy selections and compare them with the claimed ids |
| `output-finalize` | `output_finalize` | Decode the text and publish the final record |

Autoregressive generation is not a chain. It lives in `direct-infer`, which discovers the output a claim then verifies.

## Ranges

`prompt_bind` runs once, before any range. It takes the original UTF-8 prompt
and the run's configuration as committed inputs, folds SHA-256 over the text in
chunks of at most 256 bytes, renders it through the chat template unless the run is
raw, and splits it into pieces. Its output is every piece of the prompt and a
`PromptContext` holding the text's hash, the model id and the configuration.
Every prompt stage publishes `PromptState`, so the context travels with the
pieces, and the last range's `prompt_prepare` holds the record for the whole
prompt.

Run preparation chooses where to cut those pieces, and the chain checks each cut
rather than trusting it. Ranges hold at most
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

1. `prompt_seed_p{j}`: takes `prompt_bind`'s pieces, the previous range's context
   and this range's cut, an index into the pieces. It checks the cut, copies the
   range, and runs the range's first 64 merges. Because each range starts where
   the previous context ended, a range cannot be substituted, omitted or
   duplicated without the text's pieces saying so.
2. `prompt_merge_p{j}_b{b}`, `prompt_prepare_p{j}`: 64 merges per stage, then ids.
   Each merge scans only the range, so total tokenizer work is linear in the
   prompt. Every range runs the same number of batches, enough for the range
   needing the most merges, as filled in by run preparation.
3. `embed_p{j}`: `input-embedding` over the range's ids, starting where the
   previous range's embedding ended (`range_origin` for the first).
4. `aux_p{j}_l{l}`, `range_p{j}_l{l}`: the `prefill-prepare-aux` and
   `prefill-range` programs. A layer's range takes its rows from the layer below
   and its inherited K/V cache from its own previous range. Sliding-window layers
   carry only the window; global layers carry every earlier key.

A preparation that goes wrong does not stop a stage. An aborting stage has no
output to compare or replay, so it runs to completion and records the first
problem in `PromptContext.fault`: no text, an empty range, a cut past the end of
the prompt, a cut a merge could cross, a range over `range_size` tokens, or pieces
that could still merge. A run whose last context has a fault, or is not
`complete`, is rejected on its record.

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
direct generation would, and appends the selections to the previous check's transcript. It also compares each selection
with the consumed token id retained in that position's activation row and commits the result as the edge's claim
verdict: `claim_mismatches`, the previous check's count plus this range's, and
`first_claim_mismatch_position`, the earliest one. It reads the last prompt
range's context for the range size and the prompt's token count, and publishes
`check_fault` and `check_fault_position` when it cannot stand behind the edge: a
position with no predicting row, a range with no tokens or more than the range
size, or a range that does not start where the prompt or the previous range ended.
The earliest range's fault stands. The last check's `DecodeEdge` feeds
`output-finalize`.

`output_finalize`:
`output-finalize` reads the last check's edge and the last prompt range's context,
and publishes the final record: the ids and their hash, the text and
`output_text_sha256` (the SHA-256 of the text's exact UTF-8 bytes), the verdict and
check fault, `output_fault` (no tokens, over the budget, short of the budget without
EOS, past EOS, or an edge that disagrees with its transcript), and the context the
chain ran under (`input_text_sha256`, `model_id`, `raw_prompt`, `max_new_tokens`,
`range_size`, `prompt_tokens`, `prompt_complete`, `prompt_fault`,
`prompt_fault_position`). The host copies this record into `InferenceResult` and
computes none of it.

The repeat counts, the last prompt range's index, the text and configuration, and
the per-range externals (`cut_{j}`, `claimed_{k}`) are filled in by run preparation
into the frozen run
manifest. `range-origin` is the empty sequence at position 0 that the first
range of each binding starts from, because a repeat's `first` fallback must name a
stage.

## Soundness

Causal attention means row `i` sees exactly the tokens direct generation's decode
step at position `i` saw. For an honest claim every row and K/V entry equals
direct generation's, the last check's `DecodeEdge` equals its last selection, and
`output_finalize` is the same record it publishes.

The claimed ids are an input to this chain, so a claim must not be able to post
an output other than the one the chain consumed. It cannot: a claim posts no
output ids of its own. Its output is `output_finalize`'s record,
and it stands only if that record's prompt fault, check fault, `claim_mismatches`
and `output_fault` are zero and `prompt_complete` is set.

A claim chooses its own ids, and indexing past a table would abort a stage with no
output. So `input-embedding` and `prefill-prepare-aux` read the first row for an id
outside their tables, and keep the id on the row. The chain finishes, and the
verdict rejects the claim: such an id can never be the greedy selection.

`sequence-check` has no independent claimed-token input: it checks the token IDs
carried by the final-layer activation rows. The embedding stages are the only
consumers of the host-provided claimed-ID ranges.

`sequence-check` publishes the verdict rather than aborting on a disagreement,
because an aborting stage has no output to hash or replay. False claims are
caught one of three ways:

- **A doctored layer output** diverges at that (layer, range) stage, and a
  challenge replays it.
- **A doctored verdict.** A claimer who feeds non-greedy ids and commits a clean
  verdict anyway diverges at that check, and a challenge replays it.
- **Honest execution over non-greedy ids** has no divergent stage. It commits a
  non-zero verdict, so the claim refutes itself in its own final record.
  `challenge build` reports this as a rejected claimed output.

The prompt side is inside the chain. The text, the configuration and the model
id are committed inputs of `prompt_bind`, and the last range's `prompt_prepare`
context says what happened to them: the text's SHA-256, the model id, and whether
every piece was prepared without a fault.

A stage program cannot read its inputs' commitments, so the model id is a hash of
the import's commitments by role, carried as a claimed value. It is checked from
outside the chain: recompute it from the role list, and compare a stage's
`input_manifest.json` entry against the role it binds. To challenge a claim,
`challenge build` prepares the claim's prompt, token budget
and claimed ids from the model `--run` selects. A claim that names or ran other
weights therefore differs at some stage's committed inputs, and that stage is the
divergence to replay. A replay reruns the stage on the claim's own committed
inputs, so it shows what those inputs produce; what convicts a claim of using the
wrong weights is that comparison, which a contract can make without a replay.

## Cost

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
replay one page per iteration.
Attention and cache-copy sweeps keep their existing chunk size of 64.

Native-staged and native-direct retain their whole-matrix kernels and must
produce exactly the same stage outputs as Raster. To group pages, add
`chunk = N` to the corresponding `call_recur!` sites, take
`RecurInput<Block<BytesPage>>` in the step tile, and rebuild and verify the
three affected program identities as described in [development](development.md).


### Statement and token-history binding

Every nonempty stage boundary carries a versioned `Execution` record inside
`ModelContext.execution`. The shared `pipeline-core` crate defines the canonical
statement hash and position-sensitive token-prefix hash; `model-identity` checks
stage transitions. The record binds the input-text hash, model ID, raw-prompt
flag, generation budget, and range size. Embedding folds the actual consumed
IDs, so changing a token while keeping the same length and final token changes
the authenticated history.

Prompt preparation can advance only an unfinished prompt. Prompt embeddings
must reproduce the prepared range exactly; claimed embeddings begin only after
the completed prompt and extend the preceding embedding history. Transformer
inputs and auxiliary rows must name the same range, and each layer's prior
cache must end at that range's start. Cache positions are checked for exact,
ordered coverage, including sliding-window retention. Donors must name the
same range and the authorized donor layer. Layer-completion metadata remains
part of the model-identity check.

The initializer programs publish canonical empty sentinels. Consumers accept
an empty prompt/cache predecessor only at position zero, and an empty check
predecessor only at the completed prompt anchor. Reusing those sentinels later
sets a sticky execution fault; they cannot reset an existing statement/history.
Sequence checking reads claim IDs from the activation rows and carries a
cumulative checked endpoint. Finalization checks the supplied prompt against
that statement and anchor, then folds the emitted transcript independently and
requires the exact checked endpoint, position, final token, and budget.

These checks rely on authenticated predecessor provenance and approved stage
program identities. They do not authorize arbitrary externally fabricated
boundary records or replace the protocol's stage-program approval rules. A
model/execution fault makes the final claim invalid even if native and Raster
honestly execute the same malicious wiring. `test-parity` exercises that case
as well as divergence from the honest template; production RISC Zero tests in
`model_identity_guest.rs` cover the corresponding guest checks. Model imports
and prepared runs use version 6; older boundary encodings must be regenerated.
