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
| `prompt-check` | `prompt_check_p{j}` | Check a range's claimed tokens against the merge and publish their ids |
| `input-embedding` | `embed_p{j}`, `embed_c{k}` | Gather embedding rows |
| `prefill-prepare-aux` | `aux_p{j}_l{l}`, `aux_c{k}_l{l}` | Per-layer PLE inputs |
| `prefill-range` | `range_p{j}_l{l}`, `range_c{k}_l{l}` | Run one transformer layer over one range |
| `sequence-check` | `check_c{k}` | Re-derive the greedy selections and compare them with the claimed ids |
| `output-finalize` | `output_finalize` | Decode the text and publish the final record |

Autoregressive generation is not a chain. It lives in `direct-infer`, which discovers the output a claim then verifies.

## Ranges

`prompt_bind` runs once, before any range. It takes the original UTF-8 prompt,
the run's configuration and its stage schedule as committed inputs, folds SHA-256
over the text in chunks of at most 256 bytes, renders it through the chat template
unless the run is raw, and splits it into pieces. It checks the schedule and stores
it in the context (see [Stage schedule](#stage-schedule)). Its output is every piece
of the prompt and a `PromptContext` holding the text's hash, the model id and the
configuration.
Every prompt stage publishes `PromptState`, so the context travels with the
pieces, and the last range's `prompt_check` holds the record for the whole
prompt.

Run preparation tokenizes the pieces natively and cuts the tokens into ranges of
at most `range_size`. For each range it commits *marks*: one number per piece,
1 where a token starts. The chain does not replay the merge to produce the ids.
It checks that the marks are the merge's own result, and looks the ids up. A
range may end between any two tokens. The claimed ids are cut into ranges of the
same size.

The check rests on one fact about the ranked merge: a tokenization is the
merge's result exactly when every token, merged together with its left neighbour
in the segment on those two tokens' pieces alone, comes back as those two tokens,
and a token with no left neighbour merges alone to one piece. The argument and
its tests are in [the proposal](../proposals/claimed-prompt-tokens.md).
`prompt_check` runs those small merges without building them: tokens are paired
(0,1), (2,3), … in one pass and (1,2), (3,4), … in another, a piece may merge
only inside its pair and segment, and each pass is 64 rounds that merge at most
one pair of pieces per pair of tokens. A merge that joins two tokens, a token
left in more than one piece, or a pair still merging after the last round is a
fault. 64 rounds cover tokens of up to 33 pieces; Gemma 4 E2B's longest
vocabulary entry is 31 characters.

Each prompt range `p{j}`:

1. `prompt_check_p{j}`: takes `prompt_bind`'s pieces, the previous range's output
   and this range's marks. It copies as many pieces as there are marks, starting
   where the previous range ended, numbers their tokens from the marks, and
   checks every neighbouring pair of tokens as described below. It then looks
   each token up in the vocabulary and publishes the ids, and beside them the
   pieces of the range's last token for the next range to pair its first token
   with. Because each range starts where the previous context ended, a range
   cannot be substituted, omitted or duplicated without the text's pieces saying
   so.
3. `embed_p{j}`: `input-embedding` over the range's ids, starting where the
   previous range's embedding ended (`range_origin` for the first).
4. `aux_p{j}_l{l}`, `range_p{j}_l{l}`: the `prefill-prepare-aux` and
   `prefill-range` programs. A layer's range takes its rows from the layer below
   and its inherited K/V cache from its own previous range. Sliding-window layers
   carry only the window; global layers carry every earlier key.

A preparation that goes wrong does not stop a stage. An aborting stage has no
output to compare or replay, so it runs to completion and records the first
problem in `PromptContext.fault`: no text, a range with no marks, marks past the
end of the prompt, a token whose pieces are in two segments, tokens that are not
what the merge produces, a window still merging after the last round, or a range
over `range_size` tokens. A run whose last context has a fault, or is not
`complete`, is rejected on its record.

Nothing a claim supplies can abort `prompt_check`. The marks are numbers, and the
stage reads a mark only for a piece it has copied, of which there are never more
than there are marks.

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

The repeat counts, the last prompt range's index, the text, configuration and
stage schedule, and the per-range externals (`marks_{j}`, `claimed_{k}`) are filled in by run preparation
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
id are committed inputs of `prompt_bind`, and the last range's `prompt_check`
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
with the prompt: per prompt range one check, one embed and two stages per layer.
`prompt_check` runs a fixed 128 merge rounds over its range whatever the range
holds, so its cost does not depend on how many merges the range needs.

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
and prepared runs use version 8; older boundary encodings must be regenerated.

### Stage schedule

The stage schedule is which stages a run executes and in what order. It is three
counts, chosen by run preparation and committed as the `schedule` input of
`prompt_bind`: the prompt ranges `P`, the merge batches `B` each range runs after
its seed, and the claimed ranges `C`. Run preparation always sets `B` to 1. With
the model's layer count `L` they fix every stage:

1. `prompt_bind`
2. for each prompt range `j`: `prompt_check_p{j}`, which is the range's
   `prompt_seed`, `prompt_merge` batch 0 and `prompt_prepare`, then `embed_p{j}`
3. for each claimed range `k`: `embed_c{k}`
4. for each layer: the auxiliary stage of every range, then the transformer
   stage of every range
5. for each claimed range `k`: `check_c{k}`
6. `output_finalize`

`prompt_bind` checks that each count is at least one and within its limit, and
that `C` fits the token budget at this range size. It stores the schedule in the
`Execution` record beside the statement, so it reaches every later stage through
the inputs they already take. A schedule that fails is kept with the execution
fault set. `P` and `B` depend on tokenization, so they are held to by the stages
that follow.

Every stage records its step in `Execution`: its operation, range, layer and
batch. In a tile it works out that step from its committed inputs, requires the
schedule to contain it, and requires each input to carry the step the schedule
puts before it, under the same statement and schedule. The rules are in
`pipeline-core`'s `schedule.rs` and applied by `model-identity`'s `pipeline.rs`.

| Stage | Its step must be | Its inputs must carry |
| --- | --- | --- |
| `prompt_seed` | range `j < P` | `prompt_prepare` of range `j-1`, or `prompt_bind` when `j = 0`; and `prompt_bind` |
| `prompt_merge` | batch `b < B` of range `j` | batch `b-1`, or the range's `prompt_seed` when `b = 0` |
| `prompt_prepare` | range `j` | batch `B-1` of range `j`. The prompt is complete exactly when `j = P-1`. |
| `embed_p{j}` | range `j` | `prompt_prepare` of range `j`, and `embed_p{j-1}` or the origin |
| `embed_c{k}` | range `k < C` | `embed_c{k-1}`, or `embed_p{P-1}` when `k = 0` |
| `aux_*_l{l}` | layer `l < L` of its range | that range's embedding |
| `range_*_l{l}` | layer `l < L` of its range | layer `l-1` of the range (or its embedding), layer `l` of the range before (or the origin), the range's auxiliary stage for layer `l`, and the donor layer of the same range |
| `check_c{k}` | claimed range `k` | layer `L-1` of range `k` and of the range before, `check_c{k-1}` or the empty edge, and `prompt_prepare` of range `P-1` |
| `output_finalize` | last | `check_c{C-1}` and `prompt_prepare` of range `P-1` |

A failed requirement sets the execution fault, which is sticky and reaches
`model_fault` in the final record. A valid record therefore means every scheduled
stage ran, each after the stages the schedule puts before it. Two stages that do
not feed each other, such as the auxiliary stages of two layers, are not ordered
against each other. A chain with no `prompt_bind`, such as the tokenizer fixture
chain, has no schedule, and its prompt stages publish the execution fault.

`prompt_check_p{j}` is a range's seed, its one merge batch and its prepare in one
stage. It records those three steps in order, each checked against the one
before, so its output carries the `prompt_prepare` step the later stages require.
The schedule's rules and the `Execution` record are unchanged from when the three
were separate stages; `B` is simply always 1. The tokenizer fixture chain still
runs `prompt-seed`, `prompt-merge` and `prompt-prepare` as separate stages.
