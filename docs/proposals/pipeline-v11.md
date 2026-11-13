# Pipeline version 11: one re-lock for the shared changes

Proposed and implemented November 11, 2026, as import and prepared-run version 11. This
collects every pending change to code that all stage programs reach, so they cost one
re-lock, one move to version 11 and one gate. [Result](#result) has what it measured, and
[raster-chain.md](../internals/raster-chain.md) describes the chain as it is now.

## Goal

**Make every stage's size depend on a range and not on the prompt, and remove the two
places where shared stage code assumes a Gemma dense model, without changing what a Gemma
dense claim costs.**

Two kinds of change are in here only because they share a re-lock:

- long prompts: binding the prompt one range at a time;
- model families: a model with no per-layer inputs, and a chat template that is data.

Three smaller changes ride along because their programs re-lock anyway.

## What does not change

- The challenge: rerun honestly, find the first mismatching checkpoint, replay that stage.
- The layer stages, the embedding stage, and every tile's arithmetic.
- The schedule's two counts.
- A wrong claim never aborts a stage.
- Raster. The update to a newer Raster is a separate re-lock, later.

## Baseline

Gemma 4 E4B (`Q8_0`) on the current Raster, from challenges run through the backend on
November 11. These are the numbers the pass is held to.

| Stage | Trace steps | Replay | Heaviest tile | Heaviest step with transition | 128-step fault window |
| --- | --- | --- | --- | --- | --- |
| `range_p0_l23`, 512 tokens | 353,510 | 197 to 202 s | `accumulate_context`, 26.9M cycles | 52.0M | 2.39B cycles |
| `check_c0`, 7 tokens | 116,633 | 36 s | `keep_last_row`, 75.8M | 168.4M | 0.86B |
| `prompt_bind`, 817 tokens | 78,712 | 20 s | `scan_prompt_chunk`, 3.6M | 7.4M | 0.11B |
| `prompt_check_p0`, 512 tokens | about 2.5 million | 131 s | not measured | | |

Claim time was 121 s at 773 input and 7 output tokens, and 308 s at 2,684 and 65.

The `prompt_check` row is from running the stage authenticated by hand; its step count is
estimated from the size of its commitment (444 KB against 97 KB for the layer range). It
replays faster than a layer range, so it does not need to become two stages. Its trace is
seven times longer because every one of its 128 rounds walks every piece of the range,
including the rounds after the last merge.

## The changes

### 1. Bind the prompt one range at a time

As in [paged-prompt-bind.md](paged-prompt-bind.md), which has the design and the argument
for it: `prompt_bind_p{j}` per prompt range, each continuing from the previous one's hash,
scanner and split state; the text's hash, length and trim offsets as a claim the last range
checks; `prompt_check_p{j}` reading `prompt_bind_p{j}`'s pieces and what the previous check
left unconsumed.

In the schedule rules this is small. `Step::bind()` takes a range and is run for every
prompt range. A bind step follows the bind of the range before, or nothing. A check step
takes its range's bind, and the check of the range before, or the empty state. The chain
order for a prompt range is bind, check, embed.

Decisions taken since the proposal:

- One bind stage per token range. A bind range was estimated at about 72,000 steps and
  measured about 28,000, a twelfth of a layer range, so nothing is gained by grouping
  ranges and the one-to-one wiring stays fixed.
- `prompt_check` is not split.

### 2. A model may have no per-layer inputs

The model descriptor requires as many auxiliary commitments as transformer layers, and its
encoding has no room for anything else. A family without per-layer embedding inputs has
none to give.

The descriptor would carry its auxiliary count, which is zero or the layer count, and its
version moves. `ModelContext` already holds the layer count for the schedule rules; it
would also hold whether the model has auxiliary stages. With none, the schedule runs no
auxiliary step, and no layer step is asked for a per-layer input.

Nothing else is needed in shared code for this. A family's layer program takes the inputs
its layer needs; a program with no per-layer input never asks the schedule for one. A
mixture-of-experts layer is a different program for the same layer step, or a further
stage that family adds, and neither is designed here.

For Gemma dense the count is the layer count and every stage is the one it is today.

### 3. The chat template is the tokenizer's

`PROMPT_PREFIX` and `PROMPT_SUFFIX` are constants in the prompt source that five stage
programs share. A model with another template would change that file and re-lock all five.

The two strings move into the committed `TokenizerProfile`, in place of `templated`, which
becomes "the prefix or the suffix is not empty". The import's family module supplies them.
The scan seeds itself with the profile's prefix and appends the profile's suffix, and run
preparation renders with the same two values.

This changes the tokenizer's commitment, so the model id of every import changes with it.

### 4. One walk of a vocabulary bucket per atom

`prompt_bind` walks each atom's bucket twice: once to hash it against the committed
tokenizer and once to find the atom. One loop over the bucket's entries does both, carrying
the hash and the match together. Measured, trace steps per atom fell from 21.5 to 11.3.

### 5. Smaller chunks of rows in `sequence-check`

`check_c{k}` needs one row of the previous range: the last. It gets it by walking all of the
range's rows 256 at a time and keeping the last of each chunk, and at E4B's width that tile
call is the heaviest single step in the chain, three times the layer's heaviest.

Selecting the row by its index is not an option. A faulted prompt leaves the previous
range with no rows, and an index into an empty list stops the stage with no output, which
no stage here may do. So the walk stays and its chunk becomes 64 rows, about 19M cycles a
call by proportion.

The rows copied into the stage's own list are the same problem at larger outputs. Seven
rows in one call measured 11.2M cycles, about 1.6M a row, so a full chunk of 64 would be
about 100M. That chunk becomes 16 rows. Measured afterwards: 19.9M cycles for
`keep_last_row`, and 5.8M for `append_predicting_rows` on eight rows.

## Left out

- **Dropping finished windows in `prompt_check`.** It would remove most of the 2.5 million
  steps. It changes one program and no format, so it can follow on its own.
- **A newer Raster.** Waiting on its main branch. It re-locks everything by itself.
- **Paging `output_finalize`.** Sized by the output, which stays small.
- **Anything a specific new family needs:** its layer, its tokenizer scanner, its numerics.

## What changes in the repository

- `pipeline-core` (`schedule.rs`) and `model-identity` (the descriptor and `pipeline.rs`):
  the bind range, and the auxiliary count. Nearly every stage re-locks.
- `prompt-core`: the carried scan state, the text claim, the template in the profile.
- `prompt-bind` and `prompt-check`: inputs, outputs and the loops above.
- `sequence-check`: the two chunk sizes.
- `run-prep`: text slices and the text claim; ranges cut from what each slice yields.
- `staged-infer` routines for the three programs, and the native schedule enumeration.
- `model-import`: the template and the profile, the descriptor, the chain template's bind
  stage per range.
- `test-parity`: a wrong text claim, a claimed length the text never reaches, a withheld
  slice, and a long text in one range. A descriptor with no auxiliary inputs and a template
  that is not Gemma's are covered by unit tests of `model-identity` and `prompt-core`, not
  by a gate case: the parity bundle is still one Gemma model.
- Versions: imports and prepared runs to 11, the descriptor's own version, the backend pin.
- `raster-chain.md`, `claim-build.md`, `test-parity.md`, `model-families.md`.

## Order of work

1. Schedule rules and descriptor, with their unit tests. No stage builds against them yet.
2. `prompt-core` and run preparation, checked natively: the pieces of a sliced prompt equal
   the pieces of the whole one, for cuts at every position of the parity prompts.
3. Native routines, so `staged-infer` runs the new chain and direct inference still agrees.
4. The stage programs, by the Raster skill's check ladder.
5. Import, chain template, parity cases.
6. One re-lock, re-import of both models, the full gate with challenges.
7. The baseline table again.

## Decisions made while building

Kept here so the implementation can be checked against them.

- **Schedule.** `Step::bind(range)`. A bind follows the bind of the range before
  (`ModelContext::continue_bind`), and the schedule is bound in range 0 only. The first
  range's check has nothing before it and continues from its own bind's context; every
  check requires its range's bind (`require_bound`) and the check before or the origin
  (`require_previous_check`).
- **Descriptor.** Version 2: version, layer count, auxiliary count (zero or the layer
  count), then the roots. Nothing in `ModelContext` or the schedule rules changes for a
  model without auxiliary inputs: no auxiliary program runs for it, and its layer program
  asks for no per-layer input.
- **Prompt state.** `PromptState` gains `rest`, the pieces a check leaves unconsumed, and
  `bind`, what a bind stage carries to the next: the text fold, the scan state, the split
  state, the claim, and `done`. A check publishes an empty `bind`; a bind publishes an
  empty `rest`.
- **The text claim** is its own committed input of every bind stage, `TextClaim`: the
  text's SHA-256, its length, and the two trim offsets. Only range 0 reads it; later ranges
  carry it.
- **A check's pieces** are `previous.rest` then `bound.pieces`, numbered from
  `context.next_piece` as before, so the range plan and its faults keep their meaning: marks
  past what is available are a cut beyond the text. The prompt is complete when the check
  has consumed everything available and its bind is `done`.
- **Two new faults:** the text does not match its claim, and text after the text has ended.
- **No size limits.** A limit on a slice's bytes or on the unconsumed pieces cannot be met by
  every honest prompt under a tokenizer that fuses or drops unknown text, and a claimer
  already chooses how large `prompt_bind` is today. Left for when a profile needs it.
- **No origin stages.** The pass first added a one-tile `prompt-origin` stage beside
  `range-origin` and `decode-init`, because a repeat's `first` must name a stage. All three
  were then removed: a stage of three steps is not a stage. The empty values the first
  range starts from are committed constants written by the import, range 0's stages are
  written out as plain stages bound to them, and each repeat starts at range 1. No stage
  program changed for this, because the stages that read an empty value already checked
  that it was empty.
- **`prompt-bind`'s loops.** A recur that builds a list does not give back its final state,
  so the stage runs: one pass over the text for the fold and the scan's position; one for
  the atoms; one walk of a bucket per atom, building a list of atoms with what the bucket
  said; then two chunked passes over that list, one for where the split ends and whether
  every lookup was proven, one for the pieces. Sixteen tiles in place of thirty-three.
- **Template.** `TokenizerProfile.template_prefix` and `template_suffix` replace
  `templated`; a prompt is rendered through them when either is not empty and the run is
  not raw.

## Result

Version 11 changes every checkpoint, so nothing is compared byte for byte across it. The
pass was held to the baseline above on Gemma 4 E4B, on the 5,783-byte test prompt (about
1,200 tokens, three ranges) with 8 output tokens, and to the parity gate with its
challenge leg.

| | Before | After |
| --- | --- | --- |
| Stages in the claim | 348 | 348: three more bind stages, and no origin stages |
| Chain time, warm | 97.6 s | 82.6 s; the first run after an import, cold, took 135 s |
| `range_p0_l23`, 512 tokens | 353,510 steps, replay 197 to 202 s | 353,510 steps, replay 196 s |
| `check_c0` | 14,564 steps a token; `keep_last_row` 75.8M cycles | 14,564 steps a token (131,209 at 8 tokens, replay 41 s); `keep_last_row` 19.9M |
| Bind | one stage of 124,852 steps for the whole prompt | 27,746 and 27,842 steps for the two 512-token ranges, replay 17 s; heaviest tile `append_pieces`, 8.5M cycles |
| `prompt_check_p0`, 512 tokens | 444 KB commitment, replay 131 s | 2,730,134 steps, 443 KB commitment, replay 141 s; no tile above 0.4M cycles |

The chain times differ by less than two runs of the same build do, so claim time is
unchanged, not improved.

On a prompt twice as long (about 2,400 tokens, five ranges) the fourth range's bind is
27,620 steps and its check 2,727,347: a range is the same size wherever it is in the prompt.

The step counts and cycles are from `challenge build` and `challenge profile` on each stage.
The 128-step fault-window figures of the baseline were not measured again.

Two things the measurements showed that the baseline had not:

- **The layer stage's heaviest tile is `binding_copy_rows`, 45.7M cycles**, not
  `accumulate_context` at 26.9M. It copies 64 rows a call. The code is unchanged by this
  pass, and the baseline's profile had sampled fewer tiles. It is now the heaviest step in
  the chain.
- **A bind stage can be too short to replay.** On the pinned parity model a range is six
  tokens and a bind stage 107 to 112 steps, under the 128-step window, so the gate's
  challenge leg no longer replays `prompt-bind` through authenticated Raster and checks
  only that recomputation finds a lie there. On a real model the same happens to the bind
  of a final range that holds only the template's closing tokens, or a token or two of
  text before them. See [the issue](../issues/short-stage-fraud-window.md), which also
  has the sizes of the other short stages on E4B.

With the origin stages removed, all 348 checkpoints of the E4B claim are byte for byte
what they were with them.
