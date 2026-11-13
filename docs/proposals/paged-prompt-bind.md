# Binding the prompt one range at a time

Proposed November 11, 2026, and implemented the same day in import version 11. What was
decided while building it, where it departs from this text, and what it measured are in
[pipeline-v11.md](pipeline-v11.md); [raster-chain.md](../internals/raster-chain.md)
describes the chain as it is. The measurements below are from before the change, on
Gemma 4 E4B (`Q8_0`) and the 5,783-byte test prompt, about 1,200 tokens.
[What keeps it sound](#what-keeps-it-sound) is the part to review first.

## Goal

**Replace the one `prompt_bind` stage, which grows with the whole prompt, with one bind stage
per prompt range, and have each `prompt_check` read only its own range's pieces.**

The target is a 100,000-token prompt. Every other stage of the chain is already sized by a
range of at most `range_size` tokens. `prompt_bind` is the exception, and `prompt_check`
inherits its size because it takes all of `prompt_bind`'s output.

This is about the size of a stage and the cost of a challenge. It does not change
time-to-claim: natively the whole bind takes under two seconds.

## What does not change

- The challenge: rerun honestly, find the first mismatching checkpoint, replay that stage.
- The work done per character. The tiles call the same shared functions (`fold_text`,
  `scan_chunk_into`, `apply_atom`) with the same results.
- The pair check in `prompt_check` and the argument for it in
  [claimed-prompt-tokens.md](claimed-prompt-tokens.md).
- Everything from `embed_p{j}` onward, and `output_finalize`, which grows with the output
  only.
- The schedule's two counts, `P` and `C`.
- A wrong claim never aborts a stage. It publishes a fault, and the final record rejects
  the claim.
- Raster.

## Today

`prompt_bind` runs once. It folds SHA-256 over the text, scans the text into atoms (one per
character or special token), and for every atom checks the vocabulary bucket it falls in
against the committed tokenizer, looks the atom up, and appends its pieces.

Measured by running the stage authenticated and listing its steps:

| | Whole stage | Per atom |
| --- | --- | --- |
| Atoms | 5,799 | |
| Tile calls | about 113,000 | 19.5 |
| Trace steps | about 171,000 | 29.5 |
| Raw trace | about 220 MB | 38 KB |
| Raster scratch at the end | 548 MB | 94 KB |
| Published pieces | 534 KB | 92 bytes |

The text pass is 46 of those tile calls: 23 chunks hashed and 23 scanned. All the rest is
per atom, and about 8.5 of the 19.5 calls per atom walk the atom's vocabulary bucket twice,
once to check it and once to search it (4.3 entries on average).

Scaled linearly to 100,000 tokens at this prompt's 4.8 bytes per token:

| | One 512-token range | 100,000 tokens in one stage |
| --- | --- | --- |
| Atoms | about 2,450 | about 480,000 |
| Trace steps | about 72,000 | about 14 million |
| Raw trace | about 90 MB | about 18 GB |
| Raster scratch | about 230 MB | about 45 GB |
| Authenticated run | about 10 s | roughly half an hour |

Raster accepts traces of up to 2³² steps, so nothing fails outright. The stage just stops
being comparable to any other, and its scratch space would not fit on most challengers.

`prompt_check_p{j}` has the second half of the problem. It takes every piece of the prompt as
an input and walks the whole list twice, once to find where its range ends and once to copy
its range out, because a Raster stage cannot select a range whose bounds it computes. At
100,000 tokens that is a 44 MB list read twice by each of 196 stages.

For scale, a `prompt_check` range is already much heavier than a bind range would be.
Listing the steps of `prompt_check_p0` for this prompt was stopped after nine minutes with
2.2 GB of raw trace written and more to come; the whole of `prompt_bind` is 220 MB. Its real
size is not measured, and this proposal does not change it.

## The design

### The text is cut into one slice per range

Run preparation already tokenizes the prompt natively to choose the ranges. It would also
cut the original text into `P` consecutive slices and commit each as the `text` input of its
own stage, `prompt_bind_p{j}`. A slice is a list of chunks of at most 256 bytes, as the whole
text is today.

A stage can only read part of an earlier stage's list if that part is its own artifact, so
the cut has to be made in the inputs. It cannot be made by a later stage choosing an offset.

### Each bind stage continues from the one before

`prompt_bind_p{j}` runs the same three loops over its slice, starting each from the state
`prompt_bind_p{j-1}` ended in:

| State | What it holds | Size |
| --- | --- | --- |
| `TextFold` | SHA-256 chaining words, up to 63 pending bytes, byte count, first and last non-whitespace offsets | fixed |
| `ScanState` | text held back in case it starts a special token, the offset, whether the text has ended | shorter than the longest special token |
| `SplitState` | the segment number and whether the last atom was unknown | fixed |

These travel in the stage's output beside `PromptContext`. `prompt_bind_p0` starts from the
same empty states `prompt_bind` starts from today, with the template's prefix pending.

Its output is the pieces of its own slice, in order, and the context.

### What is only known at the end becomes a claim

Three things the first range needs are properties of the whole text:

- its SHA-256, which the statement is bound to in the first stage;
- its length, which tells the scan where the text ends and the template's suffix goes;
- the offsets of its first and last non-whitespace characters, which decide what a templated
  prompt keeps.

Today `prompt_bind` reads the whole text once to learn them and a second time to scan it.
With slices, run preparation commits them as an input of `prompt_bind_p0`, next to the
schedule: `input_text_sha256`, `total_bytes`, `keep_start`, `keep_end`. Every bind stage
scans with these claimed values and goes on folding the real ones. The stage that sees the
end of the text compares the two and publishes a fault if any differs.

This is how the schedule's `P` and the marks are already treated: supplied, used, and held
to by a later stage.

### `prompt_check_p{j}` reads its own range

Its inputs become `prompt_bind_p{j}`, `prompt_check_p{j-1}` and its marks, in place of the
whole prompt's pieces. The two walks over every piece go away.

A slice's pieces are not exactly a range's tokens, for two reasons. The scan holds back up to
one special token's length of text until it sees what follows, so the last few characters of
a slice produce their atoms in the next one. And a character outside the vocabulary becomes
one piece per byte, which a range may end between. So a check takes its pieces from a stream:

1. the pieces the previous check left unconsumed, then
2. the pieces of `prompt_bind_p{j}`.

It takes as many as it has marks, as it does now, and publishes the remainder for the next
check alongside the last token's pieces it already publishes. A range is therefore the whole
tokens available to it, up to `range_size`. Preparation cuts each slice just late enough for
its range to be full, so the remainder is a handful of pieces.

The prompt is complete at range `P-1` when the bind stage has seen the end of the text with
the claim matching, and the check leaves no remainder.

### The schedule

No new count. The table in [raster-chain.md](../internals/raster-chain.md#stage-schedule)
changes in two rows and gains one:

| Stage | Its step must be | Its inputs must carry |
| --- | --- | --- |
| `prompt_bind_p{j}` | range `j < P` | `prompt_bind_p{j-1}`, or nothing when `j = 0` |
| `prompt_check_p{j}` | range `j < P` | `prompt_bind_p{j}`, and `prompt_check_p{j-1}` unless `j = 0` |

The chain order for each prompt range becomes `prompt_bind_p{j}`, `prompt_check_p{j}`,
`embed_p{j}`. A claim gains `P - 1` stages and checkpoints.

### New faults

Each is recorded in `PromptContext.fault` and the stage finishes:

- the claimed hash, length or trim offsets differ from the text's;
- the text has not ended at range `P-1`, or a slice follows the end;
- a slice is larger than the limit;
- a check is handed more unconsumed pieces than the limit;
- the last check leaves pieces unconsumed.

## What keeps it sound

**The pieces are the same pieces.** Each of the three loops is a fold: its result depends
only on the state it starts in and the chunks it is fed, in order. Stopping after any chunk
and resuming in another stage from the same state gives the same atoms and pieces as running
straight through. The cut points are therefore not something a claim can get wrong; any cut
gives the unpaged result.

**The slices are the text.** The SHA-256 state is carried from slice to slice, so the final
hash is the hash of the slices in order. The schedule rule requires each bind stage to follow
the one before it under the same statement, so a slice cannot be dropped, repeated or
reordered. The last stage requires that hash to be the one the statement was bound to.

**The claimed values cannot change the result.** The scan's output is a function of the text
and the four claimed values. If they are the true ones, the atoms are today's atoms. If any
is not, the stage that reaches the end of the text finds the difference and publishes a
fault. A length that is too large means the scan never sees the end, and range `P-1` is not
complete.

**The checks see every piece once.** Each check takes its pieces from the front of the
stream and passes on what is left, so the concatenation of what the checks consume is the
concatenation of what the binds produced. The pair check's argument is about consecutive
pieces and their tokens and does not depend on where a range ends.

What this leaves to check is that the carried states really are the whole state: that
`scan_chunk_into` and `apply_atom` read nothing else that changes between chunks.

## Open points for authoring

- **Where the carried state lives.** `PromptContext` is described as all-scalar and small
  and travels through every prompt stage; the three states could join it or sit beside it in
  `PromptState`.
- **Where the unconsumed pieces live.** In the same `pieces` list after the last token's
  pieces with a count in the context, or in a second list.
- **The two limits.** A slice of a full range is about 2.4 KB here. The largest honest slice
  is `range_size` tokens of the vocabulary's longest entry in four-byte characters, about
  64 KB at 512 tokens. The unconsumed pieces are bounded by the longest special token plus
  one character's bytes.
- **A very small last slice.** A raw prompt whose last range is a few characters gives a
  bind stage under the fraud window. This is the existing
  [short-stage issue](../issues/short-stage-fraud-window.md). Because a range may hold fewer
  than `range_size` tokens, preparation can move the previous cut earlier to avoid it.
- **Whether `prompt_bind_p0` should be its own program.** It takes the configuration, the
  schedule and the text claim where the others take a previous stage. One program with an
  empty previous stage, as the layer ranges do with `range_origin`, avoids a second lock.

## What would change in the repository

- `prompt-bind` takes a previous stage and a slice, and `prompt-check` takes one bind stage
  in place of the whole prompt. Both re-lock.
- The schedule rules in `pipeline-core` and `model-identity` gain the bind range. They are
  shared code: after the last schedule change only `decode-init` and `range-origin` kept
  their locks, so expect to re-lock the rest, about two hours of machine time.
- Run preparation writes `P` text slices and the text claim. `prepared_run.json`, the import
  and the backend's pinned version move to 11. `iris claim build` takes nothing new.
- The native kernels in `staged-infer` follow the stages, and the parity gate's stage lists
  gain `prompt_bind_p{j}`.
- `test-parity` gains cases: a wrong claimed hash, length and trim offset; a dropped and a
  repeated slice; honest cuts inside a special token and inside a byte-fallback character.
  `falsified-prompt_bind` becomes a falsified bind range.
- `raster-chain.md`, `claim-build.md` and `test-parity.md`.

## Left out

- **One walk of the bucket per atom.** Each atom's bucket is walked once to check it and
  once to search it. One walk could do both, for about 12 tile calls per atom in place of
  19.5. It is independent of paging, but it changes the same program; it is taken up in
  [pipeline-v11.md](pipeline-v11.md).
- **Fusing bind into `prompt_check`.** One program per range in place of two, at the price
  of making the heaviest prompt stage heavier.
- **Slices on a fixed byte grid.** A check would then need a varying number of bind stages
  as inputs, which the chain template cannot express.
- **`output_finalize`.** It is one loop round per generated token and stays small while
  outputs stay under about a thousand tokens.

## Questions for you

- Is one bind stage per token range right, given that a bind range is about 72,000 trace
  steps and a check range is far more? A bind stage could cover several ranges, at the cost
  of the varying inputs described above.
- Should the single bucket walk go into the same change, since `prompt-bind` re-locks anyway?
- `prompt_check` turned out to be the heavy prompt stage. Should it be measured properly and
  possibly split before this is implemented, or after?
