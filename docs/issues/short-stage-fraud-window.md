# Issue: `short-stage-fraud-window` — a stage whose trace fits in one fraud window cannot be disputed

Status: open, deferred to its own project on 2026-11-05. Until then a stage too short
for the window is not used for challenges.

## The gap

`challenge build` replays the disputed stage with
`cargo raster chain run --fraud-proof-window-size 128`
([`challenge.rs`](../../crates/inference/cli/src/challenge.rs), `run_raster_replay`).
Raster's commitment writer refuses any trace whose length does not exceed the
window. `CommitmentFile::finish` returns
`InvalidWindow("trace length outside commitment limits")` when
`len <= window_size` (`raster` repo, `crates/raster-prover/src/commitment_file.rs`).
The authenticated replay panics, so no challenge can be built for that stage. A
claimer can lie about such a stage without being caught.

## Scope

Measured on Gemma 4 E2B Q8 on 2026-11-05, in the teacher-forced chain:

| stage | trace steps | at or under 128 |
| --- | --- | --- |
| `range_origin` | 3 | every claim |
| `decode_init` | 3 | every claim |
| `output_finalize` | 19 at 3 output tokens, 58 at 16 | under about 40 output tokens |
| `embed_c{k}` | 34 at 3 tokens in the range, 112 at 16 | under about 19 tokens in the range |
| `embed_p{j}` | 93 at 13 tokens in the range, 297 at 47 | under about 19 tokens in the range |
| `aux_c{k}_l{l}` | 68 at 3 tokens in the range, 315 at 16 | under about 7 tokens in the range |

The token limits are fitted to those two points. A range's last stages shrink with its
token count, so a short final range can stay under the window on a long claim. Every
other stage replayed at the smallest real claim tried, 13 prompt tokens and 3 output
tokens. On the 512-token `parity-gemma` vocabulary and its 6-token ranges, the
`embed_*`, `aux_*`, `check_*` and `output_finalize` stages are all short, and so is
each `prompt_bind_p{j}` (107 to 112 steps).

### On the version 11 chain

Measured on Gemma 4 E4B Q8 on 2026-11-11, on one claim of about 1,200 prompt tokens
in three ranges and 8 output tokens. E4B's rows are wider than E2B's, so its stages
take more steps a token and stay short for fewer tokens. Each line is fitted to the
points given.

| stage | trace steps | at or under 128 |
| --- | --- | --- |
| `embed_p{j}`, `embed_c{k}` | 138 at 8 tokens in the range, 2,440 at 185, 6,690 at 512: 13 a token and 34 | 7 tokens or fewer in the range |
| `aux_p{j}_l{l}`, `aux_c{k}_l{l}` | 297 at 8 tokens in the range, 5,961 at 185, 16,425 at 512: 32 a token and 41 | 2 tokens or fewer in the range |
| `prompt_bind_p{j}` | 9,874 at 185 tokens in the range, 27,746 at 512: about 11 for each atom of its slice of the text | a slice of about 8 atoms or fewer |

Long at every size tried: `range_*` (5,700 steps at 8 tokens in the range, 353,510
at 512), `check_c{k}` (131,209 at 8 tokens) and `prompt_check_p{j}` (2,730,134 at
512 tokens; it replays at the pinned model's six-token ranges, and was not measured
on fewer).

The chain has no origin stages and no `output_finalize` stage (see below), so no
stage is short at every size, and none is short because the output is.

The bind stages are new in version 11. Until then one
`prompt_bind` stage bound the whole prompt and was long at every real size. A bind
stage's size follows its slice of the text, and a slice ends where its range's
pieces have all been produced. The template's closing tokens are all produced when
the text ends, so a final range that holds only those (five tokens for Gemma) has
an empty slice, and one that holds a token or two of text before them has a slice
of a few characters. Both are under the window. The length of an empty slice's bind
was not measured.

Until 2026-11-11 three one-tile stages, `range_origin`, `decode_init` and from
version 11 `prompt_origin`, produced the empty values the first range starts from.
They were short at any size and came first in the chain, so a claim whose first
false checkpoint was one of them could not be challenged. They are gone: the empty
values are committed constants written by the import, and range 0's stages bind
them directly (see [raster-chain.md](../internals/raster-chain.md)). A claim that
binds another constant differs from honest execution at the stage that reads it.

`output_finalize` held the output commitment and was three steps a token and ten
(35 at 8 output tokens on E4B), so on a claim of under 40 output tokens that
commitment could not be challenged. Since version 12 every check states the output
record itself, and the claim's is the last check's, a stage of about 16,400 steps a
token.

## A direction that was tried

Raster derives `bits_per_item = ceil(128 / window_size)`, and its closed
`fraud-window-trace-ends` issue treats sizing the window as a proof-size choice, so a
smaller window should not weaken detection. Retrying a too-short replay at
`--fraud-proof-window-size 2` needed no Raster or stage change: every stage above then
replayed and matched its native checkpoint on the real model, and `challenge profile`
accepted the result. It was not carried through a fault proof, and was removed so that
this can be designed as a whole.

The rule is that a stage too short to dispute is not a stage: its work belongs inside
a bigger one, or is data. That is how `sequence_join` was folded into
`input-embedding`, and how the origin stages became constants. A `[[chain.repeat]]`
`first =` fallback must name a stage, which is why range 0 is written out as plain
stages and the repeats start at 1.
