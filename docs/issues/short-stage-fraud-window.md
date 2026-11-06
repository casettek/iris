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
`embed_*`, `aux_*`, `check_*` and `output_finalize` stages are all short.

The two constant stages are short at any size, so a claim whose first false
checkpoint is `range_origin` or `decode_init` cannot be challenged. `output_finalize`
holds the output commitment, so on a small claim that commitment cannot be challenged
either.

## A direction that was tried

Raster derives `bits_per_item = ceil(128 / window_size)`, and its closed
`fraud-window-trace-ends` issue treats sizing the window as a proof-size choice, so a
smaller window should not weaken detection. Retrying a too-short replay at
`--fraud-proof-window-size 2` needed no Raster or stage change: every stage above then
replayed and matched its native checkpoint on the real model, and `challenge profile`
accepted the result. It was not carried through a fault proof, and was removed so that
this can be designed as a whole.

The earlier rule was that a stage too short to dispute belongs inside a bigger stage.
That is how `sequence_join` was folded into `input-embedding`. `decode_init` cannot
become a committed constant as things stand, because a `[[chain.repeat]]` `first =`
fallback must name a stage.
