# Issue: `short-stage-fraud-window` — a stage whose trace fits in one fraud window cannot be disputed

Status: open for `decode_init`; resolved for the teacher-forced chain. Observed
2026-11-03 while validating the teacher-forced chain on `parity-gemma`.

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

Observed failures, each a corrupted checkpoint hash for an otherwise honest claim:

| chain | stage | why the trace is short |
| --- | --- | --- |
| autoregressive | `decode_init` | one tile, no input |
| teacher-forced (first draft) | `sequence_join` | two sweeps in 64-id chunks: one chunk each for the 15 + 8 ids tested |
| teacher-forced, `parity-gemma` only | `sequence_check` | 8 positions × 2 head pages; on Gemma 4 E2B each position sweeps 8,192 pages |

## Scope

A stage's trace grows with the tokens it handles. An audit on Gemma 4 E2B with a
3-token "say ciao" claim found `input_embedding` and `output_finalize` (both chains),
plus `decode_init`, `decode_embed_t*` and `decode_aux_t*_l*` (autoregressive),
under the window. Every other stage replayed. Production claims are far larger,
so this only affects toy-sized inputs and is not a target. Revisit only if small
claims become a use case.

## Rule

Raster is not changed for this. A stage too short to dispute belongs inside a
bigger stage.

- The join now lives inside `input-embedding`, which embeds the prompt and then the
  claimed ids. The autoregressive chain binds an empty committed claim to the same
  program.
- `sequence_check` is short only on the 512-token test vocabulary. It is a sweep
  over the head per generated token, so on any real model it is one of the longest
  stages.
- `decode_init` remains. It cannot become a committed constant, because a
  `[[chain.repeat]]` `first =` fallback must name a stage. Removing it means
  writing decode step 0 out of the repeat, with an empty committed edge as its
  `prior`.
