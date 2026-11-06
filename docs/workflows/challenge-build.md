# Challenge Build Workflow

`challenge build` consumes a compact claimed checkpoint hash list plus frozen run context, recomputes the checkpointed run, stops at the first observed divergent hash, and replays that one divergent stage through Raster.

```bash
cargo run --release -p iris-cli -- challenge build \
  --run inference.toml \
  --claim-context target/staged-infer/runs/.../claim_bundle.json \
  --checkpoints target/staged-infer/runs/.../checkpoints.txt
# or
just challenge \
  target/staged-infer/runs/.../claim_bundle.json \
  target/staged-infer/runs/.../checkpoints.txt
```

Flow:

1. Load the claimed `checkpoints.txt` and use `claim_bundle.json` only as today's local carrier for `prepared_run.json`.
2. Verify the frozen run manifest's hash, then prepare the challenger's own run: the claim's prompt, `raw_prompt`, `tokens` and claimed ids, from the model `--run` (default: `inference.toml`) selects. Run the checkpointed staged executor on that manifest. The prompt and token count are the claim's even if the current run spec has changed.
3. After each recomputed checkpoint, hash the verifier's local checkpoint record and compare it to the claimed hash at the same index.
4. If no divergence exists, report the claimed and recomputed traces.
5. If a divergence exists, seed a replay directory with all prior recomputed stages.
6. Run `cargo raster chain run --stage <stage>` for the divergent stage.
7. Compare the Raster replay checkpoint with the native recomputed checkpoint. The stage name, input commitment, output structural commitment, and output SHA-256 must all match, yielding the same checkpoint hash.
8. Write the challenge artifacts only after that parity check passes. A mismatch fails challenge construction with a native/Raster parity error identifying the stage, differing field, and both values. The replay files remain available for debugging.

Recomputing the claimer's manifest would rerun the claimer's model and agree with
it, which is why step 2 prepares a new run. The claimed ids are checked against the
frozen manifest's commitments. The challenger does not need the claimer's model files. If the selected model's id is not the
one the claim names it warns and carries on. The run then diverges at the first
stage whose committed inputs differ: `prompt_bind` for another configuration or
text, `prompt_seed_p{j}` for another cut, or the first stage reading other weights.
A claim whose `sequence_check` or layer checkpoint is doctored diverges there and
replays that stage as usual.

If every checkpoint matches, there is no stage to dispute, and the claim is
judged by its own final record, `output_finalize`. The claim posts no other
copy of its output. It is rejected, without a replay, when:

- **The prompt was not prepared correctly.** `prompt_fault` is non-zero, or
  `prompt_complete` is not set.
- **A check could not stand behind its edge.** `check_fault` is non-zero:
  a position with no predicting row, a range with no tokens or more than the range
  size, or a range that does not continue the sequence. `check_fault_position`
  says where. The range limit is what bounds each check's replay cost.
- **The committed verdict is not clean.** `claim_mismatches` counts claimed ids
  that are not the greedy selection, and `first_claim_mismatch_position` says
  where the first one is.
- **The output breaks the token budget.** `output_fault` is non-zero: the output
  must fill `max_new_tokens` unless it reaches EOS, and must not continue past EOS.

The chain establishes all of these. The challenge reads them from the final record
and computes none of them itself.

The challenge then reports `claimed output rejected` with the reason.

Challenge hash verification happens during recomputation. Sequential stages stop immediately at the first mismatch. Parallel aux waves still execute with the existing parallelism; if a mismatch is found in the middle of a completed aux batch, later batch outputs may exist on disk but are ignored for the challenge trace and replay seed.

Divergence reasons are checked in this order:

- Checkpoint hash mismatch.
- Missing claimed or recomputed checkpoint hash.

Challenge outputs live under the recomputed run's `challenge/<stage>/` directory:

- `divergence.json`
- `replay_package.json`
- `challenge_trace.json`
- `challenge_bundle.json`
- `challenge_timings.json`

`challenge_timings.json` records version 1, the selected stage and checkpoint index,
and two measured wall-clock durations in milliseconds. `checkpoint_trace_duration_ms`
covers input/context validation, checkpoint recomputation through divergence, and
trace publication. `raster_replay_duration_ms` covers the selected stage's
`cargo raster chain run` command, including any compilation. Replay-directory
seeding and final parity/artifact checks are outside these phase timers and remain
part of the caller's overall challenge duration.

During execution, stdout also includes `IRIS_CHALLENGE_PROGRESS` followed by a
version-1 JSON record at each phase transition: `checkpoint-trace`,
`replay-preparing`, `raster-replay`, and `validating`. Each record includes the
phase's `started_at_ms` Unix timestamp and the completed phase durations (or
null while not yet complete). Consumers can persist these records as they arrive
to display live elapsed times and freeze each phase independently. Ordinary
console output and the final timing artifact are retained alongside these updates.

For manual fraud tests, corrupt a copy of the claimed hash list and challenge that copy:

```bash
just corrupt target/staged-infer/runs/.../checkpoints.txt
just challenge \
  target/staged-infer/runs/.../claim_bundle.json \
  target/staged-infer/runs/.../checkpoints.corrupt.txt
```

`--claim` and `--trace` remain lower-level debug inputs that compare full checkpoint records. `--trace` prepares a run from `--run` because a trace alone has no frozen run metadata. It uses that run spec’s model, prompt, and token count, and discovers the output with direct inference. `challenge locate` remains reserved. See [Fault proving](fault-prove.md) for `challenge corrupt` and `fault prove`.

A stage whose trace is 128 steps or fewer cannot be replayed at this window, so a
challenge that lands on one fails at the replay. See
[short-stage-fraud-window](../issues/short-stage-fraud-window.md).

The authenticated challenge-stage replay explicitly uses `--fraud-proof-window-size 128`.
It builds `commit.bin` during execution with RAM working storage and no raw trace.
Checkpoint output validation and challenge artifacts are unchanged. Fault proving
runs the known stage's program directly with `run --audit`, using the disputed
commitment's window size. It stops at divergence and saves neither a raw trace
nor another commitment.
