# Challenge Build Workflow

`challenge build` consumes a compact claimed checkpoint hash list plus frozen run context, recomputes the checkpointed run, stops at the first observed divergent hash, and replays that one divergent stage through Raster.

```bash
cargo run --release -p iwebs-cli -- challenge build \
  --specification specification.toml \
  --claim-context target/procedure-pipeline/runs/.../claim_bundle.json \
  --checkpoints target/procedure-pipeline/runs/.../checkpoints.txt
# or
just challenge \
  target/procedure-pipeline/runs/.../claim_bundle.json \
  target/procedure-pipeline/runs/.../checkpoints.txt
```

Flow:

1. Load the claimed `checkpoints.txt` and use `claim_bundle.json` only as today's local carrier for `prepared_run.json`.
2. Verify the frozen run manifest's hash, then prepare the challenger's own run: the claim's prompt, `raw_prompt`, `max_new_tokens` and claimed ids, on the models `--specification` (default: `specification.toml`) names. Run the checkpointed pipeline executor on that manifest. The prompt and token count are the claim's, whatever the challenger's specification and any case beside it say.
3. After each recomputed checkpoint, hash the verifier's local checkpoint record and compare it to the claimed hash at the same index.
4. If no divergence exists, report the claimed and recomputed traces.
5. If a divergence exists, seed a replay directory with all prior recomputed stages.
6. Run `cargo raster pipeline run --stage <stage>` for the divergent stage.
7. Compare the Raster replay checkpoint with the native recomputed checkpoint. The stage name, input commitment, output structural commitment, and output SHA-256 must all match, yielding the same checkpoint hash.
8. Write the challenge artifacts only after that parity check passes. A mismatch fails challenge construction with a native/Raster parity error identifying the stage, differing field, and both values. The replay files remain available for debugging.

Recomputing the claimer's manifest would rerun the claimer's model and agree with
it, which is why step 2 prepares a new run. The claimed ids are checked against the
frozen manifest's commitments. The challenger does not need the claimer's model files. If the selected model's id is not the
one the claim names it warns and carries on. The run then diverges at the first
stage whose committed inputs differ: `prompt_bind_p0` for another configuration or
text digest, the first `prompt_bind_p{j}` whose slice of the text differs,
`prompt_check_p{j}` for other token marks, or the first stage reading other weights.
A claim whose `sequence_check` or layer checkpoint is doctored diverges there and
replays that stage as usual.

If every checkpoint matches, there is no stage to dispute, and the claim is
judged by its own final record, the last check's edge and output record. The claim posts no other
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

The pipeline establishes all of these. The challenge reads them from the final record
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
and two measured wall-clock durations in milliseconds. `checkpoint_list_duration_ms`
covers input/context validation, checkpoint recomputation through divergence, and
trace publication. `raster_replay_duration_ms` covers the selected stage's
`cargo raster pipeline run` command, including any compilation. Replay-directory
seeding and final parity/artifact checks are outside these phase timers and remain
part of the caller's overall challenge duration.

During execution, stdout also includes `IWEBS_CHALLENGE_PROGRESS` followed by a
version-1 JSON record at each phase transition: `checkpoint-list`,
`replay-preparing`, `raster-replay`, and `validating`. Each record includes the
phase's `started_at_ms` Unix timestamp and the completed phase durations (or
null while not yet complete). Consumers can persist these records as they arrive
to display live elapsed times and freeze each phase independently. Ordinary
console output and the final timing artifact are retained alongside these updates.

For manual fault tests, corrupt a copy of the claimed hash list and challenge that copy:

```bash
just corrupt target/procedure-pipeline/runs/.../checkpoints.txt
just challenge \
  target/procedure-pipeline/runs/.../claim_bundle.json \
  target/procedure-pipeline/runs/.../checkpoints.corrupt.txt
```

## From a posted claim

The inputs above come from the claimer's own directory. A challenger of a
submitted claim has none of it, and starts from the data-availability layer:

```bash
iwebs claim fetch --submission <claim_submission.json> --specification specification.toml --out fetched
iwebs challenge build --specification specification.toml --submission fetched/claim_submission.json
```

`claim fetch` writes the claim's checkpoint hashes, checkpoints and input
manifests as it posted them, the inputs its run was given (`posted_run.bin`),
and the case they hold, as `case.toml` ([the blockchains](../blockchain/overview.md#starting-from-the-blockchains)).
The specification is the challenger's own, and `challenge build --submission`
reads the case from beside the submission.
`challenge build --submission` runs every stage on the inputs the claim's
checkpoint names for it, each of which the claim posted, the procedure
pins or an earlier stage made. It tokenizes nothing and runs no native
inference: the stages check the marks and the claimed ids they are given,
and `--output-ids` is not used. It writes the pipeline the checkpoints
describe as `claim_arranged.toml` beside the run's manifest and prints a
line for each input the claim gives a stage otherwise than the pipeline does
(`checkpoint 61 (range_p1_l2) gives layer another value than the pipeline
does`). It compares the run with the posted list hash by hash, prints `stage
to replay: <stage>` for the first that differs, and replays that stage on the
same inputs. From there the flow and the outputs are the ones above. It
stops when a checkpoint names a value nobody holds or another program than
the pipeline's, which is a claim for `iwebs claim audit`.

See [Fault proving](fault-prove.md) for `test corrupt-challenge` and `fault prove`.

A stage whose trace is 128 steps or fewer cannot be replayed at this window, so a
challenge that lands on one fails at the replay. See
[short-stage-fault-window](../issues/short-stage-fault-window.md).

The authenticated challenge-stage replay explicitly uses `--fault-proof-window-size 128`.
It builds `commit.bin` during execution with RAM working storage and no raw trace.
Checkpoint output validation and challenge artifacts are unchanged. Fault proving
runs the known stage's program directly with `run --audit`, using the disputed
commitment's window size. It stops at divergence and saves neither a raw trace
nor another commitment.
