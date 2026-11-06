# Native-guided challenge profiling

The backend challenges the slowest stage across the ordered claim measurements.
The original challenge stage replay generates its commitment without profiling.
Stage durations are observed runtimes, not a wall-clock timeline: auxiliary waves
can overlap.

## Three native stage executions

1. Challenge replay: produce the original commitment and measure challenge time.
2. Native timing: measure every tile call, retaining bounded top candidates.
3. Selective capture: run normally and retain input/output bytes only for the
   exact selected invocations. Execute those samples in the guest executor after
   native execution finishes.

```sh
iris challenge profile --challenge /path/challenge_bundle.json --claim-context /path/claim_bundle.json --output /path/profile --top-tiles 3 --calls-per-tile 1 --sample-max-bytes 8589934592
```

Defaults select three tile IDs by their slowest observed native invocation, then
one invocation per ID: at most three guest measurements. With a larger call limit,
each selected tile contributes its K slowest calls. Ties use execution order.
Native runtime is a heuristic for guest cycles, not a global worst-case guarantee.
The native total includes wrapper overhead and final output-store adjustments.

Both profiling reruns must match the original output and parsed commitment. Iris
checks frozen inputs, CFS/lock identity, exact run directories, and the captured
invocation indices/coordinates before choosing a fault target. Neither pass saves
a full trace or a per-invocation native-profile stream.

The version-2 Iris summary includes native selection, guest cycles, coverage,
separate phase durations, identity checks and the eligible target. Single-sample
per-tile p50 is omitted. With multiple samples, p50 is the lower median of that
selected sample, not all executions. Older version-1 summaries remain accepted.

## Storage and failures

The sample-payload limit defaults to 8 GiB. The collector fails explicitly when
its exact selected payloads exceed the limit; it never reduces coverage silently.
Writes require a 2 GiB free-space reserve. The limit does not bound necessary live
model memory or guest-executor working memory.

The backend persists summary/error diagnostics before deleting sample payloads.
Commitments, inputs, outputs, selection/capture manifests and reports remain.
Each pass writes ownership metadata before execution so cleanup also works when
Raster never creates run.json. Cleanup is confined to this invocation; historical
runs are not swept. Standalone Raster runs retain samples for manual replay.

## Fault windows and deferral

```sh
pnpm worker:run-once --stop-before-fault-proof
pnpm worker:fault-prove <run-id>
```

The stop flag completes challenge and profiling, then waits before corruption.
Automatic workers exclude waiting runs. For a measured invocation, h = exec_index
- 1 and j = h + W - 1. Iris chooses the highest-cycle measured call satisfying
W <= j < commitment_length - W, flips one fingerprint bit at j in a copy of the
original commitment, and verifies the proof's signed window. No eligible selected
call blocks proving; it does not trigger extra measurements or middle corruption.
Legacy/manual corruption without a profile retains its original behavior.

## Validation

```sh
cargo test -p iris-cli --test challenge_profile recursive_heavy_window_end_to_end -- --ignored --nocapture
```

This isolated recursive fixture compares native-selected cycles with trace-backed
replay, verifies no full traces are created by the profiling passes, and proves
the selected fault window. It requires the matching Raster CLI, RISC Zero tools,
and proof runtime dependencies. Full-model profiling remains opt-in.

The older feature-gated standalone iris replay-profile command remains a
trace-backed debugging workflow. Its retained traces can be large; use challenge
profile for bounded native-guided profiling.

## Teacher-forced stage costs (2026-11-05)

Gemma 4 E2B Q8, one claim of 47 prompt tokens and 16 claimed tokens, 151 stages.
Each row is that stage's trace and the heaviest tile call `challenge profile`
measured in the guest. These are the selected calls, not a guaranteed maximum.

| stage | trace steps | heaviest measured tile | user cycles |
| --- | --- | --- | --- |
| `prompt_bind` | 2,871 | `append_atom_pieces` | 94,146 |
| `prompt_seed_p0` | 151,131 | `apply_ranked_merge` | 3,168,971 |
| `prompt_merge_p0_b0` | 97,454 | `finish_ranked_step` | 63,695 |
| `prompt_prepare_p0` | 1,146 | `finish_tokens` | 66,174 |
| `embed_p0` | 297 | `append_activation_row` | 5,685,372 |
| `embed_c0` | 112 | `count_row_chunk` | 9,182,832 |
| `aux_p0_l0` | 904 | `mac_weight_page` | 6,774,657 |
| `range_p0_l0` | 13,469 | `mac_q8_weight_page` | 9,626,604 |
| `range_c0_l0` | 4,636 | `mac_q8_weight_page` | 9,626,649 |
| `check_c0` | 131,245 | `score_head_page` | 2,537,947 |
| `output_finalize` | 58 | `copy_output_token_ids` | 220,701 |

The whole chain ran through authenticated Raster with 549 s of stage time, and
`chain audit --execution` reproduced every committed output. `embed_c0` and
`output_finalize` used a window of 2 for this measurement, because they are too short
for the challenge's 128.

Tokenizing dominates long prompts. At 666 prompt tokens, one 512-token range size and
8 claimed tokens the chain has 268 stages, and replays took:

| stage | trace steps | replay |
| --- | --- | --- |
| `prompt_seed_p0` | 1,607,830 | 91 s |
| `prompt_merge_p0_b0` | 1,556,799 | 86 s |
| `prompt_prepare_p0` | 12,675 | 18 s |
| `check_c0` | 65,637 | 29 s |

## Local acceptance (2026-10-31)

The eight-iteration recursive fixture reproduced both original output and
commitment, selected two tile IDs, and measured two calls. Its peak retained sample
payload was 31 bytes; profiling produced no full trace or raw native-profile
stream. Those measurements matched the trace-backed oracle. Corrupting the
selected endpoint produced a verified real fault proof with the planned window
(entries 9–10, window size 2). CPU proof/compression took about 276 seconds.

Separate capture attempts rejected a one-byte budget, wrong coordinates, and a
missing selected invocation without publishing complete run/capture manifests.
Bounded-state tests exercise 10,000 native timings and release witnesses across
1,000 recursive iterations. All ten stage program identities verified. Available
workspace space remained approximately 757 GiB. Full-model profiling was not run.
