# Verifying the stage schedule inside the Raster stages

Draft for review. November 9, 2026. Replaces the earlier drafts of the same day.

Since November 10, 2026 the schedule is two counts, prompt ranges and claimed ranges:
`prompt-check` replaced `prompt-seed`, `prompt-merge` and `prompt-prepare`, so the merge
batch count `B` and the batch coordinate described below no longer exist. The current
rules are in [raster-chain.md](../internals/raster-chain.md#stage-schedule).

## Goal

**Give the pipeline its stage schedule as an input, verify that schedule in a stage, and have every later stage use it to verify that it is being run in the right order.**

The **stage schedule** is which stages a claim runs and in what order.

## The challenge this serves

The challenge does not change, and nothing in this proposal touches it:

1. The challenger reruns the pipeline honestly.
2. It reaches the first stage checkpoint that does not match the claim. Neither the challenger nor the contract asks why.
3. The challenger replays its version of that stage in Raster and posts the fingerprint. This shows it started from the claimer's state at the last matching checkpoint and moved honestly to a different state.
4. If the challenger lied, the claimer or anyone else refutes that fingerprint with a standard Raster fault proof.

Nobody in this exchange checks the schedule, so the stages must. Each stage's check is a tile whose result is committed in the stage's output. A stage run out of order and reported honestly gives an invalid result. A stage run out of order and reported as passing is a false tile result, which a standard fault proof refutes.

## The design

### The schedule is three counts

```text
StageSchedule {
    prompt_ranges,    // P: prompt ranges
    merge_batches,    // B: merge batches each range runs after its seed
    claimed_ranges,   // C: ranges of claimed tokens
}
```

Together with the model's layer count `L`, these determine every stage and its place. The order is the one the chain already runs in:

1. `prompt_bind`
2. For each prompt range `j` in `0..P`: `prompt_seed`, `prompt_merge` batches `0..B`, `prompt_prepare`, `embed_p{j}`
3. For each claimed range `k` in `0..C`: `embed_c{k}`
4. For each layer `l` in `0..L`: the auxiliary stage of every range, then the transformer stage of every range
5. For each claimed range `k`: `check_c{k}`
6. `output_finalize`

The schedule is counts and not a list of stage entries. A list would be derived entirely from the counts, and Raster's authoring rules reject a committed collection that another input determines. Counts also need no lookup by index, and an out-of-range index in Raster aborts a stage with no output. `L` is left out for the same reason: the model descriptor already fixes it.

### It enters as an input of `prompt-bind` and is verified there

`prompt-bind` gains one committed input, `schedule`, beside `text` and `config`. Run preparation writes it from the counts it already computes for the chain manifest.

The verification lives in `prompt-bind` and not in a new program. A program that only checked three counts would have a trace too short to replay, and the house rule is that such work belongs inside a bigger stage ([short-stage issue](../issues/short-stage-fraud-window.md)). `prompt-bind` is already the first stage that takes inputs, and it already binds the statement.

A tile checks that every count is at least one and within a fixed maximum, so no later arithmetic can overflow, and that `C` is no more than the token budget allows at this range size. It then stores the schedule in the `Execution` record beside the statement. A failed check sets the execution fault, and the stage finishes.

`P` and `B` depend on how the prompt tokenizes, which is not known until the ranges are merged. Verifying them up front would mean tokenizing the whole prompt inside `prompt-bind`. Merge rounds depend on the data and a sequence cannot loop, so that cannot be written as one program. Those two counts are verified where the truth appears: the stages below hold the run to them, so a schedule with a wrong `P` or `B` cannot finish with a valid record.

### Every stage records its step and checks it against the schedule

`Execution` is the record every stage output already carries inside its [ModelContext](../../crates/model/identity/src/context.rs). It gains the schedule and a **step**: which operation produced this output, and its range, layer and batch. Because `Execution` is encoded as bytes, this changes its version and no boundary type.

The schedule and the steps travel on the connections the stages already have. Every stage descends from `prompt_bind` through its inputs, so no stage needs a new input to receive them.

Each stage, in a tile:

1. Works out its own step. The operation is a constant in the program's tile code. The coordinates come from the committed inputs: the range from the previous range's step, the layer from the authenticated layer parameters, the batch from the previous batch's step. A label supplied by the host is never used.
2. Requires that step to exist in the schedule.
3. Requires each input to carry exactly the step the schedule puts before it.
4. Records its own step in its output.

| Stage | Its step must be | Its inputs must carry |
| --- | --- | --- |
| `prompt_seed` | range `j < P` | `prompt_prepare` of range `j-1`, or `prompt_bind` when `j = 0` |
| `prompt_merge` | batch `b < B` of range `j` | batch `b-1` of range `j`, or its `prompt_seed` when `b = 0` |
| `prompt_prepare` | range `j` | batch `B-1` of range `j`. The prompt is complete exactly when `j = P-1`. |
| `input-embedding`, prompt | range `j` | `prompt_prepare` of range `j`, and `embed_p{j-1}` or the origin |
| `input-embedding`, claimed | range `k < C` | `embed_c{k-1}`, or `embed_p{P-1}` when `k = 0` |
| `prefill-prepare-aux` | layer `l < L` of range `r` | the embedding of range `r` |
| `prefill-range` | layer `l < L` of range `r` | layer `l-1` of range `r` (or its embedding), layer `l` of the range before (or the origin), the auxiliary stage of layer `l` and range `r`, and the donor layer |
| `sequence-check` | claimed range `k` | layer `L-1` of range `k` and of the range before, `check_c{k-1}` or the empty edge, and `prompt_prepare` of range `P-1` |
| `output-finalize` | last | `check_c{C-1}` and `prompt_prepare` of range `P-1` |

Most of these requirements exist today in another form: the producer role, the completed layer count, the phase and the token history. What the schedule adds is the explicit step, the batch count, a check in `prompt-merge`, which has none of its own today, and the three counts each run is held to.

A fault is sticky and reaches the final record, as the existing execution fault does. A valid final record therefore means every stage in the schedule ran, each after the stages the schedule puts before it.

### What "order" covers

Each stage verifies it against the stages whose outputs it receives, and the final stage verifies that the schedule's last step was reached. Two stages that do not feed each other, such as the auxiliary stages of two layers, are not ordered against each other, because neither receives the other's output. No output depends on which of them ran first.

Ordering those too would mean passing a value from each stage to the next in schedule order. A Raster chain connection carries a producer's whole output, and those outputs differ in type from stage to stage, so that would need a change to Raster's chain connections. This proposal does not make one.

## Authoring rules this work holds to

Re-read the Raster skill, with its `recur.md` and `data-and-io.md` references, before changing any stage.

- **No Raster change and no new program.** One program gains one input. No boundary type changes.
- **A check is tile code.** The rules are functions in `pipeline-core` and `model-identity`, which are `no_std` and shared by the tiles and the native routines. Most stages already call a transition function from a tile, such as `head_model_context` in `sequence-check`, and the new requirements go inside those functions. Only `prompt-bind` and `prompt-merge` gain a `call!`.
- **Sequences only route.** No branch, no computed argument, no field read on a binding.
- **A failed check sets the fault and the stage finishes.** No `Result`, `?`, assert or panic.
- **The schedule is a small struct of scalars**, so it passes whole into one tile. The stage's interface type is a real struct in the shared schema file the prompt stages already link, not an alias.
- **The context stays small.** It is loop-carried state in two recurs, `bind_prompt_token` and `bind_embedding_token`, and is re-committed every iteration. The schedule and step add a few dozen bytes.
- **The native routine calls the same functions**, so native and Raster outputs agree byte for byte.
- **Done means the check ladder**: both compile postures, `cargo raster cfs` with no dataflow showing as `external`, `cargo raster build --backend risc0`, `cargo raster program --verify`, and chain parity. `Raster.lock` is committed and never edited by hand.

The existing context code departs from the skill in two places, which this proposal inherits: `ModelContext` has a hand-written `Default`, and `Execution` is stored through `BytesPage::__from_parts`. Storing the schedule and step uses that same path and adds no other.

## 1 Define the schedule and its rules

Add `StageSchedule`, the step, and the rules to [pipeline-core](../../crates/inference/pipeline-core/src/schedule.rs): the limits on each count, whether a step exists in a schedule, and which step the schedule puts before a given one for each kind of input.

This step adds rules and changes nothing a stage does. `Execution` is not extended here, because that changes the bytes every stage writes and so moves every program's commitment. It belongs with step 2, where the schedule first enters a stage.

**Complete when:** unit tests show that a small schedule yields exactly one step sequence, that every step outside it is rejected, and that counts at zero, at the maximum and past it have explicit outcomes with no overflow.

**Status:** implemented on November 9, 2026, with six unit tests in `schedule.rs`.

## 2 Take the schedule as an input and verify it

Extend `Execution` with the schedule and the step, and raise its version. Then, in `prompt-bind`:

- add the `schedule` parameter to `main`, with its struct in [prompt.rs](../../crates/prompt-core/src/prompt.rs);
- add a tile that verifies it against the configuration and stores it in `Execution`, and route the context through it after `identity_bound_context`;
- do the same in the native routine.

Outside the stage:

- have [run-prep](../../crates/model/run-prep/src/lib.rs) write the schedule as a run file, as it writes `config`;
- add the binding to the `prompt_bind` stage in the chain template that [model import](../../crates/model/import/src/main.rs) writes;
- have [claim preparation](../../crates/inference/cli/src/claim.rs) fill it in, as it fills `text` and `config`;
- have direct inference store the same schedule in the context it builds, because its last selection is compared byte for byte with the chain's last check.

**Complete when:** an honest schedule gives identical `prompt-bind` outputs natively and in Raster; a zero count, a count past its maximum, and a `C` beyond the token budget each set the fault while the stage finishes; and `cfs.json` shows the verifying tile taking the schedule from the entry argument and its result reaching the output.

**Status:** implemented on November 9, 2026. `prompt-bind` is re-locked and verifies against its lock. `just test-parity` passes with exact parity across Raster, native-staged and direct, including its challenge leg. A guest test, `stage_schedule_is_checked_and_stored_in_guest`, runs the tile in RISC Zero with an honest schedule and three bad ones. The other ten programs are not re-locked: `Execution` changed the bytes they write, so their locks are stale until step 4, and the import and prepared-run versions are not bumped yet.

## 3 Record and check the step in every stage

Apply the table above, one group at a time, updating the [native routines](../../crates/inference/staged/src/routines/mod.rs) with each group.

| Group | Programs |
| --- | --- |
| Prompt | `prompt-seed`, `prompt-merge`, `prompt-prepare` |
| Embedding and layers | `input-embedding`, `prefill-prepare-aux`, `prefill-range` |
| Result | `sequence-check`, `output-finalize` |

`prompt-merge` needs a new tile in its `lib.rs` and one `call!`, because it calls no transition today. Leave `identity.rs` alone: it holds the identity tiles and is the same file in all three prompt programs.

Check one thing first. `prompt-merge` and `prompt-prepare` also run in the single-range chains, as `prompt_merge_seed` in `runtime/prompt-fixtures/ranked-bpe/Raster.toml` and in the non-ranged model template. There the pieces come from an external with no `prompt-bind` before it, so there is no schedule. Find out what those chains report today and decide what they should report after this change.

**Complete when:** for each program, a wrong program in its place, a wrong range, layer or batch, a repeated stage, a skipped stage, and a count that disagrees with the schedule each set the fault, with a native test and a guest test in `model_identity_guest.rs`; and native and Raster outputs match for each group.

**Status:** implemented on November 9, 2026, with three items left for step 4.

- The rules are in `pipeline.rs`: each transition works out its step, requires the schedule to contain it, and requires every input to carry the step the schedule puts before it. `prompt-merge` has a new `continue_range` tile. The other seven programs changed only through the transition functions their tiles already call.
- Native tests in `schedule_tests.rs` run a whole schedule honestly and then about forty ways of leaving it. Every one sets the fault.
- `just test-parity` was stopped after its comparison legs. All three cases matched exactly across Raster without authentication, native-staged and direct, with no fault in an honest record, and the prompt-fault and chain-fault legs passed.
- The single-range chains now get an execution fault from `prompt-merge`. They already got one from `prompt-prepare`, because no statement is bound there.

Left for step 4: the challenge leg of the parity gate was not run with these checks; no guest test runs the new checks inside RISC Zero yet, and the existing guest-test fixtures were updated but not run; and no program was re-locked, so every `Raster.lock` is stale, including `prompt-bind`'s.

## 4 Run the whole pipeline

Run every changed program through the check ladder and re-lock it. `model-identity` is linked by every stage, so run `cargo raster program --verify` in all of them and treat a mismatch in a program the change should not have reached as drift to investigate.

Bump the import and prepared-run version and publish the new program commitments. Old stage binaries and contexts are not compatible. Update the stage descriptions in [raster-chain.md](../internals/raster-chain.md).

Extend [test-parity](../workflows/test-parity.md)'s claim cases with a schedule that disagrees with the run. `challenge build` is not modified.

**Complete when:**

- an honest claim against an honest challenger shows no divergence;
- a claim whose run does not follow its schedule, with honest outputs, has an invalid final record;
- the same claim with a checkpoint doctored to pass is found by `challenge build` at its first divergent stage and replayed, as any other false checkpoint is. For a stage too short for the replay window, the leg checks only that recomputation finds it, as it does today.

Fault proving stays outside the gate, as it is today.

**Status:** implemented on November 9, 2026. All three completion items are met.

- All eleven programs are rebuilt, re-locked and verified against their locks. Imports and prepared runs are version 7, so every existing import must be re-imported.
- All seven guest tests pass in RISC Zero with the new checks, including the production `sequence-check` and `output-finalize` tiles against their native rules.
- `iris test-parity --skip-challenges` (a new option) passes: exact parity on all three cases, and three new chain-fault cases whose schedule names one more prompt range, merge batch or claimed range than the run. Each finishes through Raster and native-staged, agrees byte for byte, and ends with a model fault.
- `raster-chain.md` has a "Stage schedule" section, and `test-parity.md` lists the new cases and the option.

- The full gate, with its challenge leg, passes in 81 minutes: exact parity on all three cases, and 49 challenge cases found at their first divergent stage. The new `other-schedule` case, whose claimed schedule names one more merge batch than the run, first differs at `prompt_bind` and is replayed there. The ten cases that are not replayed are the same ten short stages as before this work.

The real model has not been re-imported or run; everything above used the synthetic parity model.

## Outside this proposal

- **The challenge, claim acceptance and the contract.** They are unchanged.
- **Any Raster change**, including a value passed from each stage to the next in schedule order.
- **Cuts.** Each cut is still a per-range input, checked where it is consumed. Whether the host's choice of cuts and counts is the only valid one is a separate question from whether the run follows its schedule.
- **Converting the three wirings that stop `prefill-range` into faults.**
- **Short stages and fault proving.** Both stay deferred.

## What I would implement first

Step 1, then step 2. Together they put a verified schedule in every stage's context without changing what any later stage checks, so they can be reviewed and re-locked before step 3 touches the other eight programs.
