# Replay profiling one stage

Use `replay-profile` to investigate whether an inference stage's tile boundaries
produce expensive replay units on a representative workload:

```sh
cargo run --release -p raster-inference-cli -- replay-profile \
  --run inference.toml --stage prefill_range_l13 --top-n 3
# Equivalent shortcut:
just replay-profile prefill_range_l13
```

`--run` defaults to `inference.toml`. `--stage` is required and must match an exact
expanded chain name, such as `prompt_merge_b1`, `prefill_prepare_aux_l5`, or
`prefill_range_l13`. Choose an instance present in the selected workload; there
is no automatic first-instance or all-stage survey. `--top-n` is a positive
integer and defaults to three distinct tile IDs.

## Requirements

Use a full model import, as for claims, and install `raster-cli` from the same
Raster checkout referenced by this repository's dependencies. `cargo raster run
--help` must show `--profile native` and `--profile replay`. An older installed
CLI is rejected before inference begins. With the usual sibling checkout:

```sh
cargo install --path ../raster/crates/raster-cli --locked --force
```

Replay execution also needs the RISC Zero guest toolchain (`rzup install`). No
proofs are generated. Stage libraries must expose their tiles to the guest
builder and compile without default features, as the existing stages do.

## What runs

1. Prepare and freeze the run context using the same model verification, prompt
   preparation, and manifest expansion as claim building.
2. Execute native staged inference through the selected stage, then stop. Only
   the prefix is scheduled, including when the selected stage lies inside a
   parallel auxiliary batch. Keep its actual inputs and native output baseline.
3. Run that stage's Raster program with `--profile native`, using the preserved
   input JSON and input manifest. Its output hash and structural commitment
   must match the baseline before replay profiling proceeds.
4. Group individual native tile records by tile ID and rank by each tile's
   maximum `total_duration_ns`, including Raster overhead. Break ties by tile
   ID. Choose the first N IDs, or all if fewer executed. This differs from
   Raster's cumulative-time native hot list.
5. Run the same program with `--profile replay --tile …` and the same inputs.
   This performs a second authenticated native run, followed by zkVM execution
   of selected tile invocations. Check the stage output against the baseline again.

The current Raster profiler measures **the first 128 invocations per selected
tile ID**, shared across all recursive iterations and call sites. The stage
itself still runs completely. The saved invocation limit and measured/recorded
counts show coverage. Reaching that cap is successful completion, but later
invocations may be heavier than the maximum measured. Older uncapped version 1
reports retain their original coverage when read by the report loader.

## Saved results and failures

The command prints its exact report path:

```text
<native-chain-run>/replay-profile/<stage>/
  replay-profile.json         # Guest-cycle summary and frozen run context
  native-ranking.json         # Native tile-selection timings and sizes
  native-profile.json         # Unmodified Raster native report
  raster-replay-profile.json  # Unmodified Raster replay report
  native.stdout.log
  native.stderr.log
  replay.stdout.log
  replay.stderr.log
```

The summary includes the native output checkpoint, input paths and hashes,
exact commands, selected tile IDs and ranking rule, output parity results,
and replay cycle measurements ordered by maximum measured cycles. Each replay
measurement includes its heaviest measured invocation's coordinate and executable
image ID. Native per-tile timings, sizes, and slowest-invocation coordinates are
saved in `native-ranking.json`, linked by the summary's `native_ranking_path`.
The replay-cycle fields are in `replay_tiles`, placed near the top of the JSON:

- `max_guest_cycles`: the heaviest single invocation among the measured replays.
- `total_guest_cycles`: the sum across `profiled_invocations` replays (up to 128).
- `transition_overhead.guest_cycles`: the transition-verification overhead
  measured once, using the first successfully replayed invocation. Its sampled
  coordinate, executable identity, and representative-context version are retained.

The terminal shows these guest-cycle measurements and replay progress. The native
selection pass's detailed output goes only to `native.stdout.log`. Native fields
ending in `_ns`, including `total_raster_overhead_ns`, are host timings used for
tile selection; they are saved in `native-ranking.json` and `native-profile.json`,
outside the replay summary. The overhead sample and the heaviest replay may refer
to different invocations, so both coordinates are shown.

Transition overhead is separate from replay totals. When all 128 invocations
were measured, a rough manual estimate is:

```text
replay total + 128 × sampled transition overhead
```

This excludes window initialization, authorization execution, non-tile boundary
transitions, and cryptographic proving/composition. It is neither a worst-case
bound nor a measured complete fault window. Older reports or failed overhead
measurements show `not measured`, never an invented zero. New stage summaries use
version 2. Version 1 summaries remain readable, including their inline
`native_tiles` field and optional overhead. Raw Raster replay reports retain
their own format version (currently 3).

Original Raster run directories are recorded from each command's header, never
inferred by choosing the newest directory. Keep the native run and its referenced
model/run artifacts for reproduction; the summary is not a portable input bundle.

If a profiling command, report load, or parity check fails, the command exits
nonzero and keeps available raw reports, logs, and a failed summary. Partial
replay results retain successful measurements and the failing phase/tile/coordinate.
An empty native tile set succeeds with `no_tiles` and skips replay profiling.

A failed native execution may produce no replay report at all. In that case the
summary includes the recent output from both Raster logs, because the native
program's error is printed inside `replay.stdout.log`. Check that earlier error
before treating the absent JSON file as the cause. Binary traces can occupy
several GiB per run; the native and replay passes each record a trace, and old
run directories are retained. Leave space for both passes. A trace truncated by
a failed write needs a fresh run after the underlying storage error is resolved.

Use `cargo raster analyze` on either raw profile to reopen Raster's report.
The stage summary has a separate `stage-replay-profile` kind and can be read as JSON.

## Reviewing stage authoring

Use maximum measured guest cycles to investigate expensive replay units. Inspect
their input/output sizes and native body/overhead breakdown in `native-ranking.json`
to decide whether to reduce
chunk sizes, narrow selections, or split computation. Re-run the same workload
after changing the stage; adjust its model import if the authored input layout changes.

Native timing is only a candidate-selection heuristic: an unselected tile may
cost more in the zkVM. Cycle totals cover measured invocations of selected tiles,
and include guest wrapper work. They are not proving latency, a whole-stage cost,
or a correctness guarantee for other workloads. This command supplies measurements
for manual review and has no budget threshold or automatic authoring verdict.

## Opt-in smoke test

With the matching Raster CLI on `PATH` and the RISC Zero guest toolchain installed:

```sh
cargo test --release --locked -p raster-inference-cli --test replay_profile \
  replay_profile_tokenizer_smoke -- --ignored --nocapture
```

The test imports the checked-in synthetic parity model into a fresh directory
under `target`, profiles one tokenizer stage with one selected tile, and
checks saved cycles, sampled overhead, and parity. It also checks the full command
output and replay JSON for leaked native nanosecond timings. It prints its artifact
directory for inspection.
