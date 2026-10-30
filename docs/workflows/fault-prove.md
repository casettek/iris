# Fault proving

The app automatically follows each completed challenge with a synthetic fault
proof. Iris already knows the disputed stage and invokes its program directly
with `cargo raster run --audit`. Raster compares live events against the supplied
commitment, stops at the first detected divergence, and generates a verified
fault proof. No additional commitment or raw trace is written.

Build Iris with the RISC0 proving toolchain available and **without**
`RISC0_SKIP_BUILD`. Native-only binaries built with that variable cannot prove.
`iris fault check` rejects missing embedded guests and development mode before
the app starts inference. RISC0 3.0's guest builder rejects native Windows
(`UnsupportedPlatform("unknown OS: windows")`). Use WSL/Linux for the complete
proving workflow. `scripts/build-windows.ps1` builds native inference only and
rejects `-Proving` immediately.

From a WSL shell with the RISC0 toolchain installed:

```sh
cd /home/amcas/Development/iris
unset RISC0_SKIP_BUILD RISC0_DEV_MODE
cargo build --release --locked -p iris-cli --bin iris
./target/release/iris fault check
```

For NVIDIA GPU proving in the Linux checkout, use:

```sh
./scripts/build-linux-cuda.sh
```

The script builds Iris with `--features cuda` and installs the sibling Raster
CLI with its existing `cuda` feature. Both must be rebuilt for this change. This machine
uses CUDA 13.3 and `g++-13`; the build also needs `libclang-dev` and
`protobuf-compiler`. Install the WSL CUDA compiler/development packages without
Linux display drivers. `CUDA_HOME`, `NVCC_CCBIN`, and `JOBS` can override the
script defaults. CUDA kernels target the GPU visible during the build.

CUDA accelerates RISC0 proof generation; native stage replay still uses the CPU.
Rebuilding Iris without `--features cuda` returns it to CPU proving. The backend
and frontend do not need rebuilding to use the updated Iris executable.

The backend worker must also run under Linux Node with Linux paths for
`IRIS_BINARY_PATH`, `IRIS_MODEL_MANIFEST`, and `FEED_STORAGE_DIR`. The current
Windows worker cannot directly launch a Linux executable. Keep Linux Node
dependencies and execution storage separate from Windows. Use a Linux-imported
model manifest and freshly prepared runs; Windows claim artifacts contain
Windows paths and cannot simply be reused by the Linux worker. Raster's Linux
CLI (`cargo raster`) must be installed in the worker's PATH as well.

```sh
iris challenge corrupt --challenge /path/to/challenge_bundle.json --output /path/to/fault-attempt
iris fault prove --claim-context /path/to/claim_bundle.json \
  --challenge /path/to/challenge_bundle.json \
  --commitment /path/to/fault-attempt/corrupted_commit.bin \
  --output /path/to/fault-attempt
```

Corruption XORs one bit in the middle interior fingerprint entry, excluding
revealed initial/final windows. It writes `corruption.json` and a separate
`corrupted_commit.bin`; it never edits the original challenge. Short commitments
with no interior entry are rejected. This corrupts the challenge fingerprint,
not model weights or inference outputs.

The prover reuses the selected stage's frozen input and manifest from the
challenge replay package. It verifies the frozen context and locked program
identity before launching the program. It does not run the chain or regenerate
stage inputs. Native execution stops at divergence, so there is no final replay
checkpoint check. Earlier claim/challenge checkpoint checks remain in place.
Receipt verification checks the guest identity, Finished verdict, and the
refuted commitment **header digest**. File SHA-256 is a separate artifact check.
An honest Raster audit reports `no-fault`; Iris rejects that result as an
unsuccessful fault-proof attempt rather than claiming a proof exists.

`IRIS_FAULT_PROGRESS` version 1 records include `phase`, `started_at_ms`,
`stage_replay_duration_ms`, and `proof_duration_ms`. Durations are null until
complete. Phases are `preparing`, `stage-replay`, `proving`, and `validating`.

- Stage replay measures native execution and live evidence collection through divergence.
- Proof computation measures cryptographic tile replays,
  authorization/transition proving, receipt verification, and receipt writing.
- Preparation and total wall time are also saved. Compilation belongs to the
  phase that incurs it; the two primary durations never overlap.

Challenge execution saves its honest commitment without a raw trace. Fault
execution uses a separately corrupted copy and never generates another
commitment. Its working state stays in RAM; proving can use normal scratch files.
The internal pending proof-evidence handoff is cleaned up by the Raster CLI.

Successful attempts publish `fault.receipt`, `fault_bundle.json`, and
`fault_timings.json`. Failed attempts retain diagnostic files without publishing
a successful bundle. Use a fresh output directory for a retry.

The backend uses `IRIS_FAULT_TIMEOUT_MS` (default four hours, maximum one day),
persists separate phase timings, and retains completed claim/challenge data if
proving fails. Worker recovery marks interrupted attempts failed; it does not
silently repeat proving. No blockchain transaction is submitted.

Run `pnpm validate:fault` in iris-backend for the real small profile ("Say ciao.",
one token, `prefill-finalize`) through receipt verification, without a database
or worker queue. `pnpm validate:challenge` continues to stop before proving.

Successful attempts include `raster_audit.json`, whose zero-based `detected_step`
and half-open `window_start`/`window_end` identify the proof window. Its
`execution_artifacts` field locates the native run directory. That directory has
no raw trace or new commitment. Failures can be retried in a fresh output
directory with the same disputed commitment and frozen inputs; no saved trace
or `--replay-stage` is needed. Historical attempts remain available as records.

## Historical saved-trace storage and recovery

The notes below describe the older recorded-execution path and saved-trace
utilities. They do not describe current live commitment or live audit runs.

Trace history, witnesses, outputs, and fingerprint blocks use private scratch
files. The backend sets `RASTER_SCRATCH_DIR`, `TMPDIR`, and `SQLITE_TMPDIR` to
`iris-backend/runtime/raster-scratch` by default, resolved from the backend
installation. Set `RASTER_SCRATCH_DIR` to override it. Standalone Raster commands
use `target/raster-scratch` in their working directory unless that variable is
set. Linux rejects RAM-backed tmpfs and checks available disk space at startup
and during writes, stopping with 4 GiB remaining. Scratch sessions clean up on
normal exit; a killed process may leave its session directory behind.

Commitment-only processing (`chain run`, ordinary `run --commit`, and
`trace commit`) decodes one event, derives its step using the existing recorder,
updates the commitment accumulator, and releases the event and step. It retains
only the revealed head/tail windows and one unfinished fingerprint batch in RAM;
completed fingerprint blocks are spooled for the existing Postcard file format.
It does not build step history, proof witnesses, or historical recursion snapshots.
The storage state needed to compute unchanged step commitments still uses the
shared disk-backed object/index store. Audit/proving retains its evidence path.

Witness history stores references instead of serializing each witness again.
Repeated payload bytes share disk chunks, including across different witness
encodings. This is private storage only; commitment encoding and hashing are
unchanged. The recorded trace remains in its existing format. Retained history
no longer grows in RAM with trace length; available disk, individual
program/prover working memory, and existing format/tree limits still apply.

If execution finished but commitment generation failed, reuse its saved trace:

```sh
export RASTER_SCRATCH_DIR=/home/amcas/Development/iris-backend/runtime/raster-scratch
cargo raster trace commit --project /path/to/stage-project \
  --trace /path/to/stage/trace.bin \
  --input /path/to/stage/input.json \
  --input-manifest /path/to/stage/input_manifest.json \
  --fraud-proof-window-size 2 --output /path/to/recovered-commit.bin
```

Use the original run's window size. This command does not execute the stage or
change backend job status. It prints trace-processing progress and scratch bytes.
A recovered commitment alone is not a complete IRIS challenge bundle.

### Compatibility checked for this implementation

The existing `hello-tiles` execution produced identical commitment bytes before
and after changing both the runtime and CLI (SHA-256
`86e13c978265a24320545c6a3a7afaf80c9081c2bc19477282ccf211416aa56e`).
The new recorded-trace audit reported `no-fault` against that baseline. Existing
small inference, corruption, storage, trace, and guest-verification tests passed.
Baseline files and check logs are in
`/home/amcas/Development/raster/target/large-trace-baseline/` and the parent
`target/large-trace-*.log` files.

The transition guest image ID changed because its empty storage-index root
uses the shared coordinate-index implementation whose storage access was
refactored. No guest verification source or commitment encoding changed; all
160 existing transition guest verification tests passed. Authorization and
chain-fraud guest IDs stayed unchanged. Iris and Raster were rebuilt together
with the existing CUDA script. Previously generated transition receipts still
require their original verifier/image ID; this change does not migrate them.

### Saved Gemma4 trace validation (2026-10-27)

The streaming commitment path successfully processed the existing 376,465,015,113-byte
trace (1,315,307 steps) without rerunning inference. Peak RSS was 367,048 KiB;
sampled scratch usage reached approximately 1.5 GB. The scratch contains storage
state required to derive steps, rather than proof-witness history. The small
baseline commitment remained byte-identical, its honest audit returned `no-fault`,
and this change preserved all three currently installed guest image IDs.

The recovered commitment is
`/home/amcas/Development/raster/target/stream-commit-recovery/commit.bin`.
The same directory contains `processing.log` and `validation.json`. This is a
commitment-only check; it does not publish an IRIS challenge bundle or validate
Gemma4 fault proving.
