# Fault proving

The app automatically follows each completed challenge with a synthetic fault
proof. Raster is unchanged. Iris orchestrates its existing native CLI and public
prover APIs; `src/fault/proof.rs` and `program.rs` adapt the witness/identity
orchestration currently private to Raster's CLI.

Build Iris with the RISC0 proving toolchain available and **without**
`RISC0_SKIP_BUILD`. Native-only binaries built with that variable cannot prove.
`iris fault check` rejects missing embedded guests and development mode before
the app starts inference. RISC0 3.0's guest builder rejects native Windows
(`UnsupportedPlatform("unknown OS: windows")`). Use WSL/Linux for the complete
proving workflow. `scripts/build-windows.ps1` builds native inference only and
rejects `-Proving` immediately.

From a WSL shell with the RISC0 toolchain installed:

```sh
cd /mnt/c/Users/amcas/Development/iris
unset RISC0_SKIP_BUILD RISC0_DEV_MODE
cargo build --release --locked -p iris-cli --bin iris
./target/release/iris fault check
```

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

The prover reuses the original claimer's predecessor outputs, replays only the
challenged stage, checks checkpoint parity, and audits the recorded trace. It
then assembles witnesses and calls Raster's tile replay, authorization, and
transition proof APIs. Receipt verification checks the guest identity, Finished
verdict, and refuted commitment hash. An honest audit reports `no-fault` and exits
unsuccessfully rather than claiming to have generated a proof.

`IRIS_FAULT_PROGRESS` version 1 records include `phase`, `started_at_ms`,
`stage_replay_duration_ms`, and `proof_duration_ms`. Durations are null until
complete. Phases are `preparing`, `stage-replay`, `proving`, and `validating`.

- Stage replay measures authenticated native stage execution, trace loading,
  and fault localization.
- Proof computation measures witness construction, cryptographic tile replays,
  authorization/transition proving, receipt verification, and receipt writing.
- Preparation and total wall time are also saved. Compilation belongs to the
  phase that incurs it; the two primary durations never overlap.

Successful attempts publish `fault.receipt`, `fault_bundle.json`, and
`fault_timings.json`. Failed attempts retain diagnostic files without publishing
a successful bundle. Use a fresh output directory for a retry.

The backend uses `IRIS_FAULT_TIMEOUT_MS` (default four hours, maximum one day),
persists separate phase timings, and retains completed claim/challenge data if
proving fails. Worker recovery marks interrupted attempts failed; it does not
silently repeat proving. No blockchain transaction is submitted.

Development checks use small synthetic fingerprints and subprocess fixtures.
Gemma small/medium inference and actual proving are manual acceptance checks.
