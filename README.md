# iris

`iris` provides four workflows over the same imported Gemma model inputs:

- `infer`: fast deterministic inference from a run spec and `runtime/model-artifacts/<model-id>/manifest.json`.
- `claim build`: a checkpointed deterministic rerun that emits a compact claim.
- `challenge build`: verifier rerun from a claim trace, divergence detection, and Raster replay of the first divergent stage.
- `replay-profile`: manual native and zkVM replay measurements for one exact stage instance.

`model import` is the shared setup step. It turns a model bundle into committed Raster externals and a prompt-free model manifest. Each workflow selects the model through `--run inference.toml` and verifies it once before execution. Challenges retain the claim’s frozen prompt and token count.

## Quick Start

```bash
# Import committed externals and the direct-infer manifest.
cargo run --release -p iris-cli -- model import \
  --model ../iris/assets/tiny-gemma-dev

# Fast deterministic inference, no checkpoint tree.
cargo run --release -p iris-cli -- infer --run inference.toml

# Proposer claim with every staged checkpoint.
cargo run --release -p iris-cli -- claim build --run inference.toml

# Manual replay-cost measurements for one stage (needs Raster's
# risc0-estimator branch; see docs/workflows/replay-profile.md).
cargo run --release -p iris-cli --features replay-profile -- replay-profile \
  --run inference.toml --stage prefill_range_l13 --top-n 3

# Verifier challenge from a prior claim bundle.
cargo run --release -p iris-cli -- challenge build \
  --run inference.toml \
  --claim-context target/staged-infer/runs/.../claim_bundle.json \
  --checkpoints target/staged-infer/runs/.../checkpoints.txt
```

The same commands are available through `just` recipes:

```bash
just import ../iris/assets/tiny-gemma-dev
just infer inference.toml
just claim inference.toml
just replay-profile prefill_range_l13
just challenge \
  target/staged-infer/runs/.../claim_bundle.json \
  target/staged-infer/runs/.../checkpoints.txt
```

## Workflows

- [Native Windows inference](docs/workflows/windows.md): PowerShell build, compatibility patches, and Windows model paths.
- [Three-path parity](docs/workflows/test-parity.md): run `just test-parity` to compare Raster-staged, native-staged, and native-direct inference on a pinned model and corpus.
- [Model import](docs/workflows/model-import.md): bundle inputs, generated externals, model-specific templates, and model manifests.
- [Infer](docs/workflows/infer.md): fast deterministic inference through `direct-infer`.
- [Claim build](docs/workflows/claim-build.md): checkpointed staged inference and claim artifacts.
- [Challenge build](docs/workflows/challenge-build.md): trace comparison, divergence packaging, and Raster replay.
- [Replay profiling](docs/workflows/replay-profile.md): select heavy native tiles and measure their replay costs.

## Internals

- [Raster chain](docs/internals/raster-chain.md): stage expansion, boundary types, and chain-program layout.
- [Artifacts](docs/internals/artifacts.md): stable JSON contracts and per-stage checkpoint files.
- [Development](docs/internals/development.md): workspace commands, stage checks, and lower-level debug surfaces.
- [Issues](docs/issues/README.md): known program gaps and upstream Raster handoffs.
- [Proposals](docs/proposals/): design notes and historical decisions.

## Layout

```text
Cargo.toml                  # workspace manifest
justfile                    # workflow and test shortcuts
Raster.toml                 # optional template copy for manual Raster commands
inference.toml              # default run spec
crates/
  model/
    import/                 # model bundle -> committed inputs/manifests
    run-prep/               # prompt/run preparation helpers
    detwgt/                 # DETWGT eager and mmap readers
  inference/
    cli/                    # developer-facing workflow CLI
    direct/                 # fast deterministic runtime
    staged/                 # checkpointed chain executor and parity tooling
    artifacts/              # shared JSON contracts and artifact I/O
    kernels/                # shared deterministic inference kernels
  det-num/                  # deterministic numeric primitives
raster-stages/              # verifiable Raster program crates
model-bundles/              # checked-in source model bundles for local gates
runtime/                    # imported model artifacts, prompt fixtures, generated manifest examples
manifests/                  # alternate generated manifests and examples
docs/                       # workflow, internals, issues, proposals
```

`raster-stages/*` are intentionally left outside `crates/`: each directory is a Raster program boundary with its own `Cargo.toml`, `Raster.lock`, no-std tile library, and sequence entry point. Reusable imported model externals live under `runtime/model-artifacts/<model-id>/raster/`, prompt fixtures live under `runtime/prompt-fixtures/`, and generated chain manifests point each stage at the files it needs there. Each full import saves its chain template under `runtime/model-artifacts/<model-id>/Raster.toml`. The root `Raster.toml` can be refreshed with `--manifest Raster.toml` for manual Raster commands; run-spec workflows use the model-specific template.
