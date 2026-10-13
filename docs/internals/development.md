# Development

The repo is a Cargo workspace for host/support crates plus Raster stage crates. Use the root commands for ordinary development and per-stage Raster commands when changing verifiable programs.

## Workspace Commands

```bash
cargo check --workspace
cargo test --workspace
cargo test -p direct-infer
cargo test -p staged-infer
cargo test -p inference-artifacts
cargo test -p raster-inference-cli
```

Equivalent shortcuts:

```bash
just test-workspace
just test-direct
just test-staged
just test-artifacts
just test-cli
just test-parity
```

`just test-parity` is the complete release-mode three-path regression gate. It
retains logs and reports under `target/parity/`; see the
[parity workflow](../workflows/test-parity.md) for prerequisites and exact checks.

## Raster Stage Checks

Any change to a Raster tile, sequence, stage input type, fixture, or `Raster.toml` needs the Raster check ladder appropriate to that change. At minimum, keep no-std posture intact:

```bash
cargo check --manifest-path raster-stages/<stage>/Cargo.toml --no-default-features
just stage-check <stage>
```

When a stage's verifiable behavior changes, rebuild and verify the Raster program identity deliberately:

```bash
cd raster-stages/<stage>
cargo raster cfs
cargo raster build --backend risc0
cargo raster program --verify
```

Do not hand-edit `Raster.lock`. Commit lock updates only when they are an expected consequence of a stage behavior or interface change.

## Lower-Level Surfaces

The workflow CLI is the primary API. These lower-level commands remain useful for debugging:

```bash
cargo run -p staged-infer -- chain run
cargo raster chain run --no-auth --show-output
cargo raster show target/raster/chains-no-auth/latest/output_finalize/output.bin
```

`target/`, `*.rastered`, and `*.rindex` are generated artifacts. The root `Raster.toml` is the canonical chain manifest; alternate generated manifests live under `manifests/`.
