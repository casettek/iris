# Development

The repo is a Cargo workspace for host/support crates plus Raster stage crates. Use the root commands for ordinary development and per-stage Raster commands when changing verifiable programs.

## Workspace Commands

```bash
cargo check --workspace
cargo test --workspace
cargo test -p direct-infer
cargo test -p staged-infer
cargo test -p inference-artifacts
cargo test -p iris-cli
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

## GPU Kernels

The `gpu-kernels` feature moves the layer's `Q8_0` matrix products to an NVIDIA
GPU with results identical to the CPU's (see
[numerics](numerics.md#q8_0-products-on-a-gpu)). It is separate from `cuda`,
which accelerates RISC Zero proving. Building it needs the CUDA toolkit; the
runtime is linked statically, so running needs only the NVIDIA driver.

```bash
export CUDA_HOME=/usr/local/cuda-13.3 NVCC_CCBIN=/usr/bin/g++-13
cargo build --release -p iris-cli --bin iris --features gpu-kernels
cargo test -p inference-kernels --features gpu-kernels
```

`CUDA_HOME` defaults to `/usr/local/cuda` and `NVCC_CCBIN` to `g++`. Both builds
write `target/release/iris`; a build without the feature restores the CPU-only
binary.

| Variable | Effect |
| --- | --- |
| `IRIS_GPU_KERNELS=off` | Keep every product on the CPU. |
| `IRIS_GPU_MIN_WORK` | Smallest product, in multiplications, sent to the GPU. Default `2^28`; `0` sends every `Q8_0` layer product, which is how tests and `test-parity` exercise the GPU on small models. |
| `IRIS_GPU_WEIGHT_BYTES` | Weights kept on the device before the least recently used matrix is dropped. Default 4 GiB. |

The first product on the device prints `iris: running Q8_0 layer products on the
GPU` to stderr. To check a change to the GPU path, compare `checkpoints.txt` of
the same claim built with and without the feature, then run the gate with the
GPU forced on:

```bash
IRIS_GPU_MIN_WORK=0 cargo run --release --locked -p iris-cli --features gpu-kernels -- test-parity --skip-challenges
```

## Lower-Level Surfaces

The workflow CLI is the primary API. These lower-level commands remain useful for debugging:

```bash
cargo run -p staged-infer -- chain run
cargo raster chain run --no-auth --show-output
cargo raster show target/raster/chains-no-auth/latest/check_c0/output.bin
```

`target/`, `*.rastered`, and `*.rindex` are generated artifacts. A chain manifest comes from `model import` (the template) and `claim build` (filled in for one run, under `target/iris/runs/<run>/`); run the commands above from that directory or pass it as the manifest.
