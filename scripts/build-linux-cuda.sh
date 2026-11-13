#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
export PATH="$CUDA_HOME/bin:$HOME/.cargo/bin:$HOME/.risc0/bin:$PATH"
export NVCC_CCBIN="${NVCC_CCBIN:-/usr/bin/g++-13}"
# RISC0 has a local cuda.h; preload NVIDIA driver declarations for CUDA 13 CCCL.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--include $CUDA_HOME/include/cuda.h -arch=native}"
unset RISC0_SKIP_BUILD RISC0_SKIP_BUILD_KERNELS RISC0_DEV_MODE
cd "$repo_root"
# cuda accelerates proving; gpu-kernels runs native inference layer products on the GPU.
cargo build --release --locked -p iris-cli --bin iris --features cuda,gpu-kernels -j "${JOBS:-4}"
CARGO_TARGET_DIR="$repo_root/target" cargo install --path "$repo_root/../raster/crates/raster-cli" --locked --force --features cuda -j "${JOBS:-4}"
# Installing Raster may change guest identities without changing stage sources.
# Refresh locks only here, in the explicit build workflow, never inside proving.
bash "$repo_root/scripts/build-raster-stages.sh"
./target/release/iris fault check
