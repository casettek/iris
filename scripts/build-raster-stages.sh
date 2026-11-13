#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset RISC0_SKIP_BUILD RISC0_SKIP_BUILD_KERNELS RISC0_DEV_MODE
# Refresh every stage, including dependencies and toolchain changes.
for manifest in "$repo_root"/raster-stages/*/Cargo.toml; do
  stage_dir="$(dirname "$manifest")"
  echo "Building and verifying Raster identity: $stage_dir"
  (
    cd "$stage_dir"
    cargo raster build --backend risc0
    cargo raster program --verify
  )
done
