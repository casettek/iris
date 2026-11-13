# Sourced by the test tier scripts (docs/workflows/testing.md). Sets the
# environment every tier shares and the `step` helper that runs one named
# check, keeps its log under target/test-logs, and prints one result line.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PATH="$HOME/.cargo/bin:$HOME/.risc0/bin:$PATH"
# Raster's scratch files need a real disk; /tmp is often memory-backed.
export RASTER_SCRATCH_DIR="${RASTER_SCRATCH_DIR:-$repo_root/target/raster-scratch}"
mkdir -p "$RASTER_SCRATCH_DIR"
# These would skip the proof guests or accept fake receipts.
unset RISC0_SKIP_BUILD RISC0_SKIP_BUILD_KERNELS RISC0_DEV_MODE

# IRIS_FEATURES names the `iris-cli` features every release build here uses,
# for example `cuda,gpu-kernels`. Without it a tier rebuilds target/release/iris
# as a CPU binary. GPU builds take the same compiler settings as
# scripts/build-linux-cuda.sh.
features=()
if [ -n "${IRIS_FEATURES:-}" ]; then
    features=(--features "$IRIS_FEATURES")
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    export NVCC_CCBIN="${NVCC_CCBIN:-/usr/bin/g++-13}"
    export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--include $CUDA_HOME/include/cuda.h -arch=native}"
fi

logs="$repo_root/target/test-logs"
mkdir -p "$logs"
failed=0

# step <log name> <what a pass means> <command...>
# Runs the command with its output in target/test-logs/<log name>.log and
# prints PASS or FAIL with the elapsed seconds. A failure is remembered in
# `failed` and the tier carries on, so one run reports every failure.
step() {
    local name=$1 meaning=$2
    shift 2
    local start status
    start=$(date +%s)
    "$@" > "$logs/$name.log" 2>&1 < /dev/null
    status=$?
    if [ $status -eq 0 ]; then
        printf 'PASS  %-64s %5d s\n' "$meaning" "$(( $(date +%s) - start ))"
    else
        printf 'FAIL  %-64s %5d s  (exit %d, see target/test-logs/%s.log)\n' \
            "$meaning" "$(( $(date +%s) - start ))" "$status" "$name"
        failed=1
    fi
    return 0
}

# The newest retained parity run, or the one IRIS_PARITY_RUN names.
latest_parity_run() {
    if [ -n "${IRIS_PARITY_RUN:-}" ]; then
        echo "$IRIS_PARITY_RUN"
    else
        ls -td "$repo_root"/target/parity/*/ 2>/dev/null | head -1 | sed 's:/$::'
    fi
}
