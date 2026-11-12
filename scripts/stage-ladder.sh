#!/usr/bin/env bash
# The Raster check ladder short of the lock, for each stage program, on the
# inputs a parity run kept:
#   - it compiles in both postures (with and without `std`);
#   - its control-flow schema builds and leaves no dataflow as `external`;
#   - run unauthenticated, it gives the native output byte for byte;
#   - run authenticated, it gives the same output, and its trace commits and
#     audits.
# The parity gate replays only the stages long enough for its window through
# authenticated Raster. This runs all six programs through it, with a window
# small enough for the short ones.
#
# usage: stage-ladder.sh [program ...]
# Reads the newest parity run under target/parity, or the one IRIS_PARITY_RUN
# names. Without arguments all six programs are run.
# See docs/workflows/testing.md.
source "$(dirname "${BASH_SOURCE[0]}")/test-env.sh"

run="$(latest_parity_run)"
inputs="$run/over-window/raster"
if [ ! -d "$inputs" ]; then
    echo "no parity run with stage inputs at '$inputs'; run the parity gate first" >&2
    exit 1
fi

# Stages of the parity run to execute, per program. The first is also
# committed and audited.
samples() {
    case $1 in
    prompt-bind) echo "prompt_bind_p1 prompt_bind_p0 prompt_bind_p5" ;;
    prompt-check) echo "prompt_check_p1 prompt_check_p0 prompt_check_p5" ;;
    input-embedding) echo "embed_p1 embed_p0 embed_c0" ;;
    prefill-prepare-aux) echo "aux_p1_l3 aux_p0_l0 aux_c1_l2" ;;
    prefill-range) echo "range_p1_l3 range_p0_l0 range_p3_l2 range_c1_l3" ;;
    sequence-check) echo "check_c0 check_c1" ;;
    esac
}
# Trace steps per fraud-proof window. The shortest stages here are about 110.
window="${IRIS_LADDER_WINDOW:-32}"

# In a subshell: it changes directory.
ladder() (
    local program=$1 out="$repo_root/target/stage-ladder/$1" stage first dir produced
    cd "$repo_root/raster-stages/$program" || return 1
    cargo check || return 1
    cargo check -p "$program" --no-default-features || return 1
    cargo raster cfs || return 1
    if grep -Eq '"type": ?"external"' target/raster/cfs.json; then
        echo "$program: the schema leaves dataflow as external"
        return 1
    fi
    first=""
    for stage in $(samples "$program"); do
        [ -z "$first" ] && first=$stage
        dir="$out/$stage"
        mkdir -p "$dir"
        find "$dir" -name output.bin -delete
        RASTER_AUTH=0 RASTER_OUTPUT_DIR="$dir/plain" cargo run --release -- \
            --input "$inputs/$stage/input.json" \
            --input-manifest "$inputs/$stage/input_manifest.json" || return 1
        produced=$(find "$dir/plain" -name output.bin | head -1)
        if [ -z "$produced" ] || ! cmp -s "$produced" "$inputs/$stage/output.bin"; then
            echo "$stage: the unauthenticated output differs from the native one"
            return 1
        fi
        echo "$stage: unauthenticated output matches"
    done
    dir="$out/$first"
    RASTER_OUTPUT_DIR="$dir/commit" cargo raster run \
        --input "$inputs/$first/input.json" \
        --input-manifest "$inputs/$first/input_manifest.json" \
        --commit "$dir/trace.commit" --fraud-proof-window-size "$window" \
        > "$dir/commit.log" 2>&1 || { cat "$dir/commit.log"; return 1; }
    produced="$(sed -n 's/^ *Run artifacts dir: //p' "$dir/commit.log" | head -1)/output.bin"
    if ! cmp -s "$produced" "$inputs/$first/output.bin"; then
        echo "$first: the authenticated output ($produced) differs from the native one"
        return 1
    fi
    echo "$first: authenticated output matches"
    RASTER_OUTPUT_DIR="$dir/audit" cargo raster run \
        --input "$inputs/$first/input.json" \
        --input-manifest "$inputs/$first/input_manifest.json" \
        --audit "$dir/trace.commit" > "$dir/audit.log" 2>&1 || { cat "$dir/audit.log"; return 1; }
    grep -q "Verification Success" "$dir/audit.log" || { cat "$dir/audit.log"; return 1; }
    echo "$first: trace commits and audits"
)

programs=("$@")
[ ${#programs[@]} -eq 0 ] && programs=(prompt-bind prompt-check input-embedding prefill-prepare-aux prefill-range sequence-check)
echo "== stage ladder on $run"
for program in "${programs[@]}"; do
    step "ladder-$program" "$program: runs, commits and audits as Raster" ladder "$program"
done

exit $failed
