#!/usr/bin/env bash
# Real-model check. The parity gate runs a four-layer synthetic model; this
# runs an imported Gemma model and answers two questions the gate cannot:
#   - did a change alter any honest result? A claim is built natively and every
#     checkpoint is compared with the reference kept in the repository;
#   - do the Raster stage programs still agree with native at real sizes? One
#     stage of each program is run unauthenticated on that claim's inputs and
#     its output compared with the native one, byte for byte.
#
# usage: test-real-model.sh [--update] [reference dir ...]
#   --update  write the run's checkpoints as the new reference. Only after a
#             change that is meant to alter stage outputs.
# A reference directory holds `spec.toml` (the run) and `checkpoints.txt`.
# Without arguments every directory under tests/real-model is run; one whose
# model is not imported here is reported as skipped.
# See docs/workflows/testing.md.
source "$(dirname "${BASH_SOURCE[0]}")/test-env.sh"

update=0
references=()
for arg in "$@"; do
    case $arg in
    --update) update=1 ;;
    *) references+=("$arg") ;;
    esac
done
if [ ${#references[@]} -eq 0 ]; then
    for dir in "$repo_root"/tests/real-model/*/; do
        references+=("${dir%/}")
    done
fi

# The stage of each program to run through Raster: the first prompt range,
# which is a full range, its first layer, and the first claimed range's check.
raster_samples="prompt-bind:prompt_bind_p0 prompt-check:prompt_check_p0 input-embedding:embed_p0
    prefill-prepare-aux:aux_p0_l0 prefill-range:range_p0_l0 sequence-check:check_c0"

claim() {
    local reference=$1 out=$2
    "$repo_root/target/release/iris" claim build --run "$reference/spec.toml" > "$out/claim.log" 2>&1 \
        || { tail -20 "$out/claim.log"; return 1; }
    grep -E "^faults:|^stop reason:" "$out/claim.log"
    grep -q "^faults: prompt 0, check 0, output 0, model 0; claim mismatches: 0" "$out/claim.log" \
        || { echo "the claim recorded a fault"; return 1; }
}

compare() {
    local reference=$1 out=$2 trace checkpoints line
    trace=$(sed -n 's/^checkpoint trace: //p' "$out/claim.log" | head -1)
    checkpoints="$(dirname "$trace")/checkpoints.txt"
    cp "$checkpoints" "$out/checkpoints.txt"
    if [ $update -eq 1 ]; then
        cp "$checkpoints" "$reference/checkpoints.txt"
        echo "reference updated: $(wc -l < "$checkpoints") checkpoints"
        return 0
    fi
    if cmp -s "$checkpoints" "$reference/checkpoints.txt"; then
        echo "all $(wc -l < "$checkpoints") checkpoints match the reference"
        return 0
    fi
    line=$(cmp "$checkpoints" "$reference/checkpoints.txt" | sed -n 's/.* line \([0-9]*\).*/\1/p')
    echo "checkpoints differ from the reference: $(wc -l < "$checkpoints") now," \
        "$(wc -l < "$reference/checkpoints.txt") in the reference, first difference at line ${line:-?}"
    python3 - "$trace" "${line:-0}" <<'EOF'
import json, sys
trace = json.load(open(sys.argv[1]))
checkpoints = trace["checkpoints"] if isinstance(trace, dict) else trace
index = int(sys.argv[2]) - 1
if 0 <= index < len(checkpoints):
    print("first differing stage:", checkpoints[index]["stage"])
EOF
    return 1
}

raster_stage() {
    local program=$1 stage=$2 out=$3 chain produced
    chain=$(sed -n 's/^ *dir: //p' "$out/claim.log" | head -1)
    [ -f "$chain/$stage/input.json" ] || { echo "$chain has no stage $stage"; return 1; }
    mkdir -p "$out/raster/$stage"
    find "$out/raster/$stage" -name output.bin -delete
    (cd "$repo_root/raster-stages/$program" &&
        RASTER_AUTH=0 RASTER_OUTPUT_DIR="$out/raster/$stage" cargo run --release -- \
            --input "$chain/$stage/input.json" \
            --input-manifest "$chain/$stage/input_manifest.json") || return 1
    produced=$(find "$out/raster/$stage" -name output.bin | head -1)
    if [ -z "$produced" ] || ! cmp -s "$produced" "$chain/$stage/output.bin"; then
        echo "$stage: the Raster output differs from the native one"
        return 1
    fi
    echo "$stage: Raster output matches native ($(stat -c %s "$produced") bytes)"
}

step real-model-build "the iris binary builds" \
    cargo build --release --locked -p iris-cli --bin iris "${features[@]}"

for reference in "${references[@]}"; do
    name=$(basename "$reference")
    manifest=$(sed -n 's/^model_manifest *= *"\(.*\)"/\1/p' "$reference/spec.toml")
    case $manifest in /*) ;; *) manifest="$reference/$manifest" ;; esac
    echo "== real model: $name"
    if [ ! -f "$manifest" ]; then
        echo "SKIP  $name: no import at $manifest"
        continue
    fi
    out="$repo_root/target/real-model/$name"
    mkdir -p "$out"
    step "real-model-$name-claim" "$name: an honest claim builds with no fault" claim "$reference" "$out"
    [ -f "$out/claim.log" ] && grep -q "^checkpoint trace: " "$out/claim.log" || continue
    meaning="$name: every checkpoint matches the reference"
    [ $update -eq 1 ] && meaning="$name: reference checkpoints rewritten from this run"
    step "real-model-$name-checkpoints" "$meaning" compare "$reference" "$out"
    for sample in $raster_samples; do
        step "real-model-$name-${sample#*:}" "$name: ${sample#*:} as Raster gives the native output" \
            raster_stage "${sample%%:*}" "${sample#*:}" "$out"
    done
done

exit $failed
