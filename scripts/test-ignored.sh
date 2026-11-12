#!/usr/bin/env bash
# The `#[ignore]`d tests, which a plain `cargo test` never runs because they
# need built stage binaries, the real model's tokenizer, a kept parity run, the
# RISC Zero guest toolchain or Docker. Left unrun they go stale without anyone
# noticing, so the release tier runs them.
#
# usage: test-ignored.sh [--skip-proof]
#   --skip-proof  leave out the real fault proof (needs Docker; about 4 minutes
#                 with a GPU build of the Raster CLI, longer without)
# Needs: a parity run under target/parity (or IRIS_PARITY_RUN), and the E2B
# model beside this repository (`../models/gemma-4-E2B-q8`) with its import.
# See docs/workflows/testing.md.
source "$(dirname "${BASH_SOURCE[0]}")/test-env.sh"

skip_proof=0
for arg in "$@"; do
    case $arg in
    --skip-proof) skip_proof=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

parity_run="$(latest_parity_run)"
if [ ! -d "$parity_run/over-window/challenges" ]; then
    echo "no parity run with a challenge leg at '$parity_run'; run the parity gate first" >&2
    exit 1
fi
tokenizer="${BPE_REFERENCE_TOKENIZER:-$repo_root/../models/gemma-4-E2B-q8/tokenizer.json}"
cli_features="${IRIS_FEATURES:+$IRIS_FEATURES,}replay-profile"

echo "== ignored tests (parity run: $parity_run)"
step ignored-merge-exhaustive "token check: only the merge's own grouping passes" \
    cargo test --release -p inference-kernels -- --ignored exhaust

step ignored-stage-binaries "stage programs build as host binaries" \
    cargo build -p input-embedding -p prefill-prepare-aux -p prefill-range \
    -p prompt-bind -p prompt-check -p sequence-check -p staged-infer --bins

# Benchmarks print timings and assert nothing; the helper is spawned by
# another test; the production vocabulary test runs in release below.
STAGED_INFER_REPLAY_BIN_DIR="$repo_root/target/debug" BPE_REFERENCE_TOKENIZER="$tokenizer" \
    step ignored-staged "native loaders match authenticated Raster stage binaries" \
    cargo test -p staged-infer --no-fail-fast -- --ignored --test-threads=1 \
    --skip benchmark --skip challenge_cleanup_child \
    --skip claimed_tokens_on_the_production_vocabulary

BPE_REFERENCE_TOKENIZER="$tokenizer" \
    step ignored-production-vocabulary "claimed tokens check out on the real vocabulary" \
    cargo test --release -p staged-infer --test prompt_check -- --ignored

step ignored-direct "direct inference on the real model gives the known token" \
    cargo test --release -p direct-infer --test executor -- --ignored

IRIS_PARITY_RUN="$parity_run" \
    step ignored-parity-attacks "kept attacks replay the same natively and through Raster" \
    cargo test --release -p iris-cli --features "$cli_features" --lib --no-fail-fast -- \
    --ignored --test-threads=1 parity::challenge

step ignored-guests "tiles run in the zkVM guest as they do natively" \
    cargo test --release -p iris-cli --features "$cli_features" --test model_identity_guest \
    --no-fail-fast -- --ignored --test-threads=1

# This file is compiled only with `replay-profile`, so its plain test never
# runs in the workspace tests either: run both here.
step ignored-replay-profile "a stage profiles through native and guest replay" \
    cargo test --release -p iris-cli --features "$cli_features" --test replay_profile \
    --no-fail-fast -- --include-ignored --test-threads=1

if [ $skip_proof -eq 1 ]; then
    echo "SKIP  a fault proof is produced and verifies (--skip-proof)"
else
    step ignored-fault-proof "a fault proof is produced and verifies" \
        cargo test --release -p iris-cli --features "$cli_features" --test challenge_profile \
        -- --ignored
fi

exit $failed
