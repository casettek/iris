set positional-arguments

default:
    just --list

import model:
    cargo run --release -p iris-cli -- model import --model "{{model}}"

infer run="inference.toml":
    cargo run --release -p iris-cli -- infer --run "{{run}}"

claim run="inference.toml":
    cargo run --release -p iris-cli -- claim build --run "{{run}}"

replay-profile stage top_n="3" run="inference.toml":
    cargo run --release -p iris-cli --features replay-profile -- replay-profile --run "{{run}}" --stage "{{stage}}" --top-n "{{top_n}}"

corrupt checkpoints:
    cargo run --release -p iris-cli -- claim corrupt --checkpoints "{{checkpoints}}"

challenge claim_context checkpoints run="inference.toml":
    cargo run --release -p iris-cli -- challenge build --claim-context "{{claim_context}}" --checkpoints "{{checkpoints}}" --run "{{run}}"

challenge-claim claim run="inference.toml":
    cargo run --release -p iris-cli -- challenge build --claim "{{claim}}" --run "{{run}}"

test-workspace:
    cargo test --workspace

test-direct:
    cargo test -p direct-infer

test-staged:
    cargo test -p staged-infer

test-cli:
    cargo test -p iris-cli

test-artifacts:
    cargo test -p inference-artifacts

stage-check stage:
    cargo check --manifest-path "raster-stages/{{stage}}/Cargo.toml" --no-default-features

# The three test tiers; docs/workflows/testing.md says what each one shows.
# Fast (minutes): unit tests, the stage schema check, and every Raster.lock against its sources.
test-fast:
    bash scripts/test-fast.sh

# Full (about half an hour): fast, the parity gate, and all six stage programs through authenticated Raster.
test-full:
    bash scripts/test-full.sh

# Release (about an hour): full, the ignored tests (guests, a real fault proof) and the real-model check.
test-release *args:
    bash scripts/test-release.sh {{args}}

# The Raster check ladder for stage programs, on the newest parity run's inputs: [program ...]
stage-ladder *args:
    bash scripts/stage-ladder.sh {{args}}

# The ignored tests on their own: [--skip-proof]
test-ignored *args:
    bash scripts/test-ignored.sh {{args}}

# A real model's checkpoints against the kept reference, and its stages through Raster: [--update] [reference dir ...]
test-real-model *args:
    bash scripts/test-real-model.sh {{args}}

# Full local host parity gate; all logs and outputs are kept under target/parity.
test-parity:
    cargo run --release --locked -p iris-cli -- test-parity

# Refresh proof guest identities after Raster, guest, dependency or toolchain changes.
build-raster-stages:
    bash scripts/build-raster-stages.sh
