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
    cargo run --release -p iris-cli -- replay-profile --run "{{run}}" --stage "{{stage}}" --top-n "{{top_n}}"

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

# Full local host parity gate; all logs and outputs are kept under target/parity.
test-parity:
    cargo run --release --locked -p iris-cli -- test-parity
