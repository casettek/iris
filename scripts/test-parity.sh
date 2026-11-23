#!/usr/bin/env bash
# The parity gate alone, about 25 minutes, built with the features
# IRIS_FEATURES names so that target/release/iris stays the build in use.
#
# usage: test-parity.sh [gate arguments, such as --bundle parity-mistral]
# With IRIS_TEST_CELESTIA=1 the gate posts to the local Celestia node.
# See docs/workflows/testing.md and docs/workflows/test-parity.md.
source "$(dirname "${BASH_SOURCE[0]}")/test-env.sh"

echo "== parity gate"
step parity-gate "parity gate: honest runs agree, cheating is caught" \
    cargo run --release --locked -p iris-cli "${features[@]}" -- test-parity "$@"
# The gate's own summary: what it covered, by outcome.
sed -n '/^PASS: [0-9]* cases/,$p' "$logs/parity-gate.log" | sed 's/^/      /'

exit $failed
