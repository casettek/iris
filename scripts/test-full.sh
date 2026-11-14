#!/usr/bin/env bash
# Full tier, about half an hour: the fast tier, then the parity gate on the
# small synthetic model, then every stage program through authenticated
# Raster on that run's inputs.
# See docs/workflows/testing.md.
source "$(dirname "${BASH_SOURCE[0]}")/test-env.sh"

bash "$repo_root/scripts/test-fast.sh" || failed=1

echo "== full tier"
# Three implementations agree byte for byte, bad inputs publish faults, and a
# challenger finds and replays falsified claims. Its own report says which.
step parity-gate "parity gate: honest runs agree, cheating is caught" \
    cargo run --release --locked -p iris-cli "${features[@]}" -- test-parity
# The gate's own summary: what it covered, by outcome.
sed -n '/^PASS: [0-9]* cases/,$p' "$logs/parity-gate.log" | sed 's/^/      /'

bash "$repo_root/scripts/stage-ladder.sh" || failed=1

# The same gate and ladder for each other family's small model: its layer
# program, and what the shared programs do for its tokenizer.
step parity-gate-mistral "parity gate, mistral-dense model" \
    cargo run --release --locked -p iris-cli "${features[@]}" -- test-parity --bundle parity-mistral
sed -n '/^PASS: [0-9]* cases/,$p' "$logs/parity-gate-mistral.log" | sed 's/^/      /'

bash "$repo_root/scripts/stage-ladder.sh" || failed=1

exit $failed
