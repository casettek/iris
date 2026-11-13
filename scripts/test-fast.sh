#!/usr/bin/env bash
# Fast tier, a few minutes: do the pieces behave, and do the stage programs
# in the tree match the identities the locks record?
# See docs/workflows/testing.md.
source "$(dirname "${BASH_SOURCE[0]}")/test-env.sh"

echo "== fast tier"
# Includes the schema check: no tile reads a value its program does not pin.
step unit-tests "unit tests and stage schema check" \
    cargo test --workspace --no-fail-fast

check_locks() {
    local stage status=0
    for manifest in "$repo_root"/raster-stages/*/Cargo.toml; do
        stage="$(dirname "$manifest")"
        echo "== $stage"
        (cd "$stage" && cargo raster program --verify) || status=1
    done
    return $status
}
# Rebuilds nothing when the guests are cached; a stale lock fails here.
step stage-locks "every Raster.lock matches its stage's sources" check_locks

exit $failed
