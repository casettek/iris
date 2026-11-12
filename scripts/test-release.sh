#!/usr/bin/env bash
# Release tier, about an hour: the full tier, then the ignored tests (stage
# binaries, the real tokenizer, guest execution and a real fault proof) and
# the real-model check.
#
# usage: test-release.sh [--skip-proof]
# See docs/workflows/testing.md.
source "$(dirname "${BASH_SOURCE[0]}")/test-env.sh"

bash "$repo_root/scripts/test-full.sh" || failed=1
bash "$repo_root/scripts/test-ignored.sh" "$@" || failed=1
bash "$repo_root/scripts/test-real-model.sh" || failed=1

exit $failed
