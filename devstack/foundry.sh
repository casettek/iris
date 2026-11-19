#!/usr/bin/env bash
# Run a Foundry tool (forge, cast, anvil) from the pinned image, in the
# contracts directory, on the host's network so it reaches the local EVM node.
#
# usage: foundry.sh <tool> [arguments...]
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
image="ghcr.io/foundry-rs/foundry@sha256:3a392c18f9b16e665e932de159fb6fa9ed972c4b4352f2f76a1a8c7ca9b57d39"

exec docker run --rm --network host \
  -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$here/../contracts":/work -w /work \
  --entrypoint "$1" "$image" "${@:2}"
