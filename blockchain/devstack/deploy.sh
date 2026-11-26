#!/usr/bin/env bash
# Deploy IWEBS's contracts to the local EVM node and write where they are to
# blockchain/devstack/deployment.json.
#
# The accounts are the local node's own test accounts, whose keys are
# published with it: the first deploys, owns the registry, relays Celestia's
# block roots and sends every transaction; the second is the attester, the
# trusted host whose signature stands for a blob's availability proof.
#
# The claims contract is deployed with RISC Zero's verifier and the ids of
# Raster's two proof programs, which the IWEBS binary names. A deployment
# takes fault proofs of that build of Raster only: after a Raster update,
# build IWEBS and deploy again.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$here/env.sh"

foundry() { "$here/foundry.sh" "$@"; }

# The libraries a contract is linked with, as `forge create` takes them.
link=()

deploy() {
  local contract=$1
  shift
  foundry forge create "src/$contract.sol:$contract" \
    --rpc-url "$IWEBS_EVM_RPC" --private-key "$IWEBS_SENDER_KEY" --broadcast \
    "${link[@]}" ${1:+--constructor-args "$@"} 2>&1 |
    sed -n 's/^Deployed to: //p'
}

foundry forge build > /dev/null 2>&1 || {
  foundry forge build
  exit 1
}

iwebs="${IWEBS_BINARY:-$here/../../target/release/iwebs}"
programs=$("$iwebs" evm proof-programs) || {
  echo "could not read Raster's proof programs from $iwebs: build iwebs first" >&2
  exit 1
}
program() { sed -n "s/.*\"$1\": *\"\([^\"]*\)\".*/\1/p" <<< "$programs" | head -1; }
transition=$(program transition_image_id)
authorization=$(program authorization_image_id)

oracle=$(deploy DataRootOracle "$IWEBS_SENDER_ADDRESS")
registry=$(deploy IwebsRegistry "$IWEBS_SENDER_ADDRESS")
verifier=$(deploy RasterVerifier)
# The claims contract calls these two as deployed libraries: its own code is
# near the size a contract may be.
manifest=$(deploy InputManifest)
journal=$(deploy RasterJournal)
[ -n "$manifest" ] && [ -n "$journal" ] || {
  echo "a library did not deploy" >&2
  exit 1
}
link=(--libraries "src/InputManifest.sol:InputManifest:$manifest"
  --libraries "src/RasterJournal.sol:RasterJournal:$journal")
claims=$(deploy IwebsClaims "$registry" "$oracle" "$IWEBS_ATTESTER_ADDRESS" \
  "$IWEBS_RETENTION_SECS" "$IWEBS_CHALLENGE_PERIOD_SECS" "$IWEBS_FAULT_PROOF_PERIOD_SECS" \
  "$verifier" "$transition" "$authorization")
[ -n "$oracle" ] && [ -n "$registry" ] && [ -n "$verifier" ] && [ -n "$claims" ] || {
  echo "a contract did not deploy" >&2
  exit 1
}

# The contracts' errors by selector, so that whoever sends a transaction can
# name the error a refused one reverted with.
errors=$(mktemp)
trap 'rm -f "$errors"' EXIT
for contract in DataRootOracle IwebsRegistry IwebsClaims RasterJournal InputManifest; do
  foundry forge inspect "$contract" errors --json 2> /dev/null
done > "$errors"

"$iwebs" test devstack deployment --out "$here/deployment.json" --errors "$errors" \
  --set "rpc=$IWEBS_EVM_RPC" \
  --set "oracle=$oracle" \
  --set "registry=$registry" \
  --set "claims=$claims" \
  --set "verifier=$verifier" \
  --set "transition_image_id=$transition" \
  --set "authorization_image_id=$authorization" \
  --set "relayer=$IWEBS_SENDER_ADDRESS" \
  --set "owner=$IWEBS_SENDER_ADDRESS" \
  --set "attester=$IWEBS_ATTESTER_ADDRESS" \
  --set "retention_secs=$IWEBS_RETENTION_SECS" \
  --set "challenge_period_secs=$IWEBS_CHALLENGE_PERIOD_SECS" \
  --set "fault_proof_period_secs=$IWEBS_FAULT_PROOF_PERIOD_SECS"
