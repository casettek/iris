#!/usr/bin/env bash
# Deploy iris's contracts to the local EVM node and write where they are to
# devstack/deployment.json.
#
# The accounts are the local node's own test accounts, whose keys are
# published with it: the first deploys, owns the registry, relays Celestia's
# block roots and sends every transaction; the second is the attester, the
# trusted host whose signature stands for a blob's availability proof.
#
# The claims contract is deployed with RISC Zero's verifier and the ids of
# Raster's two proof programs, which the iris binary names. A deployment
# takes fault proofs of that build of Raster only: after a Raster update,
# build iris and deploy again.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$here/env.sh"

foundry() { "$here/foundry.sh" "$@"; }

deploy() {
  local contract=$1
  shift
  foundry forge create "src/$contract.sol:$contract" \
    --rpc-url "$IRIS_EVM_RPC" --private-key "$IRIS_SENDER_KEY" --broadcast \
    ${1:+--constructor-args "$@"} 2>&1 |
    sed -n 's/^Deployed to: //p'
}

foundry forge build > /dev/null 2>&1 || {
  foundry forge build
  exit 1
}

iris="${IRIS_BINARY:-$here/../target/release/iris}"
programs=$("$iris" evm proof-programs) || {
  echo "could not read Raster's proof programs from $iris: build iris first" >&2
  exit 1
}
program() { python3 -c 'import json, sys; print(json.loads(sys.argv[1])[sys.argv[2]])' "$programs" "$1"; }
transition=$(program transition_image_id)
authorization=$(program authorization_image_id)

oracle=$(deploy DataRootOracle "$IRIS_SENDER_ADDRESS")
registry=$(deploy IrisRegistry "$IRIS_SENDER_ADDRESS")
verifier=$(deploy RasterVerifier)
claims=$(deploy IrisClaims "$registry" "$oracle" "$IRIS_ATTESTER_ADDRESS" \
  "$IRIS_RETENTION_SECS" "$IRIS_CHALLENGE_PERIOD_SECS" "$IRIS_FAULT_PROOF_WINDOW_SECS" \
  "$verifier" "$transition" "$authorization")
[ -n "$oracle" ] && [ -n "$registry" ] && [ -n "$verifier" ] && [ -n "$claims" ] || {
  echo "a contract did not deploy" >&2
  exit 1
}

# The contracts' errors by selector, so that whoever sends a transaction can
# name the error a refused one reverted with.
errors=$(
  for contract in DataRootOracle IrisRegistry IrisClaims RasterJournal; do
    foundry forge inspect "$contract" errors --json 2> /dev/null
  done | python3 -c '
import json, sys
decoder, text, errors = json.JSONDecoder(), sys.stdin.read(), {}
at = 0
while at < len(text):
    if text[at].isspace():
        at += 1
        continue
    listed, at = decoder.raw_decode(text, at)
    errors.update({selector: name for name, selector in listed.items()})
print(json.dumps(errors, indent=4, sort_keys=True))'
)

cat > "$here/deployment.json" <<EOF
{
  "rpc": "$IRIS_EVM_RPC",
  "oracle": "$oracle",
  "registry": "$registry",
  "claims": "$claims",
  "verifier": "$verifier",
  "transition_image_id": "$transition",
  "authorization_image_id": "$authorization",
  "relayer": "$IRIS_SENDER_ADDRESS",
  "owner": "$IRIS_SENDER_ADDRESS",
  "attester": "$IRIS_ATTESTER_ADDRESS",
  "retention_secs": $IRIS_RETENTION_SECS,
  "challenge_period_secs": $IRIS_CHALLENGE_PERIOD_SECS,
  "fault_proof_window_secs": $IRIS_FAULT_PROOF_WINDOW_SECS,
  "errors": $errors
}
EOF
python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); print({k: v for k, v in d.items() if k != "errors"}, len(d["errors"]), "errors")' "$here/deployment.json"
