#!/bin/bash
# A bridge node over the local validator, as celestia-app's
# scripts/single-node-bridge.sh starts one. It signs with the validator's
# funded key, so blobs can be submitted through it, and takes requests
# without a token: it is a local test node.
set -o errexit -o nounset

CHAIN_ID="test"
CORE="celestia-validator"
STORE="/home/celestia/.celestia-bridge-${CHAIN_ID}"

# The validator keeps only its recent blocks, so the first block is there to
# ask for only while the chain is young. Its hash is kept beside the store
# for every later start.
GENESIS_FILE="/home/celestia/genesis-hash-${CHAIN_ID}"
GENESIS_BLOCK_HASH=""
if [ -s "${GENESIS_FILE}" ]; then
  GENESIS_BLOCK_HASH=$(cat "${GENESIS_FILE}")
fi
while [ -z "${GENESIS_BLOCK_HASH}" ] || [ "${GENESIS_BLOCK_HASH}" = "null" ]; do
  GENESIS_BLOCK_HASH=$(curl -s "http://${CORE}:26657/block?height=1" 2>/dev/null | jq -r .result.block_id.hash) || true
  sleep 1
done
echo "${GENESIS_BLOCK_HASH}" > "${GENESIS_FILE}"
# A private network is named by an environment variable, not a flag.
export CELESTIA_CUSTOM="${CHAIN_ID}:${GENESIS_BLOCK_HASH}"

if [ ! -f "${STORE}/config.toml" ]; then
  celestia bridge init --core.ip "${CORE}" --node.store "${STORE}"
  cp /validator/.celestia-app/keyring-test/* "${STORE}/keys/keyring-test/"
fi

exec celestia bridge start \
  --core.ip "${CORE}" \
  --node.store "${STORE}" \
  --keyring.keyname validator \
  --rpc.addr 0.0.0.0 \
  --rpc.skip-auth
