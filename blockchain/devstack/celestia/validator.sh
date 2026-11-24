#!/bin/bash
# A single-validator Celestia blockchain, as celestia-app's scripts/single-node.sh
# makes one: chain id `test`, one funded key named `validator`.
set -o errexit -o nounset

APP_HOME="/home/celestia/.celestia-app"
CHAIN_ID="test"
KEY_NAME="validator"
KEYRING_BACKEND="test"

if [ ! -f "${APP_HOME}/config/genesis.json" ]; then
  celestia-appd init "${CHAIN_ID}" --chain-id "${CHAIN_ID}" --home "${APP_HOME}" > /dev/null 2>&1
  celestia-appd keys add "${KEY_NAME}" --keyring-backend="${KEYRING_BACKEND}" --home "${APP_HOME}" > /dev/null 2>&1
  celestia-appd genesis add-genesis-account \
    "$(celestia-appd keys show "${KEY_NAME}" -a --keyring-backend="${KEYRING_BACKEND}" --home "${APP_HOME}")" \
    "1000000000000000utia" --home "${APP_HOME}"
  celestia-appd genesis gentx "${KEY_NAME}" 5000000000utia \
    --fees 500utia \
    --keyring-backend="${KEYRING_BACKEND}" \
    --chain-id "${CHAIN_ID}" \
    --home "${APP_HOME}" \
    --commission-rate=0.05 --commission-max-rate=1.0 --commission-max-change-rate=1.0 > /dev/null 2>&1
  celestia-appd genesis collect-gentxs --home "${APP_HOME}" > /dev/null 2>&1
  # Index transactions and keep block results, so proofs can be queried.
  sed -i "s#\"null\"#\"kv\"#g" "${APP_HOME}/config/config.toml"
  sed -i "s#discard_abci_responses = true#discard_abci_responses = false#g" "${APP_HOME}/config/config.toml"
fi

exec celestia-appd start \
  --home "${APP_HOME}" \
  --rpc.laddr tcp://0.0.0.0:26657 \
  --grpc.enable --grpc.address 0.0.0.0:9090 \
  --api.enable \
  --force-no-bbr \
  --delayed-precommit-timeout 1s
