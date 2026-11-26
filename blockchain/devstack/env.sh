# The local stack's endpoints, accounts and parameters. Source it.
#
# The two keys are the local EVM node's first and second test accounts. They
# are published with the node and hold nothing outside it: never use them
# anywhere else.

export IWEBS_CELESTIA_RPC="${IWEBS_CELESTIA_RPC:-http://127.0.0.1:26658}"
export IWEBS_EVM_RPC="${IWEBS_EVM_RPC:-http://127.0.0.1:8545}"

# Deploys, owns the registry, relays block roots, sends every transaction.
export IWEBS_SENDER_ADDRESS="0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
export IWEBS_SENDER_KEY="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
# The attester: the trusted host whose signature stands for a blob's proof.
export IWEBS_ATTESTER_ADDRESS="0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
export IWEBS_ATTESTER_KEY="0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"

# What the claims contract is deployed with, as `AvailabilityParameters`
# defaults them in Rust.
export IWEBS_RETENTION_SECS="${IWEBS_RETENTION_SECS:-2592000}"
export IWEBS_CHALLENGE_PERIOD_SECS="${IWEBS_CHALLENGE_PERIOD_SECS:-604800}"
export IWEBS_FAULT_PROOF_PERIOD_SECS="${IWEBS_FAULT_PROOF_PERIOD_SECS:-172800}"
