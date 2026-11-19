# Local chains

The two chains iris integrates against, run locally with Docker: a
single-validator Celestia with a bridge node, and an EVM node with iris's
contracts. Everything here is for development. Nothing is deployed anywhere.

This directory is the one definition of the stack. iris-backend does not run
its own: it is given the addresses below and connects to them.

## What runs

| Service | Image | Address | What it is for |
| --- | --- | --- | --- |
| `celestia-validator` | celestia-app 10.6.0 | `127.0.0.1:26657` | Produces blocks. |
| `celestia-bridge` | celestia-node 0.35.0 | `http://127.0.0.1:26658` | Takes and serves blobs. This is the address iris uses (`IRIS_CELESTIA_RPC`). |
| `evm` | Foundry 1.9.0 (anvil) | `http://127.0.0.1:8545` | Holds the contracts. |

The images are pinned by digest in `docker-compose.yml`. Foundry's tools
(`forge`, `cast`) run from the same image through `foundry.sh`, so nothing is
installed on the host besides Docker and Python 3.

## Start it

```bash
just devstack-up        # start the three services and wait until they are healthy
just devstack-deploy    # build and deploy the contracts, write deployment.json
```

`just devstack-down` stops the services and deletes their state. The chains
start from nothing each time, so deploy again after starting.

`deployment.json` is what everything else reads: the EVM node's address, the
three contract addresses, the accounts, the three time parameters, and the
contracts' errors by selector. It is not committed.

## Who does what

The baseline has two trusted parts. Both are stand-ins for proofs, and both
are run by the same host here.

- **The attester** signs, for each blob, where it is on Celestia, the data
  root of its block, who submitted it and what it holds. The contract checks
  that signature in place of an availability proof. `iris da attest` signs
  with the key in `IRIS_ATTESTER_KEY`, after reading the blob back from
  Celestia and working out its contents itself.
- **The relayer** posts each Celestia block's height, data root and time to
  the `DataRootOracle` contract. The contract takes a blob as included only
  under a root the oracle holds. The oracle has the shape of a Blobstream
  contract, so a bridge that proves the roots can take its place.

iris holds no account on either chain and sends nothing to the EVM. It posts
blobs to Celestia, signs as the attester, and writes the bytes of each
contract call (`iris evm ...`). Whoever runs it sends them: `submit.py` here,
iris-backend for a service.

## Accounts

`env.sh` names the accounts. They are the EVM node's first two test accounts,
whose keys are published with the node. They hold nothing outside it. Never
use them anywhere else.

| Account | Role |
| --- | --- |
| `0xf39F...2266` | Deploys, owns the registry, relays block roots, sends every transaction. |
| `0x7099...79C8` | The attester. |

## Submit a claim

```bash
source devstack/env.sh
iris claim build --run spec.toml --claimer 0x...   # builds the claim, posts nothing
iris claim post --submission <dir>/claim_submission.json   # posts its data to Celestia
python3 devstack/submit.py claim --run spec.toml --submission <dir>/claim_submission.json
```

`submit.py` attests the blobs, relays their block roots, registers the run's
configuration if it is not registered, and submits the claim. It prints
whether the contract took it, or the error it refused it with.

A challenge goes the same way: `iris challenge post`, then `submit.py
challenge --challenge <dir>/challenge_submission.json`.

## Check the contracts against iris

```bash
python3 devstack/replay.py claim --run spec.toml --submission <dir>/claim_submission.json
python3 devstack/replay.py challenge --run spec.toml \
  --submission <dir>/claim_submission.json --challenge <dir>/challenge_submission.json
```

`replay.py` is the contracts' test. It asks `iris claim check` (or `iris
challenge check`) and the contract about the same submission, then about one
changed copy after another, and fails unless the two give the same answer:
both take it, or both refuse it at the same check. A few cases are the
contract's alone and are printed as such: the attester's signature, and what
the contract already holds.

## Run the parity gate against Celestia

With `IRIS_CELESTIA_RPC` set and no `--da-dir`, every iris command uses the
Celestia node in place of the stand-in directory. That includes the gate:

```bash
IRIS_CELESTIA_RPC=http://127.0.0.1:26658 just test-parity
```

## Files

| File | What it is |
| --- | --- |
| `docker-compose.yml` | The three services. |
| `celestia/validator.sh`, `celestia/bridge.sh` | How the two Celestia services start. |
| `env.sh` | Addresses, accounts and time parameters. Source it. |
| `foundry.sh` | Runs a Foundry tool from the pinned image. |
| `deploy.sh` | Deploys the contracts and writes `deployment.json`. |
| `submit.py` | Sends a claim or a challenge. |
| `replay.py` | Compares the contracts with iris, case by case. |
