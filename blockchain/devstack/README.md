# Local blockchains

The two blockchains IWEBS integrates against, run locally with Docker: a
single-validator Celestia with a bridge node, and an EVM node with IWEBS's
contracts. Everything here is for development. Nothing is deployed anywhere.

This directory is the one definition of the stack. iris-backend does not run
its own: it is given the addresses below and connects to them.

## What runs

| Service | Image | Address | What it is for |
| --- | --- | --- | --- |
| `celestia-validator` | celestia-app 10.6.0 | `127.0.0.1:26657` | Produces blocks. |
| `celestia-bridge` | celestia-node 0.35.0 | `http://127.0.0.1:26658` | Takes and serves blobs. This is the address IWEBS uses (`IWEBS_CELESTIA_RPC`). |
| `evm` | Foundry 1.9.0 (anvil) | `http://127.0.0.1:8545` | Holds the contracts. |

The images are pinned by digest in `docker-compose.yml`. Foundry's `forge`
runs from the same image through `foundry.sh`, so nothing is installed on the
host besides Docker.

## Start it

```bash
just devstack-up        # start the three services and wait until they are healthy
just devstack-deploy    # build and deploy the contracts, write deployment.json
```

`just devstack-down` stops the services and deletes their state. The blockchains
start from nothing each time, so deploy again after starting.

`deployment.json` is what everything else reads: the EVM node's address, the
three contract addresses, the accounts, the three time parameters, and the
contracts' errors by selector. It is not committed.

## Who does what

The baseline has two trusted parts. Both are stand-ins for proofs, and both
are run by the same host here.

- **The attester** signs, for each blob, where it is on Celestia, the data
  root of its block, who submitted it and what it holds. The contract checks
  that signature in place of an availability proof. `iwebs da attest` signs
  with the key in `IWEBS_ATTESTER_KEY`, after reading the blob back from
  Celestia and working out its contents itself.
- **The relayer** posts each Celestia block's height, data root and time to
  the `DataRootOracle` contract. The contract takes a blob as included only
  under a root the oracle holds. The oracle has the shape of a Blobstream
  contract, so a bridge that proves the roots can take its place.

IWEBS holds no key of an account on either blockchain. It posts blobs to Celestia,
signs as the attester, and writes the bytes of each contract call (`iwebs evm
...`). Whoever runs it sends them: iris-backend for a service. Here, `IWEBS
devstack submit` sends them itself through the account the local EVM node
keeps unlocked, which works against a development node and nothing else.

## Accounts

`env.sh` names the accounts. They are the EVM node's first two test accounts,
whose keys are published with the node. They hold nothing outside it. Never
use them anywhere else.

| Account | Role |
| --- | --- |
| `0xf39F...2266` | Deploys, owns the registry, relays block roots, sends every transaction. |
| `0x7099...79C8` | The attester. |

## Run all of it as a test

```bash
just test-blockchain                              # about 3 minutes
just test-blockchain --real-model specification.toml  # with a real model's claim, a few minutes more
```

The blockchain tier makes an honest claim and a claim of each known lie, and holds
the contracts to what IWEBS says of each
([testing](../../docs/workflows/testing.md#the-blockchain-tier)). The sections below
are the same steps by hand.

## Submit a claim

```bash
source blockchain/devstack/env.sh
iwebs claim build --specification specification.toml --case case.toml --claimer 0x...   # builds the claim, posts nothing
iwebs claim post --submission <dir>/claim_submission.json   # posts its data to Celestia
iwebs test devstack submit claim --specification specification.toml --submission <dir>/claim_submission.json
```

`iwebs test devstack submit` attests the blobs, relays their block roots, registers the specification's
procedure if it is not registered, and submits the claim. It prints
whether the contract took it, or the error it refused it with.

A challenge goes the same way: `iwebs challenge post`, then `iwebs test devstack
submit challenge --submission <claim_submission.json> --challenge
<dir>/challenge_submission.json`.

The commands read where the contracts are from `blockchain/devstack/deployment.json`.
Run them from the repository's root, or name that file with
`IWEBS_BLOCKCHAIN_DEPLOYMENT`.

## Challenge a claim from the blockchains

A challenger starts from what Celestia holds, in a directory of its own:

```bash
# Read the claim's data back, with the case it posted.
iwebs claim fetch --submission <claim_submission.json> --specification specification.toml --out fetched
# Run its stages on the inputs it posted and replay the first whose output differs.
iwebs challenge build --specification specification.toml --submission fetched/claim_submission.json
# Post the trace, then submit.
iwebs challenge post --challenge <challenge_bundle.json> \
  --submission fetched/claim_submission.json --specification specification.toml --challenger 0x...
iwebs test devstack submit challenge --specification specification.toml \
  --submission fetched/claim_submission.json --challenge <dir>/challenge_submission.json
```

An honest claim has nothing to challenge. To have one, make the claim a lying
claimer would submit, and submit it as a claim of its own:

```bash
iwebs test corrupt-claim --submission <claim_submission.json> --stage <stage> \
  --claimer 0x... --output lying
iwebs claim post --submission lying/claim_submission.json
iwebs test devstack submit claim --specification specification.toml --submission lying/claim_submission.json
```

A defender reads a challenge's trace back the same way: `iwebs challenge fetch
--challenge <challenge_submission.json> --out commit.bin`.

`iwebs da contents --submission <submission>` measures what a claim or a
challenge put on Celestia: every blob read back, and its size split among
what it holds.
[The blockchains](../../docs/blockchain/overview.md#starting-from-the-blockchains) says what
each of these checks.

## Point at a fault no replay shows

Before replaying anything, a challenger asks of the fetched claim whether
every checkpoint names an approved program and whether every input a stage
takes from an earlier one is that stage's output:

```bash
iwebs claim audit --submission fetched/claim_submission.json \
  --specification specification.toml --calls calls.json
iwebs test devstack submit fault --calls calls.json --call <n>
```

The audit runs nothing and prints each fault with the number of the call that
points at it. The contract checks the call against the claim's own
checkpoints and holds the claim void (`voided(claimId)`); there is no
challenge to post and nothing for the claimer to answer.

To have such a claim, make the lie with `iwebs test corrupt-claim --submission ...
--fault program` or `--fault input`, then post and submit it as above.

Two more lies are ones a replay shows, and an audit finds nothing in them.
`--fault arrangement [--input <name>]` gives a stage a value the claim may
name in a place the pipeline does not give it, such as another layer's weights.
`--fault verdict --stage <the last check>`, on a claim built with claimed
ids that are not the model's (`claim build --output-ids`), states its
record valid. Post and submit each as above, fetch it, and `challenge build
--submission` replays the stage on the inputs its checkpoint names.

## Answer a challenge with a fault proof

A challenge is shown false with a Raster fault proof of its trace, as `iwebs
fault prove` writes it:

```bash
iwebs test devstack submit fault-proof --submission <claim_submission.json> \
  --challenge <challenge_submission.json> --receipt <fault.receipt>
```

The contract verifies the Groth16 proof with RISC Zero's verifier and checks
that it is about that challenge: its trace, and the disputed checkpoint's
program and inputs. `iwebs evm fault-proof --receipt <fault.receipt>` alone
prints what a contract is given and what it reads from the journal.

The claims contract is deployed with the ids of Raster's proof programs as
the IWEBS binary names them (`iwebs evm proof-programs`), so `deploy.sh` needs
a built IWEBS, and a deployment made before a Raster update refuses proofs
made after it. Deploy again.

`just contracts-test` runs the contract's own test of this on a real proof
kept in `blockchain/contracts/test/fault_proof.json`. It needs Docker and no blockchain.

## Check the contracts against IWEBS

```bash
iwebs test devstack replay claim --specification specification.toml --submission <dir>/claim_submission.json
iwebs test devstack replay challenge --specification specification.toml \
  --submission <dir>/claim_submission.json --challenge <dir>/challenge_submission.json
```

`iwebs test devstack replay faults --specification specification.toml --submission
fetched/claim_submission.json` does the same for the two on-the-spot calls:
every question the audit asks, asked of the contract.

`iwebs test devstack replay` is the contracts' test. It asks `iwebs claim check` (or `iwebs
challenge check`) and the contract about the same submission, then about one
changed copy after another, and fails unless the two give the same answer:
both take it, or both refuse it at the same check. A few cases are the
contract's alone and are printed as such: the attester's signature, and what
the contract already holds.

## Run the parity gate against Celestia

With `IWEBS_CELESTIA_RPC` set and no `--da-dir`, every IWEBS command uses the
Celestia node in place of the stand-in directory. That includes the gate:

```bash
IWEBS_TEST_CELESTIA=1 just test-parity
```

## Files

| File | What it is |
| --- | --- |
| `docker-compose.yml` | The three services. |
| `celestia/validator.sh`, `celestia/bridge.sh` | How the two Celestia services start. |
| `env.sh` | Addresses, accounts and time parameters. Source it. |
| `foundry.sh` | Runs a Foundry tool from the pinned image. |
| `deploy.sh` | Deploys the contracts and writes `deployment.json`. |

Two commands of the IWEBS binary drive the stack
(`blockchain/host/src/blockchain/devstack/`): `iwebs test devstack submit` sends a claim,
a challenge, a fault proof or a fault, and `iwebs test devstack replay` compares
the contracts with IWEBS, case by case. `deploy.sh` writes `deployment.json`
with `iwebs test devstack deployment`.
