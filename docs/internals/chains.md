# The chains: Celestia and the EVM contracts

A claim's data goes to Celestia, and the claim itself to a contract on an EVM
chain. This page says what iris does on each, what the contracts check, and
which parts are still a trusted host in place of a proof.

It is the baseline integration: a claim arrives and is accepted or refused, a
challenge arrives and is admitted or refused. What happens after arrival (the
defence, the clocks, bonds, finality) is not built.

Both chains run locally from [`devstack/`](../../devstack/README.md).

## Who does what

| Step | Who | Command |
| --- | --- | --- |
| Build the claim and the blobs to post | iris | `iris claim build` |
| Post the blobs to Celestia | iris | `iris claim post` |
| Sign what each blob holds, as the attester | iris, with the attester's key | `iris da attest` |
| Read a block's root for the relayer | iris | `iris da block-root` |
| Write the bytes of each contract call | iris | `iris evm configuration`, `iris evm claim`, `iris evm challenge` |
| Send the transactions, relay the roots, keep the results | the caller | `devstack/submit.py`, or iris-backend |

iris holds no account and sends nothing to the EVM. Each command does one
thing, given the addresses it needs, and can be run again. Keys for sending,
waiting for receipts, retries and records are the caller's.

## Posting is its own step

`iris claim build` no longer posts. It writes the blobs beside the submission
(`posting-blob-<n>.bin`) with the outputs their proof is given
(`posting-outputs.json`), and makes every check a contract would on the claim
as it will be once they are posted. `iris claim post --submission <file>`
posts them and writes where they are into the submission's `availability`.
A post that fails is run again without building the claim again.

They are plain files in the claim's directory, because every subdirectory
there is a stage.

## The layer

iris reaches a data-availability layer through one interface (`PostingLayer`):
post a blob, read it back, read a block's root and time. Which one is used:

1. `--da-dir <dir>`: the stand-in directory.
2. Otherwise, `IRIS_CELESTIA_RPC=http://host:port`: a celestia-node, through
   its JSON-RPC (`blob.Submit`, `blob.Get`, `header.GetByHeight`).
3. Otherwise the default stand-in directory.

Every command that touches posted data follows that rule, including the
parity gate.

A blob is named by its height, namespace and commitment. The commitment is
Celestia's own: the blob split into 512-byte shares, a namespaced Merkle tree
over them in subtrees, and a Merkle root over the subtree roots. iris computes
it itself (`inference_artifacts::celestia::blob_commitment`), so a pointer can
be checked before anything is posted. The tests reproduce seven commitments
the local node gave.

## The contracts

Three contracts, in [`contracts/src`](../../contracts/src):

| Contract | Holds | Entry points |
| --- | --- | --- |
| `IrisRegistry` | Each registered configuration, under its id | `register(bytes)`, by the owner |
| `DataRootOracle` | Each Celestia block's data root and time, by height | `post(height, dataRoot, time)`, by the relayer |
| `IrisClaims` | Each accepted claim and admitted challenge | `submitClaim(bytes)`, `submitChallenge(bytes)` |

Each entry point takes one `bytes` argument, the ABI encoding of one struct
(`IrisTypes.sol`), and decodes it itself. iris writes that encoding
(`inference_artifacts::evm`), with its own ABI encoder: no ABI library is
involved on either side of the format.

**The registry** recomputes a configuration's id from what it is given, with
the same encoding as `Configuration::id()`, and stores it under that id. A
claim names an id; the contract judges it by what is stored there.

**A claim's id** is the SHA-256 of its ABI encoding. The contract encodes the
claim again after decoding it, so the same claim has one id however the bytes
it arrived in were laid out. `iris evm claim` prints the same id, and the
replay below compares the two.

### What `submitClaim` checks

In the order of `iris claim check`. Each error has the name of the Rust
rejection it is.

1. The configuration is registered, and the claim has the shape of its
   pipeline: one inference, or two and an agreement.
2. Each inference's statement is the configuration's model and parameters
   over the claimed input; its text is the one its record commits to; its
   record is valid.
3. The source record is the one the statement names and meets what the
   configuration requires; each result code is what the configuration's forms
   give the submitted text; so is the agreement.
4. Every blob is under a root the oracle holds, stays retrievable through the
   challenge period, carries the attester's signature, and names the claimer.
   Together they hold the new information the statement names and the claim's
   checkpoint list.
5. The records and results the blobs' proofs state are the ones the claim
   submits.

The contract recomputes every hash these checks compare: the statement, the
source record's digest, the forms and their result, the stated records.

### What `submitChallenge` checks

1. The claim is one the contract accepted, and its challenge period is open.
2. The checkpoint the challenge gives is the claim's at that stage: its hash,
   with the Merkle path given, makes the claim's checkpoint root.
3. The stage names a program the configuration approves.
4. The challenge disputes that checkpoint's output.
5. The trace body is posted under a root the oracle holds, stays retrievable
   through the fault-proof window, carries the attester's signature, and names
   the challenger.

Only the trace's fingerprint (the digest of its header) goes on chain. The
body is on Celestia.

On chain a challenge is of the checkpoint it opens. It carries no commitment
of its own that could name another one: Raster's one-stage commitment is made
from the checkpoint, not submitted. `iris evm challenge` refuses to encode a
challenge that was built about another checkpoint, since those bytes would
state a challenge nobody made. Whether the trace is a run of that
checkpoint's program on its inputs is what a defence settles.

The same claim, or the same challenge, is refused a second time.

## What a trusted host does in place of a proof

| Trusted part | What it does | What replaces it |
| --- | --- | --- |
| The attester | Reads each blob back from Celestia, works out what it holds, and signs where it is, its block's root, who submitted it and what it holds. The contract checks the signature | A receipt of a program that does the same, checked against its pinned id |
| The relayer | Posts each block's height, data root and time to `DataRootOracle` | A bridge that proves the roots. The oracle has the shape of a Blobstream contract. Celestia's own Blobstream deployments are reported as no longer maintained, so which bridge is open |

The attester's key is `IRIS_ATTESTER_KEY`. `iris da attest` signs nothing it
has not read from the layer itself: a submission that states other contents
than its blob holds is refused, not signed.

## The contracts are held to iris

`devstack/replay.py` is the contracts' test. It asks `iris claim check` (or
`iris challenge check`) and the contract about the same submission, then
about one changed copy after another, and fails unless the two give the same
answer: both take it, or both refuse it at the same check.

A few cases are the contract's alone, and the replay prints them as such:

- A blob with no signature. The signature is how the contract learns what the
  Rust model reads from the layer itself.
- A claim or a challenge submitted twice, a challenge of a claim the contract
  does not hold, and a challenge after the challenge period. The Rust model
  keeps no earlier submissions and no clock.
- A claim that names another checkpoint root. `iris claim check` reads the
  claim's directory and stops before its checks when the list there is not
  the one the submission names.

The claims and challenges to replay come from the parity gate run against
Celestia (`IRIS_CELESTIA_RPC=... just test-parity`), which leaves them posted:
a claim with a source, a consensus claim, and for each falsified checkpoint a
lying claim with the challenge of it
(`over-window/challenges/falsified-*/contract`). The lying claim posts its own
checkpoint list, so a contract takes it, as it must: nothing at arrival tells
a false output from a true one. The challenge is then admitted.

## Not built

- **After arrival.** The defence of a challenge, the clocks that close a
  claim, bonds, and what a claim nobody successfully challenged becomes.
- **The proof program.** See the trusted parts above.
- **Raster's admission of a challenge.** `iris challenge check` asks Raster
  whether it admits the challenge (`verify_challenge`). The contract does not,
  and neither does the attester, who signs only that the posted body is a
  well-formed trace with the header the challenge names. Nothing acts on an
  admitted challenge yet, so this belongs with the defence: the program that
  proves availability can make the same check.
- **Everything [availability](availability.md#open) and
  [the contract page](contract.md) list as open.** This work changes none of
  it.
