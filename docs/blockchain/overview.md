# Blockchain: Celestia and the EVM contracts

A claim's data goes to Celestia, and the claim itself to a contract on an EVM
blockchain. This page says what iris does on each, what the contracts check, and
which parts are still a trusted host in place of a proof.

What is built: a claim arrives and is accepted or refused, a claim with a
fault a contract can read is void on the spot, a challenge arrives and is
admitted or refused, and a fault proof dismisses a challenge whose trace is
wrong. What follows from any of those (clocks, bonds, finality) is not built
([what is open](../protocol/open.md)).

Both blockchains run locally from [`blockchain/devstack/`](../../blockchain/devstack/README.md).

## Who does what

| Step | Who | Command |
| --- | --- | --- |
| Build the claim and the blobs to post | iris | `iris claim build` |
| Post the blobs to Celestia | iris | `iris claim post` |
| Sign what each blob holds, as the attester | iris, with the attester's key | `iris da attest` |
| Read a block's root for the relayer | iris | `iris da block-root` |
| Write the bytes of each contract call | iris | `iris evm procedure`, `iris evm claim`, `iris evm challenge`, `iris evm fault-proof` |
| Name Raster's proof programs for a deployment | iris | `iris evm proof-programs` |
| Measure what a submission put on Celestia | iris | `iris da contents` |
| Read a submitted claim's data back, as a challenger | iris | `iris claim fetch` |
| Ask of a claim what no replay shows, and write the call for each fault | iris | `iris claim audit` |
| Read a submitted challenge's trace back, as a defender | iris | `iris challenge fetch` |
| Send the transactions, relay the roots, keep the results | the caller | `iris test devstack submit`, or iris-backend |

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
it itself (`iris_celestia::blob_commitment`), so a pointer can
be checked before anything is posted. The tests reproduce seven commitments
the local node gave.

### What a submission posts

`iris da contents --submission <claim or challenge submission>` reads every
blob the submission names back from the layer and says what each holds:

```json
{
  "bytes": 216927,
  "blobs": [
    {
      "height": 19220,
      "commitment": "149f...",
      "bytes": 216927,
      "framing_bytes": 57,
      "parts": [
        {
          "kind": "run_inputs",
          "bytes": 187,
          "inferences": 1,
          "values": 7,
          "text_bytes": 12,
          "marks_bytes": 27,
          "claimed_bytes": 5,
          "documents": 0,
          "document_bytes": 0,
          "other_bytes": 143
        },
        {
          "kind": "checkpoints",
          "bytes": 216683,
          "list_bytes": 11310,
          "trace_bytes": 83462,
          "input_manifests": 174,
          "input_manifest_bytes": 121555
        }
      ]
    }
  ]
}
```

`bytes` is everything posted: the whole of every blob. A part's `bytes` is
its size as the blob encodes it, and `framing_bytes` is the rest of the blob
(the magic, the submitter and the lengths), so a blob's parts and framing add
up to it. A claim posts its run's inputs (`kind: "run_inputs"`, with the text,
the token marks, the claimed ids and a source's documents each apart from the
rest) and its checkpoints: the list, the checkpoints and every input manifest.
A challenge
posts the challenger's trace of the stage (`kind: "trace_body"`).

The numbers above are a one-token claim of Gemma 4 E4B: 174 checkpoints. The
input manifests are over half of it, and the run's inputs are 187 bytes. On
the local blockchains that claim took about 1,074,000 gas to submit, and a
challenge of it, with a 38 KB trace posted, about 278,000.

## The contracts

Four contracts, in [`blockchain/contracts/src`](../../blockchain/contracts/src):

| Contract | Holds | Entry points |
| --- | --- | --- |
| `IrisRegistry` | Each registered procedure, under its id | `register(bytes)`, by the owner |
| `DataRootOracle` | Each Celestia block's data root and time, by height | `post(height, dataRoot, time)`, by the relayer |
| `IrisClaims` | Each accepted claim, whether a challenger showed it void, each admitted challenge, and whether a fault proof dismissed it | `submitClaim(bytes)`, `voidClaim(bytes)`, `submitChallenge(bytes)`, `submitFaultProof(bytes)` |
| `RasterVerifier` | Nothing: RISC Zero's verifier of a Groth16 proof | `verify(seal, imageId, journalDigest)`, called by `IrisClaims` |

Each entry point takes one `bytes` argument, the ABI encoding of one struct
(`IrisTypes.sol`), and decodes it itself. iris writes that encoding
(`iris_evm`), with its own ABI encoder: no ABI library is
involved on either side of the format.

**The registry** recomputes a procedure's id from what it is given, with
the same encoding as `Procedure::id()`, and stores it under that id. A
claim names an id; the contract judges it by what is stored there.

**A claim's id** is the SHA-256 of its ABI encoding. The contract encodes the
claim again after decoding it, so the same claim has one id however the bytes
it arrived in were laid out. `iris evm claim` prints the same id, and the
replay below compares the two.

### What `submitClaim` checks

In the order of `iris claim check`. Each error has the name of the Rust
rejection it is.

1. The procedure is registered, and the claim has the shape its procedure
   states: one inference per model, an agreement exactly when the result
   states one, and a relay record per inference a relay gives its prompt.
2. Each inference's statement is the procedure's model and parameters
   over the claimed input, or for an inference a relay gives its prompt, over
   that relay's record; its text is the one its record commits to; its record
   is valid.
3. The source record is the one the statement names and meets what the
   procedure requires; each result code is what the procedure's forms
   give the submitted text; so is the agreement.
4. Every blob is under a root the oracle holds, stays retrievable through the
   challenge period, carries the attester's signature, and names the claimer.
   Together they hold the claim's checkpoint list. The contract keeps one
   hash of the commitments of the run inputs they hold.
5. The records and results the blobs' proofs state are the ones the claim
   submits.

The contract recomputes every hash these checks compare: the statement, the
source record's digest, the forms and their result, the stated records.

### The ways to challenge a claim

[How iris works](../protocol/overview.md#what-catches-each-way-a-claim-can-be-wrong) has
the table: a claim that names an unapproved program or an input nobody else
holds is void on the spot (`voidClaim`), and a claim whose stage output is
wrong is challenged with a replay (`submitChallenge`), which the claimer can
answer with a fault proof (`submitFaultProof`). The sections below say what
each call checks.

`iris claim audit --submission <fetched>/claim_submission.json --specification <file>`
asks these of every checkpoint of a claim and every input its program has,
from its files and the procedure alone, with nothing run. A challenger does this first. With
`--calls <file>` it writes every call a contract can be asked, each with what
the contract is expected to answer, and the ones that point at a fault are
the ones to send (`iris test devstack submit fault --calls <file> --call <n>`).

### What `voidClaim` checks

The call takes the claim's own checkpoint at a stage, opened against the
claim's checkpoint root as a challenge opens it, and one question about it.
Nothing in it is the challenger's word. It reverts with `NoFault` when the
checkpoint is as it should be, and otherwise the claim is void of one of five
faults.

With no input named, the question is the stage's program. It must be one the
claim's procedure approves, or the claim is void (`UnapprovedProgram`).

With an input named, the question is where that input comes from. The caller
gives the bytes of the stage's input list, which the checkpoint commits to by
hash, and the checkpoint the stage's own checkpoint names as the input's
producer. The checks and their order are `CheckpointHashes::check_link`'s:

1. The list is the one the checkpoint commits to, the stage's program is
   approved, and the input is one the procedure lists for that program.
   Otherwise the call reverts: it is not this question's to judge.
2. The list is in its one canonical form and holds the input, or the claim
   is void (`UnreadableInputs`).
3. An input the program is given from fixed data holds a commitment the
   procedure pins for it, or the claim is void (`NotAPinnedValue`).
4. An input the run gives holds the commitment of a value the claim posted,
   or the claim is void (`NotAPostedValue`). For that the caller presents the
   commitments of every run input the claim posted, which the contract holds
   to the hash it kept when the claim arrived (`NotThePostedInputs`). An
   input that may be either, such as the claimed ids, which are a pinned
   empty value in a prompt range, is held by either.
5. An input from an earlier stage holds an empty value the procedure
   allows, or the output commitment of the checkpoint its own checkpoint
   names as the producer, which must be an earlier one whose program may
   produce this input. Otherwise the claim is void (`NotAnEarlierOutput`).

`iris claim audit` says more than the contract records: for an input that is
not an earlier output, whether no producer was named, it was not earlier, it
was of the wrong kind, or it produced something else.

A commitment is 32 bytes. An input list writes it as 64 lowercase hex digits,
and the contract reads it back to bytes once and compares bytes. Text that is
not 64 lowercase hex digits is no commitment and equals none.

A void claim is recorded with its fault, the stage and the time
(`voided(claimId)`), and cannot be voided again. Nothing reads that record
yet.

**The input list.** A checkpoint commits to its stage's
`input_manifest.json` by the SHA-256 of its bytes, so for those bytes to say
one thing to every reader they must be exactly the canonical encoding of what
they hold: two spaces of indent, the entries in ascending order of their
names with none repeated, the three fields in order, and nothing else. Every
name and field is plain text, printable ASCII with no quote and no backslash,
so nothing is escaped. `InputManifest.sol` reads the list byte by byte in
that form, and `read_input_manifest` in Rust holds the same rule.

**The pinned inputs.** The procedure lists, for each program, every
input the pipeline template binds to a file the import wrote, with every
commitment it is bound to: the tokenizer, the embedding, the model's
descriptor, the witness each stage checks its data with, the decoder and the
head, and one commitment per layer for the weights. They are read from the
template as the piped inputs are, so nothing about them is particular to a
model. An input bound per layer may hold any of its commitments: which layer
belongs at which stage is the schedule's, checked inside the stages, and a
challenger has every layer.

**Cost.** On the small model's 87-checkpoint claim, voiding for a program
used about 112,000 gas, for an input from an earlier stage about 686,000, for a
pinned input about 646,000, and for an input the run gives, held against 18
posted values, about 651,000. On gemma E2B, a layer stage held against the
35 pinned weight commitments used about 888,000. Most of what is left is
reading the input list byte by byte and opening the checkpoints.

### What `submitChallenge` checks

1. The claim is one the contract accepted, and its challenge period is open.
2. The checkpoint the challenge gives is the claim's at that stage: its hash,
   with the Merkle path given, makes the claim's checkpoint root.
3. The stage names a program the procedure approves. A stage that does
   not is `voidClaim`'s, and no replay of it is admitted.
4. The challenge disputes that checkpoint's output.
5. The challenger's trace is in the one shape the protocol fixes for its
   length: Raster's own admission of a challenge
   (`raster_prover::challenge::verify_challenge`). The window is 128 items,
   or for a shorter trace the largest power of two below its length, and the
   fingerprint is packed at the bits per item that window reveals 128 bits
   with. A window the challenger chose, or another packing, is a trace no
   defender is held to answer.
6. The trace body is posted under a root the oracle holds, stays retrievable
   through the fault-proof period, carries the attester's signature, and names
   the challenger.

A challenge gives the header of its trace whole, and the contract makes the
digest the posted body is found by from it. Only that header goes on chain.
The body is on Celestia.

On chain a challenge is of the checkpoint it opens. It carries no commitment
of its own that could name another one: Raster's one-stage commitment is made
from the checkpoint, not submitted. `iris evm challenge` refuses to encode a
challenge that was built about another checkpoint, since those bytes would
state a challenge nobody made. Whether the trace is a run of that
checkpoint's program on its inputs is what a fault proof settles.

The same claim, or the same challenge, is refused a second time.

### What `submitFaultProof` checks

A challenge says that honest execution of the disputed checkpoint's program
on its inputs gives another output, and posts the trace of that execution. A
Raster fault proof answers it: a window of that trace re-executed in RISC
Zero, ending at a step the trace gets wrong. `iris fault prove` writes it as
a Groth16 proof of Raster's transition program, and that proof, as it is, is
what the contract takes. Nothing else is proven for the blockchain.

1. The challenge is one the contract admitted and has not dismissed.
2. The proof's journal says the window reached the fault verdict.
3. The journal names the transition and authorization programs the contract
   was deployed with. A journal says which programs checked the window's
   earlier steps and its inputs, and a proof that names others was checked by
   nothing the contract knows.
4. The trace the window refutes is the one the challenge posted: its header
   digest is the one stored when the challenge was admitted.
5. The window runs the disputed checkpoint's program, on the disputed
   checkpoint's inputs. Both are stored when the challenge is admitted.
6. The Groth16 seal verifies over the whole journal, as a proof of the
   transition program.

Steps 2 to 5 are the checks `raster_core::pipeline::defend` makes for a
refutation. Step 4 is there because a real proof that some other trace is
false would otherwise dismiss an honest challenge. Step 5 is there because a
real proof that the trace is not a run of some other program, or on other
inputs, is true of every trace.

The challenge is then marked dismissed, with the time. Anyone may send the
proof: it says the same whoever holds it. No clock is read, and nothing
follows from a dismissal yet (see [what is open](../protocol/open.md)).

**The journal.** A fault proof's journal is Raster's `TransitionJournal` as
RISC Zero wrote it, and it begins with the window's opening state, whose size
depends on where the window opens. `RasterJournal.sol` finds the fields above
by walking it. `iris evm fault-proof` makes the same reading of the bytes
(`iris_evm::read_fault_journal`) and holds it to the decoded
journal every time it writes a call, and a unit test holds it to RISC Zero's
own encoding, so a change of Raster's journal is caught before a contract
reads one wrongly.

**The verifier.** `RasterVerifier` is RISC Zero's `RiscZeroGroth16Verifier`,
unchanged, from `risc0-ethereum` v3.0.1 (`blockchain/contracts/lib`, with the one
OpenZeppelin file it imports). That release takes proofs of the RISC Zero
circuits Raster is built with. A deployment also pins the ids of Raster's two
proof programs, which `iris evm proof-programs` prints, so it takes fault
proofs of one build of Raster: after a Raster update, build iris and deploy
again.

## Starting from the blockchains

A challenger has the procedure and the models, and nothing of the
claimer's. A defender has its own run of the disputed stage, and nothing of
the challenger's. Each reads the rest back from Celestia.

**`iris claim fetch --submission <claim_submission.json> --specification <file> --out <dir>`**
reads every blob the submission names and writes a new directory:

| File | What it is |
| --- | --- |
| `claim_submission.json` | The submission, as given |
| `checkpoints.txt`, `checkpoint_list.json` | The claim's checkpoint hashes and its checkpoint list, byte for byte as posted |
| `<stage>/input_manifest.json` | Each stage's input manifest |
| `posted_run.bin` | The inputs the claim's run was given, as it posted them |
| `case.toml` | The case the claim posted, for the reader's specification to run |

`--specification` is the reader's own specification of the procedure: the
procedure kind, the models, the parameters, and for a claim with a source the
recipe and the signer's key. The command leaves it as it is. The case it
writes holds what was posted: the prompt text the posted slices concatenate
to, in the case itself, or the source's documents with their attestations as
files beside it.

Nothing is written unless the specification describes the procedure the claim
names, every blob was posted for the claimer, and together they hold the
checkpoint list the claim names and the inputs of one run. The
outputs that travel with the stand-in proof are not read: a challenger makes
its own.

The submission's bytes are the ones of the transaction. The command takes
them as a file, and whoever runs it asks the contract whether it holds a
claim with that id (`claims(bytes32)`), as iris-backend does.

**`iris challenge build --specification <file> --submission <dir>/claim_submission.json`**
builds the challenge from that directory and the challenger's own
specification. The case is the `case.toml` beside the submission. Every stage is run on the
inputs the claim's checkpoint names for it: nothing is tokenized and no
output is found by direct inference, since the stages check what they are
given. The command writes the pipeline the checkpoints describe
(`claim_arranged.toml`, beside the run's own manifest) and prints each input
the claim gives a stage otherwise than the pipeline does. For a claim arranged
as the pipeline it is the pipeline. The run is compared with the posted list hash
by hash, and the first stage whose output differs is replayed on those same
inputs; the command checks that the replay ran on the input list the
checkpoint commits to and ended in another output. It stops when a
checkpoint names a value that is no earlier stage's output and none the pipeline
holds, or another program than the pipeline runs that stage with: such a claim
is one `iris claim audit` rules on. `iris challenge post` and `iris evm
challenge` then take the same directory as the claim's.

**`iris challenge fetch --challenge <challenge_submission.json> --out <commit.bin>`**
reads the trace the challenge names. It is written only when it was posted
for the challenger and its header is the one the challenge names, and it is
the file `iris fault prove --commitment` takes.

### Staging a dispute

Two commands make the wrong submissions a test needs. Neither is anything an
honest party runs.

- `iris test corrupt-claim --submission <claim_submission.json> --stage <stage> --claimer <address> --output <dir>`
  writes the claim a lying claimer submits: the honest claim with another
  output stated for one checkpoint, under its own list and root, with the
  honest claim's blobs carrying that list. A changed hash alone is no claim,
  since a list is posted with the checkpoints it names. The command refuses a
  checkpoint the claim's records are read from (the last stage's and its
  checks'): a contract refuses that claim on arrival, so there is nothing to
  challenge.
- `--fault program` and `--fault input` on the same command write the two
  lies no replay shows in place of a false output: a checkpoint that names a
  program outside the approved set, and one that commits to an input list in
  which one input is not its producer's output. The second posts its own
  input list beside the honest ones.
- `--fault arrangement [--input <name>]` writes a lie a replay shows on other
  inputs than the pipeline's: the stage is given, for one input, the value an
  earlier stage of the same program is given (another layer's weights,
  another range's marks, another stage's output), and its checkpoint still
  states the output of a run that was given the right one. An audit finds
  nothing in it.
- `--fault verdict --stage <the last check>` writes the lie of a claimer
  whose claimed output is not the model's. Give it a claim built with such
  ids (`claim build --direct-output-ids`), whose own record is not valid. The
  check is run on the claim's inputs and its record stated valid, with no
  claimed token differing, and the result stage is run on that record. The
  claim arrives with a valid record, and a replay of the check shows it.
  Built for a procedure kind of one inference.
- `iris challenge post --commitment <corrupted_commit.bin> --out <file>` posts
  a challenge over the copy of a trace that `iris test corrupt-challenge` changed,
  in place of the bundle's own.

A contract takes both, as it must: nothing at arrival tells them from honest
ones.

The staged dispute then ends as a real one would. `iris fault prove` proves a
window of the corrupted trace, read back from Celestia, and `iris evm
fault-proof --receipt <fault.receipt> --challenge <challenge_submission.json>
--submission <claim_submission.json>` writes the call that answers the
corrupted challenge with it. The call also carries `rejection`: the contract
error iris expects from the journal, or null.

## What a trusted host does in place of a proof

| Trusted part | What it does | What replaces it |
| --- | --- | --- |
| The attester | Reads each blob back from Celestia, works out what it holds, and signs where it is, its block's root, who submitted it and what it holds. The contract checks the signature | A receipt of a program that does the same, checked against its pinned id |
| The relayer | Posts each block's height, data root and time to `DataRootOracle` | A bridge that proves the roots. The oracle has the shape of a Blobstream contract. Celestia's own Blobstream deployments are reported as no longer maintained, so which bridge is open |

The attester's key is `IRIS_ATTESTER_KEY`. `iris da attest` signs nothing it
has not read from the layer itself: a submission that states other contents
than its blob holds is refused, not signed.

## The contracts are held to iris

`iris test devstack replay` is the contracts' test. It asks `iris claim check` (or
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

A fault proof is tested apart from the replay, since a real one takes a
prover to make. `blockchain/contracts/test/FaultProof.t.sol` (`just contracts-test`, a
few seconds, in the Foundry image) gives `submitFaultProof` a real Groth16
fault proof of a small stage, kept as `blockchain/contracts/test/fault_proof.json`:
RISC Zero's verifier takes it, the challenge it is about is dismissed once,
and the same proof is refused against a challenge of another trace, another
program or other inputs, under other pinned programs, and with its journal or
its seal changed. The proof names its own programs, so it stays one whatever
Raster's current programs are.

`iris test devstack replay faults` holds the two on-the-spot calls to iris. It takes a claim
in the directory `iris claim fetch` wrote, asks the contract every question
`iris claim audit` asks (of each checkpoint whether its program is approved,
of each piped input whether it is its producer's output), and fails unless
the contract answers each as iris does: it refuses to void an honest stage
with the same error, and takes a fault iris found. A fault found is then
sent, and the claim must read as void of that fault. An honest claim is the
case that matters most: a rule a little stricter than what honest claims do
would let anyone void one. `blockchain/contracts/test/InputManifest.t.sol` holds the
input list's form, and the trace's shape, to the same cases as the Rust
tests.

The claims and challenges to replay come from the parity gate run against
Celestia (`IRIS_CELESTIA_RPC=... just test-parity`), which leaves them posted:
a claim with a source, a consensus claim, and for each falsified checkpoint a
lying claim with the challenge of it
(`over-window/challenges/falsified-*/contract`). The lying claim posts its own
checkpoint list, so a contract takes it, as it must: nothing at arrival tells
a false output from a true one. The challenge is then admitted.

The gate also takes the challenger's side. Each lying claim is read back from
the layer into an empty directory and compared with what was posted: the
list, the checkpoints and every input manifest. For the first of them a
challenge is built from that directory alone, and must find and replay the
same stage. The claim with a source is read back too: its documents must be
the posted ones, and the case written over them must give the source
record the claim names.

## What is open

What is not built around these contracts, and every other known gap, is in
[one list](../protocol/open.md).
