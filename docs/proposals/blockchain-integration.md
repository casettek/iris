# Blockchain integration: gaps to close first, and how to integrate

Status: review, 2026-11-18. Nothing here is built. It reviews the claim and
challenge structures as they are at `4f886a3` and outlines the integration with
a locally running Celestia and EVM.

Background: [what a contract holds and checks](../internals/contract.md) and
[data availability and the registry](../internals/availability.md). Gaps those
pages already list (two commitments to one output, a stage that cannot run,
short stages, the signer's key as a run input, a source's host choices) are not
repeated here.

## Where each gap stands

Decided on 2026-11-18, and built on this branch unless it says otherwise:

| Gap | Decision | State |
| --- | --- | --- |
| 1. A withheld manifest | Post the manifests up front again | Built. The request step stays the way to cover a run input a challenger cannot make |
| 2. Reading a record out of an output | Build it as proposed [below](#reading-records-without-a-contract-decoding-anything) | Built: the stand-in proof states the records and the contract compares them at arrival. The proof program itself should bind its reading with a selection proof, not a claimer's index |
| 3. No submitter | The claimer names the address the claim is attributed to | Built, for claims and challenges |
| 4. Retention past the defence | Not ruled on | Not built |
| 5. The lifecycle | Not yet: the baseline integration first | Not built |
| 6. Proof programs not pinned | Not ruled on | Not built |
| 7. Age of documents | Not ruled on | Not built |
| 8. Encodings | Address it | Built for a checkpoint's hash, the checkpoint list and a configuration's id. The manifest and the record are receipts' work (gap 2) |

## Gaps not listed before

Each was read from the code, not found by running an attack.

### 1. One withheld input manifest makes a claim unchallengeable

This is the item set aside as "post on request", and it is wider than it was
described when the manifests stopped being posted.

A claimer states, for any stage, a checkpoint whose `input_commitment` is the
hash of a manifest it never reveals, and any output it likes. The last check is
the natural target: its output holds the claim record.

- A replay cannot be made: a stage challenge is a run on the claimer's inputs,
  which the manifest names.
- The link comparison cannot be made: `CheckpointList::check_link` first holds
  a presented manifest to the checkpoint's hash, and only the claimer has one
  that matches.

So until a claimer must reveal a manifest on request, with the claim void when
it does not, a dishonest claimer has a general way past every dispute. It is a
precondition of the integration, not an edge case. Two ways to close it:

- **The request step** (preferred, by the principle that only what a challenger
  cannot make is posted): a challenger names a position; the claimer must post
  that stage's manifest, and any run input in it the challenger cannot make,
  within a window. Honest claimers are never asked.
- **Post the manifests again** (205 KB on the E2B claim). No interaction, more
  data.

Part of it closes with no data at all. Three link faults can be read from the
posted checkpoints alone and should not need the manifest: an input that has no
pinned empty value and no producer named, a producer that is not earlier, and a
producer of the wrong kind. `check_link` should rule on those before it asks
for the manifest.

### 2. Reading a record out of a stage's output is the caller's word

`check_record_binding` and `check_result_binding` hold the presented
`output.bin` to the checkpoint's byte hash, then compare the submission with a
record "decoded from it by the caller". The host does that decoding today. A
contract cannot decode a Raster payload, and if a challenger supplies the
decoded record, a challenger can void an honest claim by misreading it.

Close it with a proof that the record is at its place in the output: a Raster
selection proof against the output's structural commitment, or a small receipt
that decodes the bytes and states the record. The first ties this to the "two
commitments" gap, which is about the same pair of hashes.

### 3. Nothing binds a submission to who submits it

`ClaimSubmission` and `ChallengeSubmission` name no submitter, and every blob is
public before the transaction that uses it. A challenger's trace body sits on
Celestia first; anyone can copy the pointer and the challenge and submit it
first, taking whatever a successful challenge earns. The same holds for a
claim. Close it by having the posted blob name the submitter's address, as part
of what its availability proof states, and the contract require it to be the
sender.

### 4. A claim's data must outlive the defence, not only the challenge period

`check_claim_availability` requires blobs to be retrievable until submission
plus the challenge period. A challenge that arrives at the end of that period
opens a fault-proof window, and anyone but the claimer who defends needs the
claim's new information then. The bound should be the challenge period plus the
fault-proof window (plus the request window of gap 1).

### 5. The dispute's lifecycle is not modelled

There are checks for a claim's arrival, a challenge's arrival, and a defence
(`check_defence`), and nothing that says what state a claim is in between them:
no defence submission, no clock, no rule for when a claim is final or void, no
rule for several challenges of one claim. A contract is mostly that state
machine. Raster's settlement proposal suggests a pure
`settle(challenge, defences, body_available_at, now, T)`; iris needs the same
over a claim, written and tested in Rust before any Solidity.

### 6. The proof programs are not pinned

A configuration pins stage programs and models. It does not pin what a
contract verifies receipts against: Raster's transition image id
(`check_defence` takes it as an argument), the availability proof's program,
and any comparison proof's. They belong in the configuration, or in a
contract-wide entry beside the registry.

### 7. Nothing bounds how old a source's documents are

An attestation's time is folded into the documents' chained hash, which the
contract never opens. A claimer can build a claim from documents attested at
any time in the past. For a settlement this matters as much as which domain
they came from. Close it with the source record stating the earliest and
latest attestation time, and a configuration bound on them, when domains and
base prompts are pinned.

### 8. Several commitments cannot be recomputed by an EVM contract as they are

Not attacks, but each is a check the Rust model makes that a contract cannot
make cheaply or safely in its current encoding:

| What | Today | For a contract |
| --- | --- | --- |
| A checkpoint's hash | SHA-256 of its JSON | A fixed binary encoding |
| The checkpoint list | Hex lines; opening one checkpoint means presenting all of them | A Merkle root, opened with a path |
| A configuration's id | SHA-256 of its JSON | A fixed binary or ABI encoding, or a Merkle root over the programs |
| A stage's input manifest | Raster's canonical JSON, parsed in the link check | A receipt that reads it and states the fault |
| A record in an output | Decoded by the caller (gap 2) | A selection proof or a receipt |

The first three change `checkpoints.txt`, the submission and the registry, so
they come before any contract is written against them.

## Reading records without a contract decoding anything

For gap 2. The claimer already makes one proof per claim, the availability
proof, and its program already reads the claim's checkpoints. Have the same
program read the two kinds of output a contract compares a submission with,
and state what they hold:

- for each inference, the claim record in its last check's output, with that
  output's hash;
- what the result stage's output states, with its hash;
- and that those hashes are the ones the claim's own checkpoints commit to: the
  last checkpoint's, which must run the pipeline's result program, and the
  checkpoints it names as each inference's check.

The contract then compares the stated records with the submission when the
claim arrives. Three things follow:

- No contract and no challenger decodes a Raster payload. The one decoder is
  the Rust the stages already use, run inside the proof.
- The two "challenge admission" comparisons about records become arrival
  checks. A claim whose submission is not what its checkpoints commit to is
  never accepted, so there is no dispute type for it and nothing for a
  challenger to misread.
- Nothing new is posted. The outputs are private inputs of the proof; a
  challenger makes its own.

It rests on the output's byte hash, which is what a replay disputes, so it
does not depend on how the "two commitments" gap is closed. Built in the
stand-in: the host reads the records as it did, now as what the proof states,
with the outputs carried beside the stand-in proof.

## What holds better than it did

With a real availability proof, the source record is no longer only the
`source` stage's word. The proof runs the source step, signatures included,
over the posted documents, and the contract holds the result to the
statement's source digest. That closes "a small source is a short stage" for
the record a contract reads.

## How to integrate

### What runs where

| Piece | Local version | In place of |
| --- | --- | --- |
| Celestia | A single-validator devnet with one bridge node; blobs posted and read through the node's blob API | The directory stand-in |
| Blobstream | The real contract-side verification of a data root against a relayed commitment, with a trusted relayer in place of the light-client proof | `blobstream.json` |
| EVM | A local node (anvil), contracts built and tested with Foundry | `iris claim check` on the host |
| Receipts | RISC Zero's verifier contract; its mock verifier while proofs run in dev mode | `AvailabilityProof::HostChecked` |

The names of the node APIs and contracts are from memory and are to be
confirmed when the work starts.

### The availability proof as one receipt

The blob's commitment on Celestia is a Merkle root over its shares, and
Blobstream attests to data roots of blocks. The simplest tie that keeps the
contract small: the proof program takes the blob's bytes and its share proofs,
checks the shares up to the block's data root, and states the data root with
what the blob holds (and, for gap 3, who posted it). The contract then checks
one receipt and one inclusion of that data root in what Blobstream attested.
The heavy hashing is in the program; the contract never sees a share.

The program needs the source step (already `no_std` in `source-core`), the
checkpoint package check, the share and namespace tree hashing, and, for a
challenger, Raster's body admission, which Raster's settlement proposal plans
to move into guest-buildable code.

### What the contract does itself, and what it takes a receipt for

- **Itself**: the registry; the claim's arrival checks that are hashes and
  string comparisons (configuration, pipeline shape, statement, output text,
  source record, forms and result, agreement); the clock, bonds and state
  machine; inclusion of a data root in Blobstream; approved-program lookups.
- **By receipt**: availability; anything that reads a Raster format (the link
  fault, the record or result binding fault); a defence, which is Raster's own
  receipt.

The rule: one implementation of every check that reads iris's or Raster's
formats, in Rust, run on the host in `iris claim check` and in a proof program
for the contract. Solidity re-implements only what is a hash or a comparison.

### Order of work

1. **Close what changes formats** (no chain needed). Done: the manifests
   (gap 1), the encodings (gap 8), the submitter (gap 3), the records
   stated by the claimer's proof (gap 2). Left: the proof program ids in a
   configuration (gap 6), the retention bound (gap 4).
2. **The lifecycle in Rust** (gap 5), set aside until the baseline integration
   runs: claim, challenge, defence and their clocks as a pure function with
   tests. The baseline needs only a claim's arrival and a challenge's arrival
   on chain.
3. **Celestia**: an `AvailabilityLayer` over a local node: the real blob
   commitment, posting, reading, share proofs. The parity gate runs against it
   behind a flag.
4. **The availability program**, in dev mode first, then proven.
5. **Contracts** on a local EVM: registry, claim and challenge entry points,
   the state machine of step 2, the Blobstream check with a trusted relayer,
   receipt verification. Each contract test replays a case the Rust model
   already decides, and both must agree.
6. **End to end** on the parity model: an honest claim that finalizes, a false
   checkpoint that is challenged and voided, a false challenge that is defended.

## What belongs in iris, and what in iris-backend

**iris**: everything that defines the protocol and has to have one source.
Formats and encodings; the Rust contract model; the proof programs; the
contracts themselves, beside the model they must agree with; the Celestia
client as a library; and commands that each do one stateless thing given
explicit endpoints (`post these blobs`, `write this submission's calldata`,
`check this claim against this chain`). The local devnet tests live here.

**iris-backend**: everything that is operation and not protocol. Keys and
wallets; sending transactions and waiting for them; the Blobstream relayer in
its trusted-host form; watching the chain for claims, challenges and deadlines;
deciding to challenge or defend and scheduling the work; retries; recording
pointers, transaction hashes and states in its database; and showing all of it
to the frontend. It already fetches and signs documents, which is the same kind
of work.

The line: iris takes inputs and produces bytes, or checks bytes, and can be run
again with the same result. The backend holds secrets, time and state, and
calls iris.

One consequence: `claim build` should stop posting. It should write the blobs
and the submission, and posting becomes its own command the backend calls, so
that a failed post is retried without building the claim again.
