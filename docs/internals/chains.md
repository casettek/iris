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
| Write the bytes of each contract call | iris | `iris evm configuration`, `iris evm claim`, `iris evm challenge`, `iris evm fault-proof` |
| Name Raster's proof programs for a deployment | iris | `iris evm proof-programs` |
| Measure what a submission put on Celestia | iris | `iris da contents` |
| Read a submitted claim's data back, as a challenger | iris | `iris claim fetch` |
| Ask of a claim what no replay shows, and write the call for each fault | iris | `iris claim audit` |
| Read a submitted challenge's trace back, as a defender | iris | `iris challenge fetch` |
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

### What a submission posts

`iris da contents --submission <claim or challenge submission>` reads every
blob the submission names back from the layer and says what each holds:

```json
{
  "bytes": 182482,
  "blobs": [
    {
      "height": 40003,
      "commitment": "da10...",
      "bytes": 182482,
      "framing_bytes": 57,
      "parts": [
        { "kind": "prompt_text", "bytes": 11, "text_bytes": 9 },
        {
          "kind": "checkpoints",
          "bytes": 182414,
          "list_bytes": 9490,
          "trace_bytes": 70459,
          "input_manifests": 146,
          "input_manifest_bytes": 102165
        }
      ]
    }
  ]
}
```

`bytes` is everything posted: the whole of every blob. A part's `bytes` is
its size as the blob encodes it, and `framing_bytes` is the rest of the blob
(the magic, the submitter and the lengths), so a blob's parts and framing add
up to it. A claim posts its prompt text or its source (`kind: "source"`, with
the documents' bodies apart from the attestations and recipe) and its
checkpoints: the list, the checkpoints and every input manifest. A challenge
posts the challenger's trace of the stage (`kind: "trace_body"`).

The numbers above are a one-token claim of Gemma 4 E2B: 146 checkpoints, and
the input manifests are over half of it.

## The contracts

Four contracts, in [`contracts/src`](../../contracts/src):

| Contract | Holds | Entry points |
| --- | --- | --- |
| `IrisRegistry` | Each registered configuration, under its id | `register(bytes)`, by the owner |
| `DataRootOracle` | Each Celestia block's data root and time, by height | `post(height, dataRoot, time)`, by the relayer |
| `IrisClaims` | Each accepted claim, whether a challenger showed it void, each admitted challenge, and whether a fault proof dismissed it | `submitClaim(bytes)`, `voidForProgram(bytes)`, `voidForInput(bytes)`, `submitChallenge(bytes)`, `submitFaultProof(bytes)` |
| `RasterVerifier` | Nothing: RISC Zero's verifier of a Groth16 proof | `verify(seal, imageId, journalDigest)`, called by `IrisClaims` |

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

### The four ways to challenge a claim

A claim can be wrong in four ways, and each has its own call.

| What is wrong | The call | How it is settled |
| --- | --- | --- |
| A stage's execution: its program on its inputs gives another output | `submitChallenge` | The challenger posts its trace of the stage. The claimer can answer with a fault proof (`submitFaultProof`) |
| A stage's program: it is outside the approved set | `voidForProgram` | On the spot: the claim is void |
| A stage's input from an earlier stage: it is not what that stage produced | `voidForInput` | On the spot: the claim is void |
| A stage's input from the run itself: it is not the pinned model's, or not what the claim posted | none yet | See [Not built](#not-built) |

The first is an assertion that can itself be false, which is why it has an
answer. The next two are facts the contract reads from the claim's own
checkpoints, so there is nothing to answer and nothing to wait for. A replay
cannot stand in for them: it runs the program a checkpoint names on the
inputs it names, and so says nothing of whether that is the right program or
where the inputs came from.

`iris claim audit --submission <fetched>/claim_submission.json --run <spec>`
asks the last two of every checkpoint of a claim, from its files and the
configuration alone, with nothing run. A challenger does this first. With
`--calls <file>` it writes every call a contract can be asked, each with what
the contract is expected to answer, and the ones that point at a fault are
the ones to send (`devstack/submit.py fault --calls <file> --call <n>`).

### What `voidForProgram` and `voidForInput` check

Both open the claim's own checkpoint at a stage against the claim's
checkpoint root, as a challenge does. Nothing in either is the challenger's
word.

`voidForProgram` looks the checkpoint's program up in the claim's
configuration. Approved, the call reverts with `ProgramApproved`. Otherwise
the claim is void (`UnapprovedProgram`).

`voidForInput` takes the bytes of the stage's input list, which the
checkpoint commits to by hash, the name of one input, and the checkpoint the
stage's own checkpoint names as that input's producer. Its checks and their
order are `CheckpointList::check_link`'s:

1. The list is the one the checkpoint commits to, the stage's program is
   approved, and the input is one the configuration says that program takes
   from an earlier stage. Otherwise the call reverts: it is not this call's
   to judge.
2. The list is in its one canonical form, or the claim is void
   (`UnreadableInputList`).
3. The list holds the input, or the claim is void (`InputMissing`).
4. If the input holds an empty value the configuration allows it, the call
   reverts with `Linked`.
5. The checkpoint names a producer for the input (`NoProducer`), and the
   producer is an earlier checkpoint (`ProducerNotEarlier`).
6. The producer's program is one that may produce this input
   (`ProducerOfWrongKind`).
7. The commitment the stage loaded the input by is the producer's output
   commitment (`NotTheProducersOutput`). If it is, the call reverts with
   `Linked`.

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

**Cost.** On a 306-checkpoint claim, `voidForProgram` used about 101,000 gas
with 1.3 KB of call data, and `voidForInput` about 1,630,000 gas with 4.1 KB.
The second reads the whole configuration from storage and the input list
byte by byte, and neither has been made cheaper.

### What `submitChallenge` checks

1. The claim is one the contract accepted, and its challenge period is open.
2. The checkpoint the challenge gives is the claim's at that stage: its hash,
   with the Merkle path given, makes the claim's checkpoint root.
3. The stage names a program the configuration approves. A stage that does
   not is `voidForProgram`'s, and no replay of it is admitted.
4. The challenge disputes that checkpoint's output.
5. The challenger's trace is in the one shape the protocol fixes for its
   length: Raster's own admission of a challenge
   (`raster_prover::challenge::verify_challenge`). The window is 128 items,
   or for a shorter trace the largest power of two below its length, and the
   fingerprint is packed at the bits per item that window reveals 128 bits
   with. A window the challenger chose, or another packing, is a trace no
   defender is held to answer.
6. The trace body is posted under a root the oracle holds, stays retrievable
   through the fault-proof window, carries the attester's signature, and names
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
what the contract takes. Nothing else is proven for the chain.

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

Steps 2 to 5 are the checks `raster_core::chain::defend` makes for a
refutation. Step 4 is there because a real proof that some other trace is
false would otherwise dismiss an honest challenge. Step 5 is there because a
real proof that the trace is not a run of some other program, or on other
inputs, is true of every trace.

The challenge is then marked dismissed, with the time. Anyone may send the
proof: it says the same whoever holds it. No clock is read, and nothing
follows from a dismissal yet (see [Not built](#not-built)).

**The journal.** A fault proof's journal is Raster's `TransitionJournal` as
RISC Zero wrote it, and it begins with the window's opening state, whose size
depends on where the window opens. `RasterJournal.sol` finds the fields above
by walking it. `iris evm fault-proof` makes the same reading of the bytes
(`inference_artifacts::evm::read_fault_journal`) and holds it to the decoded
journal every time it writes a call, and a unit test holds it to RISC Zero's
own encoding, so a change of Raster's journal is caught before a contract
reads one wrongly.

**The verifier.** `RasterVerifier` is RISC Zero's `RiscZeroGroth16Verifier`,
unchanged, from `risc0-ethereum` v3.0.1 (`contracts/lib`, with the one
OpenZeppelin file it imports). That release takes proofs of the RISC Zero
circuits Raster is built with. A deployment also pins the ids of Raster's two
proof programs, which `iris evm proof-programs` prints, so it takes fault
proofs of one build of Raster: after a Raster update, build iris and deploy
again.

## Starting from the chains

A challenger has the configuration and the models, and nothing of the
claimer's. A defender has its own run of the disputed stage, and nothing of
the challenger's. Each reads the rest back from Celestia.

**`iris claim fetch --submission <claim_submission.json> --run <spec> --out <dir>`**
reads every blob the submission names and writes a new directory:

| File | What it is |
| --- | --- |
| `claim_submission.json` | The submission, as given |
| `checkpoints.txt`, `checkpoint_trace.json` | The claim's list and checkpoints, byte for byte as posted |
| `<stage>/input_manifest.json` | Each stage's input manifest |
| `inference.toml` | The reader's run spec over what the claim posted |

`--run` is the reader's own spec of the configuration: the pipeline, the
models, the parameters, and for a claim with a source the recipe and the
signer's key. The command keeps those and replaces the rest with what was
posted: the prompt text, written into the spec itself, or the source's
documents with their attestations.

Nothing is written unless the spec describes the configuration the claim
names, every blob was posted for the claimer, and together they hold the new
information the statement names and the checkpoint list the claim names. The
outputs that travel with the stand-in proof are not read: a challenger makes
its own.

The submission's bytes are the ones of the transaction. The command takes
them as a file, and whoever runs it asks the contract whether it holds a
claim with that id (`claims(bytes32)`), as iris-backend does.

**`iris challenge build --run <dir>/inference.toml --submission <dir>/claim_submission.json`**
builds the challenge from that directory alone. The run is prepared as a
claimer prepares its own, from the spec, with the output found by direct
inference (or given with `--direct-output-ids`), and is compared with the
posted list hash by hash. The command refuses a spec whose prompt is not the
one the claim names. `iris challenge post` and `iris evm challenge` then take
the same directory as the claim's.

**`iris challenge fetch --challenge <challenge_submission.json> --out <commit.bin>`**
reads the trace the challenge names. It is written only when it was posted
for the challenger and its header is the one the challenge names, and it is
the file `iris fault prove --commitment` takes.

### Staging a dispute

Two commands make the wrong submissions a test needs. Neither is anything an
honest party runs.

- `iris claim corrupt --submission <claim_submission.json> --stage <stage> --claimer <address> --output <dir>`
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
- `iris challenge post --commitment <corrupted_commit.bin> --out <file>` posts
  a challenge over the copy of a trace that `iris challenge corrupt` changed,
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

A fault proof is tested apart from the replay, since a real one takes a
prover to make. `contracts/test/FaultProof.t.sol` (`just contracts-test`, a
few seconds, in the Foundry image) gives `submitFaultProof` a real Groth16
fault proof of a small stage, kept as `contracts/test/fault_proof.json`:
RISC Zero's verifier takes it, the challenge it is about is dismissed once,
and the same proof is refused against a challenge of another trace, another
program or other inputs, under other pinned programs, and with its journal or
its seal changed. The proof names its own programs, so it stays one whatever
Raster's current programs are.

`replay.py faults` holds the two on-the-spot calls to iris. It takes a claim
in the directory `iris claim fetch` wrote, asks the contract every question
`iris claim audit` asks (of each checkpoint whether its program is approved,
of each chained input whether it is its producer's output), and fails unless
the contract answers each as iris does: it refuses to void an honest stage
with the same error, and takes a fault iris found. A fault found is then
sent, and the claim must read as void of that fault. An honest claim is the
case that matters most: a rule a little stricter than what honest claims do
would let anyone void one. `contracts/test/InputManifest.t.sol` holds the
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
the posted ones, and the run spec written over them must give the source
record the claim names.

## Not built

The contract covers the two cases Raster covers today, a claim challenged
with a replayed checkpoint and a challenge shown false with a fault proof of
its trace, and the two faults of a claim that no replay shows. What is around
them is open, and each is its own piece of work:

- **A wrong input from the run itself.** A stage's inputs are of three
  kinds: from an earlier stage (`voidForInput`), from the model, and from
  this run (the prompt or the documents, and what the claimer derives from
  them). Nothing holds the last two to anything on chain. The model's data
  is in every stage's input list by its commitment, and the configuration
  pins the model's commitments by role, but which role belongs at which
  stage is the schedule's, and some inputs (`descriptor`,
  `witness`) are not among the pinned ones at all. A run input has no value
  on chain to be compared with: the contract knows the posted prompt by its
  SHA-256, and a stage loads it by another commitment. And a challenger
  cannot replay a stage on a run input it does not have, since only the
  input's commitment is posted. This is the same hole as the first item of
  [availability's open list](availability.md#open).

- **What follows.** Nothing reads a dismissal, or the lack of one. A claim
  with an unanswered challenge and a claim with none are held the same way.
  The clocks that close a claim or a challenge, bonds, and what a claim nobody
  successfully challenged becomes are not built. `submitFaultProof` reads no
  clock: the fault-proof window is used only for how long a challenge's trace
  must stay retrievable.
- **Several challenges of one claim.** Each is admitted and dismissed alone.
  How they are settled together belongs with the clocks and bonds.
- **A true trace with a false label.** Raster has a second defence: the
  challenger's trace is an honest run, and ends in another output than the
  challenger stated for it (`TerminalMatch`). The contract does not take it,
  so such a challenge cannot be dismissed on chain yet. It needs the stated
  output compared with the one the window ends in.
- **`iris fault prove` and the contract ask slightly different things.**
  `fault prove` holds its proof to the checkpoint of the claim bundle it is
  given, which in a staged dispute is the honest claim's, while the challenge
  on chain is of the lying claim's checkpoint. The two have the same program
  and inputs, which is all a fault proof names, so the proof serves both.
  `fault prove` also does not ask which authorization program the journal
  names; `iris evm fault-proof` and the contract do.
- **Upgrades.** A deployment takes proofs of one build of Raster and one
  release of RISC Zero's circuits.
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
