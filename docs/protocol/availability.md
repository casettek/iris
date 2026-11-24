# Data availability and the procedure registry

A claim names its inputs and its checkpoints by commitment. A challenger
cannot dispute what it cannot get, so a contract accepts a claim only with its
data posted to the data-availability layer (Celestia), and a challenge only
with its trace body posted. A claim also names, by id, the registered
procedure it is a claim of.

The rules are in `blockchain_rules::availability` and
`blockchain_rules::registry`; [the contract page](checks.md) lists the
checks in the order a contract makes them. The layer is behind one trait
(`cli/src/posting.rs`), with two implementations: a
stand-in directory, and a local Celestia node. Blobstream and the availability
proof are still stand-ins run by a trusted host. [The blockchains](../blockchain/overview.md) says
how the Celestia node and the contracts are used.

## What is posted, and what is not

**The rule.** Every input a stage is given is one of three things, and a
challenger holds all three:

| A stage's input is | Where a challenger gets it |
| --- | --- |
| An earlier stage's output | Its own run of that stage, which the claim's checkpoint for it either matches or does not |
| Fixed data the procedure pins: the model's weights and tokenizer, the constants the first stages start from | Published elsewhere, by the commitment the procedure pins |
| The run's own | Posted with the claim |

So whenever two parties disagree about a stage, they ran it on the same
inputs, and a replay settles it. A stage input that is none of the three is a
fault of the checkpoint that names it, which a contract rules on with nothing
run ([the blockchains](../blockchain/overview.md#the-ways-to-challenge-a-claim)).

**The run's own inputs** are everything a stage is given that neither an
earlier stage nor the procedure fixes. A claim posts each as the value the
stage is given, and anyone writes the file the stage loads from it:

| Run input | Given to | What it is |
| --- | --- | --- |
| `documents`, `recipe`, `signer` | `source` | The documents in chunks with their attestations, the recipe, and the key the attestations are checked against |
| `text` | each `prompt_bind` | That range's slice of the prompt text. The slices concatenate to the prompt, so the text is posted once |
| `config`, `schedule`, `claim` | each `prompt_bind` | The run's model, flags and token budget; how many prompt and claimed ranges it has; and what it claims of the whole text |
| `marks` | each `prompt_check` | For each piece of that range, whether a token starts there |
| `claimed` | each `embed_c` | That range of the claimed output ids |
| `forms` | `result` | The output texts the result is read by |

A contract does not judge what a posted input says. The stages do, as they
run: `prompt_bind` hashes the text and holds it to the text digest and the
source record, `prompt_check` holds the marks to the tokenizer's own merge,
the first bind checks the schedule, `sequence-check` compares the claimed ids
with the greedy output, and the statement a contract recomputes binds the
model, the flags and the text's hash. So a claimer who posts marks that are
not the tokenizer's is shown false by a replay of that stage on those marks:
the honest run ends in a record that says so, and the claim stated a clean
one. The signer's key, the recipe and the forms are the procedure's as
well, which a contract holds the claim's records to when it arrives.

**A value in the wrong place.** A checkpoint may name any value a challenger
holds, and the comparison does not ask whether it is the one the chain gives
that stage. So a claimer can give a stage another layer's weights, the marks
of another range, the procedure posted for another inference, or the
output of another earlier stage of the right kind. Nothing voids such a
claim. The stage's own checks say what it was given: its output carries the
fault, and every later stage carries it into the final record. A claim that
states a clean output for that stage is shown false as any other lie about a
stage is, by a replay of the stage on the inputs its checkpoint names.
`challenge build` runs every stage on exactly those: it binds each input to
the earlier stage the checkpoint points at when that stage's output is what
the input list loads, and otherwise to the file the chain holds under that
commitment, a pinned one or a posted one.

A run input is posted as its value and not as the Raster file a stage loads,
because the file's index is several times the value's size and anyone writes
both from the value. On a 1,209-token prompt the files are about 130 KB of
data and 620 KB of indexes, and the values are about a tenth of the data.

The claimer also posts what only it can state: **its checkpoints**, the
checkpoint hashes, every checkpoint they name, and every input manifest a checkpoint
commits to. A challenger finds the first checkpoint it disagrees with by hash,
and a challenge is built from the claimer's checkpoint at that position. That
checkpoint fixes which step the position is: its stage, program, producers and
the hash of its input manifest. Without it, a challenger could dispute a
position with an honest run of another step. `aux_c0_l3` and `aux_c0_l5`, for
instance, run one program on one chained input and differ only in which
layer's weights they load, so a challenger's honest `aux_c0_l5` would differ
from the claimer's honest `aux_c0_l3` with nothing for the claimer to refute.

The manifests are posted because a lying claimer's cannot be made by anyone
else. An honest claimer's are the ones a challenger makes. But a claimer could
state a checkpoint over a manifest it never reveals, and then that stage could
be neither replayed, since a replay runs on the inputs the manifest names, nor
held to its producers, since the link comparison reads the manifest. One
withheld manifest would put a claim beyond every dispute.

Every blob names the **address it was posted for**: the claimer's, which the
submission states (`claim build --claimer`, or `IRIS_CLAIMER`), or the
challenger's (`challenge post --challenger`, or `IRIS_CHALLENGER`). A contract
refuses a submission whose blobs name another address, so nobody can submit
someone else's posted data as their own. Copying a claim or a challenge then
means posting its data again under one's own address, with a proof of one's
own, which the original submitter is ahead of.

## What a claim posts

`claim build` posts, in the namespace `iris-claim`:

| Entry | What the proof states of it |
| --- | --- |
| The run's inputs (`PostedRun`): the source stage's, the forms, and each inference's procedure, schedule, text digest, text slices, marks and claimed ids | The commitment a stage's input list names each one by: the root of the Raster file written from it |
| The checkpoint hashes, with every checkpoint and every input manifest a checkpoint commits to | The root of the hashes, which is the submission's `checkpoints_root`, and that every checkpoint they name and its manifest are there |

A contract keeps one hash of the run inputs' commitments, in the order the
blobs state them (`posted_inputs_digest`). A challenger that says a stage's
run input was not posted presents the commitments, and the contract holds
them to that hash.

Entries are packed in order into blobs of at most 1 MiB, a stand-in for the
layer's own limit. A blob is `iris/blob/v3\n` followed by the postcard encoding
of the address it is posted for and its entries; a reader refuses any other
encoding of the same blob.

On the real-model check's E2B claim (a 1,209-token prompt with no source, 292
checkpoints) the parts are about 6 KB of prompt text, a 19 KB list, 142 KB of
checkpoints and 205 KB of input manifests, measured on 2026-11-18. An earlier
version also posted every Raster run file with its index (754 KB).

### What the proof states of a claim's records

The proof of the blob that holds a claim's checkpoints also states the records
those checkpoints commit to (`StatedRecords`): what the last stage's output
states, the program that checkpoint names and the checkpoints it names as its
inputs' producers, and, for each of those producers that is a check, the claim
record in its output and its program. The claimer gives the proof the outputs
themselves, and the proof holds each to the hash its checkpoint commits to
before it reads it. They are not posted, since a challenger makes its own.

A contract compares the stated records with the submission when the claim
arrives ([the contract page](checks.md#on-submission), check 7). So no
contract and no challenger decodes a stage's output, and a claim whose
submission is not what its own checkpoints commit to is never accepted.

In the stand-in, the outputs travel with the proof in the open
(`AvailabilityProof::HostChecked { outputs }`), so the host can read them
again; a receipt takes them as private inputs.

The submission's `availability` lists each blob: where it is (height,
namespace, the layer's commitment), its length, what it holds (`contents`) and
its proof.

## What a challenge posts

`iris challenge post` posts the challenger's trace body in the namespace
`iris-chall`: the trace commitment of its own run of the stage, in the encoding
Raster's admission decodes (`raster_prover::challenge::admit_body`). The proof
states the digest of its header, which is what a Raster stage challenge names
the trace by. This is the body Raster's settlement design puts on the
data-availability layer (`raster/docs/proposals/settlement-layer.md`, section 3);
the challenge itself carries only the header.

`iris challenge post --challenge <challenge_bundle.json> --submission <claim_submission.json>`
builds the stage challenge from the claimer's checkpoint and the challenger's
replay, posts the body, writes `challenge_submission.json` beside the bundle,
and prints the checks a contract makes on arrival.

## What a contract checks

On a claim, at submission:

1. **Procedure.** The claim names a registered procedure; the contract
   judges it by that one.
2. **Posted, proven, retrievable.** Each blob is included in a data root
   Blobstream has attested to, its proof holds, it was posted for the address
   the claim names as its claimer, and its block's time plus the layer's
   retention reaches past the submission time plus the challenge
   period. The blobs hold the claim's checkpoint list. The run inputs they
   hold are recorded by one hash, and not judged: a stage input that was not
   posted is a fault of the checkpoint that names it.

On a challenge, at submission: it is about the checkpoint the claim lists at
that stage, the stage names an approved program, Raster admits it, and the
trace body is posted for the challenger it names, proven and retrievable past
the submission time plus the fault-proof period.

The parameters (`AvailabilityParameters`): retention, 30 days by default (to be
confirmed against the network a contract settles on); challenge period, 7 days;
fault-proof period, 2 days.

## The procedure registry

A procedure (`Procedure`) is everything a contract judges a claim by:
the procedure kind and its approved programs, one model per inference, the raw prompt
flag and token budget, the signer a source must be signed by, and the forms the
output is read by. Its id is a hash over all of it. `iris registry add --specification
<file> --registry <file>` registers the specification's procedure and prints the id;
`claim build` writes the id into the submission; `iris claim check --registry
<file>` judges a claim by the procedure it names there.

Not pinned yet, by decision: the base prompt and the domains a source's
documents may come from. A procedure accepts any recipe. The claim still
commits to the recipe it used, through the source record its statement names,
so pinning them later is a new field and a new check, not a change to what a
claim states.

## The stand-ins

| Real system | Stand-in |
| --- | --- |
| Celestia | A directory: `IRIS_DA_DIR`, or `~/.cache/iris/stub-celestia`. Each post makes one block. One directory is one network, so a claimer and a challenger working in different places see the same blobs. The parity gate uses its own, under its run directory |
| The blob commitment | `stub_blob_commitment`: SHA-256 over a domain, the namespace and the bytes. Celestia's is a Merkle root over the blob's shares |
| Blobstream | `blobstream.json` in the directory: each block's height, time and a data root over its blobs, all attested |
| The availability proof | `AvailabilityProof::HostChecked`: the host reads the blob and recomputes what it holds. A real one is a receipt of a program that does the same, verified against its pinned image id |

`IRIS_DA_NOW` fixes the time new blocks are stamped with, for tests.

With `IRIS_CELESTIA_RPC` set and no `--da-dir`, the first two rows are the
real thing: blobs go to a Celestia node and are named by Celestia's own
commitment. The last two rows stay stand-ins there, in another form: a trusted
relayer posts block roots to a contract, and a trusted attester signs what
each blob holds ([the blockchains](../blockchain/overview.md)).

## What is open

What is not built here, and every other known gap, is in [one list](open.md).
