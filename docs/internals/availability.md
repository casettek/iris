# Data availability and the configuration registry

A claim names its inputs and its checkpoints by commitment. A challenger
cannot dispute what it cannot get, so a contract accepts a claim only with its
data posted to the data-availability layer (Celestia), and a challenge only
with its trace body posted. A claim also names, by id, the registered
configuration it is a claim of.

The rules are in `inference_artifacts::availability` and
`inference_artifacts::registry`; [the contract page](contract.md) lists the
checks in the order a contract makes them. The layer, Blobstream and the
availability proof are stand-ins run by the host
(`crates/inference/cli/src/availability.rs`), behind one trait, so that a later
integration with real systems replaces them and nothing else.

## What is posted, and what is not

A challenger reruns everything itself. It has the configuration and the
models, so it needs only what was given to the pipeline that neither fixes:
**the new information**. Everything a stage loads is made from it, the same
way by anyone.

| Given to the pipeline | Where a challenger gets it |
| --- | --- |
| A source's documents, with their attestations | Posted |
| A source's recipe (base prompt, documents, fields) | Posted, until configurations pin it |
| The prompt text of a run with no source | Posted |
| The signer's key, the forms, the raw prompt flag and token budget | The configuration |
| A sourced run's prompt text | Made by the source step from the documents and the recipe |
| Each prompt range's text and token marks, the run's configuration, schedule and text claim | Made from the text and the model's tokenizer |
| The claimed output ids | Made by greedy decoding: a claim whose ids are not the greedy output is refuted by its own record |
| Model data | Pinned by the model id, published elsewhere |
| Every Raster file's index | Made with the file |

The claimer also posts what only it can state: **its checkpoints**, the
checkpoint list and every checkpoint in it. A challenger finds the first
checkpoint it disagrees with by hash, and a challenge is built from the
claimer's checkpoint at that position. That checkpoint fixes which step the
position is: its stage, program, producers and the hash of its input manifest.
Without it, a challenger could dispute a position with an honest run of
another step. `aux_c0_l3` and `aux_c0_l5`, for instance, run one program on
one chained input and differ only in which layer's weights they load, so a
challenger's honest `aux_c0_l5` would differ from the claimer's honest
`aux_c0_l3` with nothing for the claimer to refute.

The input manifests are not posted. An honest claimer's are the ones a
challenger makes, and a replay is about a checkpoint whose manifest hash is the
challenger's own, so the challenger brings that manifest itself. Only a
dispute about a stage's inputs needs the claimer's manifest at that stage,
which joins the step set aside below ([Open](#open)).

## What a claim posts

`claim build` posts, in the namespace `iris-claim`:

| Entry | What the proof states of it |
| --- | --- |
| The source: the signer's key, the recipe and the documents with their attestations | The digest of the record the source step gives over them (`source_core::run_source`): the one the claim's statement names. One equality covers the text, the recipe, every document and the signer |
| Or, for a run with no source, the prompt text as given | Its SHA-256: the statement's input text hash |
| The checkpoint list, with every checkpoint | The SHA-256 of the list, which is the submission's `checkpoints_sha256`, and that every listed checkpoint is there |

The signer's key is the configuration's; it is posted only so that the source
step can be run over the blob alone.

Entries are packed in order into blobs of at most 1 MiB, a stand-in for the
layer's own limit. A blob is `iris/blob/v1\n` followed by the postcard encoding
of its entries; a reader refuses any other encoding of the same entries.

On the real-model check's E2B claim (a 1,209-token prompt with no source, 292
checkpoints) the parts are about 6 KB of prompt text, a 19 KB list and 142 KB
of checkpoints, measured on 2026-11-18. Earlier versions also posted the input
manifests (205 KB) and every Raster run file with its index (754 KB).

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

1. **Configuration.** The claim names a registered configuration; the contract
   judges it by that one.
2. **Posted, proven, retrievable.** Each blob is included in a data root
   Blobstream has attested to, its proof holds, and its block's time plus the
   layer's retention reaches past the submission time plus the challenge
   period. The blobs hold the new information the statement names (a source
   whose record has the statement's source digest, or a prompt text with the
   statement's input hash) and the claim's checkpoint list.

On a challenge, at submission: it is about the checkpoint the claim lists at
that stage, the stage names an approved program, Raster admits it, and the
trace body is posted, proven and retrievable past the submission time plus the
fault-proof window.

The parameters (`AvailabilityParameters`): retention, 30 days by default (to be
confirmed against the network a contract settles on); challenge period, 7 days;
fault-proof window, 2 days.

## The configuration registry

A configuration (`Configuration`) is everything a contract judges a claim by:
the pipeline and its approved programs, one model per inference, the raw prompt
flag and token budget, the signer a source must be signed by, and the forms the
output is read by. Its id is a hash over all of it. `iris registry add --run
<spec> --registry <file>` registers the run's configuration and prints the id;
`claim build` writes the id into the submission; `iris claim check --registry
<file>` judges a claim by the configuration it names there.

Not pinned yet, by decision: the base prompt and the domains a source's
documents may come from. A configuration accepts any recipe. The claim still
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

## Open

- **A stage input a challenger cannot make the same.** A claimer can feed a
  stage a run input other than the one the new information makes, such as
  token marks that are not the tokenizer's, and state a false checkpoint for
  that stage. The challenger finds another input at that stage and has no
  replay to make, since a replay starts from the claimer's input. Set aside
  for now by decision. The step planned for it: the challenger asks for that
  input, and the claimer must post it, with a proof that it has the Raster
  root the stage's input manifest names, within a window or the claim is void.
  The same step covers a stage whose input commitment differs from the
  challenger's for any reason, such as an input no earlier stage produced: the
  link check needs the claimer's manifest at that stage, which only the
  claimer has, so it posts it on request.
  The contract pins already list which inputs of each program are run inputs.
- **The proof program.** The stand-in proof runs host code (the source step,
  the checkpoint package check, and `raster_prover`'s trace header). A receipt
  needs that code in a guest-buildable crate, as Raster's settlement proposal
  plans for its body guest. The source step is already `no_std`, in
  `source-core`.
- **Two commitments to one output** (see [the contract page](contract.md)) is
  set aside for now and unchanged by this work.
