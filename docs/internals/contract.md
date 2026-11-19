# What a contract holds and checks

No contract exists yet. This page describes what one would pin, what a claimer
would submit, and the checks it would make, as the code states them today in
`inference_artifacts::contract`. `iris claim check` runs those checks on the host.

The contract verifies no proof and runs no stage. It makes one statement about a
claim:

> A claimer says that a model I trust, run with the parameters I require,
> produced this plaintext from a committed input. I do not read the input. If
> nobody successfully challenges the claim within the window, I act on the
> plaintext.

A contract may require two more things of a claim, and then the statement grows
by them: that the input was built from documents a signer it names vouches for,
by a recipe it names, and that the plaintext is read by forms it names into a
result code. See [the authenticated prompt](../proposals/authenticated-prompt.md).

## What the contract pins

A contract judges a claim by the registered configuration the claim names
(`Configuration`, see [data availability and the registry](availability.md)):
the [pipeline](pipelines.md), that pipeline's stage programs, one model per
inference the pipeline makes, and the contract's own parameters. The first
three are the pins, a file, `contract_pins.json` (version 4):

| Field | Meaning |
| --- | --- |
| `pipeline` | The pipeline's name: `single` or `consensus` |
| `programs` | The program entry, below |
| `models` | One model entry per inference, in the order the pipeline runs them |

A model import writes the pins of the `single` pipeline over that model beside
the import's `manifest.json`. `iris claim pins --run <spec> --out <file>` writes
those of the run spec's pipeline over the models it names, from their imports and
the stage projects' locks as they are when it runs.

**The program entry** is true of the stage programs whatever model they run.
Two models of one family under one pipeline have the same one.

| Value | Where it comes from |
| --- | --- |
| The approved stage programs | The `program_commitment` each stage's `Raster.lock` holds at import time |
| Each program's chained inputs | The chain template: an input is chained when any stage of the program takes it `from` another stage |
| For each chained input, the programs that may produce it | The chain template |
| For each chained input, the empty values it may hold | The fixed externals the chain's first stages are given in its place |
| Each program's run inputs | The chain template: an input bound `{ run = ... }`, or to a file under the run placeholder directory. Not checked yet: kept for the step in which a claimer posts a run input on request ([availability](availability.md#open)) |

**A model entry** is true of one model.

| Value | Where it comes from |
| --- | --- |
| Model id | The import |
| The commitments the model id hashes, by role | The import |
| Range size | The import |

**The contract's own parameters** are the raw prompt flag and the token budget,
and what it requires of the claim's origin (`RequiredOrigin`):

| Value | Meaning |
| --- | --- |
| The source, or none | The signer's key, and optionally the recipe's hash. With none, the claim must name no source. A configuration made from a run spec does not pin the recipe: the base prompt and the documents' domains are not pinned yet |
| The forms | The texts the output is read by and the code of each, in order. May be empty |

`iris claim check` reads all of them from the run spec: the signer and the
recipe from the files its `[source]` table names, and the forms from the recipe.
A contract holds the forms themselves and not only their hash, because it takes
the result step on the submitted output itself (below).

Re-locking a stage changes its program id, so import again afterwards. An import
made where the stage projects are not locked writes no pins and says so.

A contract of a pipeline over several models pins the stage programs of that
pipeline's chain over those models: both models' own programs, `prompt-source`,
and the pipeline's result program in place of `claim-result`. Each model entry
is read from its import's manifest, and the program entry from the pipeline's
chain template over the imports.

`iris claim check --pins <file>` checks a claim against a pins file, and refuses
one that is of another pipeline, of other models or of the same models in another
order. Without `--pins`, a `single` claim is checked against its model's import's
file, and a claim of several models against pins made from the imports at that
moment, as `iris claim pins` would make them.

The approved programs are a set, not a map from position to program. Which stage
runs where is the stage schedule's to enforce, inside the programs: the first
stage verifies the schedule and every later one checks its own place in it.

## What a claim commits to

A claim commits to one checkpoint per stage, in order. A checkpoint is the
boundary a Raster run of that stage is bound to:

| Checkpoint field | Meaning |
| --- | --- |
| `stage` | The stage's name |
| `program_commitment` | The program id the stage project has locked: the program a Raster replay of the stage runs |
| `input_commitment` | SHA-256 of the stage's `input_manifest.json` |
| `input_sources` | For each input taken from an earlier stage, that stage's position in the list |
| `output_commitment` | The output's structural commitment |
| `output_sha256` | SHA-256 of the stage's `output.bin` |

`claim build` runs the stages natively and does not run Raster. It still commits
to what the boundary would be if every stage had run as Raster, so a challenger
can replay any one stage in Raster from the claimer's own boundary and show
where it leaves what the claimer committed for the next. No chain-wide Raster
commitment is involved.

`checkpoint_trace.json` holds the checkpoints. `checkpoints.txt` holds one
hash per checkpoint. The claim's commitment to all of them is the root of a
tree over those hashes, with their count: `checkpoints_root`. See
[the encodings](#the-encodings-a-contract-recomputes).

## What a claimer submits

`claim build` writes `claim_submission.json` beside `claim_bundle.json`:

| Field | Meaning |
| --- | --- |
| `pipeline` | The [pipeline](pipelines.md) the claim is of |
| `configuration` | The id of the registered configuration the claim is of |
| `claimer` | The address the claim is attributed to: `0x` and 40 hex digits, the claimer's choice (`claim build --claimer`). Every blob the claim posted names it |
| `input_text_sha256` | The claimer's commitment to the input it ran. The contract takes it as given. Every inference's statement is recomputed over it |
| `source_sha256` | The digest of the source record the text was built from, or 64 zeros. The statement names it |
| `source` | The source record itself, for a claim that has one: the text's hash and length, the recipe's hash, the count and chained hash of the documents' attested facts, the signer's key, and the fault, which must be zero |
| `inferences` | One entry per inference, in the order the pipeline runs them: one for `single`, two for `consensus`. Each holds the five values below |
| `output_text` | The inference's plaintext: what the contract acts on |
| `record.statement_sha256` | `Statement::digest()` over the input text hash, the source digest, model id, raw prompt flag, token budget and range size |
| `record.output_text_sha256` | SHA-256 of the plaintext's exact UTF-8 bytes |
| `record.valid` | The chain found nothing wrong with the inference |
| `result.code` | The code of the first form the plaintext is, or zero |
| `result.forms_sha256` | A hash chained over the forms the plaintext was compared with |
| `consensus` | For a `consensus` claim: `same_input`, `agree` and `code`, as its result stage states them. Absent for `single` |
| `checkpoint_count` | How many checkpoints the claim has |
| `checkpoints_root` | The claim's commitment to its checkpoint list: the count and the root of the tree over the checkpoint hashes |
| `availability` | The blobs the claimer posted: the new information its statement names (the source, or the prompt text of a run with none), and the checkpoint list with every checkpoint, each with where it is, what it holds and its availability proof ([availability](availability.md)) |

The three `record` values are the claim record an inference's last check states
in its output (`ClaimRecord` in `raster-stages/sequence-check/src/input.rs`). The
verdict is computed in the stage from every field that can reject a claim: the
model fault, the prompt fault and completeness, the check fault, the mismatch
count and the output fault. Only the schedule's last claimed range can state a
valid one.

The two `result` values are what the chain's last stage, `result`
(`raster-stages/claim-result`), states. It reads the last check's output and the
run's forms, and states the claim record it was given beside the code
(`ResultRecord` in `crates/source-core/src/forms.rs`). The record stays the last
check's: the verdict rests on the stage that scored the output, which is long
enough to replay, and not on the few steps after it.

The source record is what the chain's first stage, `source`, states. The first
bind takes it from that stage, holds the claimed text to it, and puts its digest
in the prompt context, from where it reaches the statement.

## On submission

A contract first finds the configuration the claim names, and judges the claim
by it. It then requires the submission to name that configuration's pipeline
and to hold what a claim of it does (`check_pipeline`): one inference for
`single`, and two with their agreement for `consensus`. Checks 1 and 2 are then
made of each inference, against the model the configuration pins for it
(`check_submission`).

1. **Statement matches.** Recompute the statement digest from the pinned model id,
   range size and required parameters with the claimed input hash and source
   digest, and compare it with the record's. This one equality covers the model,
   every parameter and the source the run names. Every inference's statement is
   recomputed over the submission's one input hash and source digest, so two
   inferences that both pass ran the same input.
2. **Output matches.** The plaintext hashes to the record's output text hash, and
   the valid flag is set. A claim with an invalid record in any inference is not
   accepted.
3. **Source matches** (`check_origin`). The submitted source record hashes to
   the digest the statement names. A contract that requires a source rejects a
   claim without the record, a record with a fault, another signer or another
   recipe. A contract that requires none rejects a claim that names one.
4. **Result matches.** The contract compares the submitted plaintext with its
   own forms, as the result stage does, and rejects a claim whose forms hash or
   code is not what that gives. The result stage is a step per form, so a
   contract can afford this, and it means a false result is refused at
   submission and needs no dispute. It does this for each inference.
5. **Agreement matches**, for a `consensus` claim. The contract reads both
   submitted plaintexts by its forms and works out whether they agree, as the
   consensus result stage does, and rejects a claim that states another
   agreement or another code. A claim in which the models disagree is accepted
   as that: the contract acts on the answer `agree: false`.
6. **Data posted** (`check_claim_availability`). Every blob the claim lists is
   included in a data root Blobstream has attested to, its availability proof
   holds, and it stays retrievable past the end of the challenge period. The
   blobs hold the new information the statement names, a source whose record
   has the statement's source digest or a prompt text with its input hash, and
   the claim's checkpoint list.
7. **Records are its checkpoints'** (`check_stated_records`). A claim ends in
   its pipeline's result stage: its last checkpoint's program must be that
   pipeline's result program. That checkpoint names, for each inference, the
   earlier checkpoint whose output it read as that inference's check (`check`
   in a `single` claim, `first` and `second` in a `consensus` one), whose
   program must be the check. The record submitted for an inference must be
   the one in that check's output, and what the last stage's output states,
   every record with its result and the agreement, must be what was
   submitted. The contract decodes none of it: the claim's availability proof
   reads the records out of those outputs, each held to the hash its
   checkpoint commits to, and states them
   ([availability](availability.md#what-the-proof-states-of-a-claims-records)).
   The record is held to the check and not only to the result stage after it,
   so the verdict rests on the stage that scored the output.

The record's fields other than the signer and the recipe are not the contract's
to judge. Which documents were fetched, and that the text is what the recipe
makes of them, is what the `source` stage establishes, and a claim whose source
checkpoint is false is disputed like any other stage.

## When a challenge arrives

A challenge is about one stage. The challenger presents the claim's checkpoint
list and the checkpoints it points at, and the contract holds each to what the
claim committed: the list to `checkpoints_root`, a checkpoint to its place in
the list. A challenger cannot state a program, a boundary or a producer of its
own. Two comparisons follow, with no replay. They are the same for a claim
with a source or forms: `source` and `result` are stages like the others, with
approved programs and chained inputs, so a challenger disputes them the same
way.

- **The stage's program is approved.** If the program id the claimer's own
  checkpoint names is in the approved set, the challenge is an ordinary disputed
  one and a defender must answer it. If it is not, or the checkpoint names no
  program, the claim is void.
- **The stage's inputs are earlier outputs.** A stage keeps its own store, so
  nothing inside it can tell where an input came from: a made-up input can carry
  the labels a real one does. Without this comparison a claimer could run every
  stage honestly on inputs no earlier stage produced, and no replay would show
  it. For one chained input of the stage, the commitment the stage loaded it by
  must be the output commitment of the checkpoint the stage's own checkpoint
  names as its producer, that checkpoint must be earlier in the list, and its
  program must be one that may produce the input. Or the input holds one of its
  pinned empty values. Otherwise the claim is void.

The second comparison reads the stage's `input_manifest.json`, held to the
checkpoint's `input_commitment`. It must be exactly the canonical encoding of
what it holds: a repeated name, an extra field or other spacing voids the claim,
so that no two readers can take one input list for different inputs.

Which earlier stage is the right producer is not the contract's to judge. Every
output says which step produced it, and the consuming stage checks that against
the schedule. The contract holds only that the input is a real earlier output of
the right kind.

## The encodings a contract recomputes

A contract has to hash a checkpoint it is shown, open one checkpoint of a
list, and compute a configuration's id when one is registered. None of that is
workable over JSON or a list of hex lines, so these three follow the scheme the
stages' own digests use (`Statement::digest`, `source_id`): a tag ending in a
zero byte, then the fields in order; numbers 32-bit little-endian; text and
lists after a 32-bit count; a hash as a byte 1 and its 32 bytes, or a byte 0
when the field is empty. Everything is SHA-256. The code is
`inference_artifacts::encoding`.

- **A checkpoint's hash** (`checkpoint_hash`): tag `iris/checkpoint/v1`, the
  stage name, the program id, the input manifest hash, the producers in name
  order (each a name and a position), and the two output commitments.
- **The checkpoint list** (`checkpoints_root`): a tree of depth 20 over the
  checkpoint hashes in order. A leaf is `SHA-256(0 || hash)`, a node
  `SHA-256(1 || left || right)`, an empty position 32 zero bytes. The
  commitment is the tag `iris/checkpoints/v1`, the count and the tree's root.
  One checkpoint opens at its position with 20 hashes
  (`verify_checkpoint_opening`); the host's `CheckpointList` makes the same
  comparison with the whole list in hand.
- **A configuration's id** (`Configuration::id`): tag `iris/configuration/v2`
  and every field in the order it is declared.

A stage's input manifest stays Raster's canonical JSON, which a contract
cannot read. The link comparison over it is for a receipt, not for contract
code ([the integration proposal](../proposals/blockchain-integration.md)).
Records are read out of stage outputs by the claim's availability proof, and a
contract only compares them.

## Disputing a stage's output

The two comparisons above void a claim with no replay. What is left is a stage
whose checkpoint names an approved program and real inputs, and an output the
challenger says that program does not produce from them. That is settled by
Raster's stage challenge: the challenger posts the trace commitment of its own
authenticated run of the stage and the output it ends in, and a defender either
proves a window that dismisses it or the claim is void when the window closes.

Raster's challenge and defence name a chain commitment and a stage of it, and
read three things of that stage: its program, its input manifest and the hash of
its output. A checkpoint holds exactly those. So **a dispute about a checkpoint
is a dispute about a Raster chain of one stage built from it**
(`dispute_commitment`), and the contract recomputes that commitment's digest from
the checkpoint it opened. The encoding is a plain concatenation:

```text
01                      one stage
len name                the stage name, length as a varint
20 program              32 bytes
20 input manifest       32 bytes
00                      no bindings
20 output hash          32 bytes
20 output structural    32 bytes
00                      no manifest digest
00                      no repeats
```

Its SHA-256 is the digest a Raster challenge or defence about that checkpoint
names. Two rules follow:

- **A challenge is about a checkpoint** when the chain commitment it carries is
  that checkpoint's. Raster's own admission then applies: the output it states
  must differ from the checkpoint's, and its trace must be committed at the
  protocol's window.
- **A proof dismisses a challenge only when it is about the same checkpoint**
  (`check_defence`, which applies Raster's `defend`). The proven window must be
  against the challenger's trace commitment, run the checkpoint's program on the
  checkpoint's inputs, and end in a divergence or, on an honest trace, in an
  output other than the one the challenger stated. A window proven over another
  program or other inputs dismisses nothing: any trace diverges from a run it is
  not a trace of.

`iris fault prove` holds its proof to the second rule and records the dispute
digest in `fault_bundle.json`.

Only a differing output is a matter for a replay (`settlement`). A replay runs
the claimer's own program on the claimer's own inputs, so it says nothing about
either. A claim that names another program is settled by the program
comparison. A claim that ran a stage on other inputs than the challenger did is
not settled by the challenger's own replay: the link comparison voids it when an
input taken from an earlier stage is not that stage's output, and otherwise the
stage's own checks judge the input, and what is left to dispute is whether the
claimed output follows from the claimer's inputs.

## Running the checks

```bash
cargo run --release -p iris-cli -- claim check \
  --run inference.toml \
  --submission target/staged-infer/runs/.../claim_submission.json
```

The command reads the checkpoint files, every stage's input list and the
outputs of each inference's last check and of the result stage from the
directory the submission is in, and the posted blobs from the
data-availability layer: the stand-in directory (`--da-dir`, default
`IRIS_DA_DIR` or `~/.cache/iris/stub-celestia`), or a Celestia node when
`IRIS_CELESTIA_RPC` is set and no directory is given. It makes the configuration and pipeline checks,
the submission checks, the availability check and the challenge-time
comparisons, the link comparison for every chained input of every stage. It
prints eight lines for a `single` claim and nine for a `consensus` one, which
has a statement line per inference. `--registry <file>` judges the claim by the
configuration it names in that registry; without it, by the one the run spec
and the pins describe. `--submitted-at <seconds>` sets when the claim is taken
to be submitted.
`claim build` names its configuration, prepares the blobs it has to post, and
makes the same checks on its own claim before it reports success, when the
imports have pins. It makes them against what the prepared blobs hold, and
posts nothing: `iris claim post` posts them, and the claim passes the
availability check from then on.

`iris challenge check --challenge <challenge_submission.json> --submission
<claim_submission.json>` makes the checks of a challenge the same way, on the
submission `iris challenge post` wrote.

The contracts on a local EVM chain make these checks too, and are tested
against these two commands case by case ([the chains](chains.md)).

Add `--raster-run <dir>` with a Raster chain run of the same claim to hold every
checkpoint to the boundary that run committed: the same stages in the same order,
each with Raster's program id, input manifest commitment, producers and output
commitments. The [parity gate](../workflows/test-parity.md) does this on every
honest case.

## Known gaps

These are open. None is closed by anything on this page.

- **A stage that cannot run.** A claimer can give a stage a run input of the
  wrong shape, so that the honest program stops before it reaches its end, and
  write any output for it. As Raster's stage challenge is designed, a trace that
  does not reach the program's end is refuted as a misstated one, so a challenger
  has nothing to win with. Read from the design, not run.
- **Two commitments to one output.** A checkpoint holds the hash of a stage's
  output bytes, which a dispute is about, and its structural commitment, which
  the link comparison uses. Raster's proof guest checks the output's bytes
  against both at the end of a stage and publishes only the byte hash. So a
  claimer can list a true byte hash beside the structural commitment of a made-up
  value, feed that value to a later stage, and pass the link comparison. Closing
  it needs a proof: either Raster publishes the structural commitment in the
  window's journal, or a proof of this repository's own ties the two.
- **Short stages.** A stage whose trace is shorter than a replay window cannot be
  disputed. The claimer chooses the input, so it can make one occur.
- **No real dispute has been run end to end against a native claim**, and no
  proof of a full-size stage has been produced. The dispute rules above are
  tested on Raster's own admission and defence functions with real traces and
  made-up journals, and on one real proof of a small fixture.
- **`challenge build` replays whatever differs.** It does not yet sort a
  divergence into the program comparison, the link comparison or a replay. A
  replay aimed at a checkpoint that ran other inputs fails safely, since the
  defender can refute it.

- **The signer's key is a run input.** The `source` stage checks attestations
  against the key the run gives it and states that key in its record. No stage
  input pins the key. The contract compares the key in the submitted record
  with the one it requires, which holds as long as the `source` checkpoint is
  true.
- **A small source is a short stage.** With documents of a few hundred bytes
  the `source` stage is under the trace length a replay needs, as some check
  stages are, and its checkpoint cannot be disputed.
- **`claim build` checks its own consensus claim against pins made at that
  moment.** A contract's are the file it was given. The two differ only if a
  stage was re-locked in between.
- **The result stage is short at every size.** It is three steps and one per
  form, so a false `result` checkpoint cannot be disputed by a replay. A
  challenger's recomputation still finds it. The contract does not rest on it:
  it takes the result on the submitted plaintext itself, and for a consensus
  the agreement on the two. A result step that
  grows past what a contract can recompute, such as one that weighs several
  answers, has to be a stage long enough to replay.
- **The source's host choices are not checked.** Which documents to fetch, and
  that they were the server's answer, rest on the signer. The signer today is
  the host itself.

## Not here

- Bonds, the clock that runs the challenge window, and verifying a defence.
  These are the contract's and the settlement layer's. What is posted, and for
  how long it must stay retrievable, is in [availability](availability.md).
- Any check of the input beyond its hash and the source it names. What a
  source record stands for is in `docs/proposals/authenticated-prompt.md`.
