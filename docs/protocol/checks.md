# What a contract holds and checks

This page is the rules a contract holds a claim to: what it pins, what a
claimer submits, and each check it makes, as `blockchain_rules::checks`
states them in Rust. The contracts in `blockchain/contracts/src` make the same checks on
chain ([the blockchains](../blockchain/overview.md)), and `iris claim check` runs them on the host.
[How iris works](overview.md) is the short version.

The contract verifies no proof and runs no stage. It makes one statement about a
claim:

> A claimer says that a model I trust, run with the parameters I require,
> produced this plaintext from a committed input. I do not read the input. If
> nobody successfully challenges the claim within the window, I act on the
> plaintext.

A contract may require two more things of a claim, and then the statement grows
by them: that the input was built from documents a signer it names vouches for,
by a recipe it names, and that the plaintext is read by forms it names into a
result code. See [the design note](../proposals/authenticated-prompt.md).

## What the contract pins

A contract judges a claim by the registered procedure the claim names
(`Procedure`, see [data availability and the registry](availability.md)):
the [procedure kind](procedure-kinds.md), that procedure kind's stage programs, one model per
inference the procedure kind makes, and the contract's own parameters. The first
three are the pins, a file, `contract_pins.json` (version 5):

| Field | Meaning |
| --- | --- |
| `procedure_kind` | The procedure kind's name: `single` or `consensus` |
| `programs` | The program entry, below |
| `models` | One model entry per inference, in the order the procedure kind runs them |

A model import writes the pins of the `single` procedure kind over that model beside
the import's `manifest.json`. `iris claim pins --specification <file> --out <file>` writes
those of the specification's procedure kind over the models it names, from their imports and
the stage projects' locks as they are when it runs.

**The program entry** is true of the stage programs whatever model they run,
but for the pinned values of the inputs that hold a model's data. Two models
of one family under one procedure kind differ in those alone.

| Value | Where it comes from |
| --- | --- |
| The approved stage programs | The `program_commitment` each stage's `Raster.lock` holds at import time |
| Each program's chained inputs | The chain template: an input is chained when any stage of the program takes it `from` another stage |
| For each chained input, the programs that may produce it | The chain template |
| For each chained input, the empty values it may hold | The fixed externals the chain's first stages are given in its place |
| Each program's run inputs | The chain template: an input bound `{ run = ... }`, or to a file under the run placeholder directory. A checkpoint may name for such an input only a value the claim posted ([availability](availability.md#what-is-posted-and-what-is-not)) |
| Each program's pinned inputs, with the commitments each may hold | The chain template: an input bound to a file the import wrote, or to a chain input of such files, which lists one commitment per layer. A checkpoint may name nothing else for such an input ([the blockchains](../blockchain/overview.md#the-ways-to-challenge-a-claim)) |

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
| The source, or none | The signer's key, and optionally the recipe's hash. With none, the claim must name no source. A procedure made from a specification does not pin the recipe: the base prompt and the documents' domains are not pinned yet |
| The forms | The texts the output is read by and the code of each, in order. May be empty |

`iris claim check` reads all of them from the specification: the signer and the
recipe from its `[source]` table, and the forms from its `[[form]]` tables.
A contract holds the forms themselves and not only their hash, because it takes
the result step on the submitted output itself (below).

Re-locking a stage changes its program id, so import again afterwards. An import
made where the stage projects are not locked writes no pins and says so.

A contract of a procedure kind over several models pins the stage programs of that
procedure kind's chain over those models: both models' own programs, `prompt-source`,
and the procedure kind's result program in place of `claim-result`. Each model entry
is read from its import's manifest, and the program entry from the procedure kind's
chain template over the imports.

`iris claim check --pins <file>` checks a claim against a pins file, and refuses
one that is of another procedure kind, of other models or of the same models in another
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

`checkpoint_list.json` holds the checkpoints. `checkpoints.txt` holds one
hash per checkpoint. The claim's commitment to all of them is the root of a
tree over those hashes, with their count: `checkpoints_root`. See
[the encodings](#the-encodings-a-contract-recomputes).

## What a claimer submits

`claim build` writes `claim_submission.json` beside `claim_bundle.json`:

| Field | Meaning |
| --- | --- |
| `procedure_kind` | The [procedure kind](procedure-kinds.md) the claim is of |
| `procedure` | The id of the registered procedure the claim is of |
| `claimer` | The address the claim is attributed to: `0x` and 40 hex digits, the claimer's choice (`claim build --claimer`). Every blob the claim posted names it |
| `input_text_sha256` | The claimer's commitment to the input it ran. The contract takes it as given. The statement of every inference whose prompt is the claim's own is recomputed over it |
| `source_sha256` | The digest of the source record the text was built from, or 64 zeros. The statement names it |
| `source` | The source record itself, for a claim that has one: the text's hash and length, the recipe's hash, the count and chained hash of the documents' attested facts, the signer's key, and the fault, which must be zero |
| `relays` | For a claim of a kind that relays an output: the record each relay stage states, one per inference a relay gives its prompt. That inference's statement is recomputed over the record's text hash and digest. Absent otherwise |
| `inferences` | One entry per inference, in the order the procedure kind runs them: one for `single`, two for `consensus` and `relay`. Each holds the five values below |
| `output_text` | The inference's plaintext: what the contract acts on |
| `record.statement_sha256` | `Statement::digest()` over the input text hash, the source digest, model id, raw prompt flag, token budget and range size |
| `record.output_text_sha256` | SHA-256 of the plaintext's exact UTF-8 bytes |
| `record.valid` | The chain found nothing wrong with the inference |
| `result.code` | The code of the first form the plaintext is, or zero. Zero, with an empty forms hash, for an inference the result stage does not read |
| `result.forms_sha256` | A hash chained over the forms the plaintext was compared with |
| `consensus` | For a `consensus` claim: `same_input`, `agree` and `code`, as its result stage states them. Absent for `single` |
| `checkpoint_count` | How many checkpoints the claim has |
| `checkpoints_root` | The claim's commitment to its checkpoint list: the count and the root of the tree over the checkpoint hashes |
| `availability` | The blobs the claimer posted: the inputs its run was given, and the checkpoint hashes with every checkpoint, each with where it is, what it holds and its availability proof ([availability](availability.md)) |

The three `record` values are the claim record an inference's last check states
in its output (`ClaimRecord` in `raster-stages/sequence-check/src/input.rs`). The
verdict is computed in the stage from every field that can reject a claim: the
model fault, the prompt fault and completeness, the check fault, the mismatch
count and the output fault. Only the schedule's last claimed range can state a
valid one.

The two `result` values are what the chain's last stage, `result`
(`raster-stages/claim-result`), states. It reads the last check's output and the
run's forms, and states the claim record it was given beside the code
(`ResultRecord` in `phases/result/src/forms.rs`). The record stays the last
check's: the verdict rests on the stage that scored the output, which is long
enough to replay, and not on the few steps after it.

The source record is what the chain's first stage, `source`, states. The first
bind takes it from that stage, holds the claimed text to it, and puts its digest
in the prompt context, from where it reaches the statement.

## On submission

A contract first finds the procedure the claim names, and judges the claim
by it. It then requires the submission to name that procedure's procedure kind
and to hold what a claim of it does (`check_procedure_kind`): one inference for
`single`, two with their agreement for `consensus`, and two with the relay's
record for `relay`. A contract reads how many and which from the arrangement
the procedure states, not from the kind's name
([procedure kinds](procedure-kinds.md#what-a-contract-reads-of-a-kind)).
Checks 1 and 2 are then made of each inference, against the model the
procedure pins for it (`check_submission`).

1. **Statement matches.** Recompute the statement digest from the pinned model id,
   range size and required parameters with the claimed input hash and source
   digest, and compare it with the record's. This one equality covers the model,
   every parameter and the source the run names. An inference's statement is
   recomputed over the submission's one input hash and source digest, so two
   such inferences that both pass ran the same input. An inference a relay
   gives its prompt is recomputed over that relay's record in place of them:
   the text hash the record states, and the record's digest.
2. **Output matches.** The plaintext hashes to the record's output text hash, and
   the valid flag is set. A claim with an invalid record in any inference is not
   accepted.
3. **Source matches** (`check_origin`). The submitted source record hashes to
   the digest the statement names. A contract that requires a source rejects a
   claim without the record, a record with a fault, another signer or another
   recipe. A contract that requires none rejects a claim that names one.
   A relay's record is held too: it has no fault, is built with the relay
   recipe the procedure pins, and names the claim record of the inference
   whose output it relays, as submitted
   ([a relay claim](procedure-kinds.md#what-a-contract-checks-of-a-relay-claim)).
4. **Result matches.** The contract compares the submitted plaintext with its
   own forms, as the result stage does, and rejects a claim whose forms hash or
   code is not what that gives. The result stage is a step per form, so a
   contract can afford this, and it means a false result is refused at
   submission and needs no dispute. It does this for each inference the
   result stage reads. An inference it does not read has no result.
5. **Agreement matches**, for a `consensus` claim. The contract reads both
   submitted plaintexts by its forms and works out whether they agree, as the
   consensus result stage does, and rejects a claim that states another
   agreement or another code. A claim in which the models disagree is accepted
   as that: the contract acts on the answer `agree: false`.
6. **Data posted** (`check_claim_availability`). Every blob the claim lists is
   included in a data root Blobstream has attested to, its availability proof
   holds, and it stays retrievable past the end of the challenge period. The
   blobs hold the claim's checkpoint list. The run inputs they hold are
   recorded and not judged here: a stage input that was not posted is a fault
   of the checkpoint that names it, shown with no replay
   ([the blockchains](../blockchain/overview.md#the-ways-to-challenge-a-claim)).
7. **Records are its checkpoints'** (`check_stated_records`). A claim ends in
   its procedure kind's result stage: its last checkpoint's program must be that
   procedure kind's result program. That checkpoint names, for each inference, the
   earlier checkpoint whose output it read as that inference's check (`check`
   in a `single` claim, `first` and `second` in a `consensus` one, `check`
   for the second inference of a `relay` one), whose program must be the
   check. The record submitted for an inference must be
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
so that no two readers can take one input list for different inputs. Its names
and fields must also be plain text, printable ASCII with no quote and no
backslash: the EVM contract reads the list byte by byte and unescapes nothing.

Both comparisons are on the EVM contract as one call, `voidClaim`
([the blockchains](../blockchain/overview.md#the-ways-to-challenge-a-claim)).

Which earlier stage is the right producer is not the contract's to judge. Every
output says which step produced it, and the consuming stage checks that against
the schedule. The contract holds only that the input is a real earlier output of
the right kind.

## The encodings a contract recomputes

A contract has to hash a checkpoint it is shown, open one checkpoint of a
list, and compute a procedure's id when one is registered. None of that is
workable over JSON or a list of hex lines, so these three follow the scheme the
stages' own digests use (`Statement::digest`, `source_id`): a tag ending in a
zero byte, then the fields in order; numbers 32-bit little-endian; text and
lists after a 32-bit count; a hash as a byte 1 and its 32 bytes, or a byte 0
when the field is empty. Everything is SHA-256. The code is
`procedure_definition::encoding`.

- **A checkpoint's hash** (`checkpoint_hash`): tag `iris/checkpoint/v1`, the
  stage name, the program id, the input manifest hash, the producers in name
  order (each a name and a position), and the two output commitments.
- **The checkpoint list** (`checkpoints_root`): a tree of depth 20 over the
  checkpoint hashes in order. A leaf is `SHA-256(0 || hash)`, a node
  `SHA-256(1 || left || right)`, an empty position 32 zero bytes. The
  commitment is the tag `iris/checkpoints/v1`, the count and the tree's root.
  One checkpoint opens at its position with 20 hashes
  (`verify_checkpoint_opening`); the host's `CheckpointHashes` makes the same
  comparison with the whole list in hand.
- **A procedure's id** (`Procedure::id`): tag `iris/procedure/v4`
  and every field in the order it is declared, then the kind's arrangement
  (the result program, each inference's result input, whether the result
  states an agreement, each inference's relay) and the relay recipe the
  procedure pins.

A stage's input manifest stays Raster's canonical JSON, which a contract
cannot read as JSON. The contract reads it byte by byte in its one canonical
form ([the blockchains](../blockchain/overview.md#what-voidclaim-checks)).
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
  --specification specification.toml \
  --submission target/staged-infer/runs/.../claim_submission.json
```

The command reads the checkpoint files, every stage's input list and the
outputs of each inference's last check and of the result stage from the
directory the submission is in, and the posted blobs from the
data-availability layer: the stand-in directory (`--da-dir`, default
`IRIS_DA_DIR` or `~/.cache/iris/stub-celestia`), or a Celestia node when
`IRIS_CELESTIA_RPC` is set and no directory is given. It makes the procedure and procedure kind checks,
the submission checks, the availability check and the challenge-time
comparisons, the link comparison for every chained input of every stage. It
prints eight lines for a `single` claim and nine for a `consensus` one, which
has a statement line per inference. `--registry <file>` judges the claim by the
procedure it names in that registry; without it, by the one the specification
and the pins describe. `--submitted-at <seconds>` sets when the claim is taken
to be submitted.
`claim build` names its procedure, prepares the blobs it has to post, and
makes the same checks on its own claim before it reports success, when the
imports have pins. It makes them against what the prepared blobs hold, and
posts nothing: `iris claim post` posts them, and the claim passes the
availability check from then on.

`iris challenge check --challenge <challenge_submission.json> --submission
<claim_submission.json>` makes the checks of a challenge the same way, on the
submission `iris challenge post` wrote.

The contracts on a local EVM blockchain make these checks too, and are tested
against these two commands case by case ([the blockchains](../blockchain/overview.md)).

Add `--raster-run <dir>` with a Raster chain run of the same claim to hold every
checkpoint to the boundary that run committed: the same stages in the same order,
each with Raster's program id, input manifest commitment, producers and output
commitments. The [parity gate](../workflows/test-parity.md) does this on every
honest case.

## What is open

Every known gap in these rules and around them is in [one list](open.md).
