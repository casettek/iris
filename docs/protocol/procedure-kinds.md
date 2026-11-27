# Procedure kinds

A procedure kind is a named arrangement of the steps a procedure is made of. There are two
sets, told apart by name. A kind that verifies holds an output stated beforehand to each
model's own choice: `verify-single`, one inference on one model; `verify-consensus`, the
same inference on two models and whether they agree; and `verify-relay`, two inferences
with the first one's output inside the second one's prompt. A kind that generates finds
that output: `generate-single`, `generate-consensus` and `generate-relay`, each with the
same steps and a generation where the other has an inference. A generation runs natively
and has no stages, so it has no pipeline, no claim and no contract: everything on this page
about stages, pipelines, submissions and contracts is of the kinds that verify
([run](../workflows/run.md#two-sets-of-procedures)). Nothing pairs a kind of one set with
a kind of the other. Others can be added beside them without each command growing its own
switch.

`verify-consensus` and `verify-relay` each run directly, build a claim with a submission,
pass a contract's checks, can be challenged, and have a scenario in the parity
gate and a claim in the blockchain tier. The backend runs any kind: it passes
the kind's name on and holds no list of them.

## What a procedure kind arranges

| Segment | What it is |
| --- | --- |
| Source | A stage that states what a prompt's text was built from. `prompt-source` states it of signed documents, and is optional: a case with a prompt taken as given runs without it. `prompt-relay` states it of an earlier inference's output |
| Inferences | One or more. An inference is everything between the source and the result for one model: its binds, prompt checks, embeddings, layers and checks, ending in a check that states a claim record |
| Result program | The stage program that reads every inference's last check and states the procedure kind's result. Each procedure kind has its own |

| Procedure kind | Inferences | Result program | Its result |
| --- | --- | --- | --- |
| `verify-single` | 1 | `claim-result` | The code of the first form the output is, or zero |
| `verify-consensus` | 2, one after the other | `consensus-result` | Each output's code, and whether the two agree |
| `verify-relay` | 2, the second on the first one's output | `claim-result` | The code of the first form the second output is, or zero |

See [Raster pipeline](raster-pipeline.md) for the stages and
[the contract](checks.md) for what is checked.

## How a pipeline is put together

A model import knows nothing of procedure kinds. It writes one inference's segment of a
pipeline to its `Raster.toml`: the pipeline inputs and the stages from the first bind
to the last check. The first bind is given the constant source record, as every
later bind is, so the segment on its own is a pipeline for a text taken as given, with
no result.

A procedure kind makes the pipeline template from one segment per model
(`ProcedureKind::pipeline_template`). It follows the kind's shape, the steps its
file in `procedures/` lists:

1. the `[pipeline]` table and each segment's pipeline inputs;
2. then each step in order. A source step is one stage, named as the step
   is: `source`, or `relay`, which takes the last check of the inference its
   `check` input names. An inference step is its model's stages, with its
   first bind taking the record of the step its `source` input names in place
   of the constant. The result step is the kind's result stage, reading the
   last check of each inference its inputs name.

`claim build` then fills the template in for one run, as before. A run whose
text is taken as given removes the `source` stage again and gives the first bind
back its constant. The inputs of the `source` and result stages are written in
the template as `{ run = "..." }`, which is no binding Raster reads: a template a
run has not filled in is not a pipeline. The `relay` stage's recipe is written
the same way.

The stage projects are found beside the model's own: the template names the
directory of `prompt-bind`, and `prompt-source` and the result program are its
neighbours.

### Labels

In a procedure kind of one inference the stages keep the names an import gives them:
`prompt_bind_p0`, `check_c1`. In a procedure kind of several, each inference's names
carry its label in front, `m0_` for the first model and `m1_` for the second:
its stages, its repeats, its pipeline inputs, what its stages take from each other,
and the marks a run replaces with its range counts. `source` and the result
stage have no label. The native runner reads a stage's kind through the label
(`unlabelled`).

Run preparation fills in each inference's segment of the template from that
inference's run files, which are in a directory of their own under the run's
`prompt/` directory: `prompt/m0/`, `prompt/m1/`. A procedure kind of one inference
keeps its files in `prompt/`.

## Where the name is

| Place | Field | Notes |
| --- | --- | --- |
| Specification | `procedure_kind = "verify-consensus"` | Optional. A specification that names none is of `verify-single`. A procedure that finds its output names a `generate-` kind. A name that is no procedure kind is an error |
| `prepared_run.json` | `procedure_kind` | The claim's frozen run, which its claim bundle points at. Required, version 20 |
| `claim_submission.json` | `procedure_kind` | Required. It holds one entry per inference, for `verify-consensus` the agreement, and for `verify-relay` the relay's record |
| `iwebs claim build` | a `procedure_kind:` line | Printed before the result |

A specification names its models as `[[model]]` tables, in the order the
procedure kind runs them: one for `verify-single`, two for `verify-consensus`.

```toml
procedure_kind = "verify-consensus"
max_new_tokens = 16

[[model]]
manifest = "runtime/model-artifacts/gemma-4-E2B-q8/manifest.json"

[[model]]
manifest = "runtime/model-artifacts/ministral-3b-q8/manifest.json"
```

A `[source]` table and the forms follow, for a prompt built from signed
documents ([artifacts](artifacts.md#models-specifications-and-cases)).

`raw_prompt` and `max_new_tokens` are the run's, and the same for every
inference. So are the prompt and the source, except for an inference a relay
gives its prompt. Each model tokenizes its text with its own tokenizer and
chat template, so the inferences have different prompt tokens, range counts and
statements. In `verify-consensus` both statements name the same input text hash and
source digest.

A caller that runs procedures need hold no list of kinds. `iwebs procedure
list` names the kinds of this build, and `IWEBS procedure describe <kind>`
says what one is: its steps in order, how many models it runs, which
inference its result is of (`answer`), whether its result states an
agreement, which inference relays from which, and whether its specification
has a `[source]` or a `[relay]` table. Both print JSON with `--json`.

Every command that acts on a run takes the specification, so naming the procedure kind
there is naming it for all of them:

| Command | What it does with the name |
| --- | --- |
| `iwebs run` | Runs the procedure the specification describes: a kind that generates natively, each model in turn and then the result; a kind that verifies stage by stage, on the output its case states |
| `iwebs claim build` | Builds the procedure kind's pipeline and records the name in the frozen run and the submission |
| `iwebs claim pins` | Writes what a contract of the specification's procedure kind over its models pins |
| `iwebs claim check` | Requires the submission to be of the specification's procedure kind and to hold what a claim of it does, each inference to be of the model the specification names for it, and the claim's last stage to run that procedure kind's result program |
| `iwebs challenge build` | Refuses a specification that names another procedure kind than the claim's, then rebuilds the claim's arrangement from what the claim froze: the text, the source inputs, the forms and each inference's claimed ids |

## The consensus procedure kind

The pipeline is `source`, the first model's stages (`m0_...`), the second model's
(`m1_...`), then `result`. One checkpoint list covers all of it, so a challenger
recomputes it, finds the first checkpoint that differs and replays that stage,
as for any pipeline. Each inference is a whole claim of its own model: it has its
own statement and its last check states its own claim record.

`result` runs `consensus-result`. It reads both last checks and the run's forms
(`phases/result/src/consensus.rs`) and states a `ConsensusRecord`:

| Field | Meaning |
| --- | --- |
| `first`, `second` | For each inference, what the single result stage states: its claim record's three values, the code of the first form its output is, and the hash of the forms |
| `same_input` | Both checks' records name the same input text hash and the same source digest |
| `agree` | Both claim records are valid, `same_input`, and the outputs agree |
| `code` | The code both outputs have, when they agree and the run names forms. Zero otherwise |

The outputs agree when both are the same form: each is compared with the forms
as the single result stage compares one, without the whitespace around it, and
the two must have the same code, which is not zero. A run that names no forms
has nothing to read an output by, and there the outputs agree when they are the
same text without the whitespace around each. Two outputs that say the same
thing in different text do not agree: a model that wraps its answer in a code
fence matches no form.

A run in which the models disagree is still a claim. What makes it not one is
an invalid claim record in either inference: `claim build` refuses to write
such a claim, and a challenger rejects it on that record.

`iwebs run` on a generation and `iwebs claim build` print the answer first, then each
inference's result under an `inference:` line:

```text
procedure kind: consensus
consensus: agree            # or: disagree, or why there is no consensus
consensus code: 1
same input: true
inference: 0
...                         # the lines a single run prints
generated text:
{"verdict":"confirmed"}
inference: 1
...
```

Like the single result stage, `consensus-result` is a few steps and one per
form, so its checkpoint cannot be replayed at the challenge window
([short stages](../issues/short-stage-fault-window.md)).

## The relay procedure kind

The pipeline is `source`, the first model's stages (`m0_...`), `relay`, the
second model's (`m1_...`), then `result`.

```text
source (optional) -> first inference -> relay -> second inference -> result
```

The specification gives the text around the first output in a `[relay]`
table. The second prompt is `before`, the first inference's output exactly as
generated, `after`:

```toml
procedure_kind = "verify-relay"
max_new_tokens = 16

[[model]]
manifest = "runtime/model-artifacts/gemma-4-E2B-q8/manifest.json"

[[model]]
manifest = "runtime/model-artifacts/ministral-3b-q8/manifest.json"

[relay]
before = "A student answered: "
after = "\nIs that answer correct? Reply yes or no."
```

`relay` runs `prompt-relay` (`phases/source/src/relay.rs`). It reads the first
inference's last check and the recipe, and states a source record, the record
`prompt-source` states of signed documents:

| Field | In a relay's record |
| --- | --- |
| `text_sha256`, `text_total` | The hash and length of `before`, the output, `after` |
| `recipe_sha256` | A hash over `before` and `after`, each under its name. Where the text is cut into chunks changes nothing |
| `documents`, `documents_sha256` | One document, the first output. The hash is over the first inference's claim record: its statement and its output text hash |
| `signer` | Empty. No signer vouches for the output: the first inference's claim record does |
| `fault` | `FAULT_RELAYED` (10) when the first inference's claim record is not valid or is not of that output, `FAULT_RECIPE` when a chunk of the recipe is over its limit |

The second inference's first bind takes that record exactly as a first bind
takes the `source` stage's. It holds the text the run gives the second
inference to the record's hash and length, and the second statement names the
record's digest. So the second claim record says which first claim it was
built on, and no stage of the inference segment differs for a relay.

A first inference with no valid claim record gives a relay record with a
fault, which the second inference's bind publishes as a prompt fault. Nothing
aborts. A generation refuses such a run before the second inference, as it
refuses a source with a fault.

`result` runs `claim-result` on the second inference's last check. The first
output is in the run as the first inference's result, which has no code: the
result stage did not read it.

`iwebs run` and `iwebs claim build` print which inference the
result is of, then each inference's result under an `inference:` line, in
the order they were made:

```text
procedure kind: relay
result of inference: 1
inference: 0
...
generated text:
Paris
inference: 1
...
generated text:
Yes
```

Like `source` and `result`, `relay` is a few steps and one per chunk of the
recipe, so on a short recipe its checkpoint cannot be replayed at the
challenge window ([short stages](../issues/short-stage-fault-window.md)).

## What holds a claim to its procedure kind

The name is a label a claimer writes, so nothing rests on it alone. A contract
accepts claims of one procedure kind (`check_procedure_kind`), and then holds the shape: the
claim's last checkpoint must run that procedure kind's result program
(`check_result_binding`), and every checkpoint must name a program in the
approved set. A claim built as one procedure kind and labelled as another ends in the
wrong program.

### What a contract reads of a kind

A contract does not read the name either. A registered procedure states its
kind's arrangement (`Arrangement` in `procedure/definition/src/registry.rs`), and
the contract reads that. Registering a procedure of a new kind made of
existing segments needs no change to the contracts.

| Field | Meaning | `verify-single` | `verify-consensus` | `verify-relay` |
| --- | --- | --- | --- | --- |
| `result_program` | The stage program that states the result | `claim-result` | `consensus-result` | `claim-result` |
| `result_inputs` | For each inference, the result stage's input that reads its last check, or empty | `check` | `first`, `second` | empty, `check` |
| `agreement` | Whether the result states an agreement of two inferences | no | yes | no |
| `relayed_from` | For each inference, one more than the inference whose output its prompt holds, or zero | 0 | 0, 0 | 0, 1 |

There is one inference per model the procedure pins. The arrangement is made
from the kind's shape when a procedure is made (`Procedure::of`) and is part
of the procedure's id. The registry refuses one that says nothing of some
inference, relays from a later inference, or states an agreement of other
than two.

The checks IWEBS makes itself follow the kind's shape, which is compiled in.
So IWEBS takes no registered procedure whose arrangement is not its kind's in
this build (`Registry::get`).

### What a contract checks of a consensus claim

The submission holds both inferences' plaintexts, records and results, and the
agreement. A contract pins one model per inference and holds each statement to
its own model over the one input hash the submission names, so the two
inferences cannot be swapped and both ran the same input. It reads both
plaintexts by its own forms and refuses a claim whose codes or agreement are not
what that gives. When a challenge arrives it holds each record to the check the
result stage read for that inference. See [the contract](checks.md).

The parity gate's [consensus scenario](../workflows/test-parity.md#consensus-scenario)
runs one consensus claim over the gate's own model and the other family's.

### What a contract checks of a relay claim

The submission holds both inferences' plaintexts and records, the second
one's result, and the relay's record. A contract pins one model per inference
and the relay's recipe.

- The first statement is the first model over the input hash and source the
  submission names. The second is the second model over the relay record's
  text hash and the record's digest. So the two statements are over different
  inputs, and the second names the relay's record.
- The relay's record has no fault, is of one document, is built with the
  recipe the procedure pins, and its documents hash is over the first
  inference's claim record as submitted: its statement and its output text
  hash. A claim whose first output is not the one the second prompt was built
  on is refused here, with no stage replayed.
- The first inference has no result code, and the second one's is what the
  contract's forms give its plaintext.
- The result stage read the second inference's last check, and the second
  record is the one in that check. The first record needs no check of its own:
  the relay's record names it, the second statement names the relay's record,
  and the second record is held to its check.

A claim posts the relay's recipe with its other run inputs, so a challenger
runs the relay stage on what the claimer's did.

The parity gate's [relay scenario](../workflows/test-parity.md#relay-scenario)
builds one relay claim, and the blockchain tier submits one to the contracts.

## What is not built

For `verify-consensus`:

Nothing in IWEBS itself. `iwebs claim pins` writes a consensus contract's pins as
a file, and `iwebs claim check --pins` checks a claim against it.

For `verify-relay`:

- A second prompt that also reads the signed documents. It is fixed text
  around the first output.
- A replay of the `relay` stage on a short recipe: the stage is under the
  challenge window there, as `source` and `result` are.

## Adding a procedure kind

A procedure kind is a file, not code: `procedures/<name>.toml`, its shape.
The file lists the kind's steps in order. Each step is a use of a segment
kind (`source`, `inference`, `result`), with the stage program it runs where
its kind has several, and the earlier step each of its inputs comes from.
[`procedures/README.md`](../../procedures/README.md) has the format.

```toml
# procedures/consensus.toml
[[step]]
name = "source"
segment = "source"
program = "prompt-source"

[[step]]
name = "first"
segment = "inference"
inputs.source = "source"

[[step]]
name = "second"
segment = "inference"
inputs.source = "source"

[[step]]
name = "result"
segment = "result"
program = "consensus-result"
inputs.first = "first"
inputs.second = "second"
```

The files are compiled into `iwebs`, and a specification names one by its file
name. A shape is held to what the segment kinds declare when it is read
(`procedure/definition/src/shape.rs`): an input from a later step, a program
its segment does not have, or three inferences into the result of two is
refused there, before any run.

What a new kind needs depends on what it is made of:

| The kind uses | What to add |
| --- | --- |
| Existing segments and programs, arranged another way | Its file in `procedures/`, and its registration on the contract |
| A new program of an existing segment | The stage project under `raster-stages/`, its logic under `phases/`, its native stage in the segment's folder under `procedure/pipeline/src/segments/` with its entry in that segment's `Stage`, and the program's entry in `procedure/definition/src/segment_kind.rs`. A new result program also needs what `iwebs run` on a generation and `iwebs claim build` print for it and how a claim of it is judged when every checkpoint matches |
| A new segment kind | As a new program, with a folder of its own under `segments/` and the kind's entry in `segment_kind.rs` |

A registration on the contract is a call, not a change to it: the registered
procedure states the kind's arrangement
([what a contract reads of a kind](#what-a-contract-reads-of-a-kind)).

Two limits a shape is held to that are not the contracts': a run prepares one
source's inputs and one relay recipe, so a shape has at most one
`prompt-source` step and one `prompt-relay` step. And a result that states an
agreement is of two inferences.
