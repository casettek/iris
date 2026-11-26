# Procedure kinds

A procedure kind is a named arrangement of the segments a claim is made of. Three exist:
`single`, one inference on one model; `consensus`, the same inference on
two models and whether they agree; and `relay`, two inferences with the first
one's output inside the second one's prompt. Others can be added beside them
without each command growing its own switch.

`consensus` runs directly, builds a claim with a submission, passes a
contract's checks, can be challenged, and has a scenario in the parity gate. The
backend runs `single` and `consensus`.

`relay` runs directly and as a pipeline (`iris infer`, `iris run`). No claim
is made of one yet: see [what is not built](#what-is-not-built).

## What a procedure kind arranges

| Segment | What it is |
| --- | --- |
| Source | A stage that states what a prompt's text was built from. `prompt-source` states it of signed documents, and is optional: a case with a prompt taken as given runs without it. `prompt-relay` states it of an earlier inference's output |
| Inferences | One or more. An inference is everything between the source and the result for one model: its binds, prompt checks, embeddings, layers and checks, ending in a check that states a claim record |
| Result program | The stage program that reads every inference's last check and states the procedure kind's result. Each procedure kind has its own |

| Procedure kind | Inferences | Result program | Its result |
| --- | --- | --- | --- |
| `single` | 1 | `claim-result` | The code of the first form the output is, or zero |
| `consensus` | 2, one after the other | `consensus-result` | Each output's code, and whether the two agree |
| `relay` | 2, the second on the first one's output | `claim-result` | The code of the first form the second output is, or zero |

See [Raster chain](raster-chain.md) for the stages and
[the contract](checks.md) for what is checked.

## How a chain is put together

A model import knows nothing of procedure kinds. It writes one inference's segment of a
chain to its `Raster.toml`: the chain inputs and the stages from the first bind
to the last check. The first bind is given the constant source record, as every
later bind is, so the segment on its own is a chain for a text taken as given, with
no result.

A procedure kind makes the chain template from one segment per model
(`ProcedureKind::chain_template`). It follows the kind's shape, the steps its
file in `procedures/` lists:

1. the `[chain]` table and each segment's chain inputs;
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
run has not filled in is not a chain. The `relay` stage's recipe is written
the same way.

The stage projects are found beside the model's own: the template names the
directory of `prompt-bind`, and `prompt-source` and the result program are its
neighbours.

### Labels

In a procedure kind of one inference the stages keep the names an import gives them:
`prompt_bind_p0`, `check_c1`. In a procedure kind of several, each inference's names
carry its label in front, `m0_` for the first model and `m1_` for the second:
its stages, its repeats, its chain inputs, what its stages take from each other,
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
| Specification | `procedure_kind = "consensus"` | Optional. A specification that names none is of `single`. A name that is no procedure kind is an error |
| `prepared_run.json` | `procedure_kind` | The claim's frozen run, which its claim bundle points at. Required, version 20 |
| `claim_submission.json` | `procedure_kind` | Required, version 5. It holds one entry per inference, and for `consensus` the agreement |
| `iris infer`, `iris claim build` | a `procedure_kind:` line | Printed before the result |

A specification names its models as `[[model]]` tables, in the order the
procedure kind runs them: one for `single`, two for `consensus`.

```toml
procedure_kind = "consensus"
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
statements. In `consensus` both statements name the same input text hash and
source digest.

Every command that acts on a run takes the specification, so naming the procedure kind
there is naming it for all of them:

| Command | What it does with the name |
| --- | --- |
| `iris infer` | Runs the procedure kind directly: each inference in turn, then the result |
| `iris claim build` | Builds the procedure kind's chain and records the name in the frozen run and the submission |
| `iris claim pins` | Writes what a contract of the specification's procedure kind over its models pins |
| `iris claim check` | Requires the submission to be of the specification's procedure kind and to hold what a claim of it does, each inference to be of the model the specification names for it, and the claim's last stage to run that procedure kind's result program |
| `iris challenge build` | Refuses a specification that names another procedure kind than the claim's, then rebuilds the claim's arrangement from what the claim froze: the text, the source inputs, the forms and each inference's claimed ids |

## The consensus procedure kind

The chain is `source`, the first model's stages (`m0_...`), the second model's
(`m1_...`), then `result`. One checkpoint list covers all of it, so a challenger
recomputes it, finds the first checkpoint that differs and replays that stage,
as for any chain. Each inference is a whole claim of its own model: it has its
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

`iris infer` and `iris claim build` print the answer first, then each
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
([short stages](../issues/short-stage-fraud-window.md)).

## The relay procedure kind

The chain is `source`, the first model's stages (`m0_...`), `relay`, the
second model's (`m1_...`), then `result`.

```text
source (optional) -> first inference -> relay -> second inference -> result
```

The specification gives the text around the first output in a `[relay]`
table. The second prompt is `before`, the first inference's output exactly as
generated, `after`:

```toml
procedure_kind = "relay"
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
aborts. `iris infer` refuses such a run before the second inference, as it
refuses a source with a fault.

`result` runs `claim-result` on the second inference's last check. The first
output is in the run as the first inference's result, which has no code: the
result stage did not read it.

`iris infer` and `iris run` print each inference's result under an
`inference:` line, in the order they were made.

Like `source` and `result`, `relay` is a few steps and one per chunk of the
recipe, so on a short recipe its checkpoint cannot be replayed at the
challenge window ([short stages](../issues/short-stage-fraud-window.md)).

## What holds a claim to its procedure kind

The name is a label a claimer writes, so nothing rests on it alone. A contract
accepts claims of one procedure kind (`check_procedure_kind`), and then holds the shape: the
claim's last checkpoint must run that procedure kind's result program
(`check_result_binding`), and every checkpoint must name a program in the
approved set. A claim built as one procedure kind and labelled as another ends in the
wrong program.

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

## What is not built

For `consensus`:

Nothing in iris itself. `iris claim pins` writes a consensus contract's pins as
a file, and `iris claim check --pins` checks a claim against it.

For `relay`:

- A claim. `iris claim build` refuses a relay, because three things a claim
  needs are not written: the relay recipe among the values a claim posts, the
  rebuilding of a relay run from a claim, and the rules a contract holds a
  relay's submission to. Its two inferences name different inputs, and the
  rules hold every statement of a claim to one.
- A contract's pin of the relay recipe. The second statement names the recipe
  through the relay's record, so a claim commits to it, and nothing holds it
  to the recipe a procedure registered.

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

The files are compiled into `iris`, and a specification names one by its file
name. A shape is held to what the segment kinds declare when it is read
(`procedure/definition/src/shape.rs`): an input from a later step, a program
its segment does not have, or three inferences into the result of two is
refused there, before any run.

What a new kind needs depends on what it is made of:

| The kind uses | What to add |
| --- | --- |
| Existing segments and programs, arranged another way | Its file in `procedures/`, and its registration on the contract |
| A new program of an existing segment | The stage project under `raster-stages/`, its logic under `phases/`, its native stage in the segment's folder under `procedure/pipeline/src/segments/` with its entry in that segment's `Stage`, and the program's entry in `procedure/definition/src/segment_kind.rs`. A new result program also needs what `iris infer` and `iris claim build` print for it and how a claim of it is judged when every checkpoint matches |
| A new segment kind | As a new program, with a folder of its own under `segments/` and the kind's entry in `segment_kind.rs` |

Not yet read from the shape: a contract tells `consensus` from `single` by the
kind's name (`IrisClaims.sol`). A run prepares one source's inputs and one
relay recipe, so a shape has at most one `prompt-source` step and one
`prompt-relay` step.
