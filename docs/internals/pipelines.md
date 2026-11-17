# Pipelines

A pipeline is a named arrangement of the parts a claim is made of. One exists
today, `single`. The name is in place so that others can be added beside it
without each command growing its own switch.

## What a pipeline arranges

| Part | What it is |
| --- | --- |
| Source | Optional. The `source` stage, which states what the prompt text was built from |
| Inferences | One or more. An inference is everything between the source and the result for one model: its binds, prompt checks, embeddings, layers and checks, ending in a check that states a claim record |
| Result program | The stage program that reads every inference's last check and states the pipeline's result. Each pipeline has its own |

| Pipeline | Inferences | Result program | Its result |
| --- | --- | --- | --- |
| `single` | 1 | `claim-result` | The code of the first form the output is, or zero |

See [Raster chain](raster-chain.md) for the stages and
[the contract](contract.md) for what is checked.

## How a chain is put together

A model import knows nothing of pipelines. It writes one inference's part of a
chain to its `Raster.toml`: the chain inputs and the stages from the first bind
to the last check. The first bind is given the constant source record, as every
later bind is, so the part on its own is a chain for a text taken as given, with
no result.

A pipeline makes the chain template from one part per model
(`Pipeline::chain_template`):

1. the `[chain]` table and each part's chain inputs;
2. the `source` stage;
3. each part's stages, in order, with its first bind taking the `source` stage's
   record in place of the constant;
4. the pipeline's result stage, reading each inference's last check.

`claim build` then fills the template in for one run, as before. A run whose
text is taken as given removes the `source` stage again and gives the first bind
back its constant. The inputs of the `source` and result stages are written in
the template as `{ run = "..." }`, which is no binding Raster reads: a template a
run has not filled in is not a chain.

The stage projects are found beside the model's own: the template names the
directory of `prompt-bind`, and `prompt-source` and the result program are its
neighbours.

### Labels

In a pipeline of one inference the stages keep the names an import gives them:
`prompt_bind_p0`, `check_c1`. In a pipeline of several, each inference's names
carry its label in front, `m0_` for the first model and `m1_` for the second:
its stages, its repeats, its chain inputs, what its stages take from each other,
and the marks a run replaces with its range counts. `source` and the result
stage have no label. The native runner reads a stage's kind through the label
(`unlabelled`).

Labels are applied when the template is made and tested there. No pipeline uses
them yet, and filling in a labelled template for a run is not built.

## Where the name is

| Place | Field | Notes |
| --- | --- | --- |
| Run spec | `pipeline = "single"` | Optional. A spec that names none runs `single`. A name that is no pipeline is an error |
| `prepared_run.json` | `pipeline` | The claim's frozen run, which its claim bundle points at. Required, version 19 |
| `claim_submission.json` | `pipeline` | Required, version 4 |
| `iris infer`, `iris claim build` | a `pipeline:` line | Printed before the result |

Every command that acts on a run takes the run spec, so naming the pipeline
there is naming it for all of them:

| Command | What it does with the name |
| --- | --- |
| `iris infer` | Runs the pipeline directly |
| `iris claim build` | Builds the pipeline's chain and records the name in the frozen run and the submission |
| `iris claim check` | Requires the submission to be of the spec's pipeline, and the claim's last stage to run that pipeline's result program |
| `iris challenge build` | Refuses a run spec that names another pipeline than the claim's, then rebuilds the claim's arrangement |

## What holds a claim to its pipeline

The name is a label a claimer writes, so nothing rests on it alone. A contract
accepts claims of one pipeline (`check_pipeline`), and then holds the shape: the
claim's last checkpoint must run that pipeline's result program
(`check_result_binding`), and every checkpoint must name a program in the
approved set. A claim built as one pipeline and labelled as another ends in the
wrong program.

## Adding a pipeline

In code a pipeline is a variant of `Pipeline`
(`crates/inference/artifacts/src/pipeline.rs`) with its name, how many
inferences it makes, its result program and its result stage's bindings. The
rest is not built: a run spec that names more than one model, run preparation
and direct inference for more than one model, filling in a labelled template,
contract pins for more than one model, and the result program itself.
