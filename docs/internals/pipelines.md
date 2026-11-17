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
(`crates/inference/artifacts/src/pipeline.rs`) with its name and its result
program. The rest is not built: a run spec that names more than one model,
stage names that say which inference a stage belongs to, a chain assembled from
per-model parts, contract pins for more than one model, and the result program
itself.
