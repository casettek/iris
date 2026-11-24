# Artifact Formats

The stable format types and filenames live in `crates/inference/artifacts`.

## Model And Run Specs

`runtime/model-artifacts/<model-id>/manifest.json` records reusable model setup:

- `family`, the [model family](model-families.md) it is run as. A manifest without the field is `gemma-dense`, and one naming a family this build does not have is refused when it is loaded.
- The model's commitments by role and `model_id`, their hash.
- `range_size`, the largest range a run cuts its prompt or output into.
- Model, config, and tokenizer paths plus SHA-256 hashes.
- Model shape and fixed-point scaling parameters.
- EOS token ids.
- Optional provenance pointing back to the model's prompt-free segment of a chain, written during import. A [pipeline](pipelines.md) makes a chain template from it.

The same model-scoped directory owns the imported Raster externals under
`runtime/model-artifacts/<model-id>/raster/`. Stage program directories remain
under `raster-stages/*`; chain manifests bind those programs to explicit
external `path` / `index_path` entries in the model artifact directory.

Run-specific inputs are not model artifacts. The original text, the configuration,
each range's cut and each claimed range are written under
`target/iris/runs/<run-id>/prompt/`, and prompt-free chain templates use a
`target/iris/run-prompt-placeholder/` binding that run preparation replaces.

`inference.toml` records run-time inputs:

- `pipeline`: optional, `single` when absent. See [Pipelines](pipelines.md)
- `model_manifest`, or for a pipeline of several inferences one `[[model]]` table per model, each with a `manifest`
- Exactly one of `prompt` or `prompt_file`
- `raw_prompt`
- `tokens`

A spec may still say `chain = "teacher-forced"`. Any other `chain` is an error,
because teacher-forced is the only staged chain.

`model_manifest` must name a version 4 manifest. Older manifests are refused with
an instruction to re-import.

`prepared_run.json` (version 4) freezes the resolved prompt, rendered prompt, initial pieces, EOS ids, model manifest hash, generated run manifest path/hash, and token count for claim/challenge reproducibility.

## Checkpoints

A checkpoint trace is an ordered list of per-stage records:

- Stage name.
- The stage program's id, as its project's `Raster.lock` holds it. The native runner records it in the stage directory as `program.json`; a stage whose project is not locked names none.
- SHA-256 of `input_manifest.json`.
- For each input taken from an earlier stage, that stage's position in the trace. The native runner records it in `program.json` too.
- Output structural commitment from `output_manifest.json`.
- SHA-256 of `output.bin`.

Every checkpointed stage directory must include:

- `input.json`
- `input_manifest.json`
- `output.bin`
- `output.rindex`
- `output_manifest.json`

## Claims

`claim build` writes:

- `checkpoint_trace.json`
- `checkpoints.txt`
- `claim_bundle.json`
- `claim_submission.json`

The claim bundle records the first checkpoint input commitment, final checkpoint output commitment, checkpoint trace path, and prepared-run metadata path.

The claim submission is what a claimer would post to a [contract](contract.md): the pipeline, the input text hash, the source record of a run that has one, for each inference its output text, its last check's claim record (statement digest, output text hash, valid flag) and its result code, for a consensus the agreement, the claimer's address, and the root of the checkpoint list, which commits to every stage boundary. A model import's `contract_pins.json` is the other side: what a contract of the `single` pipeline over that model pins, as the pipeline's name, a program entry (the stage programs and the inputs each takes from earlier stages) and a model entry. `iris claim pins` writes the same file for any pipeline over the models a run spec names.

## Challenges

`challenge build` writes these files for a divergent stage:

- `divergence.json`
- `replay_package.json`
- `challenge_trace.json`
- `challenge_bundle.json`

The challenge bundle indexes the source trace, recomputed trace, divergence record, replay package, Raster commit path, and challenge trace.
