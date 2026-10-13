# Artifact Contracts

The stable contract types and filenames live in `crates/inference/artifacts`.

## Model And Run Specs

`runtime/model-artifacts/<model-id>/manifest.json` records reusable model setup:

- Model, config, and tokenizer paths plus SHA-256 hashes.
- Model shape and fixed-point scaling parameters.
- EOS token ids.
- Optional provenance pointing back to the prompt-free Raster manifest template used during import.

The same model-scoped directory owns the imported Raster externals under
`runtime/model-artifacts/<model-id>/raster/`. Stage program directories remain
under `raster-stages/*`; chain manifests bind those programs to explicit
external `path` / `index_path` entries in the model artifact directory.

Prompt-specific inputs are not model artifacts. `initial_pieces` is generated
from each run's prompt under `target/raster-inference/runs/<run-id>/prompt/`,
and prompt-free chain templates use a `target/raster-inference/run-prompt-placeholder/`
binding that run preparation replaces. The standalone prompt-prepare fixture
can point at `runtime/prompt-fixtures/default/` for local stage smoke tests.

`inference.toml` records run-time inputs:

- `model_manifest`
- Exactly one of `prompt` or `prompt_file`
- `raw_prompt`
- `tokens`

`prepared_run.json` freezes the resolved prompt, rendered prompt, initial pieces, EOS ids, model manifest hash, generated run manifest path/hash, and token count for claim/challenge reproducibility.

## Checkpoints

A checkpoint trace is an ordered list of per-stage records:

- Stage name.
- SHA-256 of `input_manifest.json`.
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

The claim bundle records the first checkpoint input commitment, final checkpoint output commitment, checkpoint trace path, and prepared-run metadata path.

## Challenges

`challenge build` writes these files for a divergent stage:

- `divergence.json`
- `replay_package.json`
- `challenge_trace.json`
- `challenge_bundle.json`

The challenge bundle indexes the source trace, recomputed trace, divergence record, replay package, Raster commit path, and challenge trace.
