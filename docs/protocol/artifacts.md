# Artifact Formats

The stable format types and filenames live in `procedure/definition` and `blockchain/rules`.

## Models, Specifications And Cases

`runtime/model-artifacts/<model-id>/manifest.json` records reusable model setup:

- `family`, the [model family](model-families.md) it is run as. A manifest without the field is `gemma-dense`, and one naming a family this build does not have is refused when it is loaded.
- The model's commitments by role and `model_id`, their hash.
- `range_size`, the largest range a run cuts its prompt or output into.
- Model, config, and tokenizer paths plus SHA-256 hashes.
- Model shape and fixed-point scaling parameters.
- EOS token ids.
- Optional provenance pointing back to the model's prompt-free segment of a pipeline, written during import. A [procedure kind](procedure-kinds.md) makes a pipeline template from it.

The same model-scoped directory owns the imported Raster externals under
`runtime/model-artifacts/<model-id>/raster/`. Stage program directories remain
under `raster-stages/*`; pipeline manifests bind those programs to explicit
external `path` / `index_path` entries in the model artifact directory.

Run-specific inputs are not model artifacts. The original text, the configuration,
each range's cut and each claimed range are written under
`target/iris/runs/<run-id>/prompt/`, and prompt-free pipeline templates use a
`target/iris/run-prompt-placeholder/` binding that run preparation replaces.

A run is described by two files. `specification.toml` defines the procedure:

- `procedure_kind`: optional, `single` when absent. See [Procedure kinds](procedure-kinds.md)
- `raw_prompt`
- `max_new_tokens`
- One `[[model]]` table per model, in the order the procedure kind runs them, each with a `manifest`
- `[source]`: optional. The recipe a prompt is built by from signed documents, and the signer's key
- `[[form]]`: optional, only with a `[source]`. The texts the output is read by, each with its `code`
- `[relay]`: in a specification of a procedure kind that relays an output, and only there. The text `before` and `after` the earlier inference's output in the prompt of the next. See [the relay procedure kind](procedure-kinds.md#the-relay-procedure-kind)

```toml
procedure_kind = "consensus"
raw_prompt = false
max_new_tokens = 4

[[model]]
manifest = "runtime/model-artifacts/gemma-4-E2B-q8/manifest.json"

[[model]]
manifest = "runtime/model-artifacts/ministral-3b-q8/manifest.json"

[source]
base = "Is mag 5 or more?\n"
tail = "\n"

[source.signer]
scheme = 1
key = "02989c0b76cb563971fdc9bef31ec06c3560f3249d6ee9e5d83c57625596e05f6f"

[[source.document]]
server = "example.org"
target = "/event/us1.json"

[[source.field]]
lead = "id: "
document = 0
path = ["id"]

[[source.field]]
lead = "\nlink: "
path = ["properties", "products", "losspager", 1, "contents", "comments.json", "url"]

[[form]]
text = "yes"
code = 1
```

A field's `document` is which of the source's documents its value is in, the
first when absent. In its `path` a quoted step is a key and a bare number an
index into an array. `[source.signer]` holds the two values of the
`signer.json` that `iris attest keygen` writes.

`case.toml` holds one case's own inputs. For a specification with no source,
exactly one of `prompt` or `prompt_file`:

```toml
prompt = "say ciao"
```

For one with a source, a `[[document]]` table per document, in the order the
source names them:

```toml
[[document]]
file = "event.bin"
attestation = "event.attestation.json"
```

Paths in a specification are relative to the specification, and paths in a
case to the case. A key neither file knows is refused. Stage inputs are built
from the values the files hold and not from their bytes, so two files that
say the same thing give the same commitments.

A `[[model]]` table's `manifest` must name a version 4 manifest. Older manifests are refused with
an instruction to re-import.

`prepared_run.json` (version 4) freezes the resolved prompt, rendered prompt, initial pieces, EOS ids, model manifest hash, generated run manifest path/hash, and token count for claim/challenge reproducibility.

## Checkpoints

A checkpoint list is an ordered list of per-stage records:

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

- `checkpoint_list.json`
- `checkpoints.txt`
- `claim_bundle.json`
- `claim_submission.json`

The claim bundle records the first checkpoint input commitment, final checkpoint output commitment, checkpoint list path, and prepared-run metadata path.

The claim submission is what a claimer would post to a [contract](checks.md): the procedure kind, the input text hash, the source record of a run that has one, for each inference its output text, its last check's claim record (statement digest, output text hash, valid flag) and its result code, for a consensus the agreement, the claimer's address, and the root of the checkpoint list, which commits to every stage boundary. A model import's `contract_pins.json` is the other side: what a contract of the `single` procedure kind over that model pins, as the procedure kind's name, a program entry (the stage programs and the inputs each takes from earlier stages) and a model entry. `iris claim pins` writes the same file for any procedure kind over the models a specification names.

## Challenges

`challenge build` writes these files for a divergent stage:

- `divergence.json`
- `replay_package.json`
- `challenge_trace.json`
- `challenge_bundle.json`

The challenge bundle indexes the source trace, recomputed trace, divergence record, replay package, Raster commit path, and challenge trace.
