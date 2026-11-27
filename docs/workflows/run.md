# Run Workflow

`run` runs a procedure with no blockchain. It makes no claim and posts nothing.

```bash
cargo run --release -p iwebs-cli -- run --specification specification.toml --case case.toml
# or
just run specification.toml case.toml
```

It reads a specification and a case, as [`infer`](infer.md) does (`specification.toml` and `case.toml` when no flag names them).

Flow:

1. **The native run.** `procedure_run::run_native` runs the procedure as `infer` does: each inference generated in turn, then what the procedure states of them. The command prints the result here, before any stage runs.
2. **The pipeline run.** `procedure_run::run_pipeline` prepares a pipeline run on the output the native run generated and runs it stage by stage, as [`claim build`](claim-build.md) does. Each stage verifies its part of that output and leaves a checkpoint.
3. **The two results are held equal.** The command fails if the pipeline run states any field of any inference's result otherwise than the native run did, or another consensus. That is a failure of parity between generation and verification, and the error names the field.
4. **The checkpoints are written**: `checkpoint_list.json` and `checkpoints.txt` in the pipeline run's directory under `target/procedure-pipeline/runs/`.

Output: everything `infer` prints, then

```text
pipeline run: 87 stages, and the result the native run stated
checkpoint list: target/procedure-pipeline/runs/<run>/checkpoint_list.json
checkpoints: target/procedure-pipeline/runs/<run>/checkpoints.txt
```

## The procedure a run is of

A specification describes a procedure: its kind's stage programs, its models, its parameters and what it requires of the source and the result. The procedure's id is a hash over all of that, and it is the id a contract registers ([the registry](../protocol/availability.md#the-procedure-registry)). So two parties that name one id have agreed to one procedure, with a blockchain or without one.

```bash
cargo run --release -p iwebs-cli -- procedure id --specification specification.toml
# procedure: 2e26...b36c
```

`--json` prints the id with the whole procedure, which is what the other party reads to see what it is agreeing to.

`run` prints the same line after the pipeline run, once it has held every checkpoint's program to the ones the procedure approves:

```text
procedure: 2e26...b36c
```

`--procedure <id>` names the procedure the run must be of. The command fails before anything runs when the specification describes another, and after the pipeline run when a stage ran a program the procedure does not approve.

The line reads `procedure: none` with the reason in two cases. A model was imported where the stage projects are not locked, so its import wrote no `contract_pins.json` and its stage programs are not pinned. Or a stage was locked again after the model was imported, so the run's programs are not the ones the import pinned. Importing the model again from the repository root settles both.

The stage programs and models are read from the pins each model's import wrote, as [`claim check`](claim-build.md) reads them. The file is named for the contract that is configured with it; nothing in it needs one.

## The record of a run

A record is what the party that made a run hands to anyone who wants to make it again.

```bash
cargo run --release -p iwebs-cli -- run --specification specification.toml --case case.toml --record my-record
# record: /path/to/my-record
```

The directory holds:

| File | What it is |
| --- | --- |
| `record.json` | The id of the procedure the run is of, each inference's result, and the record's backings. |
| `procedure.json` | The procedure itself with its id, as a registry holds one: what the id is a hash of. |
| `posted_run.bin` | The inputs the run was given, as values: the text, a source's documents with their attestations, the forms, and the output each inference is held to. These are the inputs a claim posts. |
| `checkpoint_list.json`, `checkpoints.txt` | The run's checkpoints and their hashes: what the fault provable backing is of. |

A record has two parts. **What it states** is the procedure id, the inputs and the result: the same whatever method the run was made for. **Its backings** are what stands behind that besides making the run again, one entry for each method, named by its method:

```json
"backings": [
  { "method": "fault_provable", "checkpoints": 199, "checkpoints_root": "5b1e...90aa" }
]
```

| Method | What the backing holds | Built |
| --- | --- | --- |
| None: the list is empty | Nothing. The record is backed by making the run again | Read and checked; `run` does not write one |
| `fault_provable` | The count and root of the run's checkpoints, one for each stage, so that one stage can be proven wrong. What a claim commits to on a blockchain | Yes: `run --record` writes it |

A record holds at most one backing for each method, and a build refuses a record with a method it has no check for.

The question, the criteria and the evidence of a case are inputs of the run, so they are in `posted_run.bin` and not in the procedure's id. The model's weights are the procedure's: the record names them by the model id, and the checker has its own import. A record assumes every input is available to whoever checks it, and does nothing to make it so.

`run` writes a record only of a run it can name the procedure of. It fails, before anything runs, when [the procedure has no id](#the-procedure-a-run-is-of), and after the pipeline run when a stage ran a program the procedure does not approve.

### Checking a record

```bash
cargo run --release -p iwebs-cli -- record check --record their-record --specification specification.toml
```

The specification is the checker's own, naming its own imports of the models. Flow:

1. **The record is held to itself and to the checker's procedure.** The procedure in the record hashes to the id the record states, the checker's specification describes a procedure with that id, and, when the record has the fault provable backing, its checkpoint list has the count and root that backing states. The command fails here, before anything runs, when one does not hold.
2. **The run is made again on the record's inputs.** Every file a stage is given from the run is written from the record's own value. Nothing is generated: the stages hold the output the record states to the model's own choice, token by token.
3. **Each inference's result is compared with the one the record states**, and, when the record has the fault provable backing, every checkpoint with the record's, in order.

```text
procedure: bcef...2c46
pipeline run: 199 stages, on the record's inputs
checkpoint list: target/procedure-pipeline/runs/<run>/checkpoint_list.json
inference 0: valid: true, result code: 7
record: the same at every stage, and the result it states
```

A record that does not hold ends in one of these, with status 1:

```text
record: differs first at stage `prompt_bind_p0` (checkpoint 0): input_commitment
record: inference 0: the record states `result_code` as 8, and the run on its inputs as 7
```

A record with no backing ends in `record: the result it states; it carries no checkpoints to compare`.

A record that holds says what the procedure gives on those inputs, whoever ran it. Where one does not, the check names the first stage that differs, as `--compare` does, and settles nothing more.

## Comparing two runs

`--compare` names the `checkpoint_list.json` of another run of the same specification and case: someone else's, or a claim's. The two lists are compared in order, and the command exits with status 1 if they differ:

```text
compared with theirs/checkpoint_list.json: the same at every stage
compared with theirs/checkpoint_list.json: differs first at stage `range_c0_l1` (checkpoint 11): output_commitment
```

What differs is, in the order it is looked for: the stage's name, its program, its inputs, its output.

A checkpoint names the stage's program, so two runs agree only when they were made with the same stage programs (the same `Raster.lock` files) and the same import of the model.

This finds where two runs differ. It does not settle who is right: that is what a [challenge](challenge-build.md) and a fault proof are for.

## What a claim adds

`claim build` makes the same pipeline run and leaves the same checkpoints. The [parity gate](test-parity.md) holds the two equal. A claim adds what a blockchain needs: the claim's submission and bundle, the data it has to post, and the checks a contract makes. What it posts of its run is what a record holds: the same inputs, the same checkpoints, and the same procedure id. `claim build` does not write a record or read one yet.
