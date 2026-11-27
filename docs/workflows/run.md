# Run Workflow

`run` runs a procedure with no blockchain. It makes no claim and posts nothing.

```bash
cargo run --release -p iwebs-cli -- run --specification specification.toml --case case.toml
# or
just run specification.toml case.toml
```

It reads a specification and a case (`specification.toml` and `case.toml` when no flag names them). The specification names a procedure kind, and the kind says which of two things the run is.

## Two sets of procedures

A procedure finds an output, or it verifies an output that was stated beforehand. They are different logic, so they are different procedures: each has its own kinds, told apart by name, and its own id.

| Set | Kinds | What a run does | How it runs |
| --- | --- | --- | --- |
| Generation | `generate-single`, `generate-consensus`, `generate-relay` | Runs each model forward and takes each token as the model's own choice | Natively. It has no stages and leaves no checkpoints |
| Verification | `verify-single`, `verify-consensus`, `verify-relay` | Holds an output its case states to each model's own choice, token by token | Stage by stage, with a checkpoint for each stage: as native stages here, or as Raster programs |

Nothing pairs a generation procedure with a verification procedure. Whoever needs an output verified runs the first, states what it found in the case, and runs the second. A verification does not ask where its output came from.

## A generation

```bash
cargo run --release -p iwebs-cli -- run --specification generate.toml --case case.toml
```

`procedure_run::run_native` makes each model's inference in turn, then what the procedure states of them: for a consensus, whether the two agree. The command prints each result, with the ids of the output's tokens:

```text
procedure: 0cbc...5435
generated token ids: [4443, 106]
...
generated text:
yes
```

The procedure's id is printed first, so that the output ends in the generated text. With `--record`, the `record:` line is on stderr for the same reason.

## A verification

The case states the output each model is held to, as one `[[output]]` table for each model, in the order the kind runs them. The ids are the ones a generation printed, with the token that ends the output:

```toml
[[document]]
file = "event.json"
attestation = "event.attestation.json"

[[output]]
ids = [4443, 106]
```

```bash
cargo run --release -p iwebs-cli -- run --specification verify.toml --case case-with-output.toml
```

`procedure_run::run_verification` prepares a pipeline run on that output and runs it stage by stage, as [`claim build`](claim-build.md) does. Nothing is generated. Each stage verifies its part of the output and leaves a checkpoint.

```text
inference 0: valid: true, result code: 1
inference 0: output: "yes"
pipeline run: 147 stages, on the output the case states
checkpoint list: target/procedure-pipeline/runs/<run>/checkpoint_list.json
checkpoints: target/procedure-pipeline/runs/<run>/checkpoints.txt
procedure: 2e26...b36c
```

An output that is not the model's ends the command with status 1:

```text
inference 0: valid: false, result code: 0
verification: an output the case states is not what the procedure gives
```

A case with no `[[output]]` table is refused before anything runs, and so is a generation's case that has one.

In `verify-relay`, the second model's prompt holds the first model's output, and that prompt is made of the output's text and of the record its last check states. So the first output is verified on its own first, as a run of one model, and is verified again in the run itself.

## The procedure a run is of

A specification describes a procedure, and the procedure's id is a hash over it. Two parties that name one id have agreed to one procedure, with a blockchain or without one.

| Set | What the id is a hash over |
| --- | --- |
| Generation | The kind, each model by its id, the parameters, what it requires of the source and the result, and the version of the generating logic |
| Verification | The kind, its stage programs, each model as its import pins it, the parameters, and what it requires of the source and the result. It is the id a contract registers ([the registry](../protocol/availability.md#the-procedure-registry)) |

So a case's generation and its verification have two ids.

```bash
cargo run --release -p iwebs-cli -- procedure id --specification specification.toml
# procedure: 2e26...b36c
```

`--json` prints the id with the whole procedure, which is what the other party reads to see what it is agreeing to.

`run` prints the same line. `--procedure <id>` names the procedure the run must be of: the command fails before anything runs when the specification describes another.

A verification's id names its stage programs, so `run` prints it after the pipeline run, once it has held every checkpoint's program to the ones the procedure approves. There the line reads `procedure: none` with the reason in two cases. A model was imported where the stage projects are not locked, so its import wrote no `contract_pins.json` and its stage programs are not pinned. Or a stage was locked again after the model was imported, so the run's programs are not the ones the import pinned. Importing the model again from the repository root settles both. The pins file is named for the contract that is configured with it; nothing in it needs one.

## The record of a run

A record is what the party that made a run hands to anyone who wants to make it again.

```bash
cargo run --release -p iwebs-cli -- run --specification specification.toml --case case.toml --record my-record
# record: /path/to/my-record
```

A record has two parts. **What it states** is the procedure id, the inputs and the result. **Its backings** are what stands behind that besides making the run again, one entry for each method, named by its method.

| File | In the record of | What it is |
| --- | --- | --- |
| `record.json` | Both | The id of the procedure the run is of, each inference's result, and the record's backings. |
| `procedure.json` | Both | The procedure itself with its id: what the id is a hash over. |
| `inputs.json`, `document-N.bin`, `attestation-N.json` | A generation | The case's inputs: the prompt, or the evidence with the question it is read by. |
| `posted_run.bin` | A verification | The values the run's stages were given: the text, a source's documents with their attestations, the forms, and the output each inference is held to. These are the inputs a claim posts. |
| `checkpoint_list.json`, `checkpoints.txt` | A verification | The run's checkpoints and their hashes: what the fault provable backing is of. |

```json
"backings": [
  { "method": "fault_provable", "checkpoints": 199, "checkpoints_root": "5b1e...90aa" }
]
```

| Method | What the backing holds | Whose record |
| --- | --- | --- |
| None: the list is empty | Nothing. The record is backed by making the run again | A generation's |
| `fault_provable` | The count and root of the run's checkpoints, one for each stage, so that one stage can be proven wrong. What a claim commits to on a blockchain | A verification's |

A record holds at most one backing for each method, and a build refuses a record with a method it has no check for.

The question, the criteria and the evidence of a case are inputs of the run, so they are in the record and not in the procedure's id. The model's weights are the procedure's: the record names them by the model id, and the checker has its own import. A record assumes every input is available to whoever checks it, and does nothing to make it so.

### Checking a record

```bash
cargo run --release -p iwebs-cli -- record check --record their-record --specification specification.toml
```

The specification is the checker's own, naming its own imports of the models. Flow:

1. **The record is held to itself and to the checker's procedure.** The procedure in the record hashes to the id the record states, the checker's specification describes a procedure with that id, and a verification's checkpoint list has the count and root its backing states. The command fails here, before anything runs, when one does not hold.
2. **The run is made again on the record's inputs.** A generation is made again natively, on the record's evidence and the record's question. A verification is made again stage by stage: every file a stage is given from the run is written from the record's own value, and nothing is generated.
3. **Each inference's result is compared with the one the record states**, and for a verification every checkpoint with the record's, in order.

A generation's record that holds:

```text
procedure: 0cbc...5435
generation: made again natively, on the record's inputs
inference 0: valid: true, result code: 1
record: the result it states
```

A verification's:

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

A record that holds says what the procedure gives on those inputs, whoever ran it. Where one does not, the check names the first stage or field that differs, and settles nothing more.

## Comparing two runs

`--compare` names the `checkpoint_list.json` of another run of the same verification procedure and case: someone else's, or a claim's. The two lists are compared in order, and the command exits with status 1 if they differ:

```text
compared with theirs/checkpoint_list.json: the same at every stage
compared with theirs/checkpoint_list.json: differs first at stage `range_c0_l1` (checkpoint 11): output_commitment
```

A checkpoint names the stage's program, so two runs agree only when they were made with the same stage programs (the same `Raster.lock` files) and the same import of the model. A generation leaves no checkpoints, and `--compare` refuses one.

## What a claim adds

`claim build` makes the same pipeline run as a verification and leaves the same checkpoints. The [parity gate](test-parity.md) holds the two equal. A claim adds what a blockchain needs: the claim's submission and bundle, the data it has to post, and the checks a contract makes. What it posts of its run is what a verification's record holds: the same inputs, the same checkpoints, and the same procedure id. `claim build` does not write a record or read one yet.
