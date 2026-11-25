# Run Workflow

`run` runs a procedure with no blockchain. It makes no claim, posts nothing and reads no contract's pins.

```bash
cargo run --release -p iris-cli -- run --specification specification.toml --case case.toml
# or
just run specification.toml case.toml
```

It reads a specification and a case, as [`infer`](infer.md) does (`specification.toml` and `case.toml` when no flag names them).

Flow:

1. **The direct run.** `procedure_run::run_direct` runs the procedure as `infer` does: each inference generated in turn, then what the procedure states of them. The command prints the result here, before any stage runs.
2. **The staged run.** `procedure_run::run_staged` prepares a staged run on the output the direct run generated and runs it stage by stage, as [`claim build`](claim-build.md) does. Each stage verifies its part of that output and leaves a checkpoint.
3. **The two results are held equal.** The command fails if the staged run states any field of any inference's result otherwise than the direct run did, or another consensus. That is a failure of parity between generation and verification, and the error names the field.
4. **The checkpoints are written**: `checkpoint_list.json` and `checkpoints.txt` in the staged run's directory under `target/staged-infer/runs/`.

Output: everything `infer` prints, then

```text
staged run: 87 stages, and the result the direct run stated
checkpoint list: target/staged-infer/runs/<run>/checkpoint_list.json
checkpoints: target/staged-infer/runs/<run>/checkpoints.txt
```

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

`claim build` makes the same staged run and leaves the same checkpoints. The [parity gate](test-parity.md) holds the two equal. A claim adds what a blockchain needs: the claim's submission and bundle, the data it has to post, and the checks a contract makes.
