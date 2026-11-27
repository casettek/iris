# IWEBS

IWEBS is an adjudication harness. It runs a language model as the judge of a
case, under a procedure the parties agreed to beforehand, and the verdict it
gives does not depend on who ran it.

| Word | What it is |
| --- | --- |
| Procedure | How a judgment is made, fixed before any case: the model, by a hash of its weights and tokenizer; the parameters of a run; whose signature the evidence must carry; and the answers a verdict may take. A procedure has an id, a hash over all of that. Two parties that name one id have agreed to one procedure. |
| Case | One run of a procedure: a question, and the evidence it is judged on. These are inputs of the run and are not in the procedure's id. |
| Verdict | What the procedure gives on the case. Inference is integer arithmetic with every rounding rule fixed, so the same inputs give the same bytes whoever runs them. |
| Record | What the party that ran a case hands to everyone else: the procedure's id, the case's inputs and the verdict, with what backs them. |

There are two sets of procedures. A **generation procedure** finds the
verdict: it runs the model forward, natively. A **verification procedure**
holds a verdict stated beforehand to the model's own choice, stage by stage,
so that one stage of it can be disputed. They are different logic, so they
have different names and different ids, and nothing pairs them: whoever needs
a verdict verified runs the first, then the second on what it found.

So the party that relies on a verdict need not trust the party that ran the
model. What IWEBS shows is that the procedure was followed on those inputs.
It does not show that the answer is true: that rests on the procedure the
parties chose, the model it names and the signer of the evidence. And it
assumes the model's weights and the case's inputs are available to whoever
checks.

## What backs a record

Anyone who holds a record and the model can make the run again and compare.
That needs nothing but the record. A record can also carry a backing for a
method that spares a checker the whole run, and it names the method of each.

| Method | What backs the verdict | Built |
| --- | --- | --- |
| Making the run again | Nothing but the record. Whoever doubts it runs the whole case and compares. How a generation's record is checked. | Yes: `iwebs record check` |
| Fault provable | A checkpoint for each stage of a verification, so that a dispute comes down to proving one stage wrong. Verification is cut into stages for this, each a Raster program. | Yes: `iwebs run --record` of a verification writes it |

An enclave's attestation and a validity proof would each be a further method.
Neither is built.

## Where a verdict is used

IWEBS does not enforce a verdict. Whoever relies on one does.

- **With no blockchain**, the parties put a generation procedure's id in
  whatever binds them, one of them runs the case, and the others check the
  record by making the generation again. `iwebs procedure id`, `iwebs run
  --record` and `iwebs record check` are all of it
  ([run](docs/workflows/run.md)). A record is not signed, and what happens
  when a check differs is the parties' to settle.
- **On a blockchain**, a contract registers a verification procedure. A
  claimer finds the verdict first, with a generation procedure, then submits
  a claim of it, which posts what a verification's record holds: its inputs and
  checkpoints go to Celestia, and the claim to contracts on an EVM
  blockchain. Anyone can check the claim by running the model themselves,
  and a claim that is wrong is caught: either the contract voids it on the
  spot, or a challenger replays the one stage that differs and the claimer
  has to answer with a proof.

[How IWEBS works](docs/protocol/overview.md) is the whole of both.

## The commands

Every command is `iwebs <group> <command>`. They are grouped here by who runs
them. `iwebs <group> --help` lists a group, and each command says what it
takes.

*Set up a model* and *Run a procedure* need no blockchain. The groups from
*Claim* to *Stand in for a proof* are for a verdict settled on one.

### Set up a model

| Command | What it does |
| --- | --- |
| `iwebs model convert` | Turn a GGUF model with Q8_0 weights into a model bundle. |
| `iwebs model import` | Turn a bundle into the committed files every stage reads, a model manifest, and the pins a contract would hold for it. |

### Run a procedure

| Command | What it does |
| --- | --- |
| `iwebs run` | Run a procedure with no blockchain. A generation procedure (`generate-single` and the like) finds its output natively. A verification procedure (`verify-single` and the like) holds the output its case states to the model's own choice, stage by stage with a checkpoint for each, and `--compare` names the first stage where it differs from another run. It prints the id of the procedure the run is of, and `--procedure <id>` refuses to run any other. `--record <dir>` writes the run's record. |
| `iwebs procedure id` | The id of the procedure a specification describes: a hash over its kind, its models, its parameters and what it requires of the source and the result, and for a verification its stage programs. Two parties that name one id have agreed to one procedure. `--json` prints the whole procedure. No blockchain is read. |
| `iwebs record check` | Make a recorded run again, from the record's own inputs and under your specification of its procedure, and compare the result with the record's: a generation natively, a verification stage by stage with every checkpoint compared. Anyone with the record and the models can do it. |

### Claim

| Command | What it does |
| --- | --- |
| `iwebs claim build` | Verify the output the case states stage by stage, write a checkpoint per stage, and write the claim and the data it has to post. It finds no output: the case states one. Posts nothing. |
| `iwebs claim post` | Post the claim's run inputs, checkpoint hashes and checkpoints to the data-availability layer. |
| `iwebs claim check` | Make the checks a contract makes when a claim arrives. |
| `iwebs evm claim` | Write the contract call that submits the claim. |

### Challenge

| Command | What it does |
| --- | --- |
| `iwebs claim fetch` | Read a submitted claim back from the layer into a new directory, with the case it posted. A challenger starts here. |
| `iwebs claim audit` | Ask of a fetched claim what no replay shows: that every stage names an approved program and every input is an earlier stage's output, a pinned commitment or a posted value. Nothing is run. A fault found voids the claim on the spot. |
| `iwebs challenge build` | Run every stage on the inputs the claim names, and replay through Raster the first stage whose output differs. |
| `iwebs challenge post` | Post the replay's trace and write the challenge's submission. |
| `iwebs challenge check` | Make the checks a contract makes when a challenge arrives. |
| `iwebs evm challenge` | Write the contract call that submits the challenge. |

### Defend

| Command | What it does |
| --- | --- |
| `iwebs challenge fetch` | Read the trace a challenge names back from the layer. |
| `iwebs fault prove` | Prove that a challenger's trace gets a step wrong: a Raster fault proof. |
| `iwebs fault check` | Check that this binary can prove. |
| `iwebs evm fault-proof` | Write the contract call that answers a challenge with the proof. |

### Configure a contract

| Command | What it does |
| --- | --- |
| `iwebs claim pins` | Write what a contract would pin for a run's procedure kind over its models. |
| `iwebs registry add` | Register a procedure in a registry file. |
| `iwebs procedure list` | The procedure kinds of this build. |
| `iwebs procedure describe` | One procedure kind: its steps, how many models it runs and which inference its result is of. `--json` for a caller. |
| `iwebs evm procedure` | Write the contract call that registers a procedure. |
| `iwebs evm proof-programs` | Print the ids of Raster's proof programs, which a contract pins. |

### Stand in for a proof

Three parties are trusted today where a proof belongs.

| Command | What it does |
| --- | --- |
| `iwebs attest keygen`, `iwebs attest sign` | The host that signs that a server answered a request with a document: a stand-in for a notary. |
| `iwebs da attest` | The host that signs what each posted blob holds: a stand-in for an inclusion proof. |
| `iwebs da block-root` | A block's height, data root and time, as a relayer posts them to the contract's oracle. |

### Look

| Command | What it does |
| --- | --- |
| `iwebs source render` | Check a run's source and print its record, without running the model. |
| `iwebs source inputs` | Write the source stage's inputs, to run that stage on its own. |
| `iwebs da contents` | What a submission put on the layer, measured there. |

### Test

Nothing under `iwebs test` is anything an honest party runs.

| Command | What it does |
| --- | --- |
| `iwebs test parity` | The parity gate: three implementations agree byte for byte, and cheating is caught. |
| `iwebs test claim --ids` | Build a claim of output ids that are not the model's. |
| `iwebs test corrupt-claim` | Write the claim a lying claimer would submit: another output, an unapproved program, an input from nowhere, a held value in the wrong place, or a forged verdict. |
| `iwebs test corrupt-checkpoints` | Copy a checkpoint hash list with one hash changed. |
| `iwebs test corrupt-challenge` | Flip a bit in a copy of a challenge's trace: the challenge a defender can refute. |
| `iwebs test profile` | Measure guest cycles in a completed challenge and pick a heavy window to prove. |
| `iwebs test devstack submit`, `replay`, `deployment` | Drive the local blockchains: send IWEBS's calls to the contracts there, and hold the contracts to IWEBS. |

## Quick start

```bash
# Import the judge. The example's is Gemma 4 E2B, converted to a bundle
# (docs/workflows/model-import.md). The small models under model-bundles/ are
# for the tests, and answer in nonsense words.
cargo run --release -p iwebs-cli -- model import --model <your bundle>

# The example: an insurance policy's question, judged on a signed report of
# an earthquake.
example=examples/earthquake

# The id of the generation procedure: what parties with no blockchain agree on.
cargo run --release -p iwebs-cli -- procedure id --specification $example/generation.toml

# Find the verdict, and write the record of the run.
cargo run --release -p iwebs-cli -- run --specification $example/generation.toml --case $example/case.toml --record my-record

# Check a record, as anyone who holds it and the model can.
cargo run --release -p iwebs-cli -- record check --record my-record --specification $example/generation.toml

# Verify the verdict stage by stage. This is another procedure, and its case
# states the output the generation found.
cargo run --release -p iwebs-cli -- run --specification $example/verification.toml --case $example/case-with-output.toml

# To settle it on a blockchain, build a claim of the same verification.
cargo run --release -p iwebs-cli -- claim build --specification $example/verification.toml --case $example/case-with-output.toml
```

[The example's own page](examples/earthquake/README.md) says what each file
is and what the judge is given. With no `--specification` or `--case`, a
command reads `specification.toml` and `case.toml` where it is run: the two at
the root are the smallest case there is, a generation on a prompt with no
evidence.

The local blockchains, and a claim and a challenge taken through them, are in
[blockchain/devstack/README.md](blockchain/devstack/README.md). `just --list` shows the shortcuts.

## Tests

```bash
just test-fast      # minutes: formatting, unit tests, stage locks
just test-full      # about half an hour: fast, the parity gate, the stage ladder
just test-release   # about an hour: full, a real fault proof, a real model
just test-blockchain     # about 3 minutes, with the local blockchains up: lying claims against the contracts
just fmt            # format; never `cargo fmt --all`
```

[Testing](docs/workflows/testing.md) says what a pass of each check means.

## Documentation

[docs/README.md](docs/README.md) is the index: how each workflow runs, how
the pipeline and the contracts work, and what is open.

## Layout

```text
phases/                     # the logic of each phase of a procedure, written once
  common/
    statement/              # the stage schedule and the statement a claim makes
    identity/               # binding what a stage reads to the model a claim names
  source/                   # what a prompt text was built from
  result/                   # forms, and whether two outputs agree
  inference/
    numerics/               # deterministic numbers
    tokenizer/              # the tokenizer
    prompt/                 # preparing a prompt for a model: pieces, ranges, the decoder table
    weights/                # the weight format's readers
    manifest/               # a model as an import leaves it: the manifest, the families
    kernels/                # the inference kernels generation and verification share
    import/                 # a bundle to committed files, a manifest and pins
    generation/             # fast deterministic inference: finds the output a claim states
procedures/                 # the procedure kinds: one file per kind, its steps in order
procedure/                  # a procedure and the ways to run it
  definition/               # what a procedure is, and the record of a run: specification, checkpoints,
                            # the segment kinds, a kind's shape as its file states it, and a
                            # procedure's pins and its id
  prepare/                  # a run's inputs and files, the pipeline a pipeline run is made from,
                            # and the procedure a specification describes
  native/                   # the native run: a procedure's result as fast as possible
  pipeline/                 # the pipeline run, stage by stage, that a claim and a challenge make:
                            # the runner, and each native stage under segments/{source,inference,result}
  run/                      # a procedure run with no blockchain: native, then pipeline, held equal;
                            # the record of a run, and the check of one
testing/                    # the parity gate, the lying claims, the stage schema check
cli/                        # the `iwebs` binary: one module per command group under src/commands/
raster-stages/              # one Raster program per kind of stage, each with its Raster.lock
blockchain/
  rules/                    # what a claim, a challenge and a contract hold, and the contract's rules in Rust
  host/                     # what a claimer, a challenger and a defender do: build, post, fetch, audit, prove
  evm/                      # what iwebs sends to its contracts: the ABI encoding and the calldata
  celestia/                 # Celestia's commitment to a blob, as iwebs computes it
  contracts/                # the Solidity contracts and their tests
  devstack/                 # the local Celestia and EVM blockchains
examples/                   # an adjudication from end to end: the procedure, the evidence, the case
model-bundles/              # the small models the tests run
runtime/                    # imported models
scripts/                    # the test tiers and the formatter
docs/                       # documentation
```

Each directory under `raster-stages/` is a Raster program of its own, with a
`Raster.lock` that records its identity. Formatting or editing code a stage
program is built from changes that identity, so those crates are not touched
by `just fmt` and a change to them ends in a re-lock
([testing](docs/workflows/testing.md#formatting)).
