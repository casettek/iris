# IWEBS

IWEBS makes a language model's answer something a contract can act on. A
claimer runs a model on a prompt and submits what it answered. Anyone can
check that claim by running the model themselves, and a claim that is wrong
is caught: either the contract voids it on the spot, or a challenger replays
the one stage that differs and the claimer has to answer with a proof.

Inference is cut into stages, each a Raster program. A claim
commits to one checkpoint per stage, posts its inputs and checkpoints to
Celestia, and is submitted to contracts on an EVM blockchain.

## The commands

Every command is `iwebs <group> <command>`. They are grouped here by who runs
them. `iwebs <group> --help` lists a group, and each command says what it
takes.

### Set up a model

| Command | What it does |
| --- | --- |
| `iwebs model convert` | Turn a GGUF model with Q8_0 weights into a model bundle. |
| `iwebs model import` | Turn a bundle into the committed files every stage reads, a model manifest, and the pins a contract would hold for it. |

### Run it

| Command | What it does |
| --- | --- |
| `iwebs infer` | Fast deterministic inference of a case under a specification. This is the only path that generates: it finds the output a claim then states. |
| `iwebs run` | Run a procedure with no blockchain: the result of its native run, then a pipeline run of it with a checkpoint per stage. `--compare` names the first stage where it differs from another run. |

### Claim

| Command | What it does |
| --- | --- |
| `iwebs claim build` | Run the model stage by stage, write a checkpoint per stage, and write the claim and the data it has to post. Posts nothing. |
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
# Import a model: the small one checked in here, or your own bundle.
cargo run --release -p iwebs-cli -- model import --model model-bundles/parity-gemma

# Run it. `specification.toml` names the model manifest and the token budget,
# and `case.toml` the prompt. These two files are the defaults.
cargo run --release -p iwebs-cli -- infer --specification specification.toml --case case.toml

# Build a claim of what it answered.
cargo run --release -p iwebs-cli -- claim build --specification specification.toml --case case.toml
```

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
                            # the segment kinds, and a kind's shape as its file states it
  prepare/                  # a run's inputs and files, and the pipeline a pipeline run is made from
  native/                   # the native run: a procedure's result as fast as possible
  pipeline/                 # the pipeline run, stage by stage, that a claim and a challenge make:
                            # the runner, and each native stage under segments/{source,inference,result}
  run/                      # a procedure run with no blockchain: native, then pipeline, held equal
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
