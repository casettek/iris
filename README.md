# iris

iris makes a language model's answer something a contract can act on. A
claimer runs a model on a prompt and submits what it answered. Anyone can
check that claim by running the model themselves, and a claim that is wrong
is caught: either the contract voids it on the spot, or a challenger replays
the one stage that differs and the claimer has to answer with a proof.

Inference is cut into stages, each a Raster program. A claim
commits to one checkpoint per stage, posts its inputs and checkpoints to
Celestia, and is submitted to contracts on an EVM blockchain.

## The commands

Every command is `iris <group> <command>`. They are grouped here by who runs
them. `iris <group> --help` lists a group, and each command says what it
takes.

### Set up a model

| Command | What it does |
| --- | --- |
| `iris model convert` | Turn a GGUF model with Q8_0 weights into a model bundle. |
| `iris model import` | Turn a bundle into the committed files every stage reads, a model manifest, and the pins a contract would hold for it. |

### Run it

| Command | What it does |
| --- | --- |
| `iris infer` | Fast deterministic inference from a run spec. This is the only path that generates: it finds the output a claim then states. |

### Claim

| Command | What it does |
| --- | --- |
| `iris claim build` | Run the model stage by stage, write a checkpoint per stage, and write the claim and the data it has to post. Posts nothing. |
| `iris claim post` | Post the claim's run inputs, checkpoint hashes and checkpoints to the data-availability layer. |
| `iris claim check` | Make the checks a contract makes when a claim arrives. |
| `iris evm claim` | Write the contract call that submits the claim. |

### Challenge

| Command | What it does |
| --- | --- |
| `iris claim fetch` | Read a submitted claim back from the layer into a new directory, with a run spec over what it posted. A challenger starts here. |
| `iris claim audit` | Ask of a fetched claim what no replay shows: that every stage names an approved program and every input is an earlier stage's output, a pinned commitment or a posted value. Nothing is run. A fault found voids the claim on the spot. |
| `iris challenge build` | Run every stage on the inputs the claim names, and replay through Raster the first stage whose output differs. |
| `iris challenge post` | Post the replay's trace and write the challenge's submission. |
| `iris challenge check` | Make the checks a contract makes when a challenge arrives. |
| `iris evm challenge` | Write the contract call that submits the challenge. |

### Defend

| Command | What it does |
| --- | --- |
| `iris challenge fetch` | Read the trace a challenge names back from the layer. |
| `iris fault prove` | Prove that a challenger's trace gets a step wrong: a Raster fault proof. |
| `iris fault check` | Check that this binary can prove. |
| `iris evm fault-proof` | Write the contract call that answers a challenge with the proof. |

### Configure a contract

| Command | What it does |
| --- | --- |
| `iris claim pins` | Write what a contract would pin for a run's procedure kind over its models. |
| `iris registry add` | Register a procedure in a registry file. |
| `iris evm procedure` | Write the contract call that registers a procedure. |
| `iris evm proof-programs` | Print the ids of Raster's proof programs, which a contract pins. |

### Stand in for a proof

Three parties are trusted today where a proof belongs.

| Command | What it does |
| --- | --- |
| `iris attest keygen`, `iris attest sign` | The host that signs that a server answered a request with a document: a stand-in for a notary. |
| `iris da attest` | The host that signs what each posted blob holds: a stand-in for an inclusion proof. |
| `iris da block-root` | A block's height, data root and time, as a relayer posts them to the contract's oracle. |

### Look

| Command | What it does |
| --- | --- |
| `iris source render` | Check a run's source and print its record, without running the model. |
| `iris source inputs` | Write the source stage's inputs, to run that stage on its own. |
| `iris da contents` | What a submission put on the layer, measured there. |

### Test

Nothing under `iris test` is anything an honest party runs.

| Command | What it does |
| --- | --- |
| `iris test parity` | The parity gate: three implementations agree byte for byte, and cheating is caught. |
| `iris test claim --ids` | Build a claim of output ids that are not the model's. |
| `iris test corrupt-claim` | Write the claim a lying claimer would submit: another output, an unapproved program, an input from nowhere, a held value in the wrong place, or a forged verdict. |
| `iris test corrupt-checkpoints` | Copy a checkpoint hash list with one hash changed. |
| `iris test corrupt-challenge` | Flip a bit in a copy of a challenge's trace: the challenge a defender can refute. |
| `iris test profile` | Measure guest cycles in a completed challenge and pick a heavy window to prove. |
| `iris test devstack submit`, `replay`, `deployment` | Drive the local blockchains: send iris's calls to the contracts there, and hold the contracts to iris. |

## Quick start

```bash
# Import a model: the small one checked in here, or your own bundle.
cargo run --release -p iris-cli -- model import --model model-bundles/parity-gemma

# Run it. `inference.toml` names the model manifest, the prompt and the token budget.
cargo run --release -p iris-cli -- infer --run inference.toml

# Build a claim of what it answered.
cargo run --release -p iris-cli -- claim build --run inference.toml
```

The local blockchains, and a claim and a challenge taken through them, are in
[blockchain/devstack/README.md](blockchain/devstack/README.md). `just --list` shows the shortcuts.

## Tests

```bash
just test-fast      # minutes: formatting, unit tests, stage locks
just test-full      # about half an hour: fast, the parity gate, the stage ladder
just test-release   # about an hour: full, a real fault proof, a real model
just test-chain     # about 3 minutes, with the local blockchains up: lying claims against the contracts
just fmt            # format; never `cargo fmt --all`
```

[Testing](docs/workflows/testing.md) says what a pass of each check means.

## Documentation

[docs/README.md](docs/README.md) is the index: how each workflow runs, how
the chain and the contracts work, and what is open.

## Layout

```text
crates/
  inference/
    cli/                    # the `iris` binary: one module per command group under src/commands/
    artifacts/              # what a claim, a challenge and a contract hold, and the contract's rules in Rust
    direct/                 # fast deterministic inference
    staged/                 # the stage-by-stage run a claim and a challenge make
    kernels/                # the inference kernels both share
    pipeline-core/          # the stage schedule and the statement a claim makes
  model/
    import/                 # a bundle to committed files, a manifest and pins
    run-prep/               # a run's inputs, written as the files its stages load
    identity/               # binding what a stage reads to the model a claim names
    detwgt/                 # the weight format's readers
  prompt-core/, source-core/, det-num/   # the tokenizer, sources and results, deterministic numbers
raster-stages/              # one Raster program per kind of stage, each with its Raster.lock
blockchain/
  evm/                      # what iris sends to its contracts: the ABI encoding and the calldata
  celestia/                 # Celestia's commitment to a blob, as iris computes it
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
