# What a contract holds and checks

No contract exists yet. This page describes what one would pin, what a claimer
would submit, and the checks it would make, as the code states them today in
`inference_artifacts::contract`. `iris claim check` runs those checks on the host.

The contract verifies no proof and runs no stage. It makes one statement about a
claim:

> A claimer says that a model I trust, run with the parameters I require,
> produced this plaintext from a committed input. I do not read the input. If
> nobody successfully challenges the claim within the window, I act on the
> plaintext.

## What the contract pins

A contract opts into two entries, and chooses two parameters of its own. The
model import writes both entries to `contract_pins.json`, beside the import's
`manifest.json`.

**The pipeline entry** is true of the stage programs whatever model they run.
Two models of one family have the same one.

| Value | Where it comes from |
| --- | --- |
| The approved stage programs | The `program_commitment` each stage's `Raster.lock` holds at import time |
| Each program's chained inputs | The chain template: an input is chained when any stage of the program takes it `from` another stage |
| For each chained input, the programs that may produce it | The chain template |
| For each chained input, the empty values it may hold | The fixed externals the chain's first stages are given in its place |

**The model entry** is true of one model.

| Value | Where it comes from |
| --- | --- |
| Model id | The import |
| The commitments the model id hashes, by role | The import |
| Range size | The import |

**The contract's own parameters** are the raw prompt flag and the token budget.
`iris claim check` reads them from the run spec.

Re-locking a stage changes its program id, so import again afterwards. An import
made where the stage projects are not locked writes no pins and says so.

The approved programs are a set, not a map from position to program. Which stage
runs where is the stage schedule's to enforce, inside the programs: the first
stage verifies the schedule and every later one checks its own place in it.

## What a claim commits to

A claim commits to one checkpoint per stage, in order. A checkpoint is the
boundary a Raster run of that stage is bound to:

| Checkpoint field | Meaning |
| --- | --- |
| `stage` | The stage's name |
| `program_commitment` | The program id the stage project has locked: the program a Raster replay of the stage runs |
| `input_commitment` | SHA-256 of the stage's `input_manifest.json` |
| `input_sources` | For each input taken from an earlier stage, that stage's position in the list |
| `output_commitment` | The output's structural commitment |
| `output_sha256` | SHA-256 of the stage's `output.bin` |

`claim build` runs the stages natively and does not run Raster. It still commits
to what the boundary would be if every stage had run as Raster, so a challenger
can replay any one stage in Raster from the claimer's own boundary and show
where it leaves what the claimer committed for the next. No chain-wide Raster
commitment is involved.

`checkpoint_trace.json` holds the checkpoints. `checkpoints.txt` holds one
SHA-256 per checkpoint, and its own SHA-256 is the claim's commitment to all of
them.

## What a claimer submits

`claim build` writes `claim_submission.json` beside `claim_bundle.json`:

| Field | Meaning |
| --- | --- |
| `output_text` | The plaintext the contract acts on |
| `input_text_sha256` | The claimer's commitment to the input it ran. The contract takes it as given |
| `record.statement_sha256` | `Statement::digest()` over the input text hash, model id, raw prompt flag, token budget and range size |
| `record.output_text_sha256` | SHA-256 of the plaintext's exact UTF-8 bytes |
| `record.valid` | The chain found nothing wrong with its own run |
| `checkpoints_sha256` | SHA-256 of the claim's `checkpoints.txt` |

The three `record` values are the claim record the last check states in its
output (`ClaimRecord` in `raster-stages/sequence-check/src/input.rs`). The
verdict is computed in the stage from every field that can reject a claim: the
model fault, the prompt fault and completeness, the check fault, the mismatch
count and the output fault. Only the schedule's last claimed range can state a
valid one.

## On submission

1. **Statement matches.** Recompute the statement digest from the pinned model id,
   range size and required parameters with the claimed input hash, and compare it
   with the record's. This one equality covers the model and every parameter.
2. **Output matches.** The plaintext hashes to the record's output text hash, and
   the valid flag is set.

## When a challenge arrives

A challenge is about one stage. The challenger presents the claim's checkpoint
list and the checkpoints it points at, and the contract holds each to what the
claim committed: the list to `checkpoints_sha256`, a checkpoint to its line in
the list. A challenger cannot state a program, a boundary or a producer of its
own. Three comparisons follow, with no replay:

- **The stage's program is approved.** If the program id the claimer's own
  checkpoint names is in the approved set, the challenge is an ordinary disputed
  one and a defender must answer it. If it is not, or the checkpoint names no
  program, the claim is void.
- **The stage's inputs are earlier outputs.** A stage keeps its own store, so
  nothing inside it can tell where an input came from: a made-up input can carry
  the labels a real one does. Without this comparison a claimer could run every
  stage honestly on inputs no earlier stage produced, and no replay would show
  it. For one chained input of the stage, the commitment the stage loaded it by
  must be the output commitment of the checkpoint the stage's own checkpoint
  names as its producer, that checkpoint must be earlier in the list, and its
  program must be one that may produce the input. Or the input holds one of its
  pinned empty values. Otherwise the claim is void.
- **The record is the last stage's.** The submitted record must be the one in the
  output the claim's last checkpoint commits to.

The second comparison reads the stage's `input_manifest.json`, held to the
checkpoint's `input_commitment`. It must be exactly the canonical encoding of
what it holds: a repeated name, an extra field or other spacing voids the claim,
so that no two readers can take one input list for different inputs.

Which earlier stage is the right producer is not the contract's to judge. Every
output says which step produced it, and the consuming stage checks that against
the schedule. The contract holds only that the input is a real earlier output of
the right kind.

The checkpoint list is this repository's own format. Today opening one
checkpoint means presenting the whole list of hashes.

## Disputing a stage's output

The three comparisons above void a claim with no replay. What is left is a stage
whose checkpoint names an approved program and real inputs, and an output the
challenger says that program does not produce from them. That is settled by
Raster's stage challenge: the challenger posts the trace commitment of its own
authenticated run of the stage and the output it ends in, and a defender either
proves a window that dismisses it or the claim is void when the window closes.

Raster's challenge and defence name a chain commitment and a stage of it, and
read three things of that stage: its program, its input manifest and the hash of
its output. A checkpoint holds exactly those. So **a dispute about a checkpoint
is a dispute about a Raster chain of one stage built from it**
(`dispute_commitment`), and the contract recomputes that commitment's digest from
the checkpoint it opened. The encoding is a plain concatenation:

```text
01                      one stage
len name                the stage name, length as a varint
20 program              32 bytes
20 input manifest       32 bytes
00                      no bindings
20 output hash          32 bytes
20 output structural    32 bytes
00                      no manifest digest
00                      no repeats
```

Its SHA-256 is the digest a Raster challenge or defence about that checkpoint
names. Two rules follow:

- **A challenge is about a checkpoint** when the chain commitment it carries is
  that checkpoint's. Raster's own admission then applies: the output it states
  must differ from the checkpoint's, and its trace must be committed at the
  protocol's window.
- **A proof dismisses a challenge only when it is about the same checkpoint**
  (`check_defence`, which applies Raster's `defend`). The proven window must be
  against the challenger's trace commitment, run the checkpoint's program on the
  checkpoint's inputs, and end in a divergence or, on an honest trace, in an
  output other than the one the challenger stated. A window proven over another
  program or other inputs dismisses nothing: any trace diverges from a run it is
  not a trace of.

`iris fault prove` holds its proof to the second rule and records the dispute
digest in `fault_bundle.json`.

Only a differing output is a matter for a replay (`settlement`). A replay runs
the claimer's own program on the claimer's own inputs, so it says nothing about
either. A claim that names another program is settled by the program
comparison. A claim that ran a stage on other inputs than the challenger did is
not settled by the challenger's own replay: the link comparison voids it when an
input taken from an earlier stage is not that stage's output, and otherwise the
stage's own checks judge the input, and what is left to dispute is whether the
claimed output follows from the claimer's inputs.

## Running the checks

```bash
cargo run --release -p iris-cli -- claim check \
  --run inference.toml \
  --submission target/staged-infer/runs/.../claim_submission.json
```

The command reads the checkpoint files, every stage's input list and the last
stage's output from the directory the submission is in. It makes the submission
checks and the three challenge-time comparisons, the link comparison for every
chained input of every stage. `claim build` makes the same checks on its own
claim before it reports success, when the import has pins.

Add `--raster-run <dir>` with a Raster chain run of the same claim to hold every
checkpoint to the boundary that run committed: the same stages in the same order,
each with Raster's program id, input manifest commitment, producers and output
commitments. The [parity gate](../workflows/test-parity.md) does this on every
honest case.

## Known gaps

These are open. None is closed by anything on this page.

- **A stage that cannot run.** A claimer can give a stage a run input of the
  wrong shape, so that the honest program stops before it reaches its end, and
  write any output for it. As Raster's stage challenge is designed, a trace that
  does not reach the program's end is refuted as a misstated one, so a challenger
  has nothing to win with. Read from the design, not run.
- **Two commitments to one output.** A checkpoint holds the hash of a stage's
  output bytes, which a dispute is about, and its structural commitment, which
  the link comparison uses. Raster's proof guest checks the output's bytes
  against both at the end of a stage and publishes only the byte hash. So a
  claimer can list a true byte hash beside the structural commitment of a made-up
  value, feed that value to a later stage, and pass the link comparison. Closing
  it needs a proof: either Raster publishes the structural commitment in the
  window's journal, or a proof of this repository's own ties the two.
- **Short stages.** A stage whose trace is shorter than a replay window cannot be
  disputed. The claimer chooses the input, so it can make one occur.
- **No real dispute has been run end to end against a native claim**, and no
  proof of a full-size stage has been produced. The dispute rules above are
  tested on Raster's own admission and defence functions with real traces and
  made-up journals, and on one real proof of a small fixture.
- **`challenge build` replays whatever differs.** It does not yet sort a
  divergence into the program comparison, the link comparison or a replay. A
  replay aimed at a checkpoint that ran other inputs fails safely, since the
  defender can refute it.

## Not here

- Data availability, bonds, the challenge window, and verifying a defence. These
  are the contract's and the settlement layer's.
- Any check of the input beyond its hash.
- A way to open one checkpoint without presenting the whole list.
