# Testing

What the tests are for, which command runs which, and what a pass does and does
not tell you.

## What has to hold

IWEBS makes four promises, and every check here is about one of them.

1. **An honest claimer is never convicted.** The fast native code that builds a
   claim and the Raster stage programs that judge it give the same bytes at every
   checkpoint.
2. **A dishonest claim is caught, at the right stage.** A challenger recomputing
   the pipeline finds the first wrong checkpoint and can run that stage again
   through authenticated Raster.
3. **The catch can be proven.** The stage runs in the zkVM guest as it does
   natively, and a fault proof of it verifies.
4. **The contracts say what IWEBS says.** A claim IWEBS would take the contracts
   take, a claim that lies is void or challenged on the blockchains as the Rust
   rules say it is, and a challenge IWEBS would refuse the contracts refuse.

## The tiers

```sh
just test-fast      # minutes
just test-full      # about half an hour
just test-release   # about an hour; add --skip-proof to leave out the fault proof
just test-blockchain     # about 3 minutes; needs the local blockchains
```

Each of the first three runs the one before it. The blockchain tier stands apart,
because it needs the local blockchains running (see [The blockchain tier](#the-blockchain-tier)).
Every check prints one line, `PASS` or `FAIL` with what a pass means and how
long it took, and keeps its log under `target/test-logs/`. A tier runs all of
its checks even after a failure and exits nonzero if any failed.

Set `IWEBS_FEATURES=cuda,gpu-kernels` to keep `target/release/iwebs` a GPU build;
without it a tier rebuilds that binary for the CPU. Variables such as
`IWEBS_PIPELINE_WEIGHT_CACHE_BYTES` pass through to the runs.

| Tier | Check | A pass means | Promise |
| --- | --- | --- | --- |
| fast | Formatting | Every host file is as `just fmt` leaves it. See [Formatting](#formatting). | none |
| fast | Unit tests | The pieces behave: fixed-point arithmetic against independent vectors, the tokenizer against independent references, weight loading that refuses corrupted or forged data, worker and cache reuse. | 1 |
| fast | Stage schema check (one of the unit tests) | No tile reads a value its program does not pin. See [Raster stage checks](../protocol/development.md#raster-stage-checks). | 2 |
| fast | Stage locks | Every `Raster.lock` matches its stage's sources, the Raster checkout and the guest compiler. | 2, 3 |
| full | [Parity gate](test-parity.md) | On each family's small synthetic model: three independent implementations agree byte for byte, bad inputs publish a fault instead of stopping a stage, and a challenger finds falsified claims at the expected stage. | 1, 2 |
| full | Stage ladder | Each stage program a family runs compiles with and without `std`, gives the native output when run as Raster, and its authenticated trace commits and audits. Run after each family's gate, on that run's inputs. | 1, 2 |
| release | Ignored tests | Native loaders match authenticated Raster stage binaries on multi-page weights; the real tokenizer's references hold; tiles run in the zkVM guest as they do natively; a fault proof of a small fixture is produced and verifies. | 1, 3 |
| release | Real-model check | On an imported Gemma model: an honest claim builds with no fault, every checkpoint matches the reference in `tests/real-model/`, and one stage of each program gives the native output when run as Raster. | 1 |
| pipeline | Contracts' own tests | The Solidity tests pass, among them a real fault proof against `submitFaultProof`. | 3, 4 |
| pipeline | Honest claim | The contract takes it, agrees with `iwebs claim check` on changed copies of it, and refuses every on-the-spot question as `iwebs claim audit` does. | 4 |
| pipeline | Four voided lies | A claim whose checkpoint names an unapproved program, or an input that is not its producer's output, a pinned commitment or a posted value, is taken, and voided with that fault when the contract is pointed at the checkpoint. | 2, 4 |
| pipeline | Four rearranged lies | A claim that gives a stage a value from the wrong place (another layer's weights, another range's marks, another stage's output, the empty value) is taken and survives every on-the-spot question. The challenger's replay of that stage on the inputs its checkpoint names is a challenge the contract admits. | 2, 4 |
| pipeline | Forged verdict | A claim of ids that are not the model's, with its check's record stated valid, arrives with a valid record and survives every question, and the challenger goes to replay the last check. | 2, 4 |

The gate replays a falsified stage through authenticated Raster only when the
stage is longer than its replay window. On the synthetic model that is
`prompt-check` and `prefill-range`; the gate's report says which stages replayed.
The stage ladder is what runs the other programs through authenticated Raster,
with a smaller window. That includes `prompt-source` and `claim-result`, on the
claim the gate's source scenario builds, and `consensus-result`, on the claim its
consensus scenario builds, and `prompt-relay`, on the claim its relay scenario
builds. The two result programs and the relay program are a handful of steps and are
committed at a window of two.

## The blockchain tier

```sh
just devstack-up        # the local blockchains, once
just devstack-deploy    # the contracts, once per restart of the blockchains and after any contract change
just test-blockchain
just test-blockchain --real-model specification.toml
```

The tier sends real transactions to the local EVM node and posts to the local
Celestia node ([devstack](../../blockchain/devstack/README.md)). It builds its own claims
of the newest parity run's small model, for addresses no earlier run used, so
it can be repeated on one deployment. `--deploy` deploys the contracts first.
Everything it makes is kept under `target/blockchain-test/<run>`.

The small model is the gemma family's, from the newest parity run of it. Its
check stage is under the replay window, so there the forged verdict is
followed only as far as the replay. The tier also builds a claim of the
`verify-relay` procedure kind, on the run the gate's relay scenario made: the
contract must take it and agree with IWEBS on changed copies of it, and a lie
in its second inference must be challenged by a replay of a run rebuilt from
what the claim posted. `--real-model <specification>` adds a
claim of that specification's model on the `case.toml` beside it, another layer's weights given to one of its
stages, and the forged verdict, each of which must end in a challenge the
contract accepts. With a short prompt on a GPU build that adds a few minutes.
No proof is made.

To have the other tiers post to the local Celestia node in place of the
stand-in directory, set `IWEBS_TEST_CELESTIA=1`:

```sh
IWEBS_TEST_CELESTIA=1 just test-parity
```

## Formatting

```sh
just fmt          # format the host code
just fmt-check    # list what is not formatted; the fast tier runs this
```

Never run `cargo fmt --all` here. It follows path dependencies into the
Raster checkout and rewrites it, and it rewrites the code the stage programs
are built from. A stage program's identity covers its sources down to the
line numbers in their panic messages, so formatting that code changes every
program id: a re-lock of every stage, new contract pins and a new real-model
reference.

`just fmt` formats what no stage program is built from: every Rust file
except the stage crates (`raster-stages/`), the crates they depend on
(`inference-numerics`, `common-identity`, `common-statement`), and the six files of
`inference-tokenizer` and `source-phase` they take by symbolic link. The stage code is
formatted only together with a re-lock. If a stage crate gains a dependency
inside this repository, add it to `guest_dirs` in `scripts/fmt.sh`; the stage
lock check fails if that is forgotten and the crate is then formatted.

## The pieces on their own

```sh
just test-parity                      # the gate alone
just stage-ladder                     # every program of the run's family, on the newest parity run
just stage-ladder prefill-range       # one program
just test-ignored --skip-proof
just test-real-model
```

`stage-ladder` and `test-ignored` read a retained parity run: the newest under
`target/parity/`, or the one `IWEBS_PARITY_RUN` names. Run the gate first.

`test-ignored` needs the E2B model beside this repository
(`../models/gemma-4-E2B-q8`) and its import, the RISC Zero guest toolchain, and
for the fault proof a working Docker engine. Build the Raster CLI with
`--features cuda` to prove on the GPU: the proof takes about 25 s that way and
several minutes without. The Groth16 wrapper runs in Docker either way.

### The real-model reference

`tests/real-model/<model>/` holds `specification.toml`, `case.toml` and `checkpoints.txt`,
one hash per stage of an honest claim. A model that is not imported here is
reported as `SKIP`, not as a pass.

Checkpoints do not change when a stage is only re-locked. They change when a
stage's output changes. After a change that is meant to do that, and only then:

```sh
just test-real-model --update
```

and commit the new `checkpoints.txt` with the change. A difference you did not
intend is a regression: the check names the first stage that differs.

### Two scripts no tier runs

Both are Python because each drives something outside this repository.

- `scripts/tokenizer_reference.py TOKENIZER_JSON OUTPUT_JSON [--production]`
  writes the token ids an independent tokenizer (`tokenizers` 0.22.2) gives a
  fixed set of texts. `tests/tokenizer/gemma-reference.json` and
  `synthetic-reference.json` came from it, and the tokenizer unit tests hold
  the native stages to them. Run it again only for a new tokenizer.
  `tests/tokenizer/generate_ranked.py` does the same for the adversarial
  ranked-merge tokenizer beside it.
- `scripts/test_prompt_bucket_transitions.py prompt-check <stage dir> vocab|merge`
  checks that a fault placed in `prompt-check`'s walk of one sparse bucket
  is reached by Raster's transition guest: it records an authenticated run
  with the fault, estimates its window without proving, and requires every
  transition to run and the stage's output to be unchanged. The stage
  directory is a kept Raster run of `prompt_check_p0` from a parity run
  (`<parity run>/short/raster/prompt_check_p0`). A few minutes. Run it
  after a change to how `prompt-check` walks its buckets.

## What no tier tells you

- **No fault proof of a real stage.** The proof in the release tier is of a
  two-step window in a small fixture. It shows the proving path works end to end,
  not that a window of a real stage proves, or how long that takes.
- **The guest tests cover chosen tiles**, not every tile of every program.
- **Authenticated Raster on a real model is not run.** The real-model check runs
  stages unauthenticated: an authenticated run of a full-size stage takes minutes
  to tens of minutes each.
- **Speed and memory have no threshold.** Timings are printed and recorded, never
  judged.
- **Nothing here runs the backend.** Its own checks are `pnpm typecheck`,
  `pnpm test` and `pnpm validate:gpu` in `iris-backend`.
- **No fault proof of a real dispute goes to the contracts.** The blockchain tier
  ends each dispute at the admitted challenge. The contract's own tests take
  one real proof, of the small fixture.
- **The blockchain tier runs against trusted stand-ins**: a host's signature for
  what a blob holds, and a relayer for Celestia's block roots.
- **None of this runs automatically.** There is no CI; a tier runs when someone
  runs it.

## When to run what

- Any change: `just test-fast`.
- A change to a stage program, to code a tile reaches, to the pipeline or to the
  native kernels: `just test-full`.
- A change to the contracts, to what a claim or a challenge posts, to the rules
  a claim or a challenge is checked by, or to how a challenge is built:
  `just test-blockchain`, after `just devstack-deploy` if a contract changed.
- Before a version is committed, after a Raster or toolchain update, and after a
  re-lock: `just test-release`.
