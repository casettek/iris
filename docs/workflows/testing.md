# Testing

What the tests are for, which command runs which, and what a pass does and does
not tell you.

## What has to hold

Iris makes three promises, and every check here is about one of them.

1. **An honest claimer is never convicted.** The fast native code that builds a
   claim and the Raster stage programs that judge it give the same bytes at every
   checkpoint.
2. **A dishonest claim is caught, at the right stage.** A challenger recomputing
   the chain finds the first wrong checkpoint and can run that stage again
   through authenticated Raster.
3. **The catch can be proven.** The stage runs in the zkVM guest as it does
   natively, and a fault proof of it verifies.

## The three tiers

```sh
just test-fast      # minutes
just test-full      # about half an hour
just test-release   # about an hour; add --skip-proof to leave out the fault proof
```

Each tier runs the one before it. Every check prints one line, `PASS` or `FAIL`
with what a pass means and how long it took, and keeps its log under
`target/test-logs/`. A tier runs all of its checks even after a failure and
exits nonzero if any failed.

Set `IRIS_FEATURES=cuda,gpu-kernels` to keep `target/release/iris` a GPU build;
without it a tier rebuilds that binary for the CPU. Variables such as
`STAGED_INFER_WEIGHT_CACHE_BYTES` pass through to the runs.

| Tier | Check | A pass means | Promise |
| --- | --- | --- | --- |
| fast | Unit tests | The pieces behave: fixed-point arithmetic against independent vectors, the tokenizer against independent references, weight loading that refuses corrupted or forged data, worker and cache reuse. | 1 |
| fast | Stage schema check (one of the unit tests) | No tile reads a value its program does not pin. See [Raster stage checks](../internals/development.md#raster-stage-checks). | 2 |
| fast | Stage locks | Every `Raster.lock` matches its stage's sources, the Raster checkout and the guest compiler. | 2, 3 |
| full | [Parity gate](test-parity.md) | On each family's small synthetic model: three independent implementations agree byte for byte, bad inputs publish a fault instead of stopping a stage, and a challenger finds falsified claims at the expected stage. | 1, 2 |
| full | Stage ladder | Each stage program a family runs compiles with and without `std`, gives the native output when run as Raster, and its authenticated trace commits and audits. Run after each family's gate, on that run's inputs. | 1, 2 |
| release | Ignored tests | Native loaders match authenticated Raster stage binaries on multi-page weights; the real tokenizer's references hold; tiles run in the zkVM guest as they do natively; a fault proof of a small fixture is produced and verifies. | 1, 3 |
| release | Real-model check | On an imported Gemma model: an honest claim builds with no fault, every checkpoint matches the reference in `tests/real-model/`, and one stage of each program gives the native output when run as Raster. | 1 |

The gate replays a falsified stage through authenticated Raster only when the
stage is longer than its replay window. On the synthetic model that is
`prompt-check` and `prefill-range`; the gate's report says which stages replayed.
The stage ladder is what runs the other four programs through authenticated
Raster, with a smaller window.

## The pieces on their own

```sh
just test-parity                      # the gate alone
just stage-ladder                     # all six programs, on the newest parity run
just stage-ladder prefill-range       # one program
just test-ignored --skip-proof
just test-real-model
```

`stage-ladder` and `test-ignored` read a retained parity run: the newest under
`target/parity/`, or the one `IRIS_PARITY_RUN` names. Run the gate first.

`test-ignored` needs the E2B model beside this repository
(`../models/gemma-4-E2B-q8`) and its import, the RISC Zero guest toolchain, and
for the fault proof a working Docker engine. Build the Raster CLI with
`--features cuda` to prove on the GPU: the proof takes about 25 s that way and
several minutes without. The Groth16 wrapper runs in Docker either way.

### The real-model reference

`tests/real-model/<model>/` holds `spec.toml`, the run, and `checkpoints.txt`,
one hash per stage of an honest claim. A model that is not imported here is
reported as `SKIP`, not as a pass.

Checkpoints do not change when a stage is only re-locked. They change when a
stage's output changes. After a change that is meant to do that, and only then:

```sh
just test-real-model --update
```

and commit the new `checkpoints.txt` with the change. A difference you did not
intend is a regression: the check names the first stage that differs.

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
- **None of this runs automatically.** There is no CI; a tier runs when someone
  runs it.

## When to run what

- Any change: `just test-fast`.
- A change to a stage program, to code a tile reaches, to the chain or to the
  native kernels: `just test-full`.
- Before a version is committed, after a Raster or toolchain update, and after a
  re-lock: `just test-release`.
