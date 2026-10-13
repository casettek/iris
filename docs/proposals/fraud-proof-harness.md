# Proposal: the fraud-proof harness — a container that cheats at inference

Status: **sketch** 2026-09-11. Nothing built.

A Docker image that, on every round, produces a real chain run, injects one random fault into one
random part of it, and asserts that the fraud machinery both *detects* the fault and *names the
right stage*. It touches the chain only through the public CLI — `chain run`, `chain audit`,
`chain fraud-prove`, `chain fraud-verify` — because the thing under test is what a disputing party
can actually do, not what an internal API can.

## 1. The oracle is attribution, not failure

"Did it fail?" is a bad assertion: a detector that always cries fraud passes it. Every injected
fault knows which stage it corrupted, and `chain fraud-prove` emits a receipt naming a stage, so
the assertion is the pair:

```text
receipt names injected_stage   AND   chain fraud-verify accepts the receipt
```

with a **negative control on every round**: the same run with no injection must produce no
receipt, and `chain audit --execution` must pass. Half the suite is honest runs. A round that
detects fraud in an honest chain is as much a failure as one that misses a real fault.

Three verdicts are failures, and the middle one is the reason this harness is worth building:

| verdict | meaning |
| --- | --- |
| `missed` | fault injected, detector said clean |
| `detected-wrong-stage` | detector caught the cheat and blamed the wrong party — in a real dispute this slashes an honest prover |
| `false-positive` | honest run flagged |
| `crashed` | the verifier panicked instead of returning a verdict (see §6) |

## 2. Fault families

The per-stage run directory is the attack surface. From
`target/raster/chains/<run>/<stage>/`: `input.json`, `input_manifest.json`, `output.bin`,
`output.rindex`, `output_manifest.json`, `commit.bin`, `trace.bin`, `tile_census.json`.

**A — link faults.** The boundary between stages; caught by `chain audit` with no `--execution`.

- flip bytes in stage *k*'s `output.bin`, so stage *k+1*'s committed input no longer matches
- flip `output.rindex`, so the index disagrees with the data it indexes
- copy another stage's commitment into `output_manifest.json`
- re-point a `from = "…"` binding in `Raster.toml` at a different producer
- **substitute a whole stage directory from a different honest run** (different prompt) — a
  valid-but-wrong artifact, which no amount of byte-flipping reaches

**B — execution faults.** Trace disagrees with honest re-execution; needs `--execution`.

- flip an event payload in `trace.bin`
- truncate the trace so a `TileExec` disappears
- reorder a `RecurSequenceIterationStart`/`End` pair
- flip `commit.bin` — but only its `fingerprint` region reaches `audit`; see §5

**C — malicious prover.** The realistic family, and the only one that models an adversary rather
than a corrupted disk. Patch a tile's *source*, then re-run that one stage with
`chain run --stage <name> --run <dir>` — which the CLI documents as exactly this scenario: "That is
what a dispute needs — the contested stage's trace, on demand." The prover produces a stage that is
internally perfect: its own trace, its own commitment, its own output, all mutually consistent.
Only re-execution against the *committed program identity* catches it.

- off-by-one in a recur bound
- `add_sat` → wrapping add in `prefill-prepare-aux`
- `isqrt` stopped one Newton iteration early
- the scan returns a stale `RowMatch` from the previous token
- `assert_all_tokens_embedded` neutered, so a short gather publishes

C splits in two, and they must be separate rounds because they exercise different detection paths:

- **C1, prover re-locks.** `Raster.lock` changes, the program commitment changes, and the chain's
  identity check should reject it *before* any replay. Cheap path.
- **C2, prover keeps the old lock** and runs new binaries. Only replay catches it. Expensive path.

A harness that only tests C1 would let a regression in C2 through unnoticed.

**D — committed externals.** Corrupt the model rather than the computation: flip a weight in
`layer3.rastered`. Two rounds, opposite expectations —

- without updating the commitment in `Raster.toml` → must be rejected
- *with* updating it → must be **accepted**. It is a different but legitimate chain. This is a
  negative control disguised as a fault, and it catches a detector that has started comparing
  against remembered values instead of committed ones.

**E — commitment coverage.** See §5.

## 3. Scale decides the whole design

Measured on the checked-in 35-layer manifest:

| thing | size / time |
| --- | --- |
| `prefill_range_l0/trace.bin` | **1.05 GB** — one stage |
| a run that reached stage 38 of 75 | 1.4 GB |
| `prompt-prepare` authenticated commit (cold) | 32 s |
| `prompt-prepare` audit | 10.6 s |
| `prompt-prepare/commit.bin` | 9,795 bytes |

Thirty-five `prefill_range` stages at a gigabyte each is not a container workload, and it is not a
fraud-proof workload either. So:

**The lab runs `tiny-gemma-dev` only.** 4 layers, `hidden_size` 4, vocab 280, 84 KB on disk — 13
stages at `--tokens 0`, ~11 more per generated token. A `--model` escape hatch exists for a manual
big run and is off by default.

This is not a compromise on fidelity. Fraud detection is a property of the protocol, not of the
model: a wrong `add_sat` in a 4-wide row is the same class of fault as in a 2048-wide one, and only
the tiny one lets a round finish in seconds.

## 4. The blocker, and what to build anyway

`README.md:497-501` says authenticated chain execution is blocked on Raster's open
`authenticated-chain-draft-output` — the recorder cannot replay a `ProgramEnd` finalized from a
`Draft` — affecting `decode-init`, `decode-select-token`, `decode-embed`, `output-finalize`. If
that still holds, `chain run` (authenticated) → `chain fraud-prove` does not complete, and families
A/B/C at *chain* level are blocked with it.

So step one is to confirm or deny it in the container, and the harness is built so the answer is a
config change rather than a rewrite: **fault families are data, and the runner selects the families
whose prerequisites hold.** Three tiers, in order of what unblocks:

1. **Single-stage lab on `prompt-prepare`** — the one stage that owns its inputs, so
   `run --commit` / `run --audit` works standalone. Verified working today (§3 timings). Families B
   and E run here now; family C runs here now by patching `prompt-prepare`'s tiles.
2. **Link-only lab on `--no-auth` chain runs** — family A is about `output.bin` and commitments,
   which exist under `--no-auth`. `chain audit` without `--execution` should catch link tampering
   with no trace involved.
3. **Full lab** — families A–D against `chain fraud-prove` / `fraud-verify`, once the blocker
   clears.

Tiers 1 and 2 are worth shipping on their own. They are also the honest scope of "we test fraud
proofs" until tier 3 runs.

## 5. Family E, and the two ways a naive sweep lies

`commit.bin` is postcard over `TraceCommitment { fingerprint, revealed_items }`
(`raster-core/src/trace.rs:607`). On `prompt-prepare` it is 9,795 bytes, and two honest runs
produce it byte-identical — there is no nondeterministic metadata in it.

Single-bit flips (`^= 0x01`, one per audit) across the whole file, classified by *reason*:

```text
bytes     0 … 1183   fraud-detected   ← the `fingerprint` field
bytes  1184 … 9794   ACCEPTED         ← the `revealed_items` field
```

The boundary is exact: byte 1183 is detected, 1184 is not.

**This is correct behaviour, not a gap.** `TraceVerifier::verify`
(`raster-prover/src/trace.rs:852`) recomputes the fingerprint from honest re-execution and compares
it index by index against `trace_commitment.fingerprint.bits`. It never reads `revealed_items` — it
uses only their *count*, as the window size. The revealed window is bound on the other path:
`revealed_items_commitment = sha256(postcard(revealed_items))`
(`raster-prover/src/trace.rs:430-435`), which the transition guest hashes into its journal at
`Init` (`raster-core/src/transition.rs:72-73`, and the doc comment on `TraceCommitment` says so
outright).

So every byte binds something. **Coverage is per-detector, and `audit` is not the detector that
covers the back two-thirds of the file.** Which is the point:

**Family E — commitment coverage.** For each artifact, sweep bytes and classify by
(detector × reason), then assert the result equals a checked-in expected map. The map is a matrix,
not a list: a byte range that no detector covers is a finding; a byte range that moves from one
detector to another is a finding; a silent widening of either fails CI. The map also documents the
commitment format better than any prose currently does.

Run it per artifact — `commit.bin`, `output.bin`, `output.rindex`, `output_manifest.json` — and per
detector — `run --audit`, `chain audit`, `chain audit --execution`, `chain fraud-prove`,
`chain fraud-verify`.

### Two lessons the first sweep taught, both of which the harness must encode

**The mutation operator is part of the test.** The first pass used `^= 0xFF`, and *every* result it
called "caught" was a postcard varint parse error — `Found a varint that didn't terminate`,
`Tried to parse invalid utf-8` — not a verification failure. Zero real detections, presented as
eleven. `^= 0x01` preserves the varint continuation bit, keeps the file well-formed, and gives a
clean signal. So the harness runs **two operators with different oracles**:

- *structure-preserving* (`^= 0x01`, field-aware edits) → expects `fraud-detected`
- *structure-breaking* (`^= 0xFF`, truncation, splices) → expects a clean `parse-error`, never a
  panic and never `ACCEPTED`

**"Non-zero exit" is not a verdict.** Collapsing parse errors and fraud detections into one
`caught` bucket is what produced the wrong reading. Every round classifies on the *message*:
`parse-error` | `fraud-detected` | `ACCEPTED` | `panic`. A detector that starts rejecting honest
commitments as malformed would otherwise look like it was getting better at catching fraud.

### The question this leaves open

Audit takes `window_size` from `revealed_items.len()` and ignores the contents. Nothing in the
audit path rejects a commitment whose revealed window has been rewritten. Whether that is
exploitable — a prover who reveals a window that is not the trace's actual first `window_size`
steps — depends on what the transition guest checks at `Init`, and is exactly the question tier 3
(§4) exists to answer. Flagged, not claimed.

## 6. Crashes count

`cargo raster run --commit /nonexistent-dir/c.bin` panics at `raster-cli/src/commands/run.rs:574`
with `Failed to create commitemt file: … NotFound` rather than returning an error — hit on the
first try while measuring §3. A verifier that panics on hostile input is a liveness bug in a
dispute: the honest party cannot produce a verdict. The harness therefore feeds bad paths,
truncated files, zero-length files and wrong-magic files deliberately, and treats a panic as
`crashed` — distinct from `parse-error`, which is the *correct* response to a malformed file.

## 7. Container shape

```text
raster-fraud-lab/
  Dockerfile               # rust + risc0 toolchain + cargo-raster; guests prebuilt into the image
  fixtures/tiny-gemma-dev/ # 84 KB, vendored — no network at run time
  harness/
    faults/                # one module per family; each exposes apply(run_dir) -> Fault { stage, kind }
    oracle                 # expected verdict per (family, kind)
    runner                 # seed -> rounds -> report
  corpus/                  # seeds that found something, replayable by number
  report/                  # JUnit XML for CI + NDJSON per round
```

Two modes:

```sh
docker run … --rounds 50 --seed 1234    # deterministic, a CI gate
docker run … --forever                  # soak; any failing seed lands in corpus/
```

**Build the honest run once, in the image, not once per round.** The honest authenticated run is
the expensive part; at tiny-gemma-dev scale the run directory is small enough to copy, so a round
becomes copy → inject → detect, which is seconds rather than minutes. That is what makes
`--forever` worth having. Family C is the exception — it pays one `chain run --stage` to produce
the dishonest stage.

**Every round works on a copy.** `model-import` rewrites `Raster.toml` and the `*.rastered`
externals in place; inside the container that is harmless, but snapshotting `/work` per round means
a failing round is a directory you `docker cp` out and poke at with the same CLI, which is the
difference between a red CI line and a diagnosis.

Each round emits one row:

```json
{"seed":1234,"family":"C2","stage_injected":"prefill_range_l2","detector":"chain fraud-prove",
 "exit_code":1,"stage_named":"prefill_range_l2","verdict":"detected-correct","secs":4.1}
```

## 8. Implementation order

1. Vendor `tiny-gemma-dev`; get one green honest authenticated run inside the container. This
   confirms or denies §4's blocker, and everything else is downstream of the answer.
2. Family E on `prompt-prepare` — no chain needed, runs today, and its output is the expected-map
   artifact the rest of the suite leans on.
3. Family B on `prompt-prepare` standalone.
4. Family A on a `--no-auth` chain.
5. Families C and D once authenticated chain + `fraud-prove` run.
6. `--forever`, corpus, JUnit.

## 9. Not in scope

- Proving performance. This measures detection and attribution; a fraud receipt's prove time is a
  separate benchmark with separate hardware assumptions.
- The real 35-layer model (§3).
- Faults in the RISC0 guest itself. The harness treats image ids as trusted inputs; a compromised
  guest is a different threat model and a different test.
