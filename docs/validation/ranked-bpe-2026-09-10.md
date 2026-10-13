# Ranked BPE validation — 2026-09-10

Ranked BPE is implemented in Raster, staged-native, and direct inference on
`bpe-tokenization`. The tokenizer, independent reference, three-path host parity,
and authenticated tokenizer checks pass. **Full authenticated inference-chain
acceptance remains blocked** by the existing `decode_init` finalization failure
described below. This implementation makes no changes to Raster libraries.

## Correctness and authoring

- Workspace tests: **151 passed, 0 failed, 3 ignored**. The production-tokenizer
  reference test was also run explicitly and passed. Importer tests passed again
  after moving tokenizer validation ahead of artifact writes.
- Both tokenizer libraries passed no-std compilation. Both Raster programs
  passed guest builds and `program --verify`.
- Source and generated CFS inspection confirmed eight explicit merge sequence
  calls, authentic piece/rule iteration, small scan state, authorized dictionary
  references, 256-piece application blocks, fresh finalized output drafts, and
  final convergence checking. `scripts/check_tokenizer_cfs.py` passed.
- Independent expectations use Hugging Face `tokenizers` **0.22.2**, with source
  tokenizer checksums pinned in `tests/tokenizer/`. Production coverage includes
  approximately 3,026 reference tokens. Synthetic coverage includes competing
  ranks, overlapping candidates, newly exposed merges, byte fallback, unknowns,
  special boundaries, whitespace, Unicode, empty input, and literal `</w>`.
- Both standalone tokenizer stages passed authenticated commit/audit round-trips
  with a 32-step fraud-proof window.
- **14 sequential authenticated tokenizer chains, 66 checkpoints:** every
  checkpoint and output payload matched staged-native, and every chain passed
  execution audit. These include 7/8/9 and 16/17 merge boundaries, zero repeats,
  identity after convergence, and a merge across positions 255/256. Empty output
  requires the suite's smaller two-step fraud-proof window.
- A deliberately shortened nine-merge case executed only eight operations and
  failed finalization with `tokenization budget exhausted`; it produced no
  accepted `prompt_prepare/output.bin`.
- Multiple named repeat blocks, independent decode counts, zero-repeat exports,
  frozen-run hash checking, and rejection of legacy version-1 prepared pieces
  are covered by tests. Existing legacy claims were not rewritten.

The verified program commitments are:

```text
prompt-merge    2e084c68dc52d4080e412651d79f5e6825fb03091d66f464df979365c8274e06
prompt-prepare  592e66b2ad10053c0ac3bcfbacd0fb78381591d596f3deb5ab32108f632a1a1d
```

Model import regenerated production and parity templates, affected inputs, and
recorded template provenance through the supported importer and fixture tool.
Prepared runs now declare version 2; replay explicitly rejects incompatible
older formats while leaving their original identities intact.

## Three-path and long-prompt parity

The repository's full three-path gate passed in `host_no_auth` mode:

- Short: 5 input tokens, 1 generated token, 25 Raster/staged-native checkpoints,
  14 direct boundaries, and 1 full logit vector; 89.87 seconds.
- Near-window: 15 input tokens, 8 generated tokens, 103 staged checkpoints,
  91 direct boundaries, and 8 full logit vectors; 364.27 seconds.
- Over-window: 33 input tokens, 8 generated tokens, 106 staged checkpoints,
  91 direct boundaries, and 8 full logit vectors; 242.99 seconds.

All prompt token IDs, compared downstream values, selected tokens, and final
results matched. Total gate wall time was 869.21 seconds, including 170.60 seconds
for builds and arithmetic checks. The report is
`target/parity/1789061470566699000-66638/report.json`.

The long synthetic fixture has **4,000 initial pieces and exactly 3,000 reference
tokens**. Its conservative budget schedules 500 merge batches and one final
stage. **All 501 checkpoints matched**, including input commitments, output
commitments, exact output bytes, and independently recomputed payload/index
roots. All scheduled identity batches executed.

The long comparison used actual prebuilt Raster stage executables in no-auth
mode, with four independent jobs seeded from native predecessor artifacts.
Every predecessor also has an executed and compared Raster result. This checks
the complete tokenizer dataflow but is not a sequential authenticated-chain run.
The report records executable hashes and every stage's measured duration at
`target/bpe-validation/long-program-checkpoints/report.json`.

## Cost and replay measurements

Measurements were collected during concurrent validation work on this machine;
they are observations, not isolated production benchmarks. Bytes below count
files in the indicated run directories, excluding shared model/tokenizer inputs
and build artifacts.

- Long prompt: staged-native wall time **44.51 seconds**; four-job Raster
  validation wall time **1,594.42 seconds**. Individual Raster stages had a median
  of **11.56 seconds** and a maximum of **28.99 seconds**. Their durations sum to
  **6,336.48 seconds** under concurrency; neither that sum nor parallel wall time
  is a measured sequential inference latency.
- Long checkpoint artifacts: Raster **465,167,054 bytes**, staged-native
  **465,165,834 bytes**. These are output and input-metadata files, without
  authenticated traces; duplicate per-job working copies are excluded.
- Authenticated adversarial suite: **1,748.81 seconds** for chain execution,
  excluding subsequent execution audits; **614,941,399 bytes** across Raster run
  directories, including trace/commit artifacts. Matching native directories
  occupied **2,586,421 bytes**.
- The cross-256 fixture alone scheduled 33 stages, took **1,054.86 seconds** for
  authenticated chain execution, and used **596,713,461 bytes** of Raster run
  storage. Full-list scans and scheduled identity batches remain costly even
  though tile materialization and merges per stage are bounded.
- Corrupting a copy of checkpoint index 2 (`prompt_merge_b1`) in a fresh
  version-2 claim identified exactly that first divergence and replayed only
  that stage through the existing authenticated Raster command. A measured
  replay took **1.61 seconds** for stage execution and **16.72 seconds** wall time
  including CLI/build overhead. Its output matched the uncorrupted checkpoint;
  the original claim was preserved.

Authenticated fixture results are in
`target/bpe-validation/ranked-auth2/report.json`. The challenge bundle is
`target/staged-infer/runs/01789062877504403000-pid46554/challenge/prompt_merge_b1/challenge_bundle.json`.
The insufficient-budget failure is retained at
`target/bpe-validation/under-budget`.

## Remaining acceptance blocker

The authenticated full inference regression reached stage 13 of the short
25-stage chain after successful tokenizer, embedding, and prefill commits.
The unchanged, inputless `decode_init` stage then failed:

```text
Failed to replay program output selection:
Missing storage object at coordinates CfsCoordinates([4294967295, 1])
```

Source inspection points to the Raster recorder's handling of standalone draft
finalization. No valid full-chain commitment or passing full-chain execution
audit was obtained. Later authenticated decode stages therefore remain
unvalidated in this run. The failure is documented in
[the decode-initialization issue](../issues/decode-init-authenticated-finalize.md),
with run evidence at `target/bpe-validation/full-auth-short`.

Resolving that failure and rerunning the authenticated full inference chain and
execution audit is required before calling all acceptance gates complete. The
successful tokenizer audits and no-auth three-path gate do not replace this
outstanding check. Reproduction commands for the completed tokenizer checks are
in [the implementation document](../proposals/ranked-bpe-tokenizer.md).
