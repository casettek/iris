# Model identity implementation plan

Date: November 6, 2026

Baseline: `pipeline-refactor` at `0b632b9`

Status: component verification, cross-stage context, and sparse verification are
implemented and validated through the complete three-path parity gate. Tokenizer
consumers authenticate selected complete buckets; main and
auxiliary embeddings authenticate selected pages; decoder reads authenticate
selected tokens. Dense transformer/head inputs and auxiliary projections retain
bounded complete sweeps. No Raster source change is required. See the validation
report for executed checks and the remaining large-model measurement and
fraud-proof limitations.

The shared context carries the expected model ID, compact descriptor, sticky
fault, layer count, completed layer count, and producer role. Each consuming stage
compares actual component data with its expected descriptor role. Native cache
conversions preserve that context. Final acceptance rejects a model fault even
when every execution checkpoint agrees. See [validation](model-identity-validation.md).

## Outcome and scope

Make Iris stage execution establish that every model value affecting the computation belongs to the model named in the claim. Keep one model ID, derived from an ordered package of component commitments. Each consuming stage authenticates its parameters and data against the corresponding commitment and publishes a persistent model fault on disagreement.

Implement this entirely in Iris, using Raster's existing structural hashes, selection proofs, ordinary stage inputs, and tile execution. A challenger noticing different input commitments is not the acceptance criterion: honest execution of a substituted component must itself produce an invalid result.

This work includes the model context and layer-role checks necessary to prevent components of different models or layers from being mixed. It does not claim to solve same-model prompt substitution, the independently supplied claimed-token list, arbitrary program replacement, all range/order attacks, claim-envelope authentication, or the existing short-stage fraud-window limitation. Authorized Iris program identities and authenticated predecessor connections remain prerequisites.

## Design decisions

### Commitment package

Introduce a versioned `ModelDescriptor` with the four global component roots and equal-length arrays of auxiliary and transformer roots. The expected roles are fixed:

| Role | Count |
| --- | --- |
| Tokenizer | 1 |
| Decoder, including special-token and EOS behavior | 1 |
| Embedding table and its parameters | 1 |
| Output head and its parameters | 1 |
| Auxiliary layer, indexed by layer number | L |
| Transformer layer, indexed by layer number | L |

Use the existing imported component structural roots. For the current 35-layer model this is 74 roots, or 2,368 bytes before metadata. Target a few KiB in compact encoding; selection proofs are separate artifacts.

Preserve the current model-ID preimage for compatible imports: the canonical role order and each role's lowercase hexadecimal commitment, using the existing separators in `model_id_of`. A compact package stores binary roots and reconstructs that canonical preimage. Validate the package version, exact role inventory, equal layer-array lengths, and supported bounds before using it. Derive the layer count from those arrays; any serialized count must agree. Unsupported versions and duplicate, missing, reordered, or unknown roles must not be accepted through the legacy JSON adapter.

Keep paths, proof files, and run-specific settings out of model identity. Dimensions, quantization, normalization constants, tokenizer behavior, and decoder flags remain covered by their component roots. Preserve the existing component payloads where possible. Proof-bearing execution views must not become inputs to their own identity calculation.

### Verification at consumption

Add a small `no_std` implementation under `crates/model/identity/`, used by the importer, native execution, and stage tiles. It owns descriptor validation, the canonical model-ID calculation, proof verification, bounded proof decoding, and model-fault definitions.

Raster already provides:

- `raster::core::tree::value_root` for structural hashing of an actual materialized value.
- `raster::core::input::verify_selection_reference` and `verify_selection_proof` for folding a selected value back to its component root.
- Existing page-geometry and list-metadata encodings/checks.
- Host witness generation through the public runtime APIs already used by `VerifiedSelections`.

For each consumed value, establish all of the following inside stage execution:

1. The descriptor's computed ID equals the expected model ID.
2. The proof's source root equals the expected root for the required component and layer.
3. The proof path is the exact field and index required by the operation.
4. The selected hash is computed from the actual parameters or bytes used by that operation.
5. Container metadata and traversal satisfy the requirements below.

A proof's own root, path, or leaf hash is never accepted as its own authority. Prefer verifying and consuming the value in the same tile. A separate verification tile is acceptable only when its authenticated output is the value or bounded state the computation actually consumes; a detached success flag beside unrelated raw data is insufficient.

Use direct hashing for small parameter objects. Use proofs for selected pages, tokenizer buckets, and decoder entries. Whole-component structural hashing is a small-fixture reference oracle, not a requirement to scan multi-gigabyte tables on every lookup.

Prefer proof sidecars to duplicating model weights. Establish their exact stage-visible representation in Phase 1: stage interface discovery needs local schema definitions, and a typed proof or a bounded encoded proof must work in actual guest builds. Follow the existing shared-source pattern used by `prompt-core` for any stage-local boundary definitions; do not turn this into a general schema refactor.

### Metadata and complete traversal

Authenticate every model value that can influence control flow as well as numerical values:

- Dimensions, scales, normalization vectors, attention/donor configuration, and quantization/layout parameters.
- Byte lengths, page sizes, page counts, vocabulary size, and tokenizer bucket counts.
- Full bucket contents when an operation searches a bucket: an omitted merge rule or vocabulary entry can change the result.
- Ordered coverage of complete matrix/head sweeps, with no omitted, repeated, or reordered pages.

Maintain bounded counters or an equivalent authenticated traversal check and compare the final count with authenticated metadata. A valid membership proof for each supplied page does not establish that all required pages were supplied.

Sparse lookup authenticates the selected content at the correct index. It need not scan unused entries. The guarantee is that all data influencing execution agrees with the named model, not that every unused byte in a local file was inspected.

### Model context and role binding

Propagate one model context through prompt state, activations, auxiliary results, caches, decode edges, and final output. It carries the expected model ID, validated layer count, and accumulated model fault. Do not silently replace the expected ID with a different descriptor's ID. Consumers of the final record must compare its model ID with the requested one; a valid execution for model B cannot answer a statement naming model A.

Validate the descriptor against the incoming context in each consuming stage. Merge predecessor faults in a fixed, documented argument order so native worker scheduling cannot affect the result. Finalization must reconcile the prompt context with the decode edge and preserve faults from both.

Add only the progress metadata necessary to bind layer roles:

- Embedding outputs start with zero completed transformer layers.
- A transformer derives its expected layer from its activation predecessor, checks its authenticated parameter layer index, and advances that count.
- Auxiliary outputs publish their authenticated layer index; the transformer requires the auxiliary role for its expected layer.
- Prior and donor caches identify their model and producing layer. Consumers compare those identities with authenticated layer parameters.
- Sequence checking requires the model's final layer depth before applying its head.

Initialize `range-origin` and `decode-init` from the established model context, updating template order if necessary. An empty initializer must not reset or act as a wildcard for model identity or faults.

These checks bind model roles; they do not replace the separately needed prompt, claimed-token, and range-continuity invariants.

### Fault behavior

Add a `model_fault` with stable codes and compact location information: component role, layer, field, and item index where applicable. Preserve the first fault deterministically through the final committed record. Claim acceptance requires `model_fault == 0` in addition to the existing validity checks.

Invalid membership, a descriptor mismatch, wrong role/index, malformed application proof, bad geometry, or incomplete coverage must never produce a clean result. Make representable bad inputs complete with a committed fault using bounded safe values and existing error-carrying patterns. Guard selection preconditions before indexing proof/data containers; test empty and truncated witness containers explicitly. Do not compute unsafe offsets or allocate from unauthenticated dimensions before validation.

Unreadable files or malformed Raster artifacts may fail as input-availability errors. They must not fall back to unverified execution. Keeping an invalid computation running is only for producing an explicitly invalid record.

## Implementation phases

Complete each phase in reviewable commits. Normal claim creation must not advertise the new guarantee until every model consumer is covered.

### Phase 1 Prove the verification path in a real guest

Build a tiny decoder-entry fixture and a small paged-weight fixture using existing imported formats. Generate witnesses with Raster's public host APIs. Verify actual typed values, expected roots, paths, and page geometry using the shared `no_std` helpers.

Exercise a tile inside an existing stage, avoiding a new tiny standalone verification stage. Confirm the proof representation is accepted by stage schema discovery, native execution, and the RISC Zero guest. Use a schema-valid bad proof that reaches the tile and returns a model fault.

Deliver:

- Shared verification primitives and canonical hashing reference tests.
- The chosen bounded proof encoding and stage-visible representation.
- A decoder-entry substitution demonstration and page-path/byte substitution tests.
- Initial measurements of proof size, tile cycles, and memory.

Exit criterion: an honest value passes, while changed data with the original model commitment fails through actual Raster execution. No Raster source modification is required. If an API limitation appears, first implement bounded encoding/adaptation in Iris; document a concrete blocker before proposing upstream work.

### Phase 2 Emit the descriptor and reusable witnesses at import

Update `crates/model/import/src/main.rs`, `externals.rs`, and `direct_manifest.rs` to produce the descriptor plus deterministic witness sidecars or derived verification views under the model's artifact directory.

Generate witnesses from the actual imported payloads and indices and cross-check their roots. Keep verification artifacts separate from the component payloads whose roots define the model, avoiding circular commitments. Bind the descriptor and verification artifacts as ordinary inputs in the generated model template.

Move the shared ID computation out of the host-only artifacts implementation, retaining compatible JSON adapters in `inference-artifacts`. Validate descriptor roles and counts against the import.

Exit criteria: unchanged component data retains its model ID; relocating the import leaves identity unchanged; changed component bytes or role assignment change identity. Repeated imports produce identical descriptor/proof content. A direct-only import lacking the staged package cannot be used to build a claim.

### Phase 3 Carry model context and faults end to end

Update the shared prompt source, activation/auxiliary/decode boundaries, native conversions, initializers, and final result contracts. Add the minimum layer-role markers described above.

Wire the descriptor and context through `run-prep`, the claim manifest renderer, the staged runner, and worker serialization. Make finalization preserve model faults and compare its prompt and edge identities. Extend claim construction and challenge result validation to reject a committed model fault even when every checkpoint matches.

Keep the new format unavailable for normal claims until Phases 4 and 5 cover every consumer. No optional missing-context default may produce a valid record.

Exit criteria: context disagreement and an earlier model fault survive every branch and join, including initializers and finalization. Tests cover native cached execution, auxiliary workers, and subprocess execution.

### Phase 4 Bind tokenizer, decoder, and embedding lookups

Apply the shared checks to:

| Stage | Required binding |
| --- | --- |
| `prompt-bind` | Tokenizer profile, vocabulary bucket count, and complete selected vocabulary buckets |
| `prompt-seed` | Boundary and merge metadata and complete selected buckets |
| `prompt-merge` | Merge metadata and complete selected rule buckets |
| `prompt-prepare` | Vocabulary and merge metadata and complete selected buckets |
| `input-embedding` | Hidden size, scale, table geometry, and selected embedding pages |
| `output-finalize` | Selected decoder entries, including text, special flags, and terminal flags |

Update the corresponding native routines and shared prompt logic in the same commits. Verify malformed or truncated lookup witnesses cannot bypass checking through an empty sweep or out-of-bounds selection.

Exit criterion: whole native and Raster chains run with the original declared model ID and substituted tokenizer, embedding, or decoder data, and finish with a nonzero model fault. Honest results remain identical across implementations.

### Phase 5 Bind auxiliary layers, transformer layers, and output head

Extend `prefill-prepare-aux`, `prefill-range`, and `sequence-check` with authenticated parameter blocks, selected-page checks, and complete matrix/head coverage checks.

Verify the expected layer role before consuming its data. Include all projection, normalization, PLE, attention, softcap, and cache-donor settings. Check prior/donor model and layer context and require completion through the expected final layer before head scoring.

Use shared verification rules in `inference-kernels` and staged weight loaders. Native caching may retain verified immutable values, but its keys must include the expected model/component root and relevant selection identity. Cached proof-validation results must also be tied to the supplied witness identity. A cache hit must not bypass current descriptor, context, role, coverage, or fault checks, or hide a changed/invalid witness that Raster would reject. Keep Raster's authored verification in execution regardless of native caching.

Exit criteria: altered weights, parameters, normalization values, head pages, exchanged layer roles, and shortened/repeated sweeps produce explicit model faults in both implementations. Cached and uncached runs produce identical checkpoints.

### Phase 6 Integrate artifacts and validate the complete claim path

Update `ModelManifest`, `PreparedRun`, final output/result readers, worker messages, and changed stage schemas together. Bump incompatible artifact versions, reject old artifacts explicitly, and retain instructions for replaying historical claims with their original implementation.

Refresh generated templates through import and rebuild affected Raster identities with the documented build tools. Do not hand-edit `Raster.lock`. Direct inference remains the output-discovery path; adapt its result/parity representation without claiming that native direct execution is a Raster proof.

Extend the parity challenge leg so model-substitution tests assert the committed model fault, not just a differing checkpoint. Retain tests for falsified checkpoint detection as a separate behavior.

Update the chain, import, claim, challenge, and parity docs to describe the new execution-enforced checks accurately. Record measured costs and any unresolved replay limitations in a validation report.

Exit criterion: the acceptance matrix below passes, the new package is required in normal claim workflows, and no unverified compatibility path remains.

## Acceptance matrix

For substitution tests, keep the original expected model ID and coherent Raster input commitments for the substituted artifacts. Deliberately update a test artifact's outer commitment when its content changes so the test reaches Iris's semantic check rather than merely failing Raster input loading.

| Case | Required result |
| --- | --- |
| Honest model, multiple prompt/output ranges and donor layers | Identical native/Raster checkpoints and clean model fault |
| Changed descriptor root, missing/duplicate role, incorrect layer count, unsupported version | Descriptor/model fault or explicit incompatible-format rejection |
| Different tokenizer profile, vocabulary, merge rule, or boundary data | Model fault in the consuming prompt stage |
| Different decoder text, special flag, or EOS flag | Model fault in finalization, even when decoded text happens to be unchanged |
| Different embedding, auxiliary, transformer, or head values | Model fault at the first affected consumption |
| Valid proof for another model, field, layer, page, or token index | Model fault |
| Correct proof beside separately substituted computation bytes | Model fault; verified bytes cannot be disconnected from used bytes |
| Changed dimensions, byte length, page size, bucket count, truncated bucket, omitted/duplicated/reordered sweep page | Model or coverage fault; no clean partial computation |
| Malformed, empty, or truncated application witness containers | Bounded failure or committed model fault; no unverified fallback |
| Mixed predecessor model contexts, exchanged auxiliary roles, wrong cache producer layer | Model fault retained through final output |
| Self-replay of substituted inputs through both implementations, evaluated by the shared claim-acceptance predicate | Matching checkpoints plus rejection on the final model fault; no need to invent execution divergence |
| Corrupted checkpoint from an otherwise valid claim | Existing divergence/replay behavior remains intact |
| Relocated artifacts and different native cache/worker settings, including an invalid witness after warming the cache | Same model ID and checkpoints; cached execution preserves the same faults |

Compare selective verification against whole-component structural hashing on small fixtures. Add mutation tests that alter every model field consumed by each stage, including control metadata; honest parity alone does not establish coverage.

Run focused crate tests, relevant `--no-default-features` checks, stage checks, and program identity rebuild/verification as specified in [development](../internals/development.md). Then run the complete [three-path parity gate](../workflows/test-parity.md) and authenticated Raster executions for the changed verification paths.

Exercise an actual fault proof for a forged clean result where the current replay tooling supports it. Use measured traces exceeding the current fraud window; explicitly retain short-stage limitations instead of treating skipped proof tests as success. Do not lower the production fraud window as part of this work.

## Implementation ownership

| Area | Main files |
| --- | --- |
| Shared model semantics and proof helpers | New `crates/model/identity/`; workspace `Cargo.toml` |
| Descriptor and verification artifact generation | `crates/model/import/src/{main,externals,direct_manifest}.rs` |
| Package/run/result contracts | `crates/inference/artifacts/src/{direct,run,inference}.rs` |
| Prompt model context | `crates/prompt-core/src/prompt.rs` and its stage-local shared source |
| Authored enforcement and boundary types | `raster-stages/*/src/{input,lib,main}.rs` as applicable |
| Native parity and conversions | `crates/inference/kernels/`, `crates/inference/staged/src/routines/`, `cache.rs` |
| Verified loading and cache behavior | `crates/inference/staged/src/{verified_weights,embedding,ple,head,weights}.rs` |
| Run wiring and claim rejection | `crates/model/run-prep/src/lib.rs`, `crates/inference/cli/src/{claim,challenge}.rs` |
| Adversarial and parity coverage | `crates/inference/cli/src/parity/`, staged integration tests, model-identity unit tests |

## Cost and completion

Measure compact descriptor bytes separately from witness storage. Report import time, native claim time, per-stage Raster trace steps, guest verification cycles, peak memory, and replay/proof cost on the pinned small model and a representative 35-layer model. Compare against the baseline with identical model, prompt, output, and range settings.

Avoid re-verifying an unchanged parameter block per token when a stage can verify it once and consume that authenticated result. Verify each page/bucket at the granularity actually used; do not replace sparse lookup with an unconditional full-table scan. Any further reuse must remain tied to verified bytes and cannot rely solely on a host cache flag.

The work is complete when every model-dependent read has an identified verification path, same-model role substitutions are checked, all model faults reach final acceptance, native/Raster outputs agree, and the measured cost is recorded. The separate pipeline and short-stage gaps remain explicit.

## Optional Raster improvement

A future authenticated commitment accessor could expose an input reference's structural root as a sequence value whose derivation is checked during replay. Iris could then compare whole component commitments directly and avoid much duplicate hashing and witness plumbing.

This is an optimization opportunity, not a dependency of this plan. Reading a host-side reference field or passing a claimed root into a tile would not provide that guarantee. No Raster changes are planned.
