# Iris staged pipeline implementation handoff

Date: November 5, 2026  
Baseline: `pipeline-refactor` at `fc4e894`

## Goal

Extend the teacher-forced staged pipeline so its execution establishes this relationship:

> We consumed the text identified by this input commitment, ran this committed model with these configuration parameters, and produced the text identified by this output commitment.

The native stages and their Raster-authored counterparts must perform equivalent computations and produce identical committed outputs. The relationship must be established by their execution across the complete pipeline.

The developer should create a detailed implementation plan identifying the necessary stage changes, shared types, native implementations, and tests, then implement those changes in reviewable phases.

## Scope

Keep the existing challenge mechanism: the challenger recomputes its own staged pipeline, finds the mismatching stage, replays that stage in Raster, and submits its fingerprint. This task is about what the pipeline executes and produces. It does not include designing a protocol, contract, dispute system, or separate execution-plan verifier.

Keep host preparation untrusted. The host may format the prompt, tokenize it natively to calculate token counts, determine range sizes, and construct the run manifest. The stages must establish the correctness of the prepared inputs they consume. Host checks may remain for early feedback.

Reuse the existing tokenizer, model stages, Raster commitments, and execution machinery. Keep the current range-partitioning behavior unless the stage work exposes a concrete reason to change it. Do not introduce alternate partition policies or generalize the execution system as part of this task.

Use a compact identifier for the model's committed Raster representation. It does not need to be the original GGUF or DETWGT file hash. Start with the commitments already produced during model import and add only what is needed to identify the model actually consumed.

Teacher-forced execution becomes the sole staged inference path. Retain direct autoregressive inference for generating the answer that the teacher-forced pipeline checks.

## Required pipeline behavior

### Input text and preparation

The pipeline must compute or verify the commitment to the input text and establish that this same text becomes the pieces and ranges consumed by tokenization. Hashing the text while consuming unrelated pieces is insufficient.

Use the original UTF-8 prompt bytes as the starting definition of input text. Apply the existing chat formatting and normalization behavior under the selected configuration. Make that transformation part of the stage computations or have the stages verify the host-prepared transformation.

Extend the relevant stages to check complete, ordered range coverage, safe tokenizer boundaries, completed merging, and range limits. The host can run its own tokenizer first to calculate range sizes; native and Raster stages must still establish that the resulting preparation is correct.

### Model and configuration

Identify the committed weights, tokenizer and decoder data, and configuration actually used by execution. Inspect the existing imported commitments before deciding whether a small aggregate identifier is needed.

Ensure the stage inputs and their committed connections bind those model components to their intended roles. Carry or verify the necessary model and configuration context through the pipeline. Do not attach a model identifier to the final output without connecting it to the values the stages consumed.

Keep local file paths as loading information, not model identity. Moving the same imported model to another directory should not change which model it represents.

### Stage connections and completion

Author the stages so their combined execution enforces the intended computation. Identify the context each stage needs from preceding stages, such as sequence position, layer/range position, and accumulated state. Reuse existing Raster input and output authentication.

Ensure substituted inputs, missing ranges, incorrect ordering, or incomplete work cannot produce the expected valid final result. Put the necessary checks and state propagation in both the native implementation and the Raster-authored version. The run manifest orchestrates those stages; it does not replace their checks.

Preserve the existing token-selection verdict. Correct execution over incorrect claimed tokens can produce a nonzero mismatch count in both implementations. Keep that invalid result explicit rather than inventing a checkpoint mismatch where the executions agree.

### Output text commitment

Compute the output-text commitment inside a pipeline stage and include it in the final committed output. Extending `output-finalize` is the natural starting point.

Hash the exact UTF-8 bytes of the decoded text emitted by the pipeline, using the existing decoder behavior for special tokens, byte fallback, whitespace, and EOS. Implement the same computation natively and in Raster.

The current explicit hash in `output-finalize` identifies the token-ID JSON. Preserve it where useful and add the text commitment as a distinct field. The host should copy the computed text commitment from the stage output into the result artifact; it must not supply an unrelated hash after execution.

Expose the input commitment, model/configuration identity, and output-text commitment together in the final result, using the context established during execution. Preserve the existing validity information. Move any remaining essential token-budget, EOS, and completion checks from host-only validation into the relevant stage computations.

## Recommended implementation phases

### Phase 1 Connect the input and model to the staged computation

First map the current data flow and identify exactly which checks exist only in host code. This is a focused implementation-planning step: name the stages and shared types to change, the context they must consume and emit, and the matching native code.

Implement the input-text commitment and its connection to formatting, pieces, and ranges. Reuse the existing partitioning logic. Establish the model/configuration identity from imported commitments and connect it to stage inputs.

**Completion criteria:** honest preparation preserves native/Raster parity. Altered text, substituted pieces, omitted or duplicated ranges, incorrect range sizes, and substituted model inputs are detected by stage execution or change the committed result as appropriate. The model identity is independent of file location.

### Phase 2 Complete the stage data flow and output commitment

Implement any remaining context propagation and completion checks across the model and sequence-check stages. Extend finalization to compute the output-text commitment and expose the complete input/model/output relationship in the final result.

Update the corresponding native implementations, shared output types, result readers, and claim artifact writer together. Limit artifact changes to what is needed to expose the pipeline's results.

**Completion criteria:** both implementations produce the same final fields and commitments. Independently hashing the exact emitted text reproduces the output-text commitment. Invalid token selections, stopping behavior, or incomplete execution remain distinguishable from valid results. The host does not supply the final commitments independently of stage execution.

### Phase 3 Validate parity and existing Raster replay

Extend the current tests and parity suite for the changed stages. Test honest execution and deliberately altered preparation, intermediate state, model inputs, and final output.

Use the existing challenge workflow to demonstrate that a falsified checkpoint is found by recomputation and its stage can be replayed in Raster. No new challenge mechanism is required.

Measure representative replay costs. Check the existing short-stage fraud-window limitation before introducing tiny standalone checking stages; prefer incorporating their work into suitable existing stages. Verify both small semantic fixtures and representative model sizes.

**Completion criteria:** native/Raster parity passes, output commitments are reproducible, and changed stages remain replayable through the existing tooling. Invalid results are represented consistently in both implementations.

### Phase 4 Remove staged autoregressive execution and update workflows

Retain the old staged autoregressive path temporarily as a test reference during the preceding phases. Replace the parity coverage that depends on it, then remove its exclusive templates, CLI branches, and stages. Keep direct autoregressive generation and any shared helpers or types still needed by teacher-forced execution.

Update examples, defaults, workflow documentation, and generated model templates. Version changed artifact formats where necessary and give clear errors for incompatible artifacts.

**Completion criteria:** normal staged workflows use only teacher-forced execution. Direct generation still works, replacement parity tests pass, and the documented workflow produces the input/model/output commitments from the pipeline.

## Tests to include

- Original text versus formatted prompt, including Unicode, whitespace, special tokens, and existing normalization behavior.
- Range coverage, unsafe cuts, insufficient merges, incorrect counts, and sliding-window boundaries.
- Changed weights, tokenizer/decoder data, or configuration, including incorrectly assigned model inputs.
- Incorrect predecessor state, missing or duplicated ranges, wrong ordering, and incomplete execution.
- Honest output, non-greedy claimed tokens, EOS and token-limit behavior, and altered decoded text.
- Identical commitments across native and Raster execution, with independent checks of text hashes.
- Existing checkpoint mismatch detection and Raster replay for the changed computations.

## Starting points in the repository

| Area | Relevant code |
| --- | --- |
| Host preparation | `crates/model/run-prep/src/lib.rs` |
| Run manifest construction | `crates/inference/cli/src/claim.rs` |
| Checks currently performed by the host | `crates/inference/cli/src/challenge.rs` |
| Imported model commitments | `crates/model/import/src/externals.rs`, `direct_manifest.rs`, `main.rs` |
| Tokenization stages | `raster-stages/prompt-merge/`, `raster-stages/prompt-prepare/` |
| Claimed-token checks and output | `raster-stages/sequence-check/`, `raster-stages/output-finalize/` |
| Native stage implementations | `crates/inference/kernels/`, `crates/inference/staged/` |
| Result types and claim artifacts | `crates/inference/artifacts/` |
| Parity suite | `crates/inference/cli/src/parity.rs` |
| Known replay constraint | `docs/issues/short-stage-fraud-window.md` |

The detailed implementation plan should specify concrete changes to these stages and modules, their tests, and a reviewable change sequence. Its focus is making the staged pipeline execute the required computation and emit the required commitments.
