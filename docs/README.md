# Documentation

Start with the workflow docs when you want to run the repo, and the internals docs when you need to change how the workflows are built.

## Workflows

- [Model import](workflows/model-import.md)
- [Infer](workflows/infer.md)
- [Claim build](workflows/claim-build.md)
- [Challenge build](workflows/challenge-build.md)
- [Replay profiling](workflows/replay-profile.md)
- [Three-path parity](workflows/test-parity.md)

## Internals

- [Raster chain](internals/raster-chain.md)
- [Numerics](internals/numerics.md)
- [Artifacts](internals/artifacts.md)
- [Model families](internals/model-families.md)
- [Development](internals/development.md)

## Design History

- [Issues](issues/README.md)
- [Proposals](proposals/)
- [Checking claimed prompt tokens instead of recomputing them](proposals/claimed-prompt-tokens.md): implemented; token boundaries as an input, one checking stage per range, the argument for why a pair check is enough, and the stage design.
- [Pipeline version 11: one re-lock for the shared changes](proposals/pipeline-v11.md): implemented; the bind stage per range, a model with no per-layer inputs, the chat template as tokenizer data, and two tile fixes, with the E4B numbers before and after.
- [Binding the prompt one range at a time](proposals/paged-prompt-bind.md): implemented in version 11; one `prompt_bind` stage per prompt range with the hash and scanner state carried between them, whole-text facts as a checked claim, and `prompt_check` reading only its own range.
- [Verifying the stage schedule inside the Raster stages](proposals/verified-stage-implementation.md): implementation order, code targets, and completion checks.
- [Enforcing the right stage](proposals/stage-progression.md): one check per stage execution, authenticated progression, and persistent faults.
- [Claim protocol Phase 1 specification](proposals/claim-protocol-phase-one-spec.md): review draft defining the statement, registered pipeline, package, and acceptance rules.
- [Contract claim pipeline implementation plan](proposals/contract-claim-pipeline-plan.md): phased work for canonical preparation, portable claims, anchored disputes, and verifier contracts.
- [Model identity implementation plan](proposals/model-identity.md): bind consumed model data to component commitments inside Iris stages.
- [Ranked BPE validation](validation/ranked-bpe-2026-09-10.md): reference and parity
  results, authenticated tokenizer audits and costs.
