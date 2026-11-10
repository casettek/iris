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
- [Development](internals/development.md)

## Design History

- [Issues](issues/README.md)
- [Proposals](proposals/)
- [Checking claimed prompt tokens instead of recomputing them](proposals/claimed-prompt-tokens.md): draft for review; token ids as an input, one checking stage per range, and the argument for why a pair check is enough.
- [Verifying the stage schedule inside the Raster stages](proposals/verified-stage-implementation.md): implementation order, code targets, and completion checks.
- [Enforcing the right stage](proposals/stage-progression.md): one check per stage execution, authenticated progression, and persistent faults.
- [Claim protocol Phase 1 specification](proposals/claim-protocol-phase-one-spec.md): review draft defining the statement, registered pipeline, package, and acceptance rules.
- [Contract claim pipeline implementation plan](proposals/contract-claim-pipeline-plan.md): phased work for canonical preparation, portable claims, anchored disputes, and verifier contracts.
- [Model identity implementation plan](proposals/model-identity.md): bind consumed model data to component commitments inside Iris stages.
- [Ranked BPE tokenizer](proposals/ranked-bpe-tokenizer.md): implemented fixed-count
  staging, input preparation, and replay contract.
- [Ranked BPE validation](validation/ranked-bpe-2026-09-10.md): reference and parity
  results, authenticated tokenizer audits and costs.
