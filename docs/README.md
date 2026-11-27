# Documentation

Three kinds of page. Start with [how IWEBS works](protocol/overview.md). The
[glossary](glossary.md) gives each word its one meaning.

## How it works

- [How IWEBS works](protocol/overview.md): the whole of it from the top. With
  no blockchain: what a procedure, a case and a verdict are, what a run
  shows, and the record that carries it. On a blockchain: what a claim is,
  the one rule, and what catches each way a claim can be wrong.
- [The pipeline](protocol/raster-pipeline.md): the stages a run is made of,
  what each checks, and the schedule that holds them in order.
- [Procedure kinds](protocol/procedure-kinds.md): a run of one model, and a consensus of
  two.
- [What a contract holds and checks](protocol/checks.md): the rules, as
  the Rust states them.
- [Blockchain](blockchain/overview.md): Celestia and the EVM contracts: what
  each call checks, what it costs, and which parts are a trusted host.
- [Data availability](protocol/availability.md): what a claim and a
  challenge post, and the procedure registry.
- [Formats and versions](protocol/formats.md): every versioned format and
  hash tag, in one table.
- [What is open](protocol/open.md): every known gap, in one list.

## How to run it

- [An example adjudication](../examples/earthquake/README.md): a policy's
  question judged on a signed report, from the procedure's id to a checked
  record.
- [Model import](workflows/model-import.md)
- [Run](workflows/run.md): a procedure with no blockchain. The two sets of procedures,
  one that finds an output and one that verifies it, the id of the procedure a run is of,
  and the record another party checks
- [Claim build](workflows/claim-build.md)
- [Challenge build](workflows/challenge-build.md)
- [Fault proving](workflows/fault-prove.md)
- [The local blockchains](../blockchain/devstack/README.md): a claim and a challenge taken
  through Celestia and the contracts.
- [Testing](workflows/testing.md): the tiers, and what a pass of each check
  means.
- [The parity gate](workflows/test-parity.md)
- [Replay profiling](workflows/replay-profile.md)
- [Native Windows inference](workflows/windows.md)

The [README](../README.md) lists every command by who runs it.

## Working on it

- [Development](protocol/development.md): workspace commands and the checks
  a stage program has to pass.
- [Model families](protocol/model-families.md): what a new model family
  needs.
- [Numerics](protocol/numerics.md)
- [Artifacts](protocol/artifacts.md): the files a run leaves.
- [Raster issues](issues/README.md): gaps that are Raster's to close.

## Design notes

Two designs are kept because the stage programs' source cites them, and that
source cannot be edited without re-locking the stages:

- [A prompt built from authenticated data](proposals/authenticated-prompt.md)
- [Checking claimed prompt tokens](proposals/claimed-prompt-tokens.md): with
  the argument for why a pair check is enough.

Everything else that was a proposal is built, and is described by the pages
above. The repository's history has the proposals themselves.
