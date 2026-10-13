# Authenticated decode initialization cannot commit its finalized draft

Status: open; discovered during ranked-BPE full-chain validation on 2026-09-10.

The unchanged [decode initialization sequence](../../raster-stages/decode-init/src/main.rs)
and [initialization tile](../../raster-stages/decode-init/src/lib.rs) initialize an empty decode
transcript through a draft and returns `finalize(draft)`. Host no-auth execution
produces the expected value. Authenticated full-chain execution reaches this
stage after successful tokenization and prefill, then fails while the Raster CLI
reconstructs the program output for the committed trace:

```text
Failed to replay program output selection:
Missing storage object at coordinates CfsCoordinates([4294967295, 1])
```

The observed run is
`target/bpe-validation/full-auth-short`, using the short parity corpus case.
Its manifest is recorded in that directory's `manifest-path.txt`. The first
twelve stages completed; `decode_init` is stage thirteen. No valid full-chain
commitment or successful full-chain execution audit was produced.

Source inspection points to standalone draft finalization: the host stores the
finalized value at a synthetic storage coordinate outside a recur site, while
the recorder's independent storage replica cannot resolve that coordinate at
`ProgramEnd`. This is distinct from tokenizer recur-output finalization, which
has passed authenticated commit/audit round-trips, including empty input.

Reproduce with the existing `decode_init` project and empty committed input
arguments, or run a normal authenticated inference chain. No tokenizer input is
required by this stage. The three-path host parity gate cannot detect this
problem because it intentionally uses no-auth Raster execution.

The BPE change leaves this stage and all Raster libraries unchanged. Fixing or
otherwise resolving this failure is required before claiming the **full
authenticated inference-chain** acceptance gate has passed. Tokenizer audits,
guest builds, program identity verification, selected tokenizer-stage replay,
and three-path host parity are separate successful checks.
