# Example: does a reported earthquake reach a policy's magnitude?

One adjudication from end to end, with no blockchain. An insurance policy
pays out if an earthquake of magnitude 5.0 or greater is reported. One model
judges that, on one signed report.

There are two procedures here, because finding a verdict and verifying one are
different logic. Nothing in IWEBS pairs them: the second is given the output
the first found, by whoever runs them.

| File | What it is |
| --- | --- |
| [`generation.toml`](generation.toml) | The procedure that finds the verdict: the kind `generate-single`, the judge, three tokens of output, whose signature the evidence must carry, and the answers a verdict may take |
| [`verification.toml`](verification.toml) | The procedure that verifies a verdict stage by stage: the kind `verify-single`, and otherwise what the first says |
| [`case.toml`](case.toml) | The evidence of this one case |
| [`case-with-output.toml`](case-with-output.toml) | The same evidence, with the output the verification holds |
| [`event.json`](event.json), [`event.attestation.json`](event.attestation.json) | The report a server gave, and the signature that the server gave it |

| Word | Here |
| --- | --- |
| Judge | Gemma 4 E2B with `Q8_0` weights, named by its import's manifest |
| Question | The `[source]` table: the text the judge is asked, and the three fields read from the evidence |
| Answers | `yes`, code 1, and `no`, code 2. Any other output is code 0 |
| Verdict | `yes`, code 1 |

The report, its server and its signer are made up for this example. The
signer is a key made with `iwebs attest keygen`, which stands in for a notary;
its secret half is not kept.

## What the judge is given

The source step checks the signature, reads the three fields out of the
report, and builds this text. The model's own template goes around it.

```text
An insurance policy pays out if an earthquake of magnitude 5.0 or greater is reported near the insured site. Is the magnitude in this report 5.0 or greater? Answer with one word: yes or no.

place: "14 km NE of Exampleville"
magnitude: 5.4
status: "reviewed"
```

`iwebs source render --specification examples/earthquake/generation.toml --case examples/earthquake/case.toml`
prints the source's record without running the model.

## Find the verdict

The judge has to be imported first, as `runtime/model-artifacts/gemma-4-E2B-q8`
([model import](../../docs/workflows/model-import.md)). From the repository's
root:

```bash
example=examples/earthquake

# The id of the generation procedure: what the parties to the policy agree on.
cargo run --release -p iwebs-cli -- procedure id --specification $example/generation.toml
# procedure: c6c83c15cc490d25f5f0969e6e01028139caa0b05aeabb694e23c7fd63a5a5fb

# Run the case, and write its record.
cargo run --release -p iwebs-cli -- run --specification $example/generation.toml --case $example/case.toml \
    --procedure c6c83c15cc490d25f5f0969e6e01028139caa0b05aeabb694e23c7fd63a5a5fb --record my-record
# generated token ids: [4443, 106]
# generated text:
# yes

# Check the record, as anyone who holds it and the model can: the generation
# is made again, and its result compared with the record's.
cargo run --release -p iwebs-cli -- record check --record my-record --specification $example/generation.toml
# record: the result it states
```

No stage runs in any of this. With no blockchain, it is all of it.

## Verify the verdict

A verification is another procedure, with another id. Its case states the
output it holds: `case-with-output.toml` is `case.toml` with the ids the
generation printed.

```toml
[[output]]
ids = [4443, 106]
```

```bash
cargo run --release -p iwebs-cli -- procedure id --specification $example/verification.toml
# procedure: b205ae317042e25ff895a1ac0849fa38b2ff522cec9a14f10aeb523c08780743

# Hold the stated output to the model's own choice, stage by stage.
cargo run --release -p iwebs-cli -- run --specification $example/verification.toml --case $example/case-with-output.toml \
    --record my-verification
# inference 0: valid: true, result code: 1
# pipeline run: 147 stages, on the output the case states

cargo run --release -p iwebs-cli -- record check --record my-verification --specification $example/verification.toml
# record: the same at every stage, and the result it states

# A claim on a blockchain is of this same procedure and this same case.
cargo run --release -p iwebs-cli -- claim build --specification $example/verification.toml --case $example/case-with-output.toml
```

The verification does not ask where the ids came from. Change one and it
finishes with `valid: false` and status 1.

Each id is the one above only with the same import of the model, and the
verification's only with the same stage programs. Yours is what
`procedure id` prints.

## What an id covers

Both ids cover the kind, the judge, the parameters, the signer and the
answers. Change an answer and each id is another. The verification's id also
covers its stage programs.

The question and the report's address are in the same file, and no id covers
them: they are inputs of the case
([the record of a run](../../docs/workflows/run.md#the-record-of-a-run)).
Change 5.0 to 6.0 in the question and the ids are the same. So two parties
that name an id have not yet agreed on the question. A record of a case
carries the question it was run on, and whoever checks the record reads it
there.

## Change the evidence

Change the magnitude in `event.json` and the signature is no longer of that
document. `source render` and a generation both stop there, with status 1, and
no verdict is given:

```bash
cargo run --release -p iwebs-cli -- source render --specification $example/generation.toml --case $example/case.toml
# iwebs: source fault 3 at 0: a document is not the attested one
```

A new report needs a new signature: `iwebs attest keygen`, `iwebs attest sign`,
and the new key's two values under `[source.signer]` in both specifications.
That is another signer, so each procedure has another id.
