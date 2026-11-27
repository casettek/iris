# Example: does a reported earthquake reach a policy's magnitude?

One adjudication from end to end, with no blockchain. An insurance policy
pays out if an earthquake of magnitude 5.0 or greater is reported. One model
judges that, on one signed report.

| Word | Here |
| --- | --- |
| Procedure | [`specification.toml`](specification.toml): the judge, three tokens of output, whose signature the evidence must carry, and the answers a verdict may take |
| Judge | Gemma 4 E2B with `Q8_0` weights, named by its import's manifest |
| Question | The `[source]` table: the text the judge is asked, and the three fields read from the evidence |
| Evidence | [`event.json`](event.json), the report a server gave, and [`event.attestation.json`](event.attestation.json), the signature that the server gave it |
| Case | [`case.toml`](case.toml): the evidence of this one case |
| Answers | `yes`, code 1, and `no`, code 2. Any other output is code 0 |
| Verdict | `yes`, code 1 |

The report, its server and its signer are made up for this example. The
signer is a key made with `iwebs attest keygen`, which stands in for a notary;
its secret half is not kept.

## What the judge is given

The source stage checks the signature, reads the three fields out of the
report, and builds this text. The model's own template goes around it.

```text
An insurance policy pays out if an earthquake of magnitude 5.0 or greater is reported near the insured site. Is the magnitude in this report 5.0 or greater? Answer with one word: yes or no.

place: "14 km NE of Exampleville"
magnitude: 5.4
status: "reviewed"
```

`iwebs source render --specification examples/earthquake/specification.toml --case examples/earthquake/case.toml`
prints the source's record without running the model.

## Run it

The judge has to be imported first, as `runtime/model-artifacts/gemma-4-E2B-q8`
([model import](../../docs/workflows/model-import.md)). From the repository's
root:

```bash
example=examples/earthquake

# The id of the procedure: what the parties to the policy agree on.
cargo run --release -p iwebs-cli -- procedure id --specification $example/specification.toml
# procedure: 0cbc56d66898186310b91e76af32592aca410e07469478447a40c039bfe05435

# Run the case, and write its record.
cargo run --release -p iwebs-cli -- run --specification $example/specification.toml --case $example/case.toml \
    --procedure 0cbc56d66898186310b91e76af32592aca410e07469478447a40c039bfe05435 --record my-record
# generated text:
# yes
# pipeline run: 147 stages, and the result the native run stated

# Check the record, as anyone who holds it and the model can.
cargo run --release -p iwebs-cli -- record check --record my-record --specification $example/specification.toml
# record: the same at every stage, and the result it states
```

The id is the one above only with the same stage programs and the same import
of the model. Yours is what `procedure id` prints.

## What the id covers

The id covers the judge, the parameters, the signer, the answers and the stage
programs. Change an answer and the id is another.

The question and the report's address are in the same file, and the id does
not cover them: they are inputs of the case
([the record of a run](../../docs/workflows/run.md#the-record-of-a-run)).
Change 5.0 to 6.0 in the question and the id is the same. So two parties that
name this id have not yet agreed on the question. A record of a case carries
the question it was run on, and whoever checks the record reads it there.

## Change the evidence

Change the magnitude in `event.json` and the signature is no longer of that
document. `source render` and `infer` both stop there, with status 1, and no
verdict is given:

```bash
cargo run --release -p iwebs-cli -- source render --specification $example/specification.toml --case $example/case.toml
# iwebs: source fault 3 at 0: a document is not the attested one
```

A new report needs a new signature: `iwebs attest keygen`, `iwebs attest sign`,
and the new key's two values under `[source.signer]`. That is another signer,
so it is another procedure id.
