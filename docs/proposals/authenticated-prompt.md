# A prompt built from authenticated data, and a result stage after the claim

Proposed November 16, 2026. Both stages and the tie between source and prompt are built,
natively and as Raster programs, and every stage is re-locked at manifest version 18; see
[What is built](#what-is-built).

## Goal

**One inference whose prompt holds web data a signer vouches for, with a basic version of
each segment a larger process would grow from.**

There are four segments. The third exists.

| Segment | What it does |
| --- | --- |
| Source | Checks a signed attestation that a document came from a named server |
| Prompt | Builds the prompt from a base prompt and values taken out of that document |
| Inference | Verifies the model's output on that prompt. Today's pipeline |
| Result | Turns the verified output text into a value for final use |

After this, a claim's record says which document the prompt's data came from, who signed
for it, and what the output means. The examples in iris-backend keep their shape.

## What does not change

- The challenge: rerun honestly, find the first mismatching checkpoint, replay that stage.
- The native run runs first and decides whether a claim is built.
- A wrong input never aborts a stage. It publishes a fault and the final record is invalid.
- `prompt-check`, the embedding, layer and check programs, apart from the re-lock a change
  to `PromptContext` brings.
- Raster.

## The run's new inputs

A case of a specification that has a source gives these in place of a prompt.

**The documents.** One or more response bodies, as bytes. They enter the stage as one list
of chunks of at most 4,096 bytes, each naming its document and whether it is that
document's last. Any bytes can be given and none of them can stop the stage from starting.

**An attestation for each document.** A signature over the SHA-256 of this message:

```text
"iris/attestation/v1\0"
scheme   u8    1 = ECDSA over secp256k1, message hashed with SHA-256
cipher   u8    0 = none: the records are the response body as received
time     u64   the signer's clock, Unix seconds
server   u16 length, bytes    the host name
target   u16 length, bytes    the request path and query
records_sha256   32 bytes
records_length   u64
```

**The recipe.** How the prompt is made from the documents:

```text
Recipe {
  base: List<String>             the base prompt, in chunks
  documents: List<DocumentName>  the documents a run must give, in order
  fields: List<Field>            in the order they appear in the prompt
  tail: String
}
DocumentName { server, target }  what the document's attestation must name
Field { lead: String, document: u32, path: String }
Form  { text: String, code: u32 }     the output texts the result stage recognizes
```

The prompt text is `base`, then `lead` and the field's value for each field, then `tail`.
A field names which document its value is in. A value is the bytes of the JSON value at
`path`, exactly as the document has them: a number, a string with its quotes, `true`,
`false` or `null`. A missing field, an object or array, a value over 1,024 bytes or
bytes that are not UTF-8 are faults. A lead or the tail may be 1,024 bytes, since a lead
often holds a URL.

A path's steps are separated by a newline, which no JSON key can hold, so a key with a
dot or a slash in it (`comments.json`) is one step. A step into an object is a key, and
the first member with that key is taken. A step into an array is an element's index. A
key is compared as the document writes it, with no escapes decoded.

The signer's public key is a fourth input. As built it is the run's: the record states
the key and the statement's source digest covers it. Pinning it, so it is not the run's
to choose, is not done.

## The stages

```text
source ─▶ prompt_bind_p0 … check_c{last} ─▶ result
```

### `source` (new program)

One stage holds the Source and Prompt segments for every document. Building the prompt alone
is a handful of tiles, and so is a small document: the two PAGER documents of the impact
claim are 833 and 1,456 bytes.

1. Fold SHA-256 over the chunks, and chain each document's hash and length as its last
   chunk is reached.
2. For each attestation: verify its signature against the signer's key, and chain what it
   states three ways: every fact, the server and target, and the hash and length. Compare
   the last chain with step 1's. Equal chains mean the same documents, as many, in the
   same order.
3. Fold the base prompt's chunks into a text hash.
4. Chain the server and target of each document the recipe names, and compare with step
   2's chain of them.
5. For each field, scan the chunks for the value at the field's path, passing over other
   documents' chunks and stopping where the value ends. Fold the lead and the value into
   the text hash.
6. Fold the tail. Publish the record.

Each step loops over one list. Step 5 is Raster's two-collection pattern: a recur sequence
over the fields, the chunks passed to it by reference, and a recur over the chunks inside
it with the field's document and path as its two small arguments. Each replay unit is one
field and one chunk. The scan's state is a depth, how much of the path the enclosing
objects and arrays match, the string and escape flags, and the value so far. No step
looks anything up by index, so nothing a run gives can abort the stage.

Nothing in this stage is claimed. A claimed position for a field could only be checked by
the same scan, because whether a key is a direct member of an object depends on every
brace before it.

```text
SourceRecord {
  present                    false in the constant an unsourced run uses
  text_sha256, text_total    the prompt text the recipe and documents give
  recipe_sha256              chained over the base, each document name, each field, the tail
  documents                  how many
  documents_sha256           chained over each one's attested hash, length, server, target, time
  signer
  fault, fault_position      the position is a document's index or a field's
}
```

### `prompt-bind` (one new input, one new check)

Every bind takes `source`: the source stage's record, or a committed `no_source` constant
in a run that has none. The first range's `open_bind` compares the run's text digest with
`text_sha256` and `text_total`, records a fault when they differ or when the source has a
fault, and stores a digest of the record in the context. From there it reaches the output
record and the statement, as the model id does today.

The text itself still enters the binds as run inputs, one slice per range. The last range
already holds the text it folded to the claim, so a prompt that is not the one the recipe
gives is a fault in a large stage.

`PromptContext` gains the digest. Every program that carries the context is re-locked.

### `result` (new program)

Takes the last check's output and the recipe's forms. It compares the output text with
each form and publishes the first matching form's code or 0, a hash over the forms, and
the claim record it was given. Today that is all it does. It is where several models'
answers would later be compared.

As built, the claim record does not move here. It stays the last check's, and this stage
restates it beside the code so a reader can tell which claim the code belongs to.

The stage is three trace steps and one per form, under the 128 a replay needs today at
any size. Its checkpoint cannot be disputed by a replay until Raster handles short
traces. Two things keep that from mattering now. What it carries is only the output's
code: the source, the prompt, the output and the verdict all rest on `source`,
`prompt-bind` and the checks. And a contract that holds the forms takes the comparison
on the submitted output itself, so a claim with a false code is refused when it is
submitted. A result step too large for a contract to repeat will need to be a stage long
enough to replay.

Neither new stage joins the stage schedule. `source` runs before the first bind creates
the context, and its place is held by the bind taking its record. `result` reads the last
check's record, which already says whether the output is complete.

## Where the logic lives

Two crates, `phases/source` and `phases/result`, hold every function the two new stages compute: the
attestation message, the field scan, the text fold, the form match. Tiles call it, the
the native stages call it and the native run calls it, as with `inference-tokenizer`.

| Where | What it gains |
| --- | --- |
| `iwebs attest sign` | Host only. Signs a document's hash, length, server and target with a key file. This is the trusted stand-in for the notary |
| Specification and case | A `[source]` table in the specification holds the recipe and the signer's key, and `[[form]]` tables the forms. The case names each document and its attestation, and gives no `prompt` or `prompt_file` |
| Native | Verifies the attestation, renders the prompt, runs the model, matches the forms. The rendered text is what the claim's binds are given |
| Pipeline, native | `source` and `result` routines with checkpoints, like every other stage |
| Raster | `raster-stages/source`, `raster-stages/result` |
| Backend | Fetches as it does now, calls `iwebs attest sign`, and writes a recipe where it wrote `prompt.txt` |

The backend's earthquake prompt keeps its wording. Two things change in the data lines:
strings carry their JSON quotes, and times are the document's epoch milliseconds until a
date transform is added to the recipe.

## When the notary arrives

The attestation's `cipher` names how the records are protected, and the message gains the
session's revealed keys. The signed hash then covers ciphertext. `source` gains two steps
between hashing the records and scanning them: decrypt and authenticate each record, and
strip the HTTP framing. The record, the tie in `prompt-bind` and the result stage do not
change.

## Signature cost

Measured on November 16 with `cargo raster run-tile --backend risc0` on a 160-byte
message, no accelerator. The program is `target/spikes/sig-tile`, untracked.

| Scheme | Cycles for one verification |
| --- | --- |
| Ed25519 | 3,285,870 |
| ECDSA, secp256k1 | 5,579,131 |
| ECDSA, P-256 | 11,912,645 |
| Reading the input only | 20,921 |

All three fit in one tile. The pipeline's heaviest tiles measured in the tens of millions of
cycles in the November 11 baselines. I chose secp256k1 because it is what an Ethereum contract can check natively
and, as far as I know, what TLSNotary notaries usually sign with. The `scheme` byte lets
that change.

## Ways a claim can be false

| The claimer | What shows it |
| --- | --- |
| alters a document | its hash is not the signed one: fault in `source` |
| signs with a key other than the one the run names, or not at all | the signature fails: fault in `source` |
| names its own signer key, or builds the prompt by another recipe | the submitted source record's signer or recipe is not the contract's: refused at submission |
| claims a source the contract did not ask for, or none where it asked for one | refused at submission |
| uses a document from another server or path | server or target is not the recipe's: fault in `source` |
| leaves a document out, adds one or swaps two | the documents' names are not the recipe's, in order: fault in `source` |
| gives the model a text the recipe does not produce | the bind's text hash is not the source's: fault in `prompt-bind` |
| skips `source` and makes up its record | the link check: the bind's input is no output of the source program |
| lies about a stage's output | that stage's checkpoint diverges and is replayed |
| lies in `result` | the contract's own comparison of the output with its forms gives another code: refused at submission. The checkpoint also diverges, and is not replayable while the stage is short |
| reads the output by other forms | the forms hash is not the one the contract's forms give: refused at submission |

The signer itself is trusted. A signer that signs a false document is outside all of this
until the notary replaces it.

## Build order

1. `source-phase`, `iwebs attest sign`, the `[source]` table and the native path. The
   backend's earthquake case runs through it. No pipeline change.
2. The native `source` and `result` routines, the bind's new input, checkpoints, and a
   parity case for each row of the table above.
3. The two Raster programs, authored against the Raster skill. Measure trace steps and
   the heaviest tile on a real document, choose the page size, run the check ladder,
   re-lock every program, bump the versions.
4. The challenge scenario on the tampered cases.

## What is built

Step 1, on November 16. No stage program, lock or manifest version changed, and the fast
test tier's lock check passes.

- `phases/source`: the attestation, the field scan, the text and recipe hashes, the
  record and the form match, one function per step a stage would take. It builds without
  the standard library. It is a workspace member through `procedure-prepare`'s dependency on it and
  is not yet listed in the root `Cargo.toml`, because Raster's lock check reads that file
  and listing it would make every stage lock stale before step 3 re-locks them anyway.
- `iwebs attest keygen` and `iwebs attest sign`.
- A `[source]` table in what is now the specification, with the recipe and the signer's
  key, and in the case a list of documents, each with its attestation. When this step
  was built the recipe and the key were files a run spec named.
- `iwebs source render`, which prints the record and can write the prompt text.
- `iwebs infer` and `iwebs claim build` on such a specification use the prompt the source gives. A
  source with a fault gives none, and the run stops there. `iwebs infer` also prints the
  source's facts and the result code. They are computed by the host from the same
  functions, and no pipeline states them yet.
- In iris-backend, on its own `authenticated-data` branch: the impact claim (now version
  4) builds a recipe from the three documents it already fetches, has IWEBS sign them and
  render the prompt at preparation, and gives IWEBS the source files at execution. It
  fails the run if IWEBS's prompt hash is not the prepared input's.

The impact claim by hand on Gemma 4 E2B, with the real documents of one event: an event
record of 34,108 bytes, `comments.json` of 833 and `alerts.json` of 1,456; eleven fields; a
2,388-byte prompt of 751 tokens; the output `{"verdict":"confirmed"}`, result code 1. The
run's `input_text_sha256` is the record's `text_sha256`. With two words of the comments
changed, the source reports fault 3 at document 1 and no prompt is made.

The first half of step 2, the same day: the tie.

- The statement names a source: `Statement` has a `source` digest, 32 zero bytes for a
  text taken as given, and its tag is `iris/statement/v2`.
- `prompt-bind` takes a `source` record. The first range holds the claimed text's hash
  and length to it (faults 13 and 14 in the shared prompt source) and stores the record's
  digest in `PromptContext.source_sha256`. The digest reaches the output record, the
  claim record's statement and the claim submission.
- The native run binds the same statement, so a native result and an honest claim
  agree. A challenger rebuilds a run under the source record the claim fixed.
- On the parity model, imported into a scratch directory and run natively: a claim with
  a source and one without build and pass `claim check`; they have the same text hash
  and different statements.

That left the record a run input. The source stage, below, now produces it.

What it changed on disk: every stage program that reaches the statement or the prompt
context, so every `Raster.lock`, and the model manifest, which is version 17 because the
pipeline template gives the bind a source. The batch below re-locked and re-imported.

The source stage, the same day: `raster-stages/prompt-source`, with its lock.

- **In the pipeline.** The import's template has a `source` stage whose three inputs a run
  fills in, and the first bind takes its record. A run without a source gets a pipeline
  without the stage: claim build removes it and gives the first bind the constant
  `no_source`, which is what every later bind is given in any run. The contract's wiring
  already allows an input to be an earlier stage's output or a pinned empty value.
- **Its shape.** The documents are one list of chunks, each naming its document and
  whether it ends it, so no step looks a document up by index. The stage is twelve steps:
  five loops over one list each, one recur sequence over the fields with a loop over the
  chunks inside it, and six single tiles. Every tile argument is an entry argument or an
  earlier step's whole result.
- **What changed from the plan above.** A document whose bytes are not the attested ones
  is found by comparing two chains, so the fault no longer says which document. A
  signature fault still does.
- **Native.** A stage routine runs the same functions in the same order. A challenger
  reruns the stage on the inputs the claim's prepared run names.

Checked:

| What | Result |
| --- | --- |
| Workspace tests | 73 test binaries pass |
| Parity model, native claim with a source | 198 stages, `source` first; the statement is the native run's; `claim check` passes |
| Parity model, native claim without one | 197 stages; `claim check` passes |
| The Raster program without authentication, on that claim's inputs | output equals the native stage's, byte for byte |
| The Raster program authenticated, same inputs | commit and audit succeed; the output is the same bytes |
| The Raster program on the real impact-claim documents | the record the native run computes; commit and audit succeed; 158 trace steps |

On the parity claim's small documents the stage is 63 trace steps, under the 128 a replay
needs. On the real documents it is 158 with 4,096-byte chunks. Smaller chunks would
lengthen it: `DOCUMENT_PAGE` in `source-phase` is the one number.

The batch, the same day:

| What | Result |
| --- | --- |
| Re-lock of all eight stage programs | verified, 19 minutes |
| Full test tier: unit tests, lock check, both parity gates with their challenge scenarios, the stage ladder | all pass, 41 minutes |
| A claim with a source on the parity model: native, native, and the whole pipeline through Raster | one statement; `claim check` holds every native checkpoint to the boundary the Raster run committed |
| Re-import of Gemma 4 E2B, Gemma 4 E4B and Ministral 3B at version 17 | done; the version 16 imports are in `runtime/model-artifact-backups` |
| Real-model check, with its references rewritten | E2B and E4B claims build with no fault; one stage of each program as Raster gives the native output |
| The impact claim on the real documents, Gemma 4 E2B | 219 stages in 68 s, valid, `{"verdict":"confirmed"}`; `claim check` passes |
| A challenge of that claim with a false `source` checkpoint | divergence found at `source`; the stage replayed through authenticated Raster |

On the parity model the same challenge finds `source` and cannot replay it: the stage is
under the trace length a replay needs there, like that model's bind stages.

The gate's source scenario, the same day (`docs/workflows/test-parity.md` has its cases). The
parity gate now builds one claim from two signed documents and runs it three ways, then
runs seven wrong sources through the source stage and the binds, in Raster and natively.
The stage ladder runs `prompt-source` on that claim's inputs. With these in place the
full test tier passes again, for both families, challenge scenarios included:

| Family | The claim with a source | Bad sources |
| --- | --- | --- |
| Gemma | 198 checkpoints the same through Raster and natively; its record is the native run's | 7 finish through both and record the fault |
| Mistral | 112 checkpoints, the same | 7, the same |

The result stage and the contract, the same day: `raster-stages/claim-result`, with its
lock, at manifest version 18.

- **The stage.** Three tiles: a seed, a loop over the forms, and one that states the
  record. Its logic is `phases/result/src/forms.rs`, which the stage compiles on its
  own, the native run calls and the native routine calls. It reads the last check's
  output through its own copy of the edge type, and a test holds the copy to the check's
  commitment. The forms are a run input, empty for a run that names none, so every pipeline
  ends in the stage.
- **The claim.** A result carries `result_code` and `result_forms_sha256`. A challenger
  reruns the stage on the forms the claim's prepared run holds.
- **What a claimer submits** (`claim_submission.json`, version 3): the source record
  beside its digest, and the result code with the forms hash. About 1,200 bytes for the
  impact claim.
- **What a contract checks.** At submission (`check_origin`): the source record hashes
  to the digest the statement names; its signer and recipe are the ones the contract
  requires, or there is none where the contract requires none; and the result code and
  forms hash are what the contract's own forms give the submitted output. At a
  challenge: the claim record is in the output of the checkpoint before the last, which
  must run the check, and the result is in the last, which must run the result stage.
  `iwebs claim check` takes the signer, the recipe and the forms from the specification.
- **Challenges are unchanged.** `source` and `result` are stages with approved programs
  and piped inputs. A challenger recomputes, finds the first checkpoint that differs
  and replays that stage.

Checked:

| What | Result |
| --- | --- |
| Workspace tests | 76 test binaries pass |
| Re-lock of all nine stage programs | verified |
| Full test tier | passes. Gemma: the claim with a source is 199 checkpoints, the same through Raster and natively, and its result stage states the code of the form its output is. Mistral: 113, the same. A false `result` checkpoint is found by recomputation in both |
| Stage ladder | `claim-result` gives the native output as Raster, and commits and audits at a window of two |
| Re-import of Gemma 4 E2B, Gemma 4 E4B and Ministral 3B at version 18 | done; the version 17 imports are in `runtime/model-artifact-backups` |
| Real-model check, references rewritten | E2B and E4B claims build with no fault; `result` and one stage of each other program give the native output as Raster |
| The impact claim on the real documents, Gemma 4 E2B | 220 stages in 101 s, valid, `{"verdict":"confirmed"}`, result code 1; `claim check` passes all six checks |
| The same submission with a false code, no code, other forms, another signer, a fault, or no source record | each refused at the submission check |
| Challenges of that claim with one false checkpoint | at `source`: found and replayed in 8 s. At `check_c0`: found and replayed in 64 s. At `result`: found, and the replay is refused because the trace is shorter than the window |
| iris-backend at manifest version 18 | typecheck, 146 unit tests and the 4 database tests pass, the last on a throwaway Postgres container |

Not done:

- **A gate case that falsifies the source checkpoint.** On the gate's model the source
  stage is shorter than the replay window. The by-hand challenge above, on the real
  documents, is the only replay of it.
- **Faults in how documents are cut into chunks**, such as a document that never ends,
  are covered by unit tests of the functions the tiles call, not by a gate case.

## Not verified

- **The source stage's trace length on a real document.** A small document with large
  pages could fall under 128 steps, which would leave the stage undisputable today.
- **The page size and the scan's cost per page.**
- **The recur sequence's exact shape**: carried state, an inner recur that stops early,
  and a per-field result read from the inner recur's state.
- **The output text as a tile argument.** The last check's record holds it
  (`OutputRecord.generated_text`), and `result` passes it to a tile once per form. That
  is fine for a verdict of a few dozen bytes and unmeasured for a 600-token output.
- **A replay of the result stage.** It cannot be replayed at the challenge window. The
  contract's own comparison is what refuses a false code, and that is modelled in
  `check_origin`, not run in a contract.
- **A signer other than the host.** The contract compares the signer's key, and the key
  is the host's own.
- **Decryption and HTTP framing costs.** Not measured, and TLSNotary's own attestation
  format was not studied.
- **The cost of a field's scan.** It walks every chunk before its own document's, one
  replay unit each, before it reaches the document. Fine for three documents; it grows
  with fields times chunks.
- **The impact claim's chosen product.** Its prompt names the preferred PAGER product,
  which the backend picks by sorting. The recipe names that product by its index in the
  event record, so the choice is the host's and unchecked. The prompt carries the
  product's id and document links for the model to cross-check, as it did before.
- **That the documents link to each other.** Nothing checks that the comments document's
  address is the link in the event record. The model is asked to.
- **Value sizes on other events.** The longest commentary string measured was 254 bytes,
  on one event. A value may be 1,024.
- **The backend's integration tests.** They need a database and were not run. Its unit
  tests pass.
