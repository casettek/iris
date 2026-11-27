# Formats and versions

Every format IWEBS writes that carries a version, in one place. A test holds
the first table to the constants in the code (`the_formats_table_is_current`
in `cli/src/lib.rs`), so a version cannot change without
this page changing with it.

A version changes when a reader of the old form would misread the new one.
Nothing here is read across versions: a reader that meets another version
stops and says which it wants, with one exception noted below.

## Files and records

| Format | Where it is | Version | Constant | Who must follow a change |
| --- | --- | --- | --- | --- |
| Weight file | `<model>.detwgt` in a model bundle | 3 | `DETWGT_VERSION` | Bundles are converted again. Version 2 is still read. |
| Deterministic number specification | The weight file's header | 1 | `DET_NUM_SPEC_VERSION` | Bundles are converted again. |
| Model descriptor | `model-descriptor.bin` in an import, and a stage input | 2 | `DESCRIPTOR_VERSION` | Models are imported again, and every stage that reads it is re-locked. |
| Model manifest | `manifest.json` in an import | 21 | `MODEL_MANIFEST_VERSION` | Models are imported again. iris-backend pins the same number (`IRIS_MODEL_MANIFEST_VERSION`). |
| Prepared run | `prepared_run.json` beside a claim | 22 | `PREPARED_RUN_VERSION` | A claim built before cannot be challenged from its own directory. |
| Execution context | Inside every stage's output | 3 | `EXECUTION_VERSION` | Every stage is re-locked, and every checkpoint changes. |
| Contract pins | `contract_pins.json` in an import | 7 | `CONTRACT_PINS_VERSION` | Pins are written again (`iwebs claim pins`), which changes the procedure's id. |
| Procedure | What a contract registers, by its id | 4 | `PROCEDURE_VERSION` | The contracts' decoding, and every registered procedure. |
| Registry file | The file `iwebs registry add` writes | 4 | `REGISTRY_VERSION` | Registry files are written again. |
| Run record | `record.json` in the directory `iwebs run --record` writes | 1 | `RUN_RECORD_VERSION` | Records written before cannot be checked: the run is made and recorded again. |
| Claim submission | `claim_submission.json`, and what `submitClaim` takes | 8 | `CLAIM_SUBMISSION_VERSION` | The contracts' decoding, and iris-backend where it reads the file. |
| Challenge submission | `challenge_submission.json`, and what `submitChallenge` takes | 2 | `CHALLENGE_SUBMISSION_VERSION` | The contracts' decoding, and iris-backend where it reads the file. |
| Posted blob | Every blob on the data-availability layer | 4 | `BLOB_MAGIC` | Blobs posted before cannot be read: `iris/blob/v4`. |

## Hash tags

A hash that is compared across parties starts with a tag that names what is
hashed and its version, so that no two kinds of value can share a hash. A
tag changes when what is hashed after it changes. The contracts hash the
same bytes, so a tag that both sides compute changes in both.

| Tag | What is hashed | Also computed by |
| --- | --- | --- |
| `iris/statement/v2` | What a claim states: the input text's hash, the model, the flags and the budget | The contracts, and the stages |
| `iris/checkpoint/v1` | One checkpoint: its stage, program, inputs, producers and output | The contracts |
| `iris/checkpoints/v1` | A claim's checkpoint list, as a tree | The contracts |
| `iris/procedure/v4` | A procedure, for its id | The contracts |
| `iris/stated-records/v1` | The records a claim's checkpoints commit to | The contracts |
| `iris/blob-attestation/v2` | What the trusted host signs of a posted blob | The contracts |
| `iris/source/v1` | A source record, for its id | The contracts, and the stages |
| `iris/attestation/v1` | What a source's signer signs of a document | The `prompt-source` stage |
| `iris/token/v1`, `iris/token-prefix/v1` | The transcript's tokens, one at a time | The stages |
| `iris/stub-celestia-blob/v1` | A blob's commitment on the stand-in layer | Nothing else: the real layer has its own |
| `raster/trace-commitment/v2` | A trace commitment's header. Raster's, not IWEBS's | The contracts |

## Not versioned

- The checkpoint hashes (`checkpoints.txt`) and a stage's input list
  (`input_manifest.json`) have one canonical form each and no version: a
  change to either is a change to `iris/checkpoint/v1` or to Raster.
- A stage program has no version. Its identity is its `Raster.lock`, which
  changes whenever its code does.
- The worker protocol between the pipeline runner and its helper processes
  (`VERSION` in `pipeline/src/worker.rs`) never leaves one run of one binary.
