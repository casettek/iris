# Model identity validation

November 6, 2026; baseline `pipeline-refactor` at `0b632b9`.

Every model-consuming stage performs component identity checks. Sparse tokenizer,
main-embedding, auxiliary-embedding, and decoder reads now use reusable membership
paths. The earlier full-component scan results below are retained as historical
validation; the sparse integration results are recorded separately.

## Sparse integration

The importer writes separate tokenizer, embedding, and per-layer auxiliary witness
artifacts and binds them into every consuming stage. These files do not change
component roots or the model ID. Small actual metadata fields plus witness list
roots reconstruct the expected component root. A bounded path then authenticates
the complete selected bucket or exact consumed page, including page geometry.
Requested indices and sibling directions come from the numerical/tokenizer lookup.

The four prompt stages check vocabulary, merge, and boundary buckets at the same
indices their tokenization uses. Hashing a selected bucket is a bounded recursive
sweep, so omitted, duplicated, reordered, or changed entries fail even if the
matching entry itself is unchanged. Merge faults survive later better matches and
all 64 rounds. Separate vocabulary preflights retain faults when an unknown atom
emits no pieces. Range slicing initializes the merge-fault carrier explicitly.
Main and auxiliary embedding preflights use the same immutable page references
and byte-offset functions as numerical execution. Dense auxiliary projections,
transformer matrices, and head projections still use complete sweeps.

Native execution verifies current witness inputs on every invocation, including
cache hits. Unread-page changes can legitimately pass with original valid paths;
changed consumed pages fail, and newly rebuilt self-consistent paths cannot
replace the component root in the statement. Verification is proportional to
consumed pages/buckets and logarithmic paths, not complete sparse table bytes.
Repeated lookups can still reverify the same page in Raster; witness paths are
stored independently rather than as a shared proof tree.

The final default workspace run passes 270 tests, with 19 opt-in tests excluded.
All 11 no-default-features stage checks pass. The four explicit guest tests pass,
including the six-stage sparse matrix; authenticated multi-page and repeated-range
worker tests pass separately.

Executed sparse checks include shared membership tests over all list lengths
0–129, malformed/extra/truncated paths, index/count substitution, full-bucket
mutation, warm-cache witness replacement, authenticated multi-page embedding and
auxiliary parity, repeated-range worker parity, and an authenticated 64-round merge
whose early membership fault survives a later valid match. All six production
sparse guests also execute honest and adversarial cases. Guest case logs are
`/tmp/iris-sparse-guests.log`; authenticated tests are in
`/tmp/iris-sparse-auth.log`, `/tmp/iris-sparse-auth-adversarial.log`, and
`/tmp/iris-sparse-merge-auth.log`.

On the pinned four-layer model, the new embedding witness payload is 300 bytes;
each of four auxiliary witnesses is 300 bytes; the tokenizer witness is 62,384
bytes. The descriptor remains 392 bytes and the existing decoder witness remains
181,380 bytes. These figures exclude `.rindex` files. On a three-element list and
1,024-byte selected value, sparse prompt membership guest execution measured about
98,000 cycles. Full 195,840-byte page verification with a two-sibling path measured
8,041,604 cycles in `input-embedding` and 8,041,867 in `prefill-prepare-aux`.
The historical complete-page hash measured 8,006,098 cycles: the sparse path adds
about 35,500 cycles while avoiding all unread pages. These are verification
fixtures, not a 35-layer runtime benchmark. The full-page test log is
`/tmp/iris-sparse-page-cost.log`.

The complete corrected three-path gate passes: 26, 60, and 96 matching checkpoints
on the three honest cases, each with a clean model fault. All 12 model-substitution
self-replays have 96 matching native/Raster checkpoints, preserve the original
model ID, publish `model_fault = 1`, and are rejected by the shared acceptance
predicate. Checkpoint-challenge, range, budget, unknown-token, and dropped-cache
regressions also pass. The final report is
`target/parity/1794017152997858926-1811081/report.json`; total validation wall time
was 595.55 seconds with the build reuse described below. This is not an inference
benchmark. All 11 Raster programs were rebuilt and verified; prompt-seed was
reverified after correcting its range-slice draft initialization.

The tokenizer substitution fixtures explicitly alter all bucket contents. Replacing
only unused vocabulary data in a seed/merge stage can correctly leave its model
verdict clean under sparse verification; the old full-scan fixture expectation was
removed. A retained-fixture sanity test verifies all four tokenizer substitutions
produce a model fault before the complete gate runs.

The final gate uses a temporary Cargo wrapper under `target/iris-host-build-cache`
to avoid redundant stage compilation. It first runs real Cargo for each source/
environment snapshot and only reuses the same resulting binary after checking its
SHA-256. It does not skip any stage execution, replay, or test assertion. The
wrapper is not part of Iris or Raster production code.

## Implemented path

Full imports produce the bounded descriptor and a separate reusable decoder
witness. The canonical component ordering and model-ID preimage are unchanged.
Finalization binds that descriptor to the model ID from the prompt context, then
authenticates the exact token text and flags consumed at each generated ID.
Proof depth and sibling directions are derived from the requested position.

The authored sequence calls `begin_authenticated_finalize` and
`consume_authenticated_decoder`; these are production tiles, not an unused
prototype. Native finalization calls the same pure implementations. Verification
continues after EOS. Failures preserve the supplied model ID, use a safe empty
decoder token, and set a sticky `model_fault` on the final record. Claim build
and challenge acceptance reject that fault even without checkpoint divergence.

Model manifests and prepared runs are v5. A result missing `model_fault` is
rejected rather than defaulted to success. Direct-only imports can still
discover output text, but their records carry a nonzero model fault and cannot
authorize staged claims. No Raster source changes were made.

## Component and context implementation

The prompt context binds the descriptor to the expected model ID. ModelContext
then carries that descriptor, ID, sticky fault, layer count, completed layer count,
and producer role. Missing contexts do not default to an accepted model.
Embedding, auxiliary, transformer, and head outputs preserve it; prompt branches
and cache joins check agreement. Transformer layers check the expected auxiliary
role, their own progression, and the selected donor's producer layer.
Origin/embedding cache sentinels must have empty KV lists; an origin decode edge
must have an empty token history. Finalization checks the complete decode context
against its prompt and descriptor.

Dense identity sequences stream the same immutable input reference consumed by
numerical execution. List hashing uses Raster's duplicate-last AppendFrontier,
encoded in a bounded page (at most 2,088 bytes). Component field names/order,
list lengths, and page content/index/offset/length affect the expected root.

Each stage has a source-only directory link to the shared identity crate. Raster's
guest-cache fingerprint traverses it, so shared verification changes invalidate
cached guests. Separate schema-only links expose context and sparse witness types
without changing Raster. Native cache conversions preserve the full model context.

## Historical full-scan validation

The expanded workspace regression suite passes (263 tests, 17 opt-in tests
skipped in the default run). All 11 stage no-default-features checks pass.
Host sequence binaries explicitly require the std feature; proof guest libraries
remain no_std-compatible.

Explicit guest execution passes for production tokenizer-context, embedding,
auxiliary, transformer, and head verification tiles, including changed roots,
wrong layer roles, and sticky faults. The page-hashing guest test checks the
replay-proven carried-state commitments, including a full 195,840-byte page.
The final-context guest test rejects a different model, incomplete layer
progression, and an inherited model fault. Authenticated complete finalization
also passes its cached/uncached native comparison on honest and hostile inputs.

All 11 changed Raster programs were rebuilt and verified against generated
Raster.lock files after the shared-source cache links were installed. Builds used
the installed RISC Zero compiler, with temporary Cargo wrappers under target/
to reuse dependency artifacts; no Raster source or compiler changes were made.

The expanded three-path parity gate passed all three cases: 26, 60, and 96
matching checkpoints with clean model faults. All 12 model-substitution cases
also passed: 96 matching native/Raster checkpoints per case, model_fault = 1,
and rejection by the shared claim-acceptance predicate. Coverage includes each
prompt consumer, both embedding range types, the head, auxiliary and transformer
weights, exchanged auxiliary roles, and decoder substitution with original and
rebuilt proofs. Existing checkpoint-divergence, bad-range, and dropped-cache
regressions passed too.

The report is target/parity/1794009164117288573-4019749/report.json. Total gate
wall time was 3,746.38 seconds, including compilation and the expanded challenge
matrix while program rebuilds ran concurrently. This is a validation duration,
not a controlled performance comparison with the earlier decoder-only gate.

Earlier decoder-only results below are retained as historical validation.

## Executed checks

- Workspace tests passed (255 passed, 14 opt-in tests ignored) with disk-backed temporary storage:
  `TMPDIR=$PWD/target/test-tmp cargo test --workspace`.
- `model-identity` and the `output-finalize` library pass no-default-features
  checks.
- The finalization CFS was generated; the RISC Zero program was rebuilt and
  verified against its generated `Raster.lock`.
- `model_identity_guest` explicitly executes both production verification
  tiles in RISC Zero and compares their outputs with native execution.
- `staged-infer --test model_identity` explicitly executes the complete authored
  finalization sequence with authenticated Raster storage. Honest data, changed
  hidden special-token text, changed EOS flags, a wrong descriptor, and a wrong
  proof agree byte-for-byte with cached and uncached native finalization.
  Changed artifacts receive their own correct outer Raster commitments, so
  these tests exercise Iris verification rather than an input-loading failure.
- Unit regressions cover index relabeling, odd-width trees, malformed paths,
  descriptor truncation, unsupported versions, role ordering, sticky faults,
  verification after EOS, and rejection of old result records.

Commands for the explicit execution tests:

```sh
cargo build -p output-finalize
TMPDIR=$PWD/target/test-tmp \
STAGED_INFER_REPLAY_BIN_DIR=$PWD/target/debug \
cargo test -p staged-infer --test model_identity -- --ignored --nocapture
cargo test -p iris-cli --test model_identity_guest -- --ignored --nocapture
# Recheck only the full-chain model-fault leg after a completed parity run:
IRIS_PARITY_RUN=$PWD/target/parity/<run> \
cargo test -p iris-cli decoder_self_replay_rejects_matching_dishonest_execution -- --ignored --nocapture
```

The complete three-path gate passed all three pinned cases and its existing
challenge leg in 1,187.50 seconds. The cases compared 26, 60, and 96 checkpoints
respectively. Artifacts are under
`target/parity/1794005311872411063-3336848/`. Both added model-fault self-replay cases also passed independently against those
retained fixtures: 96 matching checkpoints per case, followed by rejection on
the committed model fault. One case substitutes decoder data; the other also
substitutes its rebuilt membership proofs.

Decoder's selective checks authenticate consumed values. Changing only unread entries,
while supplying valid proofs for the unchanged consumed entries, need not cause
a model fault. The self-replay fixture therefore changes every entry, including
hidden special-token text, to guarantee changed consumed bytes.

A repeated pinned-model import produced 35 byte-identical files; importing under
a different artifact root retained the same model ID.

## Measurements

Expanded guest measurements: tokenizer context verification took 160,726 cycles;
embedding verification with context checks took 133,346; auxiliary and transformer
role checks took 115,499 and 115,742; head verification took 114,080. Valid final
context verification took 158,774 cycles. Streaming a 195,840-byte page into the
canonical list frontier took 8,006,098 cycles. These are individual test-tile
measurements, not full-model performance estimates.


The descriptor is exactly `136 + 64 × layers` bytes: 200 bytes for one layer,
392 for the pinned four-layer model, and 2,376 for 35 layers.

On the one-entry guest fixture, model binding took 73,938 cycles when valid and
73,637 for a wrong expected ID. Authenticating and consuming a valid token took
118,175 cycles. Changed text, special flag, and EOS flag took 117,257, 117,188,
and 117,192 cycles respectively. These are fixture measurements, not a
representative model benchmark.

Decoder membership paths are additional storage. The current sidecar stores
one path per token: `32 × vocab_size × ceil(log2(vocab_size))` sibling bytes,
plus Raster encoding and index overhead. The pinned 512-entry model's encoded
witness payload is 181,380 bytes. For a 262,144-entry vocabulary, sibling hashes
alone would occupy 144 MiB. This has not been optimized into a shared proof tree.

## Remaining limitations

Sparse membership is implemented. Witnesses still store independent sibling paths
rather than a shared proof tree, and repeated Raster lookups can repeat verification.

The full E2B/35-layer performance benchmark is left to the user’s manual run, as
agreed on November 6. No actual fault proof for a forged clean model verdict has
been completed. The short-stage fraud-window
limitation remains. Authorized Iris program identities and authenticated predecessor
connections are prerequisites; arbitrary program replacement, same-model prompt
substitution, claimed-token binding, and broader range/order attacks are separate
from model identity.
