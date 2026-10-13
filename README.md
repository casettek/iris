# raster-chain-inference

A chain project that runs a real Gemma model — `tiny-gemma-dev` from
`raster-inference` — as a verifiable Raster chain: committed prompt in,
exactly N generated token IDs and decoded text out, one stage per inference
phase, expanded per layer and generated token.

```sh
cargo run --manifest-path model-import/Cargo.toml -- \
  --model ../raster-inference/assets/tiny-gemma-dev --prompt "hello raster"
cargo raster chain run --show-output && cargo raster chain audit --execution
```

`--show-output` is what prints the generated text; without it a chain run
reports only per-stage commitments. See [Generating N tokens](#generating-n-tokens).

```text
   tokenizer      embedding      ple layers    layer weights     head
    + pieces        table          │   │          │    │           │
        │             │            ▼   ▼          ▼    ▼           ▼
  ┌────────────┐ ┌──────────┐ ┌────────────┐ ┌───────────┐ ┌──────────────┐
  │  prompt_   │→│  input_  │→│  prefill_  │ │  prefill_ │→│   prefill_   │
  │  prepare   │ │ embedding│ │ prepare_aux│ │   range   │ │   finalize   │
  └────────────┘ └──────────┘ │  _l0  _l1  │ │ _l0 → _l1 │ └──────────────┘
       ids            │       └────────────┘ └───────────┘   PrefillLogits
                      └──────────ActivationSequence─┘
```

Every stage that passes activations on shares one type, `ActivationSequence
{ rows, errors, kv, start_position }` — defined field-for-field in each crate, because the chain
links stages by structural commitment. That is what lets `prefill_range`
instances chain into each other.

The root `Raster.toml` holds the `[chain]` table and no `[program]` table;
`prompt-prepare/` and `input-embedding/` are ordinary Raster programs (own
`Cargo.toml` + `Raster.lock`, no per-member `Raster.toml`). Stage 2's `prompt`
parameter is bound `from = "prompt_prepare"`, so the two `PromptTokenization`
definitions must stay field-for-field identical — the link is by structural
commitment, not type name.

## Stage 1 — `prompt-prepare`

Two passes, each one recur sequence:

```text
  pieces ──recur seq──▶ [ pair · scan merge table · emit · advance ] ──▶ MergedPieces
  merged ──recur seq──▶ [ query · scan vocabulary  · append       ] ──▶ PromptTokenization
```

With the checked-in fixture (`prompt "hello▁world"`):

| step | value |
| --- | --- |
| initial pieces | `h e l l o ▁ w o r l d </w>` |
| after merge pass | `hello`, `▁`, `world` |
| after vocab pass | `[10, 1, 11]` |

## Model decisions worth knowing

**The prompt is wrapped in Gemma's turn format.** An instruction-tuned model
was trained to answer inside a turn it was asked to open. A bare prompt is a
fragment to it, so the continuation it produces is a turn break rather than an
answer: `What is ZK?` committed as five raw tokens generated `\n\n` and then
`<turn|>`, which is the correct continuation of that input and not an answer to
it. `model-import` therefore renders one user message plus the generation
prompt the way the model's `chat_template.jinja` does —

```text
<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n
```

— and `split_prompt` emits each special token as a whole piece. It has to: a
special token is one vocabulary entry several characters long and no merge rule
mentions one, so the merge pass could never reassemble `<|turn>` from `<`, `|`,
`t`, … There is no Jinja engine here, so that one shape is written out directly
rather than rendered from the bundle's template. `--raw-prompt` commits the
prompt exactly as given; a tokenizer with no turn markers in its vocabulary
(`tiny-gemma-dev`) falls back to that automatically.

**The merge pass is a single left-to-right pass, not multi-round BPE.**
Classic BPE repeats "find the globally best merge, rewrite the whole piece
list" until no rule applies. That shape is not expressible in Raster: each
round's rewritten piece list would have to be carried into the next round, and
a recur carries only small `state` (re-committed every iteration) or an
append-only `output` draft — never a collection. Rather than emulate rounds
with a synthetic counter list (a fake recur), the stage applies the merge table
the way one pass can: a pending token is extended by the incoming piece
whenever `(pending, piece)` is a rule, lowest rank first. Merge rules are
therefore have to be prefix extensions — `h+e→he`, `he+l→hel` — for the pass to
apply them. `tiny-gemma-dev` has no merges at all, so the pass is a no-op there
and this only bites on a tokenizer whose merge table is rank-priority.

**The piece list must end with a terminator.** A left-to-right pass only knows
a token is finished when a piece fails to extend it, so the last token would
otherwise stay in the loop-carried cursor after the final iteration, with no
step left to append it. `model-import` terminates every prompt with the
end-of-word marker `</w>`, which appears in no merge rule: it flushes the last
real token and is itself dropped with the final cursor.

**Lookups use the two-collection pattern.** Both passes need a second
collection (the merge table, the vocabulary) per item. Each is passed as a
recur-*sequence* argument, so it travels as an `AuthRef` and materializes
nothing; inside, `call_recur!` walks it with `chunk = 4`, so one replay unit
sees one `Block` of rules/entries plus one small scalar item.

**Unresolved pieces become `<unk>` (id 0).** A recur sequence cannot propagate
a `Result`, so the vocabulary scan starts from `VocabMatch { found: false,
token_id: UNK_TOKEN_ID }` and a miss simply keeps that id.

**`PromptTokenization` carries only `token_ids`.** A draft's scalar fields are
set-once and the loop never knows it is on the last element, so a
`token_count` field could not be written by the pass that builds the list.

## Stage 2 — `input-embedding`

One pass: for each prompt token id, gather its row from the committed
embedding table.

```text
  token_ids ──recur seq──▶ [ query · scan embedding rows · append ] ──▶ ActivationSequence
  errors    ──recur─────▶ summarise ──▶ assert none
```

With the checked-in fixture (`hidden_size = 4`, one row per token id `0..12`,
row `id` = `[id*100, id*100+1, id*100+2, id*100+3]`):

| step | value |
| --- | --- |
| prompt token ids | `[10, 1, 11]` |
| gathered rows | `1000..1003`, `100..103`, `1100..1103` (hex-packed) |

**The gather is a scan, not an index lookup — and that is a model limit, not a
style choice.** `select!` accepts integer *literal* indexes only, so "row
number `token_id`" cannot be selected from a runtime value; there is no map
type either. The stage therefore walks the table and matches on
`row.token_id`. Every replay unit stays bounded (one `Block` of rows plus one
token id), but the pass costs `prompt_tokens × table_rows` iterations — fine
for this fixture, impossible at a real 262k-row vocabulary. Closing that gap
needs a protocol primitive: dynamic-index selection, where the index is bound
by an inclusion proof against the list root. Until it exists, the honest
alternatives are this scan or moving the gather outside the authorization
chain.

**Embedding rows are hex-packed into a single leaf.** A `List<i32>` field
would make a row a collection-bearing struct that can never cross a tile
boundary, and it would put every scalar in its own index node. `values_hex`
keeps a row `Materializable` and the index O(rows).

**`RowMatch` is the one non-scalar recur state in the repo.** A scan has
nowhere else to carry its answer while it keeps looking, so the matched row
rides in `state` and is re-committed each chunk iteration. The same
dynamic-index primitive would remove it.

**A missing row fails the program.** The recur sequence records the failure in
`errors`; `main` folds that list into a count plus the first message, and
`assert_all_tokens_embedded` turns a non-zero count into a terminal error, so a
partial gather is never published as an authorized output.

## Stage 3 — `prefill-prepare-aux`

Gemma's per-layer-embedding (PLE) prefill inputs. Per layer, per token:

```text
  embedded  = ple_embedding[token_id] * embedding_scale
  projected = rms_norm(activation × projection * projection_scalar, w, eps)
  row       = (embedded +sat projected) * input_scale
```

```text
  tokens ──recur seq──▶ [ task · scan PLE table · project · combine ] ──▶ PleLayerInputs
  errors ──recur─────▶ summarise ──▶ assert none
```

**The layer loop lives in the chain manifest, not in the program.** The native
routine nests layers × tokens; a recur cannot. Nesting one inside another does
not work either: the inner loop *finalizes* its draft when it ends, so the
outer loop has nothing to keep appending to, and `List<List<T>>` cannot be
drafted at all. So the program handles one layer, and the manifest instantiates
it per layer — `prefill_prepare_aux_l0`, `prefill_prepare_aux_l1`, same
`project`, different committed `layer` external. Stage names must be unique;
projects may repeat. This is the static expansion a chain generator would emit
per model, and the two instances share one program identity (`1d0acf104a66…`),
so adding layers costs manifest entries, not new programs.

**Numerics are Q16.16 integers.** Tiles must replay bit-identically in the
zkVM, so floats are out (SKILL.md §3) — including in the square root, which is
an integer Newton `isqrt`. The kernels in `src/input.rs` mirror the *shape* of
Gemma's `det_num` (`scale_act`, `add_sat`, `rms_norm_in_place`) without being a
port of its quantisation, which belongs with the weights it was tuned for.

**Failures travel as data, not panics.** A recur sequence has no fallible form,
and `.expect()` inside one would kill the program — which publishes nothing at
all. So `project_activation_row` returns its error in the row, `combine_ple_row`
appends it to `errors`, and `main` folds that list and fails the program with
the first message. A short layer input is never published.

**The per-token cost is honest but fixture-scale.** Each token re-materializes
`PleLayerParams` — including the whole `ple_width × hidden_size` projection —
into two tiles. At 4×4 that is nothing; at model width the matvec wants its own
recur over projection rows, which needs somewhere to accumulate a vector that
is not growing recur state. Same unresolved shape as the gather in stage 2.

## Stage 4 — `prefill-range`

One transformer layer over the prompt, following `det_layer_prefill`:

```text
  normed = rms_norm(x, w_input)
  q,k,v  = normed × W_q, W_k, W_v
  attn   = causal_softmax(q·kᵀ · scale) × v
  x      = rms_norm(attn × W_o, w_post_attn) + x
  ff     = (gelu(normed × W_gate) ⊙ (normed × W_up)) × W_down
  x      = rms_norm(ff, w_post_ffw) + x
```

```text
  rows   ──recur────▶ project Q/K/V                        ──▶ KvSequence
  kv     ──recur seq─▶ [ query · attend kv · finish · mlp ] ──▶ ActivationSequence
  errors ──recur────▶ summarise ──▶ assert none
```

**Attention forces two passes.** A token attends over *every* token's keys, so
the K/V sequence must be a finished, storage-backed list before any attention
step can read it: pass 1 drafts it, pass 2 takes it as a recur-sequence arg and
walks it. The same list is both what pass 2 iterates and what it attends over —
both uses are references, so nothing is materialized twice.

**Pass 1 is a recur *tile*, not a sequence, for one reason:** `input.index()`
is the token's position, and only a tile can read it. That position is what the
causal mask compares against in pass 2. A recur *sequence* body sees an opaque
handle, so a position would have to be threaded through state.

**The softmax is streaming.** Keeping a running max and sum (and rescaling the
accumulator when a larger score arrives) means one replay unit sees one chunk
of keys instead of the whole sequence. The cost is the repo's largest recur
state: the running weighted sum is `hidden_size` wide. It is fixed-width, not
growing — the alternative, collecting all scores and normalising afterwards,
would need a second list and a second pass per token.

**`exp` is a polynomial, `sqrt` is Newton's method.** Both are integer-only, so
the softmax and GELU replay bit-identically in the guest. `e^x` is evaluated as
`2^(x·log2 e)`: shift by the integer part, cubic on the fraction.

**Layers chain head to tail.** A layer's output type *is* its input type
(`ActivationSequence`), so `prefill_range_l1` takes `prefill_range_l0`'s
authorized output — visible in the audit as `activations ⇐ prefill_range_l0`.
Both instances share one program identity (`b0be08ebf80f…`).

Not modelled, each being another dimension to decompose: RoPE, multi-head
attention, sliding windows, KV sharing between layers, and the PLE gate block
(`prefill_prepare_aux`'s output is computed but not yet folded back in — that
needs its rows zipped against the activations).

## Stage 5 — `prefill-finalize`

The output head, following `det_hidden_to_logits`:

```text
  position = last row of the prompt's activations
  normed   = rms_norm(position, w_final)
  logit[t] = softcap(normed · projection[t])
```

```text
  rows       ──recur──▶ track final position ──▶ normalise
  projection ──recur──▶ score chunk                ──▶ PrefillLogits
  errors     ──recur──▶ summarise ──▶ assert none
```

With the checked-in fixture the head produces `decode_position = 3` and one
logit per vocabulary entry, highest for token 11.

**"The last row" is a fold, not an index.** `select!` takes literal indexes
only, so there is no way to select the final position directly. The fold
overwrites its state on every row and what survives the loop is the last one —
and the count it accumulates on the way *is* the decode position, which the
native routine gets from `prompt_token_ids.len()`.

**This stage needs no expansion and no second collection.** Both loops are
plain `call_recur!` over real data — the prompt rows, then the vocabulary — and
the only thing riding along is the normalised position, one row wide. It is the
simplest stage in the repo, and the shape every stage would have if lookups
were addressable.

**The projection is tied to the embedding table.** Gemma's `TiedEmbedding`
logits projection *is* the input embedding. `tiny-gemma-dev` sets
`tie_word_embeddings: false`, so `model-import` reads `lm_head.weight`; with a
tied model it would read the embedding table instead.

**`prefill_range_finalize` is deliberately absent.** In the source routine it
is a checkpoint — `next_layer_idx`, the activation hash, the KV-cache hash —
recorded so a host can resume or verify mid-loop. A chain already does that:
every `prefill_range_lN` stage emits a committed `output.bin` and a checkpoint
that `chain audit` verifies by name, link, and identity. Porting it would add a
stage that computes nothing.

## The model

`model-import/` is a host-side tool (not a Raster program) that turns a Gemma
bundle into this chain's committed externals and prints the manifest:

| bundle file | becomes |
| --- | --- |
| `tokenizer.json` | `prompt-prepare`'s vocabulary + merges, and the prompt's initial pieces |
| `model.detwgt` | the embedding table, per-layer PLE tables, per-layer transformer weights, the output head |
| `config.json` | shapes, `layer_types`, sliding window, softcap, rms-norm epsilon |

**Nothing requantises on the way in.** detwgt stores each weight as its
canonical **Q16.16 bit pattern in an i32** — the same representation the
stages compute in — so the importer only reshapes and packs. An i16-stored
tensor is sign-extended on read, which the format guarantees changes no value.

`tiny-gemma-dev` is 4 layers, hidden 4, 2 query heads over 1 KV head,
head_dim 2, vocab 280 (24 base tokens + 256 byte-fallback), alternating
sliding/full attention, PLE width 2, softcap 7.5. Its tokenizer has **zero
merges**, so the merge pass is a legitimate no-op — merge-less BPE emits every
piece as it arrives, which is exactly what the pass does with an empty table.

With `--prompt "hello raster"` the chain tokenizes to
`h e l l o ▁ r a s t e r` → `[9, 10, 11, 11, 6, 3, 5, 14, 15, 8, 10, 5]` and
finishes with `decode_position = 12` and 280 logits.

### What is faithful, and what is not

Modelled: grouped-query multi-head attention, per-head Q/K norms, alternating
sliding windows, `attention_k_eq_v` (layers without a stored `v_proj` use
`k_proj`), the layer scalar, softcap, untied `lm_head`.

Not modelled — each changes values, not structure, and each is a further
dimension to decompose:

- **RoPE.** Positions drive the causal and sliding masks but do not rotate Q/K.
- **KV sharing between layers.** `num_kv_shared_layers` is ignored; every layer
  computes its own K/V. Wiring it needs a stage to consume an earlier stage's
  K/V, which means the layer output type has to carry it.
- **The PLE gate block.** `prefill_prepare_aux` computes each layer's PLE
  inputs, but `prefill_range` does not fold them back in: its rows would have
  to be zipped against the activations, and two parallel lists cannot be walked
  together.
- **`det_num` quantisation.** The kernels here are Q16.16 with an integer
  `exp`/`isqrt`; the reference uses `det_num`'s Act/Wgt/Acc rounding. Values
  are in the right range but are not bit-parity with a reference decode.
- **PLE scales.** `embedding_scale` / `projection_scalar` / `input_scale` are
  derived by the model loader rather than stored in the artifact, so the
  importer writes 1.0 for each.

## Fixture convention

**Only the first stage has inputs of its own; later stages reuse upstream
outputs.** `model-import` writes each stage's externals — the tokenizer and
prompt pieces for stage 1, the embedding table for stage 2, per-layer weights
for the rest. Anything a stage receives from upstream is bound `from = "<stage>"`,
and the chain synthesizes that binding at run time: it points the parameter at
the producer's `output.bin` / `output.rindex` and carries the producer's output
commitment into the stage's manifest.

So a downstream stage carries no `input.json` and no local copy of its
upstream input. A hand-maintained mirror would let a stage be run standalone,
but it drifts silently the moment the upstream fixture or logic changes, and a
green standalone run would then prove nothing about the chain. Downstream
stages are verified through `chain run` + `chain audit --execution`, which
re-runs each stage against its committed trace.

Practical consequence for the ladder: rungs 3–4 (`cargo raster run`, the
commit/audit round trip) apply only to stage 1. Rungs 1, 2 and 5 (`cargo
check`, `cargo raster cfs`, `cargo raster program --verify`) work for every
stage, since none of them needs input values.

## Layout

```text
Raster.toml                 # [chain] table — generated by model-import
model-import/               # host-side bundle -> committed externals
prompt-prepare/
  src/input.rs              # Rastered data types (List<T> fields, Selectable)
  src/lib.rs                # no_std tile library — all computation
  src/main.rs               # sequences + #[sequence] fn main
  input.json                # private: entry arg -> file paths
  input_manifest.json       # public: entry arg -> commitment
  *.rastered / *.rindex     # committed input values
  Raster.lock               # program identity claim — commit it
input-embedding/            # same layout, minus input.json / input_manifest.json:
                            # `embedding` is this stage's own external, and
                            # `prompt` is bound from stage 1's output
prefill-prepare-aux/        # one program, instantiated once per PLE layer;
                            # layer<N>.rastered is that instance's external
prefill-range/              # one program, instantiated once per transformer
                            # layer; instances chain into each other
prefill-finalize/           # the output head — one instance
```

## Commands

```bash
# regenerate every committed external + the root manifest from a model bundle
cargo run --manifest-path model-import/Cargo.toml -- \
  --model ../raster-inference/assets/tiny-gemma-dev --prompt "hello raster"

# any stage (from prompt-prepare/ or input-embedding/) — no inputs needed
cargo check && cargo check --lib --no-default-features
cargo raster cfs
cargo raster build --backend risc0   # rebuild guests, re-lock Raster.lock
cargo raster program --verify

# stage 1 only — the stage that owns its inputs (from prompt-prepare/)
cargo raster run --input input.json --input-manifest input_manifest.json
cargo raster run --input input.json --input-manifest input_manifest.json \
  --commit commit.bin --fraud-proof-window-size 32
cargo raster run --input input.json --input-manifest input_manifest.json \
  --audit commit.bin

# chain (from the repo root)
cargo raster chain run --no-auth
```

Any change to a tile body, a sequence, or `main`'s signature changes the
program identity: rebuild with the risc0 backend and commit the new
`Raster.lock` together with the source change.

## Generating N tokens

`model-import` writes a complete manifest whose decode section is one
`[[chain.repeat]]`. Selection is the first operation in each iteration, so `--tokens N`
means exactly N selected tokens and N transformer transitions:

```sh
cargo run --manifest-path model-import/Cargo.toml -- \
  --model ../casettek/raster-inference/assets/tiny-gemma-dev \
  --prompt "hello raster" \
  --tokens 3 \
  --manifest Raster.toml

cargo raster chain run --no-auth --show-output    # fast functional check
```

The final `output_finalize` stage publishes `generated_token_count`,
`generated_token_ids`, their reference-compatible SHA-256, decoded text, and
`stop_reason` — but a chain run prints per-stage digests, not values, so
without `--show-output` the generated text never reaches the terminal.
`--show-output` renders the last stage's `output.bin` when the run finishes:

```text
Program output value (output_finalize):
  {
    generated_token_count: 2u32
    generated_token_ids: [2] [ 236775u32 229361u32 ]
    generated_token_ids_sha256: "fd22a039…"
    generated_text: "\"ZK"
    stop_reason: "max_new_tokens"
  }
  commitment 2b5bb726c88cf7fd…  ✓ matches .rindex
```

Every stage's artifact stays on disk, so the same reader works after the fact
on any of them — which is how you read a middle stage without re-running:

```sh
cargo raster show target/raster/chains-no-auth/latest/output_finalize/output.bin
cargo raster show target/raster/chains-no-auth/latest/prompt_prepare/output.bin
```

`stop_reason` is `eos` when a generated token is in the bundle's
`eos_token_id` set (`model-import` reads it from `config.json` and marks those
ids `terminal` in the decoder table), and `max_new_tokens` otherwise. The
repeat count is a static unroll, so decoding cannot actually stop early:
everything generated after the terminal token stays in `generated_token_ids`
and its SHA-256 — it was generated — but it is a new turn, not the answer, so
`generated_text` ends where the model ended. `Raster.toml.prefill-only` is the generated `--tokens 0` boundary
fixture; it still runs `decode_init` and `output_finalize` so empty generation is
tested end to end.

For the 35-layer model, prefill/init/output cost 75 stages once and every generated
token expands to 73 more stages (`select + embed + 35 aux + 35 range + finalize`).
The repeat keeps the manifest compact; expansion deliberately keeps every execution
and audit stage.

KV-sharing layers receive both committed donor candidates. The layer's committed
`kv_donor_layer` selects the exact candidate inside the tile, which removes the old
non-templatable donor map and makes a wrong manifest edge fail closed.

**Before you trust the output**, build the guests with the non-default backend —
`cargo raster build` defaults to `--backend native`, which discovers the tiles and
prints "Build complete!" while writing no image ids, after which the chain refuses to
run with advice that does not fix it:

```sh
for d in prompt-prepare input-embedding prefill-prepare-aux prefill-range \
         prefill-finalize decode-init decode-select-token decode-embed \
         output-finalize; do
  (cd $d && cargo raster build --backend risc0)
done

# Functional execution:
cargo raster chain run --no-auth --show-output
```

Authenticated chain execution is currently blocked by Raster's open
`authenticated-chain-draft-output` issue (re-verified 2026-09-11 against a CLI
built from the `raster` checkout). A value returned from the free `finalize()`
lives at the synthetic coordinate `[u32::MAX, n]`, and the recorder's storage
replica has no object there, so any replay that resolves that coordinate panics.
Two call sites reach it:

| site | raised by |
| --- | --- |
| `recorder.rs:619` — program output selection | returning the finalized draft |
| `recorder.rs:1289` — recur source metadata at site open | feeding a list *selected out of* a finalized draft into a `call_recur!` |

Reproduces on Raster's own `chain-example` (stage 3, `phase3-report`). Here it
blocks three programs, not four:

| program | authenticated | why |
| --- | --- | --- |
| `decode-init` | ✗ `recorder.rs:619` | returns `finalize(draft)` |
| `decode-select-token` | ✗ `recorder.rs:619` | returns `finalize(draft)` |
| `decode-embed` | ✗ `recorder.rs:1289` | `finalize(embedded)`, then `select!`s `errors` off it into a recur |
| `output-finalize` | ✓ rc=0 | never calls `finalize()` — its value is what `call_recur!`'s implicit finalize returns, which lands at an ordinary coordinate (`"5"`) |

`output-finalize` is the useful data point: across these three programs the
failure tracks the *free* `finalize()`, not "the output is a draft". Whether
routing the other three through the macro's finalize is a legitimate workaround
or only moves the coordinate is untested.

It fails closed — the run aborts rather than record an output commitment the
replica cannot substantiate. `--no-auth` output and no-auth `chain audit
--execution` have been validated, but they are not a substitute for the missing
authenticated gate.

```sh
# the canonical repro, from any scratch directory
cargo-raster raster chain run path/to/raster/examples/chain-example/Raster.toml

# one stage here, non-destructively, against a finished no-auth run
cd decode-init && cargo raster run \
  --input  ../target/raster/chains-no-auth/latest/decode_init/input.json \
  --input-manifest ../target/raster/chains-no-auth/latest/decode_init/input_manifest.json \
  --commit /tmp/decode_init.bin --fraud-proof-window-size 16
```
