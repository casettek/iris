# Proposal: multi-token decode — the generation loop as chain stages

Status: **implemented** 2026-08-28.

Ports the decode loop of the reference implementation
(`casettek/raster-inference`, `src/runtime/executors/native.rs:265-450`) onto this chain, using the
same decomposition prefill already uses: one chain stage per (step × layer), the layer loop
unrolled into the manifest, the weight sweep a `call_recur!` over pages.

## Decision (2026-08-28)

**Option A — a finalize-deferring recur output, upstream in `raster` — plus dense positional
scoring here.** Recorded because the reasoning is easy to lose and expensive to re-derive:

> A is less code here, fewer stages, and leaves nothing to delete later. B's dense-scoring half is
> worth doing regardless, so if you pick A, do that piece anyway.

The three parts of that, spelled out:

- **Less code here.** B adds a whole program (`decode-kv-merge`) whose only job is to work around a
  missing primitive — new crate, new `Raster.lock`, new program commitment, guest builds, and a
  module whose central invariant (`fresh.kv[0]` is the *only* key) it cannot itself check. A adds
  nothing to this repo.
- **Fewer stages.** B costs +35 chain stages per generated token — 73 → 108, or 6,912 instead of
  4,672 for a 64-token response — to move data A moves for free. The *compute* delta is only ~0.9%
  (~350 replay units per token); the cost is structural, and it is paid on every stage's process
  launch, input-manifest resolution, `output.bin` write and commitment.
- **Nothing to delete later.** B is a workaround. When A eventually lands — and it is the natural
  fix, not a speculative one — the merge stages, their program, and their manifest wiring all have
  to be unwound, and every chain commitment that recorded them is a different shape.

**Dense positional scoring (§4.2b) is done either way.** It is mandatory under B and merely correct
under A: it bounds `score_key`'s accumulator instead of letting it grow with the key count (closing
`docs/issues/append-shaped-accumulators.md` §4 from the other direction), makes every sweep
order-independent, and turns a cache with a hole into a committed error instead of silently wrong
attention.

### Correction to A's shape

A cannot key off syntax. The first sketch had `call_recur!` return `Draft<S>` "when its `output` is
an already-open draft rather than `new!(T)`" — but `raster-stages/prefill-finalize/src/main.rs:34-42` is exactly
that shape today and *relies* on finalization, selecting `.errors` off the result. Inferring intent
from the output expression would silently break it.

So the deferral is an **explicit opt-in**, placed before the mandatory-last `args`:

```rust
let draft = call_recur!(
    tile = carry_cached_key, input = prior_keys, chunk = 64,
    output = draft, finalize = false, args = (start_position, sliding_window)
);
let out = call_recur_seq!(sequence = attend_token, input = queries, output = draft, args = (...));
```

Explicit is also the right posture for a verifier: whether a recur closed its draft is a fact about
the program's shape, so it belongs in the CFS rather than being re-derived from how the call site
was spelled.

Rests on `raster` `docs/proposals/chain-repeat.md` (**implemented** 2026-08-27) — which was written
against this exact problem and whose §2 sketch is close to §5 below. §6 names the one thing that
sketch cannot express.

Related:
- `docs/issues/append-shaped-accumulators.md` — the per-iteration cost this design multiplies by
  `35 × max_new_tokens`. Decode is where §4 of that issue stops being a tuning matter.
- `raster` `docs/proposals/recur-sequence-break.md` — EOS. Out of scope here; see §8.

## 1. The loop being ported

The reference generates greedily, with exactly one stop condition
(`decode_select_token/native/tiles.rs:46` — `generated_token_count >= max_new_tokens`; there is no
EOS check anywhere in the repo). Per generated token:

```text
decode.select_token       argmax over 262,144 logits          → next_token
decode.layer_range        embed(next_token), then 35 layers,
                          each appending one K/V row to its cache
decode.transition_finalize  final norm + logits projection    → 262,144 logits
```

carried between steps as `TransformerDecodeState { layer_caches, position, token_count }`
(`shared/model/transformer.rs`), with `position` advanced by one per step
(`decode_transition_finalize/native/tiles.rs:58`).

**The trip count is a run parameter with no data dependence.** That is the fact this whole design
rests on: `max_new_tokens` is known before the first stage runs, so the loop can be unrolled into
the manifest exactly the way the 35 layers already are. Nothing here needs a `while`.

## 2. What carries over unchanged

More than expected. A decode step is a prefill pass over a one-token prompt, so three of the five
programs are already correct for it:

| program | decode use | change |
| --- | --- | --- |
| `prefill-prepare-aux` | one PLE row for the new token | **none** |
| `prefill-finalize` | `track_final_position` over a 1-row sequence | **none** |
| `decode-select-token` | argmax over the step's logits | **none** |
| `input-embedding` | embeds `List<u32>` from `prompt_prepare` | needs a `SelectedToken` entry — §4.4 |
| `prefill-range` | one token, but over a **prior** K/V cache | §4.1–4.3 |

The per-layer externals are reused verbatim. `raster-stages/prefill-range/layer{l}.rastered` and
`raster-stages/prefill-prepare-aux/layer{l}.rastered` are properties of the model, not of the step, so every
decode step at layer `l` binds the same file with the same commitment. **No new artifacts are
imported and `model-import` emits no new weights.**

`prefill-range` also already publishes a full K/V cache from every layer: `finish_layer`
(`raster-stages/prefill-range/src/lib.rs:1008`) pushes `own_key` into `output.kv()` unconditionally. The doc
comment on `ActivationSequence.kv` claiming the field is "empty on stages that donate to nobody" is
stale — the cache decode needs is already on the wire.

## 3. The three things that break

### 3.1 There is no channel for the layer's own prior cache

`prefill-range::main` takes `donor_kv` — a second cache for the 20 KV-sharing layers — but nothing
for "the cache this layer built on the previous step". Pass 1 (`project_token`) builds `keys` from
`activations.rows` alone, so at decode step `t` the layer would attend over one key: its own.

### 3.2 Positions restart at zero

`main` seeds the position counter with `call!(zero_u32)` (`raster-stages/prefill-range/src/main.rs:230`). At decode
step `t` the token's true position is `prompt_len + t`, and position feeds RoPE
(`apply_rope`) and sliding-window visibility (`visible`, `:629`). Starting at zero silently produces
the wrong rotation and makes every cached key invisible.

### 3.3 `ple_rows[position]` goes out of range — and that is fatal, not an error

```rust
let position = select!(u32, query.clone().position);
let ple_row  = select!(PleRow, ple_rows[position]);      // raster-stages/prefill-range/src/main.rs:163-164
```

`ple` for a decode step holds exactly **one** row, while `position` is `prompt_len + t`. Per the
authoring rules, "an out-of-range index aborts the run with no output — it cannot be handled,
because a list has no non-membership proof." So this is not a committed error a verifier sees; it is
a run that produces nothing. `own_key = select!(KeyRow, keys[position])` (`:180`) has the same
shape and the same fix.

The rule this violates: **an absolute position is not an index into a per-stage list.** Every
dynamic index in this program must be an index into the list the *stage itself* built.

## 4. Design — one program, both phases

Keep `prefill-range` as the single transformer program. A decode stage is the same program with a
non-empty prior cache and a one-row activation sequence; prefill is the degenerate case where the
prior cache is empty. This is the property the repo already relies on for `donor_kv` — *"Every stage
of this one program shares `main`'s signature, so every one needs a `donor_kv` binding"*
(`crates/model/import/src/main.rs:833`) — extended one step.

### 4.1 `main` gains `prior_kv`

```rust
fn main(
    activations: ActivationSequence,   // this stage's input tokens (prefill: prompt; decode: 1)
    prior_kv:    ActivationSequence,   // this layer's cache before this stage
    donor_kv:    ActivationSequence,   // unchanged: the sharing layers' donor
    layer:       TransformerLayer,
    ple:         PleLayerInputs,
) -> Result<ActivationSequence>
```

Prefill binds `prior_kv = { from = "input_embedding" }` — the same empty-`kv` stage `donor_kv`
already uses for non-sharing layers, so **prefill behaviour and prefill wiring are unchanged**.

### 4.2 The cache is read from `prior_kv` — republishing it is blocked

**Reading** the inherited cache needs nothing new. `attend_token` gains a third key list and sweeps
it before its own, and `score_key`/`accumulate_context` already carry the "exactly one side
contributes" discriminator:

```rust
let prior_keys = select!(List<KeyRow>, prior_kv.kv);
let scores = call_recur!(tile = score_key, input = prior_keys, chunk = 64, state = scores,
                         args = (query.clone(), params.clone(), false));
let scores = call_recur!(tile = score_key, input = keys,       chunk = 64, state = scores,
                         args = (query.clone(), params.clone(), false));
```

`prior_keys` comes first because both `softmax_scores` and `accumulate_context` read the score list
as a dense window indexed from `window_start`, so the sweep must visit positions in ascending
order. **This half is implemented and compiles.**

**Republishing** the grown cache was a hard blocker until 2026-08-28, when
`raster`'s `docs/proposals/recur-deferred-finalize.md` landed an opt-in `finalize = false` on
`call_recur!` (branch `feature/recur-deferred-finalize`). See
`docs/issues/two-recurs-one-draft.md` for what was wrong and why the fix turned out to be small.

The stage now builds its output with two writers sharing one draft:

```rust
let draft = call!(begin_layer_output, new!(ActivationSequence), start_position);
let draft = call_recur!(
    tile = carry_cached_key, input = prior_keys.clone(), chunk = 64,
    output = draft, finalize = false, args = (start_position_arg, sliding_window)
);
let prepared = call_recur_seq!(
    sequence = attend_token, input = queries, output = draft, args = (prior_keys, keys, ..)
);
```

`carry_cached_key` carries the inherited cache forward, pruned to the window; `attend_token` then
appends one row and one key per token. So `rows: n` and `kv: prior + n` coexist — the shape that
was inexpressible. Prefill is the degenerate case: an empty `prior_kv` makes the first sweep a
no-op and the output identical to before, which the gate confirms.

**Implemented, and prefill-neutral by measurement.**

### 4.2b Order-independent scoring

`score_key` currently *appends*, so the score list is positional only because the sweep is
ascending. Making `zero_scores` allocate a dense `num_heads × window_len` array — `window_len` is
computable from `query.position` and `sliding_window` — and having `score_key` write at
`slot = key.position - window_start` removes the ordering requirement entirely.

It also **bounds the state**: `score_key`'s accumulator stops growing with the key count, which is
the `docs/issues/append-shaped-accumulators.md` §4 fix arrived at from the other direction, and it
turns a cache with a hole into a committed error (`softmax_scores` asserts `filled == window_len`)
instead of a plausible-looking distribution over zeros. **Implemented and gated.**

It is also what makes §4.2c's pruning safe: once a key's slot comes from its position, dropping
retired rows cannot shift any other key's score.

### 4.2c Prune sliding layers to the window

Visibility already discards keys older than `sliding_window`, so a sliding layer's cache never
needs more than 512 rows. Dropping the rest at the stage boundary caps 28 of the 35 layers at 1 MiB
regardless of how long the generation runs; only the three own-cache full-attention layers (4, 9,
14) grow. It belongs in whichever tile ends up carrying the cache forward, pruned against the
stage's `start_position` — exact for decode, a harmless superset for prefill.

### 4.3 Indexes become stage-local

Replace the recur-sequence state `u32` with a scalar-small pair, both set by pass 1:

```rust
pub struct TokenCursor {
    pub position: u32,   // absolute: RoPE + window visibility
    pub local:    u32,   // index within this stage: position - start_position
}
```

`QueryRow` carries `local` alongside `position`; §3.3's two selects become `ple_rows[local]` and
`keys[local]`, both indexes into lists this stage built. `ActivationSequence` gains
`start_position: u32` — set by `input-embedding` to 0 and by `decode-embed` to
`selected.decode_position + 1` — so a stage knows where it sits without counting anything.

A counter cannot be recovered from a sweep, which is why the position is carried explicitly rather
than derived: `run_recur_list_with_state` and its three siblings return `AuthRef<S>` — the output
draft only — and **discard the final state** (`raster/src/input.rs:2732`, `:2877`). A recur's `state`
is loop-carried, not readable at the call site.

Prefill is unaffected: with `start_position = 0`, `position == local` throughout.
**Implemented and compiles.**

### 4.4 One new program: `decode-embed`

`input-embedding` reads `PromptTokenization.token_ids`; `decode-select-token` emits
`SelectedToken { decode_position, token_id, value }`. The link is by structural commitment, so the
types must match field-for-field and they do not. `decode-embed` is `input-embedding`'s
`embed_prompt_token` over a one-element input — the same embedding external, ~90 lines, no new
kernels.

The alternative — widening `SelectedToken` into a `PromptTokenization`-shaped output — makes
`decode-select-token` emit a token list it did not build, and would change the prefill chain's last
stage. Not worth it for one stage.

## 5. The manifest

`chain-repeat` landed today with `start`, `first`, `{ident-1}`, nested static inner indexes, and
named exports, so the decode loop is expressible as one block:

```toml
[[chain.repeat]]
name  = "decode"
index = "t"
count = 64                          # or { from = "plan_stage", max = 256 }

  [[chain.repeat.stage]]
  name    = "decode_embed_t{t}"
  project = "raster-stages/decode-embed"
  inputs.selected  = { from = "decode_select_t{t-1}", first = "decode_select_token" }
  inputs.embedding = { input = "embedding" }

  [[chain.repeat.stage]]
  name    = "decode_aux_t{t}_l{l}"
  index   = "l"
  count   = 35
  project = "raster-stages/prefill-prepare-aux"
  inputs.embedded = { from = "decode_embed_t{t}" }
  inputs.layer    = { input = "aux_layer_{l}" }

  [[chain.repeat.stage]]
  name    = "decode_range_t{t}_l{l}"
  index   = "l"
  count   = 35
  project = "raster-stages/prefill-range"
  inputs.activations = { from = "decode_range_t{t}_l{l-1}", first = "decode_embed_t{t}" }
  inputs.prior_kv    = { from = "decode_range_t{t-1}_l{l}", first = "prefill_range_l{l}" }
  inputs.donor_kv    = { ... }                            # §6 — not expressible
  inputs.ple         = { from = "decode_aux_t{t}_l{l}" }
  inputs.layer       = { input = "layer_{l}" }

  [[chain.repeat.stage]]
  name    = "decode_finalize_t{t}"
  project = "raster-stages/prefill-finalize"
  inputs.activations = { from = "decode_range_t{t}_l34" }
  inputs.head        = { input = "head" }

  [[chain.repeat.stage]]
  name    = "decode_select_t{t}"
  project = "raster-stages/decode-select-token"
  inputs.logits = { from = "decode_finalize_t{t}" }

  [chain.repeat.exports.transcript]
  stage = "decode_select_t{t}"
  entry = "decode_select_token"
```

Two `first =` clauses carry the whole prefill→decode seam, and they are the two irregularities the
reference implementation spends its `init_state` / `resolve_decode_donor_cache` code on:
`activations` restarts from the embed stage at every layer 0, and `prior_kv` comes from prefill only
at `t = 0`.

`prefill_range_l*` and `prefill_prepare_aux_l*` collapse into the same shape with `count = 1`, which
is what makes prefill and decode one mechanism rather than two.

## 6. The gap: `donor_kv` is not expressible

A sharing layer's donor is *the last layer before 15 with the same attention type*: layer 13 for the
28 sliding layers, layer 14 for the 7 full-attention ones (4, 9, 14, 19, 24, 29, 34 — `l ≡ 4 mod 5`).
As a binding that is:

```
l in  0..14   →  input_embedding        (computes its own; the tile ignores the value)
l in 15..34   →  decode_range_t{t}_l13  or  decode_range_t{t}_l14, by attention type
```

Templates substitute `{l}` and subtract one. There is no conditional, no modulus, and `start`/`count`
cannot carve out a stride-5 subset — so this binding cannot be written. Three ways out, in
preference order:

1. **Extend `chain.repeat` with a positional binding table**, in the exact idiom the indexed
   `[chain.input]` already uses for its 35 `commitments`:

   ```toml
   inputs.donor_kv = { from_index = [
     "input_embedding", "input_embedding", …,       # l0 … l14
     "decode_range_t{t}_l13", …, "decode_range_t{t}_l14", …
   ] }
   ```

   One entry per index, positional, verified by length against `count`. Same argument
   `chain-repeat` §1 makes for keeping the commitments written out: the irregularity is the one
   thing a verifier must read, and compressing it is the only compression that costs something.
   **Recommended**; small, and it retires the twenty hand-written donor bindings the current
   `Raster.toml` carries *and* the ones this proposal would add. Filed as
   `docs/issues/donor-binding-not-templatable.md`.

2. **Carry both donor caches down the activation stream.** Split `ActivationSequence.kv` into
   `kv_sliding` / `kv_global`; layers 13 and 14 set theirs, every other layer passes both through
   unchanged, and `attend_token` sweeps three lists with the existing "exactly one contributes"
   discriminator (`score_key`'s `donor_pass`, `raster-stages/prefill-range/src/lib.rs:411`). Bindings become
   uniform and option 1 is unnecessary. Costs ~3 MiB of pure transport per layer per step —
   105 MiB/step against §7's 5.26 GB, so ~2%, but it is 35 copies of data that 34 layers do not read.

3. **Keep generating the manifest.** `model-import` already writes the 20 donor bindings
   (`crates/model/import/src/main.rs:836`), so this works today with no upstream change — at the cost of a
   `73 × max_new_tokens`-stage `Raster.toml` (4,672 stages at 64 tokens) and the generator staying a
   second source of truth for the pipeline's shape, which is what `chain-repeat` set out to remove.

**Option 1 blocks nothing.** Options 2 and 3 both ship without it, so §9 does not wait on upstream.

## 7. Cost

Per generated token, at `hidden=1536`, `ffn=6144`, 28 sliding + 7 full layers, 192 KiB pages:

| | bytes streamed | page replays |
| --- | --- | --- |
| 35 `decode_range` (28 × 144.7 MB + 7 × 173.0 MB) | 5.26 GB | 26,768 |
| `decode_finalize` (the 262,144-row head) | 1.61 GB | 8,192 |
| 35 `decode_aux` (projection only; the 268 MB PLE table is **indexed**, one page) | 55 MB | 280 |
| **total** | **6.9 GB** | **35,240** |

73 stages per token; 4,672 stages for a 64-token response, on top of prefill's 74.

**Decode is weight-bandwidth bound and prefill is not.** A prefill layer streams its 144.7 MB once
and amortizes it over every prompt token; a decode layer streams the same 144.7 MB for one token.
That is the whole cost story, it is a property of the model rather than of this design, and no
stage-splitting changes it.

**What this design controls is the other half — replay units that are not weight pages.** Three
recurs sweep the K/V cache once each per layer per step, and all three are single-`KeyRow` loops
over lists that are already contiguous:

| source | units/step, unchunked | with `chunk = 64` |
| --- | --- | --- |
| `score_key`, 35 layers x <=512 keys | 17,920 | 280 |
| `accumulate_context`, same sweep again | 17,920 | 280 |
| cache copy-in + copy-out, 15 own-cache layers | 15,360 | 240 |
| **total** | **51,200** | **800** |

Against a step's full budget:

| | unchunked | `chunk = 64` |
| --- | --- | --- |
| 35 x `decode_range` weight pages | 26,768 | 26,768 |
| `decode_finalize` head | 8,192 | 8,192 |
| `decode_aux`, `decode_select`, scalar tiles | 3,104 | 3,104 |
| cache + key sweeps | 51,200 | 800 |
| **replay units per token** | **89,264** | **38,864** |

Unchunked, the sweeps are **57% of every replay unit in a step** while moving 0.3% of the bytes.
`chunk = N` is the sanctioned fix and costs nothing a verifier can see, since the bound is a literal
pinned in the CFS — the same argument `decode-select-token` already makes for `chunk = 256` over the
vocabulary (`raster-stages/decode-select-token/src/main.rs:8-12`). It cuts the whole step by 2.3x from one keyword
in three places. **Chunk all three before measuring anything else.**

Cache transport itself stays small: with §4.2's window prune, 12 sliding + 3 full own-cache layers
carry **18.9 MB per step**, flat in `t` for 28 of 35 layers.

## 8. Not in scope

- **EOS.** The reference has no EOS check, so a fixed trip count is faithful to it. Stopping early
  needs `chain-repeat` §8's "iterate until a stage says stop", which that proposal separates
  deliberately. A chain that runs the full `max_new_tokens` and truncates at detokenize is correct,
  just wasteful.
- **Sampling.** `validate_sampling_config` (`runtime/pipeline.rs:22`) rejects everything but greedy;
  `decode-select-token` matches.

## 9. Implementation order

Revised 2026-08-28. Steps 1-10 are done; the decode chain generates an exact fixed token count.

1. ~~**`chunk = 64` on `score_key` and `accumulate_context`.**~~ **Done.**
2. ~~**`TokenCursor` + stage-local indexes** (§4.3).~~ **Done.**
3. ~~**`prior_kv` + the third key sweep** (§4.2).~~ **Done.**
4. ~~**Dense positional scoring** (§4.2b).~~ **Done.**
5. ~~**Deferred finalize + `carry_cached_key` + window prune.**~~ **Done**, on `raster` branch
   `feature/recur-deferred-finalize`.
6. ~~**`decode-embed`** (§4.4).~~ **Done.** Straight-line, not a recur — a decode step embeds one
   token, so there is no collection to sweep. `begin_logits` also had to change: it set
   `decode_position` from `token_count` alone, which is the next token's position only when the
   stage starts at zero. A decode stage sees one row and would have reported position 1 forever.
7. ~~**One decode step against the reference oracle.**~~ **Done** — see below.
8. ~~**Multi-step generation.**~~ **Done.** Each repeat iteration starts with selection and ends
   with the corresponding transition, so `count = N` means exactly N generated tokens.
9. ~~**Uniform donor wiring and `[[chain.repeat]]`.**~~ **Done** without positional manifest
   bindings: both committed donor candidates are bound uniformly and the tile accepts only the
   exact `kv_donor_layer`.
10. ~~**Transcript and output finalization.**~~ **Done.** `decode-init` supplies the zero-token
    fallback, selections append IDs to the carried edge, and `output-finalize` publishes count,
    IDs, the reference-compatible token-ID SHA-256, decoded text and stop reason.

The `N = 0, 1, 2, 3` unauthenticated gates pass. On the tiny fixture they publish `[]`, `[1]`,
`[1, 1]` and `[1, 1, 1]`, with positions advancing from 12.

Authenticated chain execution remains blocked upstream by Raster's open
`docs/issues/authenticated-chain-draft-output.md`: the recorder cannot replay a `ProgramEnd`
selection stored at a finalized draft coordinate. During this work a separate nested-recur CFS bug
was fixed—the recorder now resolves `chunk = 64` for a recur tile inside a recur-sequence iteration
instead of treating the outer iteration as the item—but the run then reaches the known draft-output
failure. No inference-side sentinel or synthetic sweep should paper over that verifier gap.

### Parity against the reference (2026-08-28)

`tiny-gemma-dev`, prompt `hello raster`, reference run with `--deterministic` (the same Q16.16
contract this chain computes in):

| | token 0 | token 1 | token 2 |
| --- | --- | --- | --- |
| reference `generated_token_ids` | 1 | 1 | 1 |
| chain `decode_select_*` | 1 | 1 | 1 |
| chain `decode_position` | 12 | 13 | 14 |

Positions advance correctly off a 12-token prompt, and the per-step logits differ
(116880 / 116775 / 116835) — which is the evidence that the carried K/V cache is actually
reaching the computation. If the cache were not threaded, the three steps would see identical
context and produce identical logits.

**This is not yet strong parity.** `tiny-gemma-dev` is degenerate — it selects token 1 whatever it
is shown — so matching argmax three times is weaker than it looks. The sharp check is the
reference's `--commit-checkpoints`, which emits `det_current_logits_sha256` per decode step;
comparing those against the chain's logits requires hashing the raster payload the same way
`build_det_vector_commitment` does, and has **not** been done. Do it before trusting real output.

### The gate — passed for steps 1-3, 2026-08-28

Baseline (HEAD) and modified trees both run the tiny chain to completion, and `prefill_finalize`
emits **identical commitments**:

```
baseline: payload=c9124ff6db19...  structural=b806106fdc3b...
modified: payload=c9124ff6db19...  structural=b806106fdc3b...
```

`PrefillLogits`'s schema is untouched by any of these changes, so an identical payload commitment
means all 280 logits are bit-identical — a strictly stronger statement than "the same token was
selected". Steps 1-3 are prefill-neutral, proven rather than argued.

Two mechanical notes for whoever runs this next:

- **`cargo raster build` defaults to `--backend native`**, which discovers the tiles and prints
  "Build complete!" while writing no image ids. The chain then refuses to run with
  ``Raster.lock records no image id for tile 'x' — run `cargo raster build` `` — advice that does
  not fix it. Use `cargo raster build --backend risc0`.
- **The gate tree must sit where `../../../raster` resolves natively.** Reaching the raster crates
  through a symlink makes the guest build fail with `package collision in the lockfile: packages
  raster v0.1.0 (...) and raster v0.1.0 (...) are different`.

`tiny-gemma-dev` (`casettek/.../assets/tiny-gemma-dev`: 4 layers, hidden 4, vocab 280, ple 2,
sliding window 2, 2 KV-shared layers) exercises both attention types, KV sharing and a sliding
window in a chain that imports in seconds. Run it with `cargo raster chain run --no-auth`, which
"stages still link through their real `output.bin`, so the chain computes the same values" — token
equality without per-stage guest builds. The authenticated run builds a RISC0 guest per stage and
took 15 minutes to reach stage 3 of 11; it is not an iteration loop.
