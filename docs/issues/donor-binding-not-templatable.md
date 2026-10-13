# Issue: `donor-binding-not-templatable` — the KV donor cannot be written as a repeat template

Status: **closed 2026-08-28** by uniform dual-donor bindings.

Related:
- **`raster` repo, `docs/proposals/chain-repeat.md`** (implemented 2026-08-27) — supplies the
  templating this issue finds one case short of. §2's templating rules are the constraint;
  §1's indexed `[chain.input]` is the idiom a fix should follow.
- `docs/proposals/multi-token-decode.md` §6 — the motivating design, and the two workarounds
  that ship without a fix.

## The gap

`chain.repeat` substitutes a bound index and subtracts one from it — *"`{ident-1}` is the only
arithmetic. It exists to express the sequential dependency that makes a repeat a chain rather than
a fan-out, and deliberately stops there"*. That covers every binding in this repo's per-layer
stages except one.

Gemma 3n's last 20 layers borrow their K/V from the last layer before the sharing boundary **with
the same attention type** (`crates/model/import/src/main.rs:260`, mirroring
`casettek/.../shared/model/gemma/io.rs:171`). For this model that resolves to:

| layer `l` | `inputs.donor_kv` |
| --- | --- |
| 0 … 14 | `input_embedding` — computes its own cache; the tile ignores the value |
| 15 … 34, sliding | `prefill_range_l13` |
| 19, 24, 29, 34 (full attention) | `prefill_range_l14` |

The full-attention layers are `l ≡ 4 (mod 5)`. A template has no conditional and no modulus, and
`start` / `count` select a contiguous run, so the third row cannot be carved out of the second.
The 20 bindings stay hand-written in `Raster.toml`, and nothing cross-checks them — the cost
`chain-repeat`'s own problem statement names: *"Every one of those is a `from = "..."` string that
is correct or silently wrong. `raster-inference` already carries twenty such hand-written donor
bindings."*

Multi-token decode multiplies the exposure by `max_new_tokens`: the same 20 bindings, re-derived
per step, 1,280 of them for a 64-token response.

## Why it is not a naming problem

A wrong donor is not a crash. Layer 19 attending over layer 13's cache instead of layer 14's has
the right shapes, the right widths, and a valid proof over the wrong activation — the reason
`model-import` prints the sharing map at import time rather than trusting it
(`crates/model/import/src/main.rs:734`). Templating removes the transcription step for 15 of the 35
layers and leaves the 20 that matter.

## What is not the gap

- **Not a request for expressions in manifests.** `chain-repeat` rejects that explicitly and it is
  the right call.
- **Not blocking.** `docs/proposals/multi-token-decode.md` §6 options 2 and 3 both ship today.
  This issue is about deleting a workaround, not unblocking one.

## Resolution

`prefill-range` now takes two donor candidates on every stage. The committed
`LayerParams.kv_donor_layer` is compared against each candidate's committed
layer number inside `score_key` and `accumulate_context`; exactly one candidate
can contribute. Layers before the sharing boundary bind both candidates to the
empty `input_embedding`, while sharing layers bind the two real donor stages.

That makes the topology uniform enough for `[[chain.repeat]]` without adding
conditionals, modulus, or a positional lookup table to the manifest language.
