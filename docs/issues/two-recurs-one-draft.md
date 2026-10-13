# Issue: `two-recurs-one-draft` — a stage output cannot hold two lists of different lengths

Status: **closed 2026-08-28** by `raster` `docs/proposals/recur-deferred-finalize.md`
(branch `feature/recur-deferred-finalize`). Kept because the shape rule it names is still
undocumented anywhere else, and because the reasoning in "Why this is cheap" below is what made
the fix small.

Related:
- `docs/proposals/multi-token-decode.md` §4.2 — the design this invalidates as written, and §6's
  workaround options.
- **`raster` repo, `docs/proposals/program-chain.md`** — "Linear chains only for v1 (single output
  → single named input per link)" (`:79`); multi-output chains are an explicit non-goal (`:388`).
- **`raster` repo, `docs/proposals/program-end.md`** — "Multiple named outputs (a single `output`
  artifact for now)" (`:309`).
- `raster` repo, `docs/proposals/incremental-draft-witness.md` `:16` — notes that `finalize`
  *severs provenance*, which is presumably why it is one-way.

## The gap

Every `call_recur!` / `call_recur_seq!` form finalizes its output draft and returns
`AuthRef<S>`, not `Draft<S>`:

| driver | `raster/src/input.rs` | returns |
| --- | --- | --- |
| `run_recur_list` | `:2452` | `AuthRef<S>` |
| `run_recur_list_with_state` | `:2732` | `AuthRef<S>` |
| `run_recur_sequence_list` | `:2788` | `AuthRef<S>` |
| `run_recur_sequence_list_with_state` | `:2877` | `AuthRef<S>` |

and nothing converts a finalized value back into a draft — `IntoDraft` is implemented only for
`Draft<S>` (`:1805`) and `RecurSequenceOutput<S>` (`:1814`), both of which are already open handles.

So **one draft admits at most one recur.** A `call!` chain may thread a draft freely
(`prefill-prepare-aux` seeds one with `begin_ple_layer` before its recur), but the first recur ends
it.

The consequence is a shape rule that is nowhere written down:

> Every list in a stage output that is built by a recur is built by the *same* recur, over the
> *same* input list — so they all have that list's length.

`prefill-range` satisfies it by accident. `finish_layer` (`raster-stages/prefill-range/src/lib.rs:1008`) pushes
exactly one `ActivationRow` and one `KeyRow` per query, so `rows` and `kv` are both `queries.len()`.

## Why decode needs to break it

A decode stage processes **one** token and must publish a K/V cache holding **every prior token
plus that one**:

```
ActivationSequence { rows: 1 entry, kv: prompt_len + t + 1 entries }
```

Those lengths differ, so they need two recurs — one over `queries`, one over the inherited cache —
into one draft. That is the thing that cannot be expressed. The cache is not optional: stage `t`'s
`kv` is stage `t+1`'s `prior_kv`, and without it a decode step attends over a single key.

Attention itself is fine. `prefill-range::main` now takes a `prior_kv` parameter and sweeps it as a
third key list, so a decode stage can *read* an inherited cache correctly today. It is only
*republishing* the grown cache that has no expression.

## What is not the gap

- **Not `select!`-side.** Reading the cache is already solved.
- **Not chunking or cost.** The copy is ~8 chunked replay units per layer per step (§7 of the
  proposal). It is affordable; it is unsayable.
- **Not `prefill-range`-specific.** Any stage whose output has one list per input item and one list
  that accumulates across stages hits this. Decode is just the first.

## Hands off to

Either of two upstream changes would close it, and they are not equivalent:

1. **A finalize-deferring recur output.** Let `call_recur!` return `Draft<S>` when its `output` is
   an already-open draft binding rather than `new!(T)`, leaving `finalize(draft)` — which is
   already in the sequence grammar — to end it. This is the smaller change and the one that fits
   the existing vocabulary. It needs an argument about what a partially-built draft commits to
   between recurs, which is where `incremental-draft-witness.md`'s note about `finalize` severing
   provenance becomes load-bearing.
2. **Multi-output stages** (`program-chain.md` `:388`, `program-end.md` `:309`). Larger, already
   contemplated, and would let the cache be its own artifact rather than a field.

Until one lands, the workaround is a separate `kv-merge` stage per layer per step — which pays 35
extra chain stages per generated token and needs order-independent attention scoring to be correct,
because a draft can only append and the merged order would be fresh-then-prior.

## Resolution (2026-08-28)

Closed upstream by an opt-in `finalize = false` on `call_recur!`, which hands the draft back
instead of closing it. `prefill-range` now builds its output with two writers — `carry_cached_key`
carries the inherited cache forward, then `attend_token` appends one row and one key per token — so
`rows: n` and `kv: prior + n` coexist.

**The design question this issue raised turned out to be already answered in the code.** "What does
a partially built draft commit to between recurs" looked like the hard part. But
`verify_draft_transition` (`raster-prover/guests/transition/src/checks/drafts.rs`) attaches draft
witnesses to **`TileExec` steps**, not to the recur wrapper: every iteration carries a
`DraftReplayTransition` with `root_before`, and the guest asserts root continuity across steps.
Appends inside a recur were always attested one iteration at a time. `finalize` materializes a
value and binds an `AuthRef`; it is not what makes the appends sound. Deferring it moves no
attestation.

Verified prefill-neutral on `tiny-gemma-dev`: identical `prefill_finalize` commitments
(`payload=c9124ff6db19…`, `structural=b806106fdc3b…`) before and after.
