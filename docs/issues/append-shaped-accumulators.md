# Issue: `append-shaped-accumulators` — three recur tiles carry in `state` what they could append

Status: open 2026-08-25. Unowned. **§4 (`score_key`) resolved 2026-08-28** — not by moving to
`RecurOutput`, but by making the carrier stop growing: scores are now written into a dense
`num_heads × window_len` array addressed by `key.position - window_start`, so the accumulator is a
fixed size regardless of how many keys the sweep sees. §3 (`mac_weight_page`) and §5
(`accumulate_context`) are unchanged.

Related:
- **`raster` repo, `docs/issues/recur-accumulator-slots.md`** — the upstream gap: no loop-carried
  slot is both readable and incrementally committed. That issue cites this repo as its motivating
  program. §4 here is the part of that citation that does *not* hold: two of the three tiles it
  points at are not stuck on `state` at all. §5 is the part that does.
- `raster` repo, `docs/proposals/incremental-draft-witness.md` — landed the frontier that makes
  `RecurOutput` pay only for the increment. It is what makes §3's rewrite worth doing; before it,
  both slots were expensive and the choice did not matter.
- `raster` repo, `docs/proposals/recur-sequence-break.md` / `recur-progress-commitment.md` — an
  output-only `call_recur!` still accepts `RecurControl` (`raster-macros/src/recur.rs:141`), so
  the rewrites in §3 and §4 keep the early exit, and gain one the tiles do not have today.

## 1. The test

A `call_recur!` keeps nothing between iterations. Iteration *i+1* sees only what iteration *i*
serialized and returned, so "carrying" a value means **re-emitting all of it, every time**. A
`RecurState<T>` costs `2 · N · |T|` committed bytes over `N` iterations regardless of how much of
`T` the iteration touched. A `RecurOutput<S>` costs only the increment, but cannot be read back.

One question decides which slot a loop needs:

> **Does what the iteration writes depend on what is already there?**

- **No** — the iteration computes its slice from `input` and `args` alone, and the slice lands
  after everything written so far. The accumulator is an *append*, and belongs in `output`. Any
  read of the previous state is transport, not computation: it exists only to re-emit the parts
  this iteration did not touch.
- **Yes** — the write is a function of the accumulated value (`+=`, a running max, a rescale).
  `output` cannot serve, because a draft is append-only by verification rule
  (`raster-core/src/draft.rs:944` rejects a second `Set`). This is the upstream gap.

A computed index is not by itself a reason to need `state`. An index that only ever moves forward
by a fixed step is an append with extra arithmetic.

## 2. Every recur tile in this repo, classified

| tile | site | write depends on what's there? | verdict |
| --- | --- | --- | --- |
| `mac_weight_page` | `raster-stages/prefill-prepare-aux/src/lib.rs:93`, `raster-stages/prefill-range/src/lib.rs:150` | **no** — `out[out_idx] = dot(..)`, disjoint rows in ascending order | append-shaped — §3 |
| `score_key` | `raster-stages/prefill-range/src/lib.rs:405` | **no** — `scores.push(dot(..))`, literally an append | append-shaped — §4, and the expensive one |
| `accumulate_context` | `raster-stages/prefill-range/src/lib.rs:519` | **yes** — `*dst = mac_bits(*dst, weight, *value)` over every head × lane | genuinely stuck — §5 |
| `attend_kv_chunk` | `raster-stages/prefill-range/src/lib.rs:642` | **yes** — a new head max rescales every value already accumulated | genuinely stuck, **and dead** — §6 |
| `summarise_errors` | `raster-stages/prefill-range/src/lib.rs:1069` | **yes** — reads `state.count` to decide whether to set `first` | correct as written — `\|ErrorSummary\|` is a `u32` and one `String` |

`summarise_errors` is the shape `state` is for: a read-back the loop genuinely needs, on a carrier
small enough that `2 · N · |T|` is noise. The other four are the interesting cases.

## 3. `mac_weight_page` — the clear case

`raster-stages/prefill-prepare-aux/src/lib.rs:93`, and seven more call sites in `raster-stages/prefill-range/src/main.rs`
(`:35`, `:42`, `:49`, `:116`, `:133`, `:140`, `:149`, `:167`, `:176`). The tile sweeps a row-major
weight matrix a page at a time and writes each output row at `page.offset() / stride` (`:135`).

The write is an assignment, not an accumulation:

```rust
let mut out = unpack_i32s(&state.values)?;   // lib.rs:134 — decode all of it
let start_row = (page.offset() / stride) as usize;
for local in 0..rows_in_page {
    out[start_row + local] = dot(&row, w)?;  // = , not +=
}
state.values = pack_i32s(&out);              // lib.rs:145 — re-encode all of it
```

Iteration *i* uses **none** of the values iteration *i−1* wrote. It decodes the whole accumulator
so that it can re-encode the entries it did not touch, because it is the only party that gets to
write the buffer handed forward. That is transport.

And the writes are append-shaped: `projection_pages` is swept in list order, page offsets ascend,
rows within a page are contiguous, so `start_row` is always `input.index() * rows_per_page`. An
`AppendOnlyVec` of per-page chunks reproduces the vector exactly, and the ordering becomes *more*
checkable than it is now — `start_row == input.index() * rows_per_page` is an assertion the tile
can make from `RecurInput::index` instead of trusting `page.offset()`.

Numbers for the PLE projection (`hidden` 2048, `ple_width` 256, `PAGE_SIZE` 196_608 from
`raster-stages/prefill-prepare-aux/src/input.rs:23`):

```text
matrix   256 × 2048 × 4 = 2_097_152 B  →  10 full pages of 24 rows + 1 of 16 = 11 iterations
accumulator             = 256 × 4     = 1_024 B

today, as state    11 × 2 × 1_024 B  = 22_528 B      22×
as an output draft 11 ×      96 B    =  1_056 B
```

The rewrite costs one flatten tile downstream, because `finish_ple_projection` wants a flat
256-vector rather than a list of chunks.

It also fixes a real waste on the error path. The tile latches into `state.error` (`:105`) and
short-circuits (`:100`), so after a bad page the remaining iterations still run — each
materializing a 192 KiB weight page to reach line 100 and turn around. Output-only mode accepts
`RecurControl`, so the same condition becomes a `Break`.

## 4. `score_key` — the same shape, quadratic

`raster-stages/prefill-range/src/lib.rs:405`. `ScoreAccum.scores` starts empty (`zero_scores`, `:389`) and the
tile appends:

```rust
for head in 0..heads {
    scores.push(dot(q_head, k_head)?);       // lib.rs:445
}
state.scores = pack_i32s(&scores);
state.count += 1;
```

`push` on a `Vec` that is unpacked from the previous state and repacked. Nothing already in
`scores` is read. `state.count` is a counter that `softmax_scores` recovers as
`scores.len() / heads` anyway (`:487` already checks exactly that identity).

Unlike §3 the carrier **grows**, so the amplification is not a constant — it is the iteration
count. For one query row over `L` keys with `p` of them visible, the carry is
`2 · Σᵢ |state at i|`; taking a full-attention layer where the query sees everything before it:

```text
useful output    L · heads · 4 B
carried state    ≈ L² · heads · 4 B        amplification ≈ L
```

At `heads = 8` (Gemma 3n E2B — confirm against `config.json` at import):

```text
L = 512    16 KB of scores,   8.4 MB carried      512×
L = 2048   64 KB of scores,   134 MB carried     2048×
```

Per query row, per layer — and `attend_token` runs it twice, over `keys` and `donor_keys`
(`main.rs:84`, `:90`), with the state chained across both. Note that the non-contributing
iterations are not free: a key that fails `visible()` or the `donor_pass` check returns `state`
unchanged (`:414`–`:419`) and still pays `2 · |state|`.

As an output draft the same loop pushes `heads` values per visible key and pays `L · heads · 4 B`
total — the amplification disappears entirely rather than shrinking.

## 5. `accumulate_context` — genuinely stuck, and it belongs upstream

`raster-stages/prefill-range/src/lib.rs:519`. This one answers **yes** to §1:

```rust
let dst = &mut acc[head * head_dim + lane];
*dst = mac_bits(*dst, weight, *value);       // lib.rs:577
```

Every visible key adds into every head × lane. The write is a function of the accumulated value,
the touched region is the whole accumulator, and a draft cannot express it — `apply_draft_ops`
rejects the second write to a field. There is no rewrite here.

`CtxAccum.acc` is `heads · head_dim` `i64`s — 16 KiB at 8 heads × 256 — fixed for the loop, and
carried `2 · L` times per query row per layer:

```text
L = 2048   16 KB of context,   67 MB carried       ≈ 4000×
```

This is the case the upstream issue is about, and nothing in this repo can close it. Worth
recording here so that fixing §3 and §4 is not mistaken for fixing the problem: the remaining
cost after both rewrites is this tile, and it is the larger of the two today.

## 6. `attend_kv_chunk` is dead

`raster-stages/prefill-range/src/lib.rs:642` and its `softmax_context` finisher (`:755`) implement streaming
softmax with a running max, and are **never called** — `attend_token` uses the two-pass
`score_key` → `softmax_scores` → `accumulate_context` path instead (`main.rs:82`–`:110`), which
the comment at `:77` explains as matching the reference. The `attend_kv_chunk` comment at `:648`
describes the same donor two-pass arrangement, so it reads as a superseded implementation left in
the tree.

```bash
grep -rn 'attend_kv_chunk\|softmax_context' --include=*.rs . | grep -v /target/
```

Either delete it or say in a comment why it is kept. It is also the sharpest example of §5's
category — a new max rescales values already written — so if it is deleted, that example should
move into the upstream issue rather than vanish.

## 7. Directions, none chosen

1. **Rewrite `mac_weight_page` output-only**, chunk per page, plus a flatten tile. Answers: eight
   more call sites move with it, and every downstream consumer of `ProjAccum` changes shape.
2. **Rewrite `score_key` output-only.** Answers: `softmax_scores` must derive `count` from the
   list length, and the donor two-pass chaining has to become two drafts merged, or one draft
   threaded through both `call_recur!` sites — check that `RecurOutput` supports the second.
3. **Do 2 only.** The quadratic one is the whole cost; §3 is 22× on a slot that is ~1% of its
   replay unit.
4. **Nothing, pending measurement.** §3–§5 are arithmetic over declared shapes, not a benchmark.
   `raster-runtime`'s `profiling` feature writes per-tile `input_bytes` / `output_bytes`; the
   flatness of those across a loop is the claim, and it has not been observed.

## 8. Reproducing

```bash
# the test applied, tile by tile
sed -n '128,148p' raster-stages/prefill-prepare-aux/src/lib.rs    # out[i] = dot(..)      — append-shaped
sed -n '427,450p' raster-stages/prefill-range/src/lib.rs          # scores.push(..)       — append-shaped
sed -n '570,582p' raster-stages/prefill-range/src/lib.rs          # *dst = mac_bits(*dst, — stuck
sed -n '705,745p' raster-stages/prefill-range/src/lib.rs          # rescale on new max    — stuck, dead
sed -n '1069,1081p' raster-stages/prefill-range/src/lib.rs        # reads state.count     — correct

# the call sites
grep -n 'call_recur!' raster-stages/prefill-range/src/main.rs
grep -rn 'attend_kv_chunk' --include=*.rs . | grep -v /target/    # no call site

# the shapes
grep -n -A6 'struct ScoreAccum\|struct CtxAccum\|struct AttnState' raster-stages/prefill-range/src/input.rs
grep -n 'PAGE_SIZE' raster-stages/prefill-prepare-aux/src/input.rs
```

Measured figures, which this issue does not have. With `raster-runtime`'s `profiling` feature
enabled, `cargo raster run` writes `profile.json` into `target/raster/runs/<run_id>/`. For any of
these tiles the carried state per iteration is `input_bytes` minus the driving item and the args,
and `output_bytes` is the accumulator in full. For `mac_weight_page` and `accumulate_context` both
should be flat across the loop; for `score_key` both should climb linearly. That is the issue.
