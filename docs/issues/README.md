# Issues — gaps without an owner

Index of `docs/issues/`. Last reviewed 2026-08-27.

Same convention as the `raster` repo's `docs/issues/README.md`: an **issue** names a gap and stops
before the design. Every claim is cited to a file and line in this repo. Issues here are about
*this program* — how it uses the model. A gap in the model itself belongs in `raster`'s
`docs/issues/`, and an issue here should say which upstream issue it hands off to.

## Open

| issue | opened | subject | upstream |
| --- | --- | --- | --- |
| [`short-stage-fraud-window`](./short-stage-fraud-window.md) | 2026-11-03 | An authenticated stage replay with `--fraud-proof-window-size 128` cannot commit a trace of 128 items or fewer, so such a stage cannot be challenged. Some per-range stages are short on small ranges. The one-tile origin stages, short at every size, and `output_finalize`, short under 40 output tokens, are no longer stages (2026-11-11). Deferred to its own project. | none |
| [`append-shaped-accumulators`](./append-shaped-accumulators.md) | 2026-08-25 | `mac_weight_page` and `score_key` carry an accumulator in `RecurState` that they never read back — the writes are appends, so they belong in `RecurOutput` and pay only their increment. `score_key`'s carrier grows, making the amplification the iteration count rather than a constant. `accumulate_context` genuinely needs read-back and is stuck. | `raster` `docs/issues/recur-accumulator-slots.md` |

## Closed

| issue | opened | closed | subject |
| --- | --- | --- | --- |
| [`two-recurs-one-draft`](./two-recurs-one-draft.md) | 2026-08-27 | 2026-08-28 | One draft admitted one recur, so every list in a stage output came out the same length — a decode stage could not publish 1 activation row beside a cache of `prompt_len + t + 1`. Closed by `raster`'s `recur-deferred-finalize`. |
| [`decode-init-authenticated-finalize`](./decode-init-authenticated-finalize.md) | 2026-09-10 | 2026-11-05 | `decode_init` could not commit its finalized draft under authentication. No longer reproduces: whole chains finish authenticated and pass `chain audit --execution`. |
| [`donor-binding-not-templatable`](./donor-binding-not-templatable.md) | 2026-08-27 | 2026-08-28 | Closed by binding both donor candidates uniformly and selecting the exact committed `kv_donor_layer` inside the tiles; decode now uses `[[chain.repeat]]`. |
