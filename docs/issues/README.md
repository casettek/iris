# Issues — gaps without an owner

Index of `docs/issues/`. Last reviewed 2026-11-23.

Same convention as the `raster` repo's `docs/issues/README.md`: an **issue** names a gap and stops
before the design. Every claim is cited to a file and line in this repo. Issues here are about
*this program* — how it uses the model. A gap in the model itself belongs in `raster`'s
`docs/issues/`, and an issue here should say which upstream issue it hands off to.

## Open

| issue | opened | subject | upstream |
| --- | --- | --- | --- |
| [`short-stage-fault-window`](./short-stage-fault-window.md) | 2026-11-03 | An authenticated stage replay with `--fault-proof-window-size 128` cannot commit a trace of 128 items or fewer, so such a stage cannot be challenged. Some per-range stages are short on small ranges. The one-tile origin stages, short at every size, and `output_finalize`, short under 40 output tokens, are no longer stages (2026-11-11). Deferred to its own project. | none |
| [`append-shaped-accumulators`](./append-shaped-accumulators.md) | 2026-08-25 | `mac_weight_page` and `score_key` carry an accumulator in `RecurState` that they never read back — the writes are appends, so they belong in `RecurOutput` and pay only their increment. `score_key`'s carrier grows, making the amplification the iteration count rather than a constant. `accumulate_context` genuinely needs read-back and is stuck. | `raster` `docs/issues/recur-accumulator-slots.md` |

Closed issues are not kept here: the repository's history has them.
