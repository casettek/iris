## Current State

The staged-infer path is already in decent shape: `staged-infer chain run` expands `Raster.toml` to 221 stages, runs supported stages directly, writes normal Raster-compatible `output.bin`/`output.rindex`/`output_manifest.json` checkpoints, and can leave one Raster stage in the run for parity comparison.

From the latest timing artifact I found:

- Wall time: `246.53s`
- Stage-sum: `358.44s`
- Aux waves: `151.9s` stage-sum collapsed to `39.1s` wall with 4-way parallelism
- Selected Raster reference: `prefill_range_l13` took `23.75s`
- Staged-infer shadow for that same checkpoint: `543.95ms`, with byte parity `MATCH`

So the current direct kernel for `prefill_range_l13` is not the bottleneck. Most remaining time is scheduling, repeated artifact loading, and large external materialization.

## Highest-Impact Work

1. Pipeline aux waves into range execution.

Right now the runner waits for all 35 `prefill_prepare_aux` jobs to finish before starting `prefill_range_l0`. But each range layer only needs its own aux output plus the previous range output. A dependency-aware scheduler could start `prefill_range_l0` as soon as `aux_l0` finishes, then continue layer-by-layer while later aux jobs are still running.

This preserves checkpoint parity because every aux and range stage still emits the same per-stage artifact. It just changes staged-infer scheduling from manifest-order batching to topological execution. Likely win: much of the current `~39s` aux-wave wall time can be hidden under the already-serial range chains.

2. Invert the hybrid parity mode for performance runs.

Today `--raster-stage prefill_range_l13` puts the Raster implementation on the critical path, then reruns staged-infer for comparison. That single choice costs about `23.2s` versus the matching staged-infer output.

A faster parity-preserving mode would run staged-infer as the pipeline source, write the normal checkpoint, and run the Raster reference as a shadow comparison against the same synthesized inputs. If the shadow mismatches, fail the run/report. Same parity evidence, but the slow Raster stage no longer blocks downstream staged-infer execution.

3. Share external resolver/cache state across stages.

The runtime supports mmap/ranged reads and resolver caching, but `with_stage_sequence_scope` installs a fresh file resolver per stage. That means repeated externals are repeatedly resolved/materialized: embedding/head artifacts are about `1.61GB`, transformer layers around `145MB`, PLE layers around `270MB`.

A shared cache keyed by `(path, index_path, commitment)` would preserve manifest commitment checks while avoiding repeated index parsing/mmap setup. A typed or semi-typed cache would help even more for embedding, final head, transformer layer, and PLE layer reuse across prefill plus decode steps.

4. Stop materializing whole huge inputs when the stage only selects pages.

The Raster stage implementations use `select!` to pull specific fields/pages. Direct-native loaders mostly call `materialize_auth_return` on whole arguments. This is especially expensive for `decode_embed`, which only needs one embedding row but currently pays to load the embedding table shape, and for final projection paths that turn huge `Bytes` regions into `Vec<i32>` matrices.

A staged-infer loader that mirrors the Raster stage’s selected access pattern would keep checkpoint parity because output encoding stays unchanged.

5. Reduce post-write rereads and in-memory clones.

For in-process direct stages, `write_output` already returns the structural commitment, but `finish_stage` rereads `output.bin` and recomputes roots. Also, `StageOutputCache` clones typed values and converts between near-identical activation structs. These are smaller than the scheduling/cache wins, but they add up across 221 stages.

## Guardrails

I would avoid “fusing away” checkpoints. The safe line is: compute however you want internally, but still emit the same canonical per-stage Raster artifacts and compare structural commitments/bytes against stage implementations. The biggest wins above stay inside that boundary.
