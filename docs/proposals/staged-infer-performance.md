## Implementation update

The checkpointed path now uses run-owned, byte-bounded weight caches and persistent
auxiliary worker processes. Checkpoint encoding and post-write verification are
unchanged. See [claim-build execution and validation](../workflows/claim-build.md#weight-reuse-and-auxiliary-workers)
for memory configuration, instrumentation, and parity tests. The automatic cache
budget is 25% of available RAM. Main embeddings now retain only small verified
metadata; prompt and decode stages select the pages their tokens require.

PLE preparation now has a staged-only selective loader. Both prefill and decode
auxiliary stages read just the token-addressed embedding pages and cache the
small verified parameters/projection. The eager path remains available through
`STAGED_INFER_PLE_LOADING=eager`. All output persistence and verification remains
in place. See [selective PLE validation](../workflows/test-parity.md#selective-ple-loading-in-staged-inference)
for the authenticated small-model gate and performance comparison procedure.

Main embedding selection and output-head paging are now the standard Raster-input
implementation, with no new execution switches. Finalization verifies the page
count and scores all vocabulary rows in batches of at most 32 pages, preserving
Raster row boundaries and MAC order. Only metadata enters the weight cache.
Postcard compatibility uses the existing eager kernels. See
[paged weight validation](../workflows/test-parity.md#selective-main-embeddings-and-paged-output-head)
for preserved-build comparisons and the manual Gemma procedure.

### Validation measurements (2026-10-17)

The new staged suite passes 90 tests. The three `parity-gemma` cases cover
25, 103, and 106 checkpoints: every output/index/manifest byte, commitment, and
checkpoint hash matches preserved baseline, embedding-only, updated, and
unchanged authenticated Raster execution. Zero/constrained cache budgets and
serial execution also match. Multi-page native tests cover saturation and
rejection of invalid selections before publication. All 259 protected source
and lock files are unchanged from the implementation's starting snapshot.

Seven-iteration release microbenchmarks, with four Rayon threads and no retained
weight cache, gave these results on approximately 25 MB fixtures:

| Operation | Eager mean | Updated mean |
|---|---:|---:|
| Main embedding loading and computation | 39.319 ms | 6.779 ms |
| Head loading and computation | 46.504 ms | 46.975 ms |

Embedding loading improved about 5.8x. Head timing was essentially flat on this
fixture; page verification dominated its time. Its confirmed improvement is
removal of the complete matrix allocation, with each raw page batch capped at
6 MiB. That bound excludes decoded page buffers, proof scratch space, mappings,
and the required logits output. Real-Gemma runtime improvement remains to be
measured under the actual memory pressure.

The small-model runs below are single runs in baseline/embedding/updated order,
so OS cache and startup effects make them informational rather than speedup
claims. Sampled memory includes the process tree and is not an allocator bound.

| Case | Baseline time | Updated time | Baseline/updated sampled RSS |
|---|---:|---:|---:|
| short | 0.440 s | 0.252 s | 58.3 / 57.9 MiB |
| near-window | 0.988 s | 0.908 s | 60.0 / 59.8 MiB |
| over-window | 1.046 s | 0.931 s | 60.9 / 60.6 MiB |

Reports and preserved binaries are under `target/staged-pages-validation/`.
`summary.json` records the measurements, and `parity/report.json` records hashes,
per-run timings, memory samples, and the unchanged reference binaries.

The measurements and recommendations below describe the earlier 221-stage
baseline. They are not measured speedups for the new implementation.

## Historical baseline

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
