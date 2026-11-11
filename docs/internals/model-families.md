# Model families

A model family is everything about a model that the chain cannot read from a number in its
config: how a layer computes, how its tokenizer scans text, what its bundle contains, and
which stage programs run it. Two Gemma 4 sizes are one family. A mixture-of-experts Gemma
or a Qwen model would each be another.

The chain itself is not per family. The schedule, the ranges, the commitments, the
challenge flow, the exact integer numerics and the backend are shared, and a new family
must not change what an existing one runs.

Today there is one family, `gemma-dense`: Gemma 3n and Gemma 4 dense models, imported from
E2B, E4B and the parity bundle.

## Why families get their own programs

One layer program that handled every architecture would be simpler to list and worse on
each number a claim is judged by:

- **Longest replay.** A stage's trace holds every step it takes, including the branches a
  given model never needs, for every token of every layer.
- **Heaviest fault proof.** The proof covers one tile, so the heaviest tile sets it. Tiles
  written for one family's shapes can be sized for them. The page sizes in
  [numerics.md](numerics.md) are already a compromise between two widths of one family.
- **Stability.** A stage program's identity covers all of its code. With one shared
  program, a change made for one family re-locks and re-versions every other.

So a family brings its own layer and tokenizer programs and shares the building blocks
they are made of: `det-num`, the weight-page products, attention scoring and the model
identity checks.

## Where a family is recorded and checked

| Where | What it does |
| --- | --- |
| `inference_artifacts::ModelFamily` | The list of families this build has. `ALL`, `name` and `parse` are the only way a name becomes a family. |
| `manifest.json`, field `family` | Written by the import. A manifest without it is `gemma-dense`. `ModelManifest::load_verified` refuses a family this build does not have, before any inference. |
| `model-import`, `bundle_family` | Reads `model_type` from a bundle's `config.json` and refuses one that is not a family this build imports. |
| `model-import`, `gemma::require_known_tensors` | Refuses a bundle with language-model weights the family's import does not read. |

The last check exists because two families can share a `model_type`. A dense and a
mixture-of-experts Gemma 4 both declare `gemma4`. An import only reads the tensors it asks
for, so without the check a bundle with expert matrices would import cleanly as a dense
model that computes something else.

## What is shared and what is not

| Part | Shared | Per family today |
| --- | --- | --- |
| Schedule and step rules (`pipeline-core`, `model-identity`) | yes | assumes one auxiliary stage per layer |
| `range-origin`, `decode-init`, `input-embedding`, `sequence-check`, `output-finalize` | yes | the head's softcap and the embedding scale are manifest values |
| `prompt-bind`, `prompt-check` | the stages | the scanner in `prompt-core` implements Gemma's rules; the chat template is two constants |
| `prefill-range`, `prefill-prepare-aux` | the weight-page and scoring tiles | the layer's mathematics is Gemma's |
| Native kernels (`inference-kernels`, `direct-infer`, `staged-infer`) | batched products, the GPU path, caching | `prefill_range.rs` is Gemma's layer |
| `model-import` | command line, layout, paging, tokenizer externals, commitments, chain template, manifest | `gemma.rs`: config, tensors, layer and head externals, manifest shape |
| Manifest `shape` | | the family's own; `DirectInferShape` is `gemma-dense`'s |
| Backend | yes; `IRIS_MODEL_MANIFEST` chooses the model | |

## Adding a family

In this order, because each step can be checked before the next:

1. **Read the real config and weights.** List the bundle's tensor names and shapes. Write
   down where the layer differs from an existing family in exact integer terms: every new
   function needs a definition in `det-num` that the native kernel and the guest share.
2. **Add the variant** to `ModelFamily` and its name. Nothing else compiles against it
   yet; every `match` on the family will say where it is needed.
3. **Import.** Add the family's module to `model-import` beside `gemma.rs`: its config, its
   tensor inventory, how its tensors become stage externals, and its manifest shape. Teach
   `bundle_family` to recognise it. Choose its page sizes for its own widths.
4. **Native layer.** Add its layer to `inference-kernels` on the shared products, and run
   direct inference against a reference implementation's output before anything is staged.
5. **Stage programs.** Author its layer program, and its tokenizer program if its
   tokenizer differs, from the shared tiles. Follow the Raster skill and its check ladder.
   The import's chain template names the family's programs.
6. **A parity bundle.** `model-bundles/parity-gemma` is a four-layer model small enough
   for `test-parity` to run all three paths and the challenge leg. A family needs its own,
   with its own falsified-stage cases, or its programs are not under the gate.
7. **Measure the three numbers** on a real model before tuning: claim time, the trace of
   the heaviest stage (`cargo raster run --commit … --fraud-step list` counts steps by
   tile), and the cycles of its heaviest tile.

## Changes that re-lock every stage

These are shared code that every stage program reaches, so each one re-locks nearly all
programs and moves imports and prepared runs to a new version. Make them together.

- **Which stages a layer has.** The schedule requires one auxiliary stage before every
  layer stage, which only models with per-layer embedding inputs need. A family without
  them needs the schedule to say so, and a mixture-of-experts layer would add an expert
  stage after attention. Both are one change: the schedule naming the stages of a layer.
- **Binding the prompt one range at a time.** See
  [the proposal](../proposals/paged-prompt-bind.md). It changes the same step rules.
- **A new `det-num` function** that a shared tile calls. A function only a family's own
  tiles call re-locks only that family's programs.

## What the candidates would need

From public descriptions, not from configs that have been read here:

- **Gemma 4 26B A4B**, mixture of experts. Reported as 30 layers with 128 experts of which
  8 run per token, about 3.8 billion of 25.2 billion weights per token. Compute per token
  is near E4B's; the weights are about three times larger. New: a router and top-k pick, an
  expert path where each token opens only its experts' pages, per-expert externals, and
  native products grouped by expert.
- **Qwen.** A byte-level tokenizer with a regular-expression split and its own chat
  template, a different feed-forward activation, one RoPE base, no per-layer inputs, and
  full attention in every layer. The last is a property of the model: on a long prompt
  every layer stage is as heavy as Gemma's heaviest.
