# Model families

A model family is everything about a model that the chain cannot read from a number in its
config: how a layer computes, how its tokenizer scans text, what its bundle contains, and
which stage programs run it. Two Gemma 4 sizes are one family. A mixture-of-experts Gemma
or a Qwen model would each be another.

The chain itself is not per family. The schedule, the ranges, the commitments, the
challenge flow, the exact integer numerics and the backend are shared, and a new family
must not change what an existing one runs.

Today there are two families:

- `gemma-dense`: Gemma 3n and Gemma 4 dense models, imported from E2B, E4B and the parity
  bundle. It has every stage program and can be claimed.
- `mistral-dense`: Mistral's Ministral 3 models, imported from the 3B and the
  `parity-mistral` bundle. It has its own layer program and shares every other one. Its
  small model passes the parity gate; a claim on the 3B itself has not been built yet
  ([what it has so far](#mistral-dense-so-far)).

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

So a family brings its own layer program and shares the building blocks it is made of:
`det-num`, the weight-page products, attention scoring and the model identity checks.

The rule for any stage: **a family gets its own program when its steps differ, and shares
a program when only the rules inside the steps differ.** A layer's steps differ from
family to family. A tokenizer's do not: every one hashes the text, cuts it into units,
looks each unit up, and turns the units into pieces. So `prompt-bind` and `prompt-check`
are one program each for all families, and a value in the committed tokenizer profile
chooses the rules. Sharing holds only while it costs a family nothing it would not pay
alone: if a shared program makes a family's trace noticeably longer or its heaviest tile
heavier than its own program would be, that family gets its own.

## Where a family is recorded and checked

| Where | What it does |
| --- | --- |
| `inference_artifacts::ModelFamily` | The list of families this build has. `ALL`, `name` and `parse` are the only way a name becomes a family. |
| `manifest.json`, field `family` | Written by the import. A manifest without it is `gemma-dense`. `ModelManifest::load_verified` refuses a family this build does not have, before any inference. |
| `manifest.json`, field `shape` | The family's own numbers, as `inference_artifacts::FamilyShape`. A manifest whose shape is not its family's is refused. |
| `model-import`, `bundle_family` | Reads `model_type` from a bundle's `config.json` and refuses one that is not a family this build imports. |
| `model-import`, `gemma::require_known_tensors` | Refuses a bundle with language-model weights the family's import does not read. |

The last check exists because two families can share a `model_type`. A dense and a
mixture-of-experts Gemma 4 both declare `gemma4`. An import only reads the tensors it asks
for, so without the check a bundle with expert matrices would import cleanly as a dense
model that computes something else.

## What is shared and what is not

| Part | Shared | Per family today |
| --- | --- | --- |
| Schedule and step rules (`pipeline-core`, `model-identity`) | yes; a model's descriptor has one auxiliary root per layer or none | a layer's step is `ModelContext::layer` when the model has per-layer inputs and may borrow a cache, and `plain_layer` when it has neither |
| `input-embedding`, `sequence-check` (which also states the output record) | yes | the head's softcap and the embedding scale are manifest values |
| `prompt-bind`, `prompt-check` | the stages and the merge check; the chat template, the scanner and the longest token are data in the tokenizer profile | two scanners in `prompt-core`: one atom a character with Gemma's space rules (`prompt.rs`), and one atom a word for a byte-level tokenizer (`tekken.rs`) |
| The layer program | the steps' shape: project Q/K/V, score and softmax over the keys, the feed-forward matvecs, and the weight-page tile | `prefill-range` with `prefill-prepare-aux` is Gemma's; `layer-mistral` is Ministral's. The tiles both need are copied between them, not shared |
| Native kernels (`inference-kernels`, `direct-infer`, `staged-infer`) | batched products, the GPU path, caching | `prefill_range.rs` is Gemma's layer |
| `model-import` | command line, layout, paging, tokenizer externals, commitments, chain template, manifest | `gemma.rs`: config, tensors, layer and head externals, manifest shape |
| Manifest `shape` | | the family's own: `DirectInferShape` is `gemma-dense`'s, `MistralShape` is `mistral-dense`'s |
| Backend | yes; `IRIS_MODEL_MANIFEST` chooses the model, and its one-shot commands take `--model <import-name>` for another import beside it | |

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
5. **Stage programs.** Author its layer program from the shared tiles. If its tokenizer
   cuts text by rules the prompt stages do not have, add a scanner to `prompt-core` and a
   value for it in the profile; a tokenizer whose steps differ gets its own program.
   Follow the Raster skill and its check ladder. The import's chain template names the
   family's programs.
6. **A parity bundle.** `model-bundles/parity-gemma` is a four-layer model small enough
   for `test-parity` to run all three paths and the challenge leg. A family needs its own,
   with its own falsified-stage cases, or its programs are not under the gate.
7. **Measure the three numbers** on a real model before tuning: claim time, the trace of
   the heaviest stage (`cargo raster run --commit … --fraud-step list` counts steps by
   tile), and the cycles of its heaviest tile.

## `mistral-dense` so far

Ministral 3 is a plain pre-norm decoder: full attention in every layer, a gated-SiLU
feed-forward block, one rotation table with YaRN scaling that turns consecutive pairs, a
byte-level tokenizer, and a head tied to the embedding table with no cap.

What exists:

| Part | Where |
| --- | --- |
| Conversion from GGUF | `iris model convert`, unchanged: every tensor name was already mapped |
| Import | `model-import`'s `mistral.rs`: the shape, the tensors a bundle may hold and the layer externals. The tokenizer, the embedding table, the head and the chain template are the shared code's, told by `ChainLayers` that this family has no auxiliary stage and no donor caches |
| Layer stage program | `raster-stages/layer-mistral`: three inputs, the activations, the layer and its own cache from the range before |
| Native twin of the stage | `inference-kernels`' `mistral::stage`, run by `staged-infer`'s `layer_mistral` routine. It reads the committed weight pages in place, on the CPU |
| Parity bundle | `model-bundles/parity-mistral`: four layers, a tekken-style vocabulary of 512 with a forty-byte token, three cases. `iris test-parity --bundle parity-mistral` runs the gate on it |
| Layer | `inference-kernels`' `mistral::layer`, and in `det-num`'s `mistral.rs` the integer SiLU, rotation and score scale that only this family uses |
| Tokenizer, reference | `run-prep`'s `tekken.rs`: the pattern as a backtracking matcher and the merge, which direct inference runs |
| Tokenizer, prompt stages | `prompt-core`'s `tekken.rs`: the same split as a scanner that takes one character at a time, with the Unicode categories in `unicode_classes.rs` (written by `scripts/gen-unicode-classes.py`) |
| Output decoding | `sequence-check`: a decoder entry says its text is bytes, and a character that spans tokens is put together |
| Direct inference | `direct-infer`, which branches on the family where it loads layers and runs them |

How it is checked (`crates/inference/direct/tests/mistral_reference.rs`, an ignored test
that needs an imported model): the integer logits are compared with the same weights
computed in `f64`, and with another implementation's logits read from a file. `run-prep`'s
ignored `ids_match_a_reference_tokenizer` compares token ids the same way, both the
reference tokenizer's and the ones the prompt stages' pieces merge to.

How the prompt stages read the tokenizer:

- **Words.** `prompt-bind` cuts plain text into words by the tokenizer's pattern, one
  character at a time, and makes each word one unit with one vocabulary lookup. A word
  that is a token is one piece, which is the tokenizer's own rule. Any other word is one
  piece a byte. Each word is a segment, so no merge crosses words. `run-prep`'s tests
  compare the scanner with the pattern on every string of up to five character kinds, and
  an ignored test on every string of seven.
- **Long words.** A word longer than the longest token (76 bytes) is no token. It is put
  out in parts of at most that length, all in one segment.
- **Text the scanner cannot place.** Two things need unbounded look-ahead: blanks after a
  line break belong to the break's word only if another break follows, and capitals after
  a letter with no case belong to its word only if such a letter or a lower-case one
  follows. The scanner holds up to 1,024 bytes of either. Past that it places them as if
  the text had ended and the bind records the prompt fault `FAULT_WORD_SPLIT`.
- **Merges.** `prompt-check` is unchanged apart from its round count.

What a claim still needs:

- **A reference for the real model.** The 3B imports in full, short claims on it build
  with no fault, and a falsified layer checkpoint is found and its stage replayed under
  authenticated Raster. It has no entry under `tests/real-model` yet, so
  `just test-real-model` does not hold it to reference checkpoints, and no long prompt
  has been claimed.
- **Speed.** The stage's native twin runs its matrix products on the CPU. Gemma's go
  through a weight cache and the GPU.
- **Page sizes for the 8B.** The 3B's rows fit every page in [numerics.md](numerics.md).
  The 8B's do not: its rows 4096 wide do not fit the logits-head page, and its rows 14336
  wide do not fit the matrix page.
- **Positions past 16,384.** From there the model scales its queries by a logarithm of the
  position. The import records the limit and the layer refuses positions beyond it.

## Changes that re-lock every stage

These are shared code that every stage program reaches, so each one re-locks nearly all
programs and moves imports and prepared runs to a new version. Make them together.

- **A stage kind the schedule does not have.** A model without per-layer inputs has no
  auxiliary roots in its descriptor and runs no auxiliary stage; its layer program takes
  its step with `plain_layer`, which asks the schedule for no per-layer input and refuses
  a descriptor that has auxiliary roots. A layer made of two stages, as a
  mixture-of-experts layer might be, needs a step the schedule rules do not name yet.
- **The descriptor's roles.** It names a tokenizer, a decoder, an embedding, a head, and
  per layer a transformer root and optionally an auxiliary one. A family that commits
  something else per layer either folds it into the layer's root or changes this.
- **A new `det-num` function** that a shared tile calls. A function only a family's own
  tiles call re-locks only that family's programs.

## What the candidates would need

From public descriptions, not from configs that have been read here:

- **Gemma 4 26B A4B**, mixture of experts. Reported as 30 layers with 128 experts of which
  8 run per token, about 3.8 billion of 25.2 billion weights per token. Compute per token
  is near E4B's; the weights are about three times larger. New: a router and top-k pick, an
  expert path where each token opens only its experts' pages, per-expert externals, and
  native products grouped by expert.
- **Qwen3** (for example Qwen3-4B). A plain decoder close to `mistral-dense`: a byte-level
  tokenizer with a regular-expression split and its own chat template, a gated-SiLU
  feed-forward block, one RoPE base, no per-layer inputs, and full attention in every
  layer. It adds a norm on each query and key head.
- **Qwen3.5** (for example Qwen3.5-4B). Not a plain decoder. Its config, read on
  2026-11-13, has three of every four layers as Gated DeltaNet, a linear attention with a
  state carried from token to token and a short convolution, and the fourth as gated full
  attention. New: sigmoid, softplus, an exponential and an L2 norm in exact integer form,
  and a per-layer state handed from one range to the next, which the schedule has no step
  for.

Full attention in every layer is a property of these models: on a long prompt every layer
stage is as heavy as Gemma's heaviest.
