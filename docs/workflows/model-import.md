# Model Import

`model import` is the shared setup step for every workflow. It reads a Gemma bundle and writes the committed artifacts the repo uses afterward.

Weight imports memory-map `model.detwgt` and pack the current external directly
into Raster pages. They do not load the entire file or retain decoded tensors
for the whole model. Per-layer embeddings are selected from the mapped table
one layer at a time, as whole `Q8_0` blocks. Signed 16-bit Q16.16 weights are
widened to the same canonical 32-bit values, preserving artifact bytes and
commitments. Raster serialization
still needs memory for the current external and its encoding buffers; this is
not a constant-memory streaming serializer. Keep the bundle unchanged during
import, as with direct inference's memory-mapped reader.

```bash
cargo run --release -p iris-cli -- model import \
  --model ../iris/assets/tiny-gemma-dev
```

Inputs:

- `model.detwgt`: deterministic weights, detwgt v3. The embedding tables, head,
  and attention and MLP matrices are `Q8_0` blocks; the rest is Q16.16 (see
  [numerics](../internals/numerics.md)). Write one with `iris model convert`.
- `config.json`: model shape, attention/window settings, softcap, and normalization parameters.
- `tokenizer.json`: vocabulary, merges, added tokens, and EOS metadata.

Outputs:

- `runtime/model-artifacts/<model-id>/Raster.toml`, the prompt-free [chain](../internals/raster-chain.md) template, saved on every full import. Its path and SHA-256 are recorded in the model manifest. `--range-size <positions>` (default 512) sets the largest range, in prompt tokens or claimed ids. It is recorded in the manifest and a run's configuration, so it is a per-model setting, not a per-claim one.
- Reusable Raster externals under `runtime/model-artifacts/<model-id>/raster/*`, written as `.rastered` and `.rindex` pairs with manifest commitments.
- `runtime/model-artifacts/<model-id>/manifest.json`, the prompt-free model manifest consumed by run specs.

## Converting a GGUF model

`model convert` turns a GGUF file with `Q8_0` weights into a bundle that
`model import` reads:

```bash
iris model convert --gguf ../models/gemma-4-E2B-it/gemma-4-E2B-it-Q8_0.gguf \
  --metadata ../models/gemma-4-E2B-detwgt --out ../models/gemma-4-E2B-q8
```

It copies `config.json`, `tokenizer.json` and `chat_template.jinja` from
`--metadata`, after checking that the GGUF's layer count, widths, head counts
and vocabulary match `config.json`. It then writes `model.detwgt`:

- `Q8_0` tensors are copied block for block.
- F32 and BF16 tensors become Q16.16 through `det_num`'s exact conversion.
- GGUF names map to the Hugging Face names the importer reads.
- `rope_freqs` is skipped, because RoPE comes from `config.json`.
- Any tensor without a mapping fails the conversion.

The block tensors must already be `Q8_0`. `model import` refuses a bundle whose
embedding or head tensors are Q16.16, so a model must be converted first.

By default, `<model-id>` is the sanitized model bundle directory name. Pass `--model-id <id>` to choose it explicitly, or `--artifact-root <dir>` to place model-scoped artifact directories somewhere other than `runtime/model-artifacts`.

Pass `--manifest <path>` to also save a template copy for manual Raster commands. Claims always use the model-specific template, so importing another model or overwriting that copy cannot change an existing model’s selection. Saved templates resolve stage and external paths against the importing workspace.

`infer`, `claim build`, and `challenge build` select their model through `--run inference.toml`. They verify the bundle’s recorded weight, config, and tokenizer hashes once before execution. Claims and challenges also require the recorded template hash to match. Missing provenance or changed files require a full import.

A full import records `model_commitments`, every committed model input by role,
and `model_id`, the hash of that list, in the model manifest (version 4). Paths are
not part of it, so moving an import does not change the id. Manifests from older
imports are refused: re-import them.

The prompt is deliberately not imported with the model. Its text, cuts and the
claimed ids are written under each run's `prompt/` directory during run preparation.
Legacy frozen prepared runs are rejected.

Useful partial modes:

- `--only-direct`: refresh only `runtime/model-artifacts/<model-id>/manifest.json` for direct inference. This clears template provenance, even if `--manifest` is supplied: existing staged artifacts cannot be bound to a refreshed bundle without a full import. Run a full import before building claims or challenges.
- `--only-tokenizer`: refresh tokenizer/decoder externals.
- `--only-layers`, `--only-embedding`, `--only-ple`: refresh targeted weight externals.

Partial external refreshes do not refresh model identity. Run a full import before using changed bundle files in a claim or challenge.

`model import` no longer accepts `--prompt`, `--raw-prompt`, or `--tokens`. Runtime prompt text and token count live in `inference.toml`.

## Model identity artifacts

Full imports now write a v5 model manifest, `model-descriptor.bin`, and Raster
descriptor externals for prompt binding and output finalization, plus decoder
membership witnesses. The descriptor travels in the authenticated model context
through the intervening stages. The descriptor is
`136 + 64 × layers` bytes (2,376 bytes for 35 layers). Component roots and the
derived model ID retain their previous encoding. Decoder proofs are separate
from that package and do not change the decoder's root.

Reimport models and prepare new runs after this schema change. v4 model/run
artifacts are rejected; replay historical claims with their original Iris
revision and Raster identities. Direct-only imports still support discovery,
but cannot produce a verified claim package.

All component roles are checked by their consuming stages. Imports include separate
reusable membership witnesses for tokenizer buckets, embedding pages, auxiliary
embedding pages, and decoder tokens. Sparse reads verify their selected objects;
dense projections and transformer weights use bounded complete sweeps. See the
[validation report](../proposals/model-identity-validation.md) for test coverage and costs.
