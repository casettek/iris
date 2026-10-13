# Model Import

`model import` is the shared setup step for every workflow. It reads a Gemma bundle and writes the committed artifacts the repo uses afterward.

```bash
cargo run --release -p raster-inference-cli -- model import \
  --model ../raster-inference/assets/tiny-gemma-dev
```

Inputs:

- `model.detwgt`: deterministic Q16.16 model weights.
- `config.json`: model shape, attention/window settings, softcap, and normalization parameters.
- `tokenizer.json`: vocabulary, merges, added tokens, and EOS metadata.

Outputs:

- `runtime/model-artifacts/<model-id>/Raster.toml`, a prompt-free chain template saved on every full import. Its path and SHA-256 are recorded in the model manifest.
- Reusable Raster externals under `runtime/model-artifacts/<model-id>/raster/*`, written as `.rastered` and `.rindex` pairs with manifest commitments.
- `runtime/model-artifacts/<model-id>/manifest.json`, the prompt-free model manifest consumed by run specs.

By default, `<model-id>` is the sanitized model bundle directory name. Pass `--model-id <id>` to choose it explicitly, or `--artifact-root <dir>` to place model-scoped artifact directories somewhere other than `runtime/model-artifacts`.

Pass `--manifest <path>` to also save a template copy (for example, the root `Raster.toml` for manual Raster commands). Claims always use the model-specific template, so importing another model or overwriting that copy cannot change an existing model’s selection. Saved templates resolve stage and external paths against the importing workspace.

`infer`, `claim build`, and `challenge build` select their model through `--run inference.toml`. They verify the bundle’s recorded weight, config, and tokenizer hashes once before execution. Claims and challenges also require the recorded template hash to match. Missing provenance or changed files require a full import; there is no fallback to the root template.

`initial_pieces` is deliberately not imported with the model. It is generated
from each prompt during run preparation and written under that run's `prompt/`
directory.

Tokenization now uses `prompt_merge_seed`, the fixed repeat named `tokenize`,
and final `prompt_prepare`. Run preparation derives the tokenizer repeat count
from its frozen `{text, segment}` pieces independently of the requested decode
count. Re-import older model templates before creating ranked-BPE claims;
single-stage tokenizer templates and legacy frozen prepared runs are rejected.

Useful partial modes:

- `--only-direct`: refresh only `runtime/model-artifacts/<model-id>/manifest.json` for direct inference. This clears template provenance, even if `--manifest` is supplied: existing staged artifacts cannot be bound to a refreshed bundle without a full import. Run a full import before building claims or challenges.
- `--only-tokenizer`: refresh tokenizer/decoder externals.
- `--only-layers`, `--only-embedding`, `--only-ple`: refresh targeted weight externals.

Partial external refreshes do not refresh model identity. Run a full import before using changed bundle files in a claim or challenge.

`model import` no longer accepts `--prompt`, `--raw-prompt`, or `--tokens`. Runtime prompt text and token count live in `inference.toml`.
