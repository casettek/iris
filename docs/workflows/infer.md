# Infer Workflow

`infer` is the fast deterministic path. It uses `direct-infer`, reads a run spec such as `inference.toml`, loads the referenced `runtime/model-artifacts/<model-id>/manifest.json`, and does not write staged checkpoint artifacts.

```bash
cargo run --release -p raster-inference-cli -- infer --run inference.toml
# or
just infer inference.toml
```

Flow:

1. `raster-inference-cli` dispatches `infer` to `direct_infer::DirectInferenceExecutor`.
2. `direct-infer` loads the run spec and its model manifest, verifies the recorded weight, config, and tokenizer hashes once, then loads tokenizer metadata and mmap-backed DETWGT weights.
3. The executor prepares prompt pieces from the run spec, then runs embedding, prefill layers, decode, and output finalization in host memory.
4. The command prints an `InferenceResult` and timing summary.

Output:

- Generated token count.
- Generated token ids.
- SHA-256 of the generated token id list.
- Stop reason.
- Generated text.
- Direct-infer timing summary.

This path is deterministic inference only. It is intentionally free of the per-stage checkpoint tree used by claims and challenges.

Bundle verification adds a single streaming read at startup (included in model-loading time). Tensor access, prefill, decode, and staged execution do no additional model-identity checks. Direct-only imports remain usable by `infer`; claims and challenges require a full import with template provenance.
