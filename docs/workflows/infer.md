# Infer Workflow

`infer` is the fast deterministic path. For a run of the `consensus` [procedure kind](../protocol/procedure-kinds.md) it makes each model's inference in turn and then states whether they agree. It uses `direct-infer`, reads a specification and a case (`specification.toml` and `case.toml` when no flag names them), loads the referenced `runtime/model-artifacts/<model-id>/manifest.json`, and does not write staged checkpoint artifacts.

```bash
cargo run --release -p iris-cli -- infer --specification specification.toml --case case.toml
# or
just infer specification.toml case.toml
```

Flow:

1. `iris-cli` dispatches `infer` to `procedure_direct::run_infer`, which runs `direct_infer::DirectInferenceExecutor` once per inference of the procedure kind.
2. `direct-infer` loads the specification and its model manifest, verifies the recorded weight, config, and tokenizer hashes once, then loads tokenizer metadata and mmap-backed DETWGT weights.
3. The executor prepares prompt pieces from the case, then runs embedding, prefill layers, decode, and output finalization in host memory.
4. The command prints an `InferenceResult` and timing summary.

Direct decoding stops immediately after selecting any EOS token listed in the model manifest. The specification's `max_new_tokens` value is a maximum, used only if EOS has not appeared. The reported count and token IDs include the terminal token and contain no post-EOS generation. `claim build` discovers its claimed output this way, so a claim's specification uses the same `max_new_tokens`.

Output:

- Input token count after chat templating and tokenization (measured, never supplied as a limit).
- Generated token count.
- Generated token ids.
- SHA-256 of the generated token id list.
- Stop reason.
- Generated text.
- The same record a claim's final stage publishes: the input text's SHA-256, the model id, the run's configuration and the output text's SHA-256.
- Direct-infer timing summary.

This path is deterministic inference only. It is intentionally free of the per-stage checkpoint tree used by claims and challenges.

Bundle verification adds a single streaming read at startup (included in model-loading time). Tensor access, prefill, decode, and staged execution do no additional model-identity checks. Direct-only imports remain usable by `infer`; claims and challenges require a full import with template provenance.
