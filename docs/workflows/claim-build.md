# Claim Build Workflow

`claim build` reruns inference through the checkpointed staged path and writes the phase-two claim artifacts.

```bash
cargo run --release -p raster-inference-cli -- claim build --run inference.toml
# or
just claim inference.toml
```

Flow:

1. The CLI loads the run spec and its model manifest, then verifies the weight, config, tokenizer, and model-specific template hashes once before preparing the run. Missing template provenance is an error; claims never fall back to the root `Raster.toml`.
2. Prompt pieces and `prepared_run.json` are written under `target/raster-inference/runs/<run>/`.
3. A run-specific `Raster.toml` is generated in the same run directory with the requested prompt artifact and token count.
4. `staged-infer` reads the run manifest and expands the chain stages.
5. Each stage synthesizes `input.json` and `input_manifest.json`, runs the host implementation, and writes Raster-compatible output artifacts.
6. `inference-artifacts` scans the completed chain directory and writes the claim files.

Per-stage checkpoint files live under `target/staged-infer/runs/<run>/<stage>/`:

- `input.json`
- `input_manifest.json`
- `output.bin`
- `output.rindex`
- `output_manifest.json`

Claim files live beside the generated checkpoint tree:

- `checkpoint_trace.json`: ordered checkpoint records.
- `checkpoints.txt`: one SHA-256 per checkpoint record.
- `claim_bundle.json`: whole-claim input/output commitments plus the checkpoint trace and prepared-run metadata references.

`checkpoints.txt` is the compact public checkpoint reference used by the hash-based challenge flow. For manual challenge testing, copy and corrupt one hash:

```bash
cargo run --release -p raster-inference-cli -- claim corrupt \
  --checkpoints target/staged-infer/runs/.../checkpoints.txt \
  --random
# or
just corrupt target/staged-infer/runs/.../checkpoints.txt
```

The helper writes a corrupted hash list plus a local `*.corruption.json` manifest that records the selected checkpoint for debugging. Stage-based selection is also available when `checkpoint_trace.json` is beside the hash list:

```bash
cargo run --release -p raster-inference-cli -- claim corrupt \
  --checkpoints target/staged-infer/runs/.../checkpoints.txt \
  --stage decode_select_t42
```

The claim path must preserve per-stage artifacts even if the host execution becomes faster internally. Those files are the surface the challenge path verifies.
