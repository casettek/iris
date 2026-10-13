# Challenge Build Workflow

`challenge build` consumes a compact claimed checkpoint hash list plus frozen run context, recomputes the checkpointed run, stops at the first observed divergent hash, and replays that one divergent stage through Raster.

```bash
cargo run --release -p raster-inference-cli -- challenge build \
  --run inference.toml \
  --claim-context target/staged-infer/runs/.../claim_bundle.json \
  --checkpoints target/staged-infer/runs/.../checkpoints.txt
# or
just challenge \
  target/staged-infer/runs/.../claim_bundle.json \
  target/staged-infer/runs/.../checkpoints.txt
```

Flow:

1. Load the claimed `checkpoints.txt` and use `claim_bundle.json` only as today's local carrier for `prepared_run.json`.
2. Check that the model selected by `--run` (default: `inference.toml`) matches the frozen model manifest, verify the bundle and model template once, and verify the frozen run manifest. Run the checkpointed staged executor with that frozen run manifest; its prompt and token count are preserved even if the current run spec has changed.
3. After each recomputed checkpoint, hash the verifier's local checkpoint record and compare it to the claimed hash at the same index.
4. If no divergence exists, report the claimed and recomputed traces.
5. If a divergence exists, seed a replay directory with all prior recomputed stages.
6. Run `cargo raster chain run --stage <stage>` for the divergent stage.
7. Compare the Raster replay checkpoint with the native recomputed checkpoint. The stage name, input commitment, output structural commitment, and output SHA-256 must all match, yielding the same checkpoint hash.
8. Write the challenge artifacts only after that parity check passes. A mismatch fails challenge construction with a native/Raster parity error identifying the stage, differing field, and both values. The replay files remain available for debugging.

Challenge hash verification happens during recomputation. Sequential stages stop immediately at the first mismatch. Parallel aux waves still execute with the existing parallelism; if a mismatch is found in the middle of a completed aux batch, later batch outputs may exist on disk but are ignored for the challenge trace and replay seed.

Divergence reasons are checked in this order:

- Checkpoint hash mismatch.
- Missing claimed or recomputed checkpoint hash.

Challenge outputs live under the recomputed run's `challenge/<stage>/` directory:

- `divergence.json`
- `replay_package.json`
- `challenge_trace.json`
- `challenge_bundle.json`

For manual fraud tests, corrupt a copy of the claimed hash list and challenge that copy:

```bash
just corrupt target/staged-infer/runs/.../checkpoints.txt
just challenge \
  target/staged-infer/runs/.../claim_bundle.json \
  target/staged-infer/runs/.../checkpoints.corrupt.txt
```

`--claim` and `--trace` remain lower-level debug inputs that compare full checkpoint records. `--trace` prepares a run from `--run` because a trace alone has no frozen run metadata. It uses that run spec’s model, prompt, and token count. `challenge locate` and `fault prove` are reserved commands; `challenge build` is the implemented verifier workflow today.
