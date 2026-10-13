//! Compare every tokenizer checkpoint, including its authenticated payload root.
use anyhow::{ensure, Context, Result};
use inference_artifacts::{build_checkpoint_trace, read_json};
use raster_runtime::{read_raster_artifact_from_bytes, ReadLimits};
use serde_json::Value;
use std::{fs, path::Path};

fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    ensure!(args.len() == 2 || (args.len() == 3 && args[0] == "--chains"), "usage: tokenizer_compare FIXTURE_DIR RASTER_RUN_DIR | --chains NATIVE_RUN_DIR RASTER_RUN_DIR");
    let metadata: Value = if args.len() == 2 {
        read_json(&Path::new(&args[0]).join("fixture.json"))?
    } else {
        serde_json::json!({"native_chain":args[1],"case":"full inference"})
    };
    let native = Path::new(
        metadata["native_chain"]
            .as_str()
            .context("missing native run")?,
    );
    let raster = Path::new(args.last().unwrap());
    let manifest = native.join("Raster.toml");
    let left = build_checkpoint_trace(native, &manifest)?;
    let right = build_checkpoint_trace(raster, &manifest)?;
    let expected = if args.len() == 2 {
        let repeats = metadata["repeat_count"].as_u64().context("missing count")?;
        let mut names = vec!["prompt_merge_seed".to_string()];
        names.extend((0..repeats).map(|b| format!("prompt_merge_b{b}")));
        names.push("prompt_prepare".into());
        names
    } else {
        let timing: Value = read_json(&native.join("execution-times.json"))?;
        timing["stages"]
            .as_array()
            .context("missing native stage order")?
            .iter()
            .map(|s| {
                s["name"]
                    .as_str()
                    .map(str::to_string)
                    .context("invalid stage name")
            })
            .collect::<Result<Vec<_>>>()?
    };
    ensure!(
        left.checkpoints
            .iter()
            .map(|c| &c.stage)
            .eq(expected.iter()),
        "native checkpoint inventory differs"
    );
    ensure!(
        right
            .checkpoints
            .iter()
            .map(|c| &c.stage)
            .eq(expected.iter()),
        "Raster checkpoint inventory differs"
    );
    for (a, b) in left.checkpoints.iter().zip(&right.checkpoints) {
        ensure!(
            a == b,
            "first divergent checkpoint {}: native={a:?}, Raster={b:?}",
            a.stage
        );
        let mut payloads = Vec::new();
        for root in [native, raster] {
            let dir = root.join(&a.stage);
            let bytes = fs::read(dir.join("output.bin"))?;
            let index = fs::read(dir.join("output.rindex"))?;
            let artifact =
                read_raster_artifact_from_bytes(&bytes, &index, &ReadLimits::unbounded())?;
            ensure!(
                artifact.roots_agree() && artifact.structural_root == a.output_commitment,
                "{}: invalid payload or index commitment",
                a.stage
            );
            payloads.push(bytes);
        }
        ensure!(
            payloads[0] == payloads[1],
            "{}: output bytes differ",
            a.stage
        );
    }
    println!(
        "PASS: {} exact checkpoints for {}",
        expected.len(),
        metadata["case"]
    );
    Ok(())
}
