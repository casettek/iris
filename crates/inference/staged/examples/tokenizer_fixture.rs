//! Generate and execute a tokenizer-only staged fixture using a pinned oracle case.
use anyhow::{ensure, Context, Result};
use prompt_prepare::input::{BpePiece, BpePieces};
use serde_json::{json, Value};
use staged_infer::{
    cache::CachedStageValue,
    chain_runner::{self, StagedExecutionBackend},
};
use std::{
    fs,
    path::{Path, PathBuf},
};

fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    ensure!(args.len() == 4 || (args.len() == 5 && args[4] == "--install-stage-inputs"), "usage: tokenizer_fixture OUTPUT_DIR TOKENIZER_JSON CORPUS_JSON CASE_NAME [--install-stage-inputs]");
    let out = PathBuf::from(&args[0]);
    fs::create_dir_all(&out)?;
    let out = out.canonicalize()?;
    let tokenizer: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let corpus: Value = serde_json::from_slice(&fs::read(&args[2])?)?;
    ensure!(
        inference_artifacts::file_sha256(Path::new(&args[1]))? == corpus["tokenizer_sha256"],
        "oracle tokenizer checksum differs"
    );
    ensure!(
        corpus["reference_version"] == "0.22.2",
        "unsupported oracle version"
    );
    let case = corpus["cases"]
        .as_array()
        .context("missing cases")?
        .iter()
        .find(|c| c["name"] == args[3])
        .context("missing case")?;
    let pieces =
        run_prep::split_prompt(case["text"].as_str().context("missing text")?, &tokenizer)?;
    let repeats = run_prep::tokenizer_repeat_count(pieces.len())?;
    let table = run_prep::prompt_tokenizer(&tokenizer)?;
    let initial = BpePieces {
        pieces: pieces
            .iter()
            .map(|p| BpePiece {
                text: p.text.clone(),
                segment: p.segment,
            })
            .collect::<Vec<_>>()
            .into(),
    };
    let tok = raster::write_raster_files(
        &table,
        &out.join("tokenizer.rastered"),
        &out.join("tokenizer.rindex"),
    )?;
    let pcs = raster::write_raster_files(
        &initial,
        &out.join("initial.rastered"),
        &out.join("initial.rindex"),
    )?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()?;
    let project_merge = root.join("raster-stages/prompt-merge");
    let project_final = root.join("raster-stages/prompt-prepare");
    let token_binding = format!("inputs.tokenizer = {{ external = {{ path = {:?}, index_path = {:?}, commitment = {:?} }} }}", out.join("tokenizer.rastered"), out.join("tokenizer.rindex"), tok);
    let manifest = format!(
        r#"[chain]
name = "tokenizer-fixture"
[[chain.stage]]
name = "prompt_merge_seed"
project = {project_merge:?}
{token_binding}
inputs.initial_pieces = {{ external = {{ path = {data:?}, index_path = {index:?}, commitment = {pcs:?} }} }}
[[chain.repeat]]
name = "tokenize"
index = "b"
count = {repeats}
[[chain.repeat.stage]]
name = "prompt_merge_b{{b}}"
project = {project_merge:?}
{token_binding}
inputs.initial_pieces = {{ from = "prompt_merge_b{{b-1}}", first = "prompt_merge_seed" }}
[chain.repeat.exports.pieces]
stage = "prompt_merge_b{{b}}"
entry = "prompt_merge_seed"
[[chain.stage]]
name = "prompt_prepare"
project = {project_final:?}
{token_binding}
inputs.merged_pieces = {{ from = "tokenize.pieces" }}
"#,
        data = out.join("initial.rastered"),
        index = out.join("initial.rindex")
    );
    let manifest_path = out.join("Raster.toml");
    fs::write(&manifest_path, manifest)?;
    std::env::set_current_dir(&out)?;
    let started = std::time::Instant::now();
    let run = chain_runner::run(
        &manifest_path,
        None,
        &std::env::current_exe()?,
        StagedExecutionBackend::InProcess,
    )?;
    let Some(CachedStageValue::PromptTokenization(tokens)) = run.final_output else {
        anyhow::bail!("missing final token IDs");
    };
    ensure!(
        serde_json::to_value(tokens.token_ids.as_slice())? == case["token_ids"],
        "reference token IDs differ"
    );
    if args.len() == 5 {
        let producer = if repeats == 0 {
            "prompt_merge_seed".into()
        } else {
            format!("prompt_merge_b{}", repeats - 1)
        };
        let producer = run.chain_dir.join(producer);
        fs::copy(producer.join("output.bin"), out.join("merged.rastered"))?;
        fs::copy(producer.join("output.rindex"), out.join("merged.rindex"))?;
        let output_manifest: Value =
            serde_json::from_slice(&fs::read(producer.join("output_manifest.json"))?)?;
        for (project, param, stem, commitment) in [
            (&project_merge, "initial_pieces", "initial", pcs.as_str()),
            (
                &project_final,
                "merged_pieces",
                "merged",
                output_manifest["output"]["commitment"]
                    .as_str()
                    .context("missing piece commitment")?,
            ),
        ] {
            let relative = out
                .strip_prefix(&root)
                .context("installed fixture must be inside the repository")?;
            let paths = Path::new("../..").join(relative);
            let inputs = json!({"tokenizer":{"path":paths.join("tokenizer.rastered"),"index_path":paths.join("tokenizer.rindex"),"load_preference":"read"}, (param):{"path":paths.join(format!("{stem}.rastered")),"index_path":paths.join(format!("{stem}.rindex")),"load_preference":"read"}});
            let manifests = json!({"tokenizer":{"type":"sha256","encoding":"raster","commitment":tok}, (param):{"type":"sha256","encoding":"raster","commitment":commitment}});
            fs::write(
                project.join("input.json"),
                serde_json::to_vec_pretty(&inputs)?,
            )?;
            fs::write(
                project.join("input_manifest.json"),
                serde_json::to_vec_pretty(&manifests)?,
            )?;
        }
    }
    let result = json!({"native_chain":run.chain_dir,"piece_count":pieces.len(),"repeat_count":repeats,"token_count":tokens.token_ids.len(),"native_seconds":started.elapsed().as_secs_f64(),"case":args[3]});
    fs::write(
        out.join("fixture.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!("{}", result);
    Ok(())
}
