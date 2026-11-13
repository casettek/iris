//! `iris model convert`: a GGUF model into a model bundle `model import` reads.

use std::fs;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use direct_infer::detwgt::convert::convert;
use direct_infer::detwgt::gguf::Gguf;

/// Bundle files taken from `metadata` alongside the converted weights.
const METADATA_FILES: &[&str] = &["config.json", "tokenizer.json", "chat_template.jinja"];

pub fn convert_bundle(
    gguf_path: &Path,
    metadata: &Path,
    out: &Path,
    dequantize: bool,
) -> Result<()> {
    let config: serde_json::Value = serde_json::from_slice(
        &fs::read(metadata.join("config.json"))
            .with_context(|| format!("{} has no config.json", metadata.display()))?,
    )?;
    let text = config.get("text_config").unwrap_or(&config);
    check_shape(&Gguf::open(gguf_path)?, text)?;

    fs::create_dir_all(out).with_context(|| format!("failed to create {}", out.display()))?;
    for file in METADATA_FILES {
        let source = metadata.join(file);
        if source.exists() {
            fs::copy(&source, out.join(file))
                .with_context(|| format!("failed to copy {}", source.display()))?;
        }
    }
    let converted = convert(gguf_path, &out.join("model.detwgt"), dequantize)?;
    println!(
        "wrote {}: {} block tensors{}, {} Q16.16 tensors",
        out.join("model.detwgt").display(),
        converted.block_tensors,
        if dequantize {
            " (dequantized to Q16.16)"
        } else {
            " (Q8_0)"
        },
        converted.q16_tensors
    );
    Ok(())
}

/// The GGUF and config.json must describe the same model.
fn check_shape(gguf: &Gguf, text: &serde_json::Value) -> Result<()> {
    let arch = match gguf.metadata.get("general.architecture") {
        Some(direct_infer::detwgt::gguf::Value::Str(arch)) => arch.clone(),
        _ => anyhow::bail!("GGUF has no general.architecture"),
    };
    for (gguf_key, config_key) in [
        ("block_count", "num_hidden_layers"),
        ("embedding_length", "hidden_size"),
        (
            "embedding_length_per_layer_input",
            "hidden_size_per_layer_input",
        ),
        ("attention.head_count", "num_attention_heads"),
        ("attention.head_count_kv", "num_key_value_heads"),
    ] {
        let key = format!("{arch}.{gguf_key}");
        let Ok(value) = gguf.int(&key) else { continue };
        let expected = text
            .get(config_key)
            .and_then(serde_json::Value::as_i64)
            .with_context(|| format!("config.json has no {config_key}"))?;
        ensure!(
            value == expected,
            "GGUF {key} = {value} but config.json {config_key} = {expected}"
        );
    }
    let vocab = text
        .get("vocab_size")
        .and_then(serde_json::Value::as_u64)
        .context("config.json has no vocab_size")?;
    let embedding = gguf
        .tensors
        .get("token_embd.weight")
        .context("GGUF has no token_embd.weight")?;
    ensure!(
        embedding.dims.get(1) == Some(&vocab),
        "GGUF token embedding has {:?} rows but config.json vocab_size = {vocab}",
        embedding.dims.get(1)
    );
    Ok(())
}
