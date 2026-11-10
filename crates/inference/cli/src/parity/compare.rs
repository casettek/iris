use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use raster_runtime::{read_raster_artifact_from_bytes, RasterValue, ReadLimits};
use serde_json::{json, Value};

#[derive(Debug)]
pub(super) struct Boundary {
    pub bytes: Vec<u8>,
    pub commitment: String,
    pub value: Value,
}

/// The shape of a ranged teacher-forced chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RangedShape {
    pub layers: u32,
    pub prompt_ranges: u32,
    pub claimed_ranges: u32,
}

impl RangedShape {
    /// Reads the shape back from an executed chain's stage inventory.
    pub(super) fn from_names(names: &[String]) -> Self {
        let count =
            |prefix: &str| names.iter().filter(|name| name.starts_with(prefix)).count() as u32;
        Self {
            layers: count("range_p0_l"),
            prompt_ranges: count("embed_p"),
            claimed_ranges: count("embed_c"),
        }
    }
}

/// The ranged teacher-forced chain: every prompt range tokenized and
/// embedded, every claimed range embedded, each layer across all ranges, then
/// each claimed range checked.
pub(super) fn teacher_forced_names(shape: RangedShape) -> Vec<String> {
    let mut names = vec![
        String::from("range_origin"),
        String::from("decode_init"),
        String::from("prompt_bind"),
    ];
    for j in 0..shape.prompt_ranges {
        names.extend([format!("prompt_check_p{j}"), format!("embed_p{j}")]);
    }
    names.extend((0..shape.claimed_ranges).map(|k| format!("embed_c{k}")));
    // Each layer runs across every range before the next layer.
    for l in 0..shape.layers {
        names.extend((0..shape.prompt_ranges).map(|j| format!("aux_p{j}_l{l}")));
        names.extend((0..shape.claimed_ranges).map(|k| format!("aux_c{k}_l{l}")));
        names.extend((0..shape.prompt_ranges).map(|j| format!("range_p{j}_l{l}")));
        names.extend((0..shape.claimed_ranges).map(|k| format!("range_c{k}_l{l}")));
    }
    names.extend((0..shape.claimed_ranges).map(|k| format!("check_c{k}")));
    names.push(String::from("output_finalize"));
    names
}

pub(super) fn check_inventory(expected: &[String], actual: &[String]) -> Result<()> {
    let mut seen = BTreeSet::new();
    for name in actual {
        ensure!(seen.insert(name), "duplicate boundary {name}");
    }
    let required: BTreeSet<_> = expected.iter().collect();
    if let Some(name) = required.difference(&seen).next() {
        bail!("missing boundary {name}");
    }
    if let Some(name) = seen.difference(&required).next() {
        bail!("unexpected boundary {name}");
    }
    Ok(())
}

pub(super) fn read_boundaries(
    dir: &Path,
    expected: &[String],
    direct: bool,
) -> Result<BTreeMap<String, Boundary>> {
    // Current Raster also writes public chain-commitment metadata in no-auth
    // mode. Only per-stage execution traces/commit.bin imply authentication.
    let names = if direct {
        inference_artifacts::read_json::<Vec<String>>(&dir.join("boundary-order.json"))?
    } else {
        inference_artifacts::read_stage_order(dir)?.context(
            "missing recorded stage inventory: expected chain-commitment or execution-times.json",
        )?
    };
    check_inventory(expected, &names)?;
    if !direct {
        ensure!(
            names == expected,
            "staged execution order differs from expected chain"
        );
    }
    let dirs = fs::read_dir(dir)?
        .map(|entry| {
            let entry = entry?;
            Ok(if entry.file_type()?.is_dir() {
                Some(entry.file_name().to_string_lossy().into_owned())
            } else {
                None
            })
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    check_inventory(expected, &dirs)?;
    expected
        .iter()
        .map(|stage| {
            let path = dir.join(stage);
            for file in ["trace.bin", "commit.bin"] {
                ensure!(
                    !path.join(file).exists(),
                    "{stage}: authenticated artifact {file} in host no-auth run"
                );
            }
            if !direct {
                for file in ["input.json", "input_manifest.json"] {
                    ensure!(path.join(file).is_file(), "{stage}: missing {file}");
                    let _: Value = inference_artifacts::read_json(&path.join(file))?;
                }
            }
            let boundary = read_boundary(&path)
                .with_context(|| format!("boundary {stage} ({})", path.display()))?;
            validate_errors(stage, &boundary.value)?;
            Ok((stage.clone(), boundary))
        })
        .collect()
}

pub(super) fn read_boundary(dir: &Path) -> Result<Boundary> {
    let bytes = fs::read(dir.join("output.bin"))?;
    let index = fs::read(dir.join("output.rindex"))?;
    let manifest: Value = inference_artifacts::read_json(&dir.join("output_manifest.json"))?;
    decode_boundary(
        bytes,
        &index,
        manifest["output"]["commitment"]
            .as_str()
            .context("missing output commitment")?,
    )
}

fn decode_boundary(bytes: Vec<u8>, index: &[u8], recorded: &str) -> Result<Boundary> {
    let artifact = read_raster_artifact_from_bytes(&bytes, index, &ReadLimits::unbounded())?;
    ensure!(
        artifact.roots_agree(),
        "output.rindex commitment differs from recomputed payload commitment"
    );
    ensure!(
        artifact.structural_root == recorded,
        "recorded commitment {recorded} differs from recomputed commitment {}",
        artifact.structural_root
    );
    Ok(Boundary {
        bytes,
        commitment: artifact.structural_root,
        value: value_json(&artifact.value)?,
    })
}

pub(crate) fn value_json(value: &RasterValue) -> Result<Value> {
    Ok(match value {
        RasterValue::Unit => Value::Null,
        RasterValue::Bool(v) => json!(v),
        RasterValue::Int { value, .. } => match u64::try_from(*value) {
            Ok(v) => json!(v),
            Err(_) => json!(i64::try_from(*value)?),
        },
        RasterValue::Str {
            value,
            truncated: false,
        } => json!(value),
        RasterValue::Bytes {
            data,
            truncated: false,
            ..
        } => json!({"bytes": hex::encode(data)}),
        RasterValue::Struct {
            fields,
            truncated: false,
            ..
        } => {
            let mut object = serde_json::Map::new();
            for (key, child) in fields {
                ensure!(
                    object.insert(key.clone(), value_json(child)?).is_none(),
                    "duplicate field {key}"
                );
            }
            Value::Object(object)
        }
        RasterValue::List {
            elements,
            truncated: false,
            ..
        } => Value::Array(elements.iter().map(value_json).collect::<Result<_>>()?),
        RasterValue::Map {
            entries,
            truncated: false,
            ..
        } => {
            let mut object = serde_json::Map::new();
            for (key, child) in entries {
                let key = value_json(key)?
                    .as_str()
                    .context("non-string map key")?
                    .to_string();
                ensure!(
                    object.insert(key.clone(), value_json(child)?).is_none(),
                    "duplicate map key {key}"
                );
            }
            Value::Object(object)
        }
        _ => bail!("unsupported or truncated inference boundary value"),
    })
}

pub(super) fn validate_errors(stage: &str, value: &Value) -> Result<()> {
    if stage != "prompt_prepare"
        && stage != "prompt_bind"
        && stage != "decode_init"
        && stage != "output_finalize"
        && stage != "range_origin"
        && !stage.starts_with("prompt_check_p")
        && !stage.starts_with("check_c")
        && !stage.starts_with("decode_select_")
    {
        let errors = value["errors"]
            .as_array()
            .with_context(|| format!("{stage}: missing errors collection"))?;
        ensure!(
            errors.is_empty(),
            "{stage}: nonempty stage errors: {errors:?}"
        );
    }
    Ok(())
}

pub(super) fn compare(stage: &str, left: &Boundary, right: &Boundary) -> Result<()> {
    let offset = left
        .bytes
        .iter()
        .zip(&right.bytes)
        .position(|(l, r)| l != r)
        .or_else(|| {
            (left.bytes.len() != right.bytes.len())
                .then_some(left.bytes.len().min(right.bytes.len()))
        });
    if let Some(offset) = offset {
        let detail = first_difference("output", &left.value, &right.value).unwrap_or_default();
        bail!(
            "{stage}: {detail}; first differing byte offset {offset}: left={:?}, right={:?}",
            left.bytes.get(offset),
            right.bytes.get(offset)
        );
    }
    ensure!(
        left.commitment == right.commitment,
        "{stage}: structural commitments differ"
    );
    ensure!(
        left.value == right.value,
        "{stage}: decoded values differ despite equal payloads"
    );
    Ok(())
}

fn first_difference(path: &str, left: &Value, right: &Value) -> Option<String> {
    if left == right {
        return None;
    }
    match (left, right) {
        (Value::Object(l), Value::Object(r)) => {
            for key in l.keys().chain(r.keys()) {
                if let Some(diff) = first_difference(
                    &format!("{path}.{key}"),
                    l.get(key).unwrap_or(&Value::Null),
                    r.get(key).unwrap_or(&Value::Null),
                ) {
                    return Some(diff);
                }
            }
        }
        (Value::Array(l), Value::Array(r)) => {
            for i in 0..l.len().max(r.len()) {
                if let Some(diff) = first_difference(
                    &format!("{path}[{i}]"),
                    l.get(i).unwrap_or(&Value::Null),
                    r.get(i).unwrap_or(&Value::Null),
                ) {
                    return Some(diff);
                }
            }
        }
        _ => {
            return Some(format!(
                "{path}: left={}, right={}",
                short(left),
                short(right)
            ))
        }
    }
    Some(format!("{path}: structure differs"))
}

fn short(value: &Value) -> String {
    value.to_string().chars().take(160).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boundary(value: &Value) -> Boundary {
        raster::init();
        let (data, index, root) = raster::encode_raster_value(value).unwrap();
        decode_boundary(data, &index, &root).unwrap()
    }

    #[test]
    fn changed_non_winning_logit_reports_its_exact_value() {
        let left = boundary(
            &json!({"logits": [{"token_id": 0u32, "value": 100i32}, {"token_id": 1u32, "value": -4i32}]}),
        );
        let right = boundary(
            &json!({"logits": [{"token_id": 0u32, "value": 100i32}, {"token_id": 1u32, "value": -3i32}]}),
        );
        let error = compare("prefill_finalize", &left, &right)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("logits[1].value: left=-4, right=-3"),
            "{error}"
        );
        assert!(error.contains("byte offset"));
    }

    #[test]
    fn byte_corruption_and_truncation_fail() {
        let left = boundary(&json!(10u32));
        let mut right = boundary(&json!(10u32));
        right.bytes[0] ^= 1;
        assert!(compare("stage", &left, &right)
            .unwrap_err()
            .to_string()
            .contains("offset 0"));
        right.bytes = left.bytes[..left.bytes.len() - 1].to_vec();
        assert!(compare("stage", &left, &right).is_err());
    }

    #[test]
    fn incorrect_commitment_and_malformed_payload_fail() {
        raster::init();
        let (data, index, root) = raster::encode_raster_value(&12u32).unwrap();
        assert!(decode_boundary(data.clone(), &index, "wrong")
            .unwrap_err()
            .to_string()
            .contains("recorded commitment"));
        assert!(decode_boundary(data[..2].to_vec(), &index, &root).is_err());
    }

    #[test]
    fn missing_duplicate_and_unexpected_boundaries_fail() {
        let names = direct_infer::diagnostics::boundary_names(4, 8);
        assert_eq!(names.len(), 91);
        check_inventory(&names, &names).unwrap();
        assert!(check_inventory(&names, &names[..90]).is_err());
        let mut duplicate = names.clone();
        duplicate.push(names[0].clone());
        assert!(check_inventory(&names, &duplicate)
            .unwrap_err()
            .to_string()
            .contains("duplicate"));
        let mut extra = names.clone();
        extra.push("unexpected".into());
        assert!(check_inventory(&names, &extra).is_err());
        assert!(!names.contains(&"decode_finalize_t7".into()));
    }

    #[test]
    fn ranged_inventory_round_trips_its_shape() {
        let shape = RangedShape {
            layers: 4,
            prompt_ranges: 3,
            claimed_ranges: 2,
        };
        let names = teacher_forced_names(shape);
        // origin + init + bind, 3 × (check + embed + 8 layer stages),
        // 2 × (embed + 8 layer stages + check), output.
        assert_eq!(names.len(), 3 + 3 * 10 + 2 * 10 + 1);
        assert_eq!(RangedShape::from_names(&names), shape);
        assert_eq!(names.last().map(String::as_str), Some("output_finalize"));
        assert!(!names
            .iter()
            .any(|name| name.starts_with("decode_") && name != "decode_init"));
    }

    #[test]
    fn stage_error_even_when_both_paths_agree_is_failure() {
        assert!(validate_errors("prefill_range_l0", &json!({"errors": ["page failed"]})).is_err());
        assert!(validate_errors("prefill_range_l0", &json!({})).is_err());
        validate_errors("prefill_range_l0", &json!({"errors": []})).unwrap();
    }

    #[test]
    fn no_auth_allows_public_chain_metadata_but_rejects_execution_traces() {
        use raster::core::chain::{ChainCommitment, ChainShape, StageCheckpoint};
        raster::init();
        let dir = std::env::temp_dir().join(format!("parity-no-auth-{}", std::process::id()));
        let stage = dir.join("prompt_prepare");
        fs::create_dir_all(&stage).unwrap();
        let (data, index, root) =
            raster::encode_raster_value(&json!({"token_ids": [1u32, 2u32]})).unwrap();
        fs::write(stage.join("output.bin"), data).unwrap();
        fs::write(stage.join("output.rindex"), index).unwrap();
        for file in ["input.json", "input_manifest.json"] {
            inference_artifacts::write_json(&stage.join(file), &json!({})).unwrap();
        }
        inference_artifacts::write_json(
            &stage.join("output_manifest.json"),
            &json!({"output": {"commitment": root}}),
        )
        .unwrap();
        let mut chain = ChainCommitment {
            stages: vec![StageCheckpoint {
                name: "prompt_prepare".into(),
                program_commitment: vec![],
                input_manifest_commitment: vec![],
                input_bindings: Default::default(),
                output_payload_commitment: vec![],
                output_structural_commitment: vec![],
            }],
            shape: ChainShape {
                spec_digest: vec![],
                repeats: vec![],
            },
        };
        let commitment = dir.join("chain-commitment");
        fs::write(&commitment, postcard::to_allocvec(&chain).unwrap()).unwrap();
        let expected = vec!["prompt_prepare".to_string()];
        assert!(!dir.join("execution-times.json").exists());
        read_boundaries(&dir, &expected, false).unwrap();
        chain.stages.push(chain.stages[0].clone());
        fs::write(&commitment, postcard::to_allocvec(&chain).unwrap()).unwrap();
        assert!(read_boundaries(&dir, &expected, false)
            .unwrap_err()
            .to_string()
            .contains("duplicate boundary"));
        chain.stages.pop();
        fs::write(&commitment, postcard::to_allocvec(&chain).unwrap()).unwrap();
        fs::write(stage.join("commit.bin"), b"execution commitment").unwrap();
        assert!(read_boundaries(&dir, &expected, false)
            .unwrap_err()
            .to_string()
            .contains("authenticated artifact"));
        fs::remove_dir_all(dir).unwrap();
    }
}
