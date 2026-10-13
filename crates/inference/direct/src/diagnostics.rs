//! Opt-in canonical boundaries; ordinary inference does not serialize them.
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use anyhow::{bail, ensure, Context, Result};
use inference_artifacts::InferenceRunReport;
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct DirectBoundaryRecord {
    pub stage: String,
    pub output_bytes: Vec<u8>,
    pub output_index: Vec<u8>,
    pub output_commitment: String,
}

#[derive(Debug)]
pub struct DirectDiagnosticRunReport {
    pub report: InferenceRunReport,
    pub boundaries: Vec<DirectBoundaryRecord>,
}

/// Semantic inventory, independent of the order in which PLE work is scheduled.
pub fn boundary_names(layers: u32, tokens: u32) -> Vec<String> {
    let mut names = vec!["prompt_prepare".into(), "input_embedding".into()];
    for layer in 0..layers {
        names.push(format!("prefill_prepare_aux_l{layer}"));
        names.push(format!("prefill_range_l{layer}"));
    }
    names.extend(["prefill_finalize".into(), "decode_init".into()]);
    for token in 0..tokens {
        names.push(format!("decode_select_t{token}"));
        if token + 1 < tokens {
            names.extend(decode_pass_names(layers, token));
        }
    }
    names.push("output_finalize".into());
    names
}

fn decode_pass_names(layers: u32, token: u32) -> Vec<String> {
    let mut names = vec![format!("decode_embed_t{token}")];
    for layer in 0..layers {
        names.push(format!("decode_aux_t{token}_l{layer}"));
        names.push(format!("decode_range_t{token}_l{layer}"));
    }
    names.push(format!("decode_finalize_t{token}"));
    names
}

pub(crate) struct DirectTrace {
    print: bool,
    collect: bool,
    expected: Option<BTreeMap<String, String>>,
    required: BTreeSet<String>,
    seen: BTreeSet<String>,
    pub records: Vec<DirectBoundaryRecord>,
}

impl DirectTrace {
    pub fn new(collect: bool) -> Result<Self> {
        let expected = std::env::var_os("DIRECT_INFER_COMPARE_TRACE")
            .map(|path| {
                let path = PathBuf::from(path);
                let trace = inference_artifacts::read_checkpoint_trace(&path)
                    .with_context(|| format!("failed to read {}", path.display()))?;
                let mut expected = BTreeMap::new();
                for checkpoint in trace.checkpoints {
                    ensure!(
                        expected
                            .insert(checkpoint.stage.clone(), checkpoint.output_commitment)
                            .is_none(),
                        "duplicate reference boundary {}",
                        checkpoint.stage
                    );
                }
                Ok::<_, anyhow::Error>(expected)
            })
            .transpose()?;
        let print = std::env::var("DIRECT_INFER_TRACE_COMMITMENTS")
            .map(|value| !value.eq_ignore_ascii_case("off") && value != "0")
            .unwrap_or(false)
            || expected.is_some();
        if print || collect {
            raster::init();
        }
        Ok(Self {
            print,
            collect,
            expected,
            required: BTreeSet::new(),
            seen: BTreeSet::new(),
            records: Vec::new(),
        })
    }

    fn enabled(&self) -> bool {
        self.print || self.collect
    }

    pub fn configure(&mut self, layers: u32, tokens: u32, tokenizer_repeats: u32) -> Result<()> {
        if !self.enabled() {
            return Ok(());
        }
        self.required = boundary_names(layers, tokens).into_iter().collect();
        if let Some(expected) = &self.expected {
            // A checkpointed chain includes the unused pass following its last
            // selection. It must be present in the reference, but direct mode
            // must not execute it just to satisfy diagnostics.
            let mut reference_names = self.required.clone();
            reference_names.insert("prompt_merge_seed".into());
            reference_names.extend((0..tokenizer_repeats).map(|b| format!("prompt_merge_b{b}")));
            if tokens > 0 {
                reference_names.extend(decode_pass_names(layers, tokens - 1));
            }
            check_names(
                &reference_names,
                &expected.keys().cloned().collect(),
                "reference",
            )?;
        }
        Ok(())
    }

    pub fn record<T: Serialize>(&mut self, stage: &str, value: &T) -> Result<()> {
        if !self.enabled() {
            return Ok(());
        }
        ensure!(
            self.required.contains(stage),
            "unexpected direct boundary {stage}"
        );
        ensure!(
            self.seen.insert(stage.to_string()),
            "duplicate direct boundary {stage}"
        );
        let (output_bytes, output_index, output_commitment) = raster::encode_raster_value(value)?;
        if let Some(expected) = &self.expected {
            let reference = expected.get(stage).context("missing reference boundary")?;
            ensure!(reference == &output_commitment,
                "direct-infer boundary {stage}: MISMATCH expected={reference} actual={output_commitment}");
        }
        if self.print {
            eprintln!("direct-infer trace {stage}: structural={output_commitment}");
        }
        if self.collect {
            self.records.push(DirectBoundaryRecord {
                stage: stage.into(),
                output_bytes,
                output_index,
                output_commitment,
            });
        }
        Ok(())
    }

    pub fn finish(&self) -> Result<()> {
        if self.enabled() {
            check_names(&self.required, &self.seen, "direct")?;
        }
        Ok(())
    }
}

fn check_names(required: &BTreeSet<String>, actual: &BTreeSet<String>, label: &str) -> Result<()> {
    if let Some(name) = required.difference(actual).next() {
        bail!("missing {label} boundary {name}");
    }
    if let Some(name) = actual.difference(required).next() {
        bail!("unexpected {label} boundary {name}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace() -> DirectTrace {
        DirectTrace {
            print: false,
            collect: true,
            expected: None,
            required: BTreeSet::new(),
            seen: BTreeSet::new(),
            records: Vec::new(),
        }
    }

    #[test]
    fn inventory_and_trailing_pass_are_explicit() {
        let mut trace = trace();
        let mut names = boundary_names(4, 8);
        assert_eq!(names.len(), 91);
        names.extend(decode_pass_names(4, 7));
        assert_eq!(names.len(), 101);
        names.push("prompt_merge_seed".into());
        trace.expected = Some(
            names
                .into_iter()
                .map(|name| (name, String::new()))
                .collect(),
        );
        trace.configure(4, 8, 0).unwrap();
        trace
            .expected
            .as_mut()
            .unwrap()
            .remove("decode_range_t6_l3");
        assert!(trace
            .configure(4, 8, 0)
            .unwrap_err()
            .to_string()
            .contains("missing reference"));
    }

    #[test]
    fn missing_duplicate_unexpected_and_mismatched_boundaries_fail() {
        let mut trace = trace();
        trace.configure(4, 1, 0).unwrap();
        assert!(trace.finish().is_err());
        assert!(trace.record("unknown", &1u32).is_err());
        trace.record("prompt_prepare", &1u32).unwrap();
        assert!(trace.record("prompt_prepare", &1u32).is_err());
        trace.expected = Some(BTreeMap::from([("input_embedding".into(), "wrong".into())]));
        assert!(trace
            .record("input_embedding", &1u32)
            .unwrap_err()
            .to_string()
            .contains("MISMATCH"));
    }

    #[test]
    fn empty_reference_is_not_a_disabled_comparison() {
        let mut trace = trace();
        trace.expected = Some(BTreeMap::new());
        assert!(trace.configure(4, 1, 0).is_err());
    }
}
