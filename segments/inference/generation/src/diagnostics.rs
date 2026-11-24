//! Opt-in canonical boundaries; ordinary inference does not serialize them.
use std::collections::BTreeSet;

use anyhow::{bail, ensure, Result};
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
/// The boundaries a run records, in order. `auxiliary` says whether the model's
/// layers each have a per-layer input prepared before them.
pub fn boundary_names(layers: u32, tokens: u32, auxiliary: bool) -> Vec<String> {
    let mut names = vec!["prompt_prepare".into(), "input_embedding".into()];
    for layer in 0..layers {
        if auxiliary {
            names.push(format!("prefill_prepare_aux_l{layer}"));
        }
        names.push(format!("prefill_range_l{layer}"));
    }
    names.extend(["prefill_finalize".into(), "decode_init".into()]);
    for token in 0..tokens {
        names.push(format!("decode_select_t{token}"));
        if token + 1 < tokens {
            names.extend(decode_pass_names(layers, token, auxiliary));
        }
    }
    names.push("output_finalize".into());
    names
}

fn decode_pass_names(layers: u32, token: u32, auxiliary: bool) -> Vec<String> {
    let mut names = vec![format!("decode_embed_t{token}")];
    for layer in 0..layers {
        if auxiliary {
            names.push(format!("decode_aux_t{token}_l{layer}"));
        }
        names.push(format!("decode_range_t{token}_l{layer}"));
    }
    names.push(format!("decode_finalize_t{token}"));
    names
}

pub(crate) struct DirectTrace {
    print: bool,
    collect: bool,
    required: BTreeSet<String>,
    seen: BTreeSet<String>,
    auxiliary: bool,
    pub records: Vec<DirectBoundaryRecord>,
}

impl DirectTrace {
    pub fn new(collect: bool) -> Result<Self> {
        let print = std::env::var("DIRECT_INFER_TRACE_COMMITMENTS")
            .map(|value| !value.eq_ignore_ascii_case("off") && value != "0")
            .unwrap_or(false);
        if print || collect {
            raster::init();
        }
        Ok(Self {
            print,
            collect,
            required: BTreeSet::new(),
            seen: BTreeSet::new(),
            auxiliary: true,
            records: Vec::new(),
        })
    }

    fn enabled(&self) -> bool {
        self.print || self.collect
    }

    pub fn configure(&mut self, layers: u32, tokens: u32, auxiliary: bool) {
        self.auxiliary = auxiliary;
        if self.enabled() {
            self.required = boundary_names(layers, tokens, auxiliary)
                .into_iter()
                .collect();
        }
    }

    pub fn stop_at_eos(&mut self, layers: u32, generated_tokens: u32) {
        if self.enabled() {
            self.required = boundary_names(layers, generated_tokens, self.auxiliary)
                .into_iter()
                .collect();
        }
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

    #[test]
    fn inventory_has_no_pass_after_the_last_selection() {
        let names = boundary_names(4, 8, true);
        assert_eq!(names.len(), 91);
        // A model with no per-layer inputs records one boundary a layer.
        let plain = boundary_names(4, 8, false);
        assert_eq!(plain.len(), 91 - 4 - 7 * 4);
        assert!(!plain.iter().any(|name| name.contains("aux")));
        assert!(names.contains(&String::from("decode_range_t6_l3")));
        assert!(!names.contains(&String::from("decode_embed_t7")));
    }

    #[test]
    fn missing_duplicate_and_unexpected_boundaries_fail() {
        let mut trace = DirectTrace {
            print: false,
            collect: true,
            required: BTreeSet::new(),
            seen: BTreeSet::new(),
            auxiliary: true,
            records: Vec::new(),
        };
        trace.configure(4, 1, true);
        assert!(trace.finish().is_err());
        assert!(trace.record("unknown", &1u32).is_err());
        trace.record("prompt_prepare", &1u32).unwrap();
        assert!(trace.record("prompt_prepare", &1u32).is_err());
    }
}
