//! Direct inference, `iris infer`, and how a result is printed.

use crate::*;

#[derive(Debug, Args)]
pub(crate) struct InferArgs {
    /// The specification of the procedure to run.
    #[arg(long, default_value = procedure_definition::SPECIFICATION_TOML)]
    specification: PathBuf,

    /// The case to run it on: the prompt, or the documents of the
    /// specification's source.
    #[arg(long, default_value = procedure_definition::CASE_TOML)]
    case: PathBuf,
}

/// What a consensus states of its two inferences. Printed before either
/// inference's result, whose last lines are its generated text.
pub(crate) fn print_consensus(
    first: &staged_infer::InferenceResult,
    consensus: &procedure_definition::ConsensusOutcome,
) {
    println!("consensus: {}", consensus.description(first));
    println!("consensus code: {}", consensus.code);
    println!("same input: {}", consensus.same_input);
}

/// One inference's result. `inference` is which of the run's models made it,
/// whose tokenizer the ids decode by.
pub(crate) fn print_infer_result(
    result: &staged_infer::InferenceResult,
    specification_path: Option<&Path>,
    inference: usize,
) {
    println!("generated token count: {}", result.generated_token_count);
    println!("generated token ids: {:?}", result.generated_token_ids);
    println!(
        "generated token ids sha256: {}",
        result.generated_token_ids_sha256
    );
    if let Some(specification_path) = specification_path {
        match render_token_trace_for_run(result, specification_path, inference) {
            Ok(trace) => print!("{trace}"),
            Err(error) => eprintln!("generated token trace unavailable: {error:#}"),
        }
    }
    println!("stop reason: {}", result.stop_reason);
    println!("output text sha256: {}", result.output_text_sha256);
    println!("input text sha256: {}", result.input_text_sha256);
    println!("source sha256: {}", result.source_sha256);
    println!("model id: {}", result.model_id);
    println!(
        "faults: prompt {}, check {}, output {}, model {}; claim mismatches: {}",
        result.prompt_fault,
        result.check_fault,
        result.output_fault,
        result.model_fault,
        result.claim_mismatches
    );
    // The claim record a contract reads, beside the fields it is stated from.
    println!("statement sha256: {}", result.statement_sha256);
    println!("valid: {}", result.valid);
    // What the result stage made of the output.
    println!("result code: {}", result.result_code);
    println!("result forms sha256: {}", result.result_forms_sha256);
    println!("generated text:");
    println!("{}", result.generated_text);
}

pub(crate) fn render_token_trace_for_run(
    result: &staged_infer::InferenceResult,
    specification_path: &Path,
    inference: usize,
) -> Result<String> {
    let base_dir = std::env::current_dir().context("failed to read current directory")?;
    let specification_path = absolute(&base_dir, specification_path);
    let specification = procedure_definition::read_specification(&specification_path)?;
    let specification_dir = specification_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("specification has no parent directory"))?;
    // The ids are of this one of the run's models, and decode by its tokenizer.
    let model_manifest_path = absolute(
        specification_dir,
        specification.model_manifest_of(inference)?,
    );
    let model_manifest: model_manifest::ModelManifest =
        procedure_definition::read_json(&model_manifest_path)?;
    let model_manifest_dir = model_manifest_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("model manifest has no parent directory"))?;
    let tokenizer_path = absolute(model_manifest_dir, &model_manifest.bundle.tokenizer_path);
    let tokenizer: serde_json::Value = procedure_definition::read_json(&tokenizer_path)?;
    let decoder = TokenDecoder::from_tokenizer(&tokenizer, &model_manifest.eos_token_ids)?;
    Ok(render_token_trace(result, &decoder))
}

pub(crate) fn render_token_trace(
    result: &staged_infer::InferenceResult,
    decoder: &TokenDecoder,
) -> String {
    let mut out = String::from("generated tokens:\n");
    let mut stopped = false;
    for (idx, token_id) in result.generated_token_ids.iter().copied().enumerate() {
        let display = decoder.display(token_id, stopped);
        if display.kind == TokenKind::Terminal {
            stopped = true;
        }
        out.push_str(&format!(
            "  {idx}: id={token_id} token={} kind={} rendered={}\n",
            display.token,
            display.kind.as_str(),
            display.rendered
        ));
    }
    out
}

#[derive(Debug)]
pub(crate) struct TokenDecoder {
    tokens: BTreeMap<u32, String>,
    special_ids: BTreeSet<u32>,
    terminal_ids: BTreeSet<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TokenKind {
    Text,
    Special,
    Terminal,
    Byte,
    Unknown,
    AfterStop,
}

pub(crate) struct TokenDisplay {
    token: String,
    kind: TokenKind,
    rendered: String,
}

impl TokenKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Special => "special",
            Self::Terminal => "terminal",
            Self::Byte => "byte",
            Self::Unknown => "unknown",
            Self::AfterStop => "after_stop",
        }
    }
}

impl TokenDecoder {
    pub(crate) fn from_tokenizer(tokenizer: &serde_json::Value, eos_ids: &[u32]) -> Result<Self> {
        let mut tokens = tokenizer
            .get("model")
            .and_then(|model| model.get("vocab"))
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| anyhow::anyhow!("tokenizer.json has no vocab"))?
            .iter()
            .map(|(token, id)| {
                Ok((
                    id.as_u64()
                        .ok_or_else(|| anyhow::anyhow!("token '{token}' has non-integer id"))?
                        as u32,
                    token.clone(),
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let mut special_ids = BTreeSet::new();
        if let Some(added_tokens) = tokenizer
            .get("added_tokens")
            .and_then(serde_json::Value::as_array)
        {
            for entry in added_tokens {
                let Some(id) = entry.get("id").and_then(serde_json::Value::as_u64) else {
                    continue;
                };
                if let Some(content) = entry.get("content").and_then(serde_json::Value::as_str) {
                    tokens
                        .entry(id as u32)
                        .or_insert_with(|| content.to_string());
                }
                if entry
                    .get("special")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
                {
                    special_ids.insert(id as u32);
                }
            }
        }
        Ok(Self {
            tokens,
            special_ids,
            terminal_ids: eos_ids.iter().copied().collect(),
        })
    }

    pub(crate) fn display(&self, token_id: u32, stopped: bool) -> TokenDisplay {
        let Some(token) = self.tokens.get(&token_id) else {
            return TokenDisplay {
                token: String::from("<unknown>"),
                kind: TokenKind::Unknown,
                rendered: String::from("<unknown>"),
            };
        };
        if stopped {
            return TokenDisplay {
                token: quote_token(token),
                kind: TokenKind::AfterStop,
                rendered: String::from("<skipped>"),
            };
        }
        if self.terminal_ids.contains(&token_id) {
            return TokenDisplay {
                token: quote_token(token),
                kind: TokenKind::Terminal,
                rendered: String::from("<stops text>"),
            };
        }
        if self.special_ids.contains(&token_id) {
            return TokenDisplay {
                token: quote_token(token),
                kind: TokenKind::Special,
                rendered: String::from("<skipped>"),
            };
        }
        if let Some(byte) = byte_fallback(token) {
            return TokenDisplay {
                token: quote_token(token),
                kind: TokenKind::Byte,
                rendered: format!("<byte 0x{byte:02X}>"),
            };
        }
        TokenDisplay {
            token: quote_token(token),
            kind: TokenKind::Text,
            rendered: quote_token(&token.replace('\u{2581}', " ")),
        }
    }
}

pub(crate) fn byte_fallback(token: &str) -> Option<u8> {
    let bytes = token.as_bytes();
    if bytes.len() != 6 || &bytes[..3] != b"<0x" || bytes[5] != b'>' {
        return None;
    }
    Some(hex_nibble(bytes[3])? * 16 + hex_nibble(bytes[4])?)
}

pub(crate) fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

pub(crate) fn quote_token(token: &str) -> String {
    format!("{token:?}")
}

pub(crate) fn print_infer_timing_summary(timings: &staged_infer::InferenceTimings) {
    eprint!("{}", render_infer_timing_summary(timings));
}

pub(crate) fn render_infer_timing_summary(timings: &staged_infer::InferenceTimings) -> String {
    if is_direct_infer_timing(timings) {
        return render_direct_infer_timing_summary(timings);
    }

    let mut out = String::new();
    out.push_str("infer timing:\n");
    out.push_str(&format!(
        "  total: {}\n",
        format_duration(timings.total_duration)
    ));
    if !timings.aux_waves.is_empty() {
        for wave in &timings.aux_waves {
            out.push_str(&format!(
                "  aux wave {}: {} stages, parallelism {}, wall {}, stage-sum {}",
                wave.name,
                wave.stage_count,
                wave.parallelism,
                format_duration(wave.wall_duration),
                format_duration(wave.stage_duration_sum)
            ));
            out.push('\n');
        }
    }

    let synthesis = timings
        .stages
        .iter()
        .map(|stage| stage.input_synthesis_duration)
        .sum();
    let input_load = timings
        .stages
        .iter()
        .map(|stage| stage.input_load_duration)
        .sum();
    let kernel = timings
        .stages
        .iter()
        .map(|stage| stage.kernel_duration)
        .sum();
    let encode = timings
        .stages
        .iter()
        .map(|stage| stage.encode_duration)
        .sum();
    out.push_str(&format!("  stages: {}\n", timings.stages.len()));
    out.push_str(&format!(
        "  input synthesis: {}\n",
        format_duration(synthesis)
    ));
    out.push_str(&format!("  input load: {}\n", format_duration(input_load)));
    out.push_str(&format!("  kernels: {}\n", format_duration(kernel)));
    out.push_str(&format!("  encode: {}\n", format_duration(encode)));
    out
}

pub(crate) fn is_direct_infer_timing(timings: &staged_infer::InferenceTimings) -> bool {
    !timings.stages.is_empty()
        && timings
            .stages
            .iter()
            .all(|stage| stage.routine == "direct-infer")
}

pub(crate) fn render_direct_infer_timing_summary(
    timings: &staged_infer::InferenceTimings,
) -> String {
    let named = |name: &str| {
        timings
            .stages
            .iter()
            .find(|stage| stage.stage == name)
            .map(|stage| stage.total_duration)
            .unwrap_or_default()
    };
    let accounted = named("load_direct_model")
        + named("prompt_prepare")
        + named("input_embedding")
        + named("prefill")
        + named("decode")
        + named("output_finalize");
    let kernel_records = timings
        .stages
        .iter()
        .filter(|stage| !matches!(stage.stage.as_str(), "prefill" | "decode"))
        .count();

    let mut out = String::new();
    out.push_str("direct-infer timing:\n");
    out.push_str(&format!(
        "  total: {}\n",
        format_duration(timings.total_duration)
    ));
    out.push_str(&format!("  phase records: {kernel_records}\n"));
    out.push_str(&format!(
        "  model load: {}\n",
        format_duration(named("load_direct_model"))
    ));
    out.push_str(&format!(
        "  prompt prepare: {}\n",
        format_duration(named("prompt_prepare"))
    ));
    out.push_str(&format!(
        "  input embedding: {}\n",
        format_duration(named("input_embedding"))
    ));
    out.push_str(&format!(
        "  prefill: {}\n",
        format_duration(named("prefill"))
    ));
    out.push_str(&format!("  decode: {}\n", format_duration(named("decode"))));
    out.push_str(&format!(
        "  output finalize: {}\n",
        format_duration(named("output_finalize"))
    ));
    out.push_str(&format!(
        "  accounted phase time: {}\n",
        format_duration(accounted)
    ));
    out
}

pub(crate) fn format_duration(duration: std::time::Duration) -> String {
    if duration.as_secs() >= 1 {
        format!("{:.2}s", duration.as_secs_f64())
    } else if duration.as_millis() >= 1 {
        format!("{}ms", duration.as_millis())
    } else if duration.as_micros() >= 1 {
        format!("{}us", duration.as_micros())
    } else {
        format!("{}ns", duration.as_nanos())
    }
}

/// What a direct run found: the procedure kind, the source, and each
/// inference's result and timings.
pub(crate) fn print_direct_run(
    run: &procedure_direct::InferRun,
    specification_path: &Path,
    case_path: &Path,
) -> Result<()> {
    println!("procedure kind: {}", run.procedure_kind);
    let source = source::run_source_of(specification_path, case_path)?;
    match (run.reports.as_slice(), run.consensus.as_ref()) {
        ([report], None) => {
            if let Some(count) = report.input_token_count {
                println!("input token count: {count}");
            }
            // Before the result: its last lines are the generated text.
            if let Some(source) = &source {
                source::print_source_lines(source);
            }
            print_infer_result(&report.result, Some(specification_path), 0);
            print_infer_timing_summary(&report.timings);
        }
        ([first, second], Some(consensus)) => {
            if let Some(source) = &source {
                source::print_source_lines(source);
            }
            print_consensus(&first.result, consensus);
            for (inference, report) in [first, second].into_iter().enumerate() {
                println!("inference: {inference}");
                if let Some(count) = report.input_token_count {
                    println!("input token count: {count}");
                }
                print_infer_result(&report.result, Some(specification_path), inference);
                print_infer_timing_summary(&report.timings);
            }
        }
        _ => anyhow::bail!(
            "the `{}` procedure kind gave no result to print",
            run.procedure_kind
        ),
    }
    Ok(())
}

pub(crate) fn run(args: InferArgs) -> Result<ExitCode> {
    warn_if_debug_build("infer");
    let run = run_infer(args.specification.clone(), args.case.clone())?;
    print_direct_run(&run, &args.specification, &args.case)?;
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_infer_timing_summary_uses_phase_language() {
        let timings = staged_infer::InferenceTimings {
            total_duration: std::time::Duration::from_millis(10),
            stages: vec![
                direct_timing("load_direct_model", 1),
                direct_timing("prompt_prepare", 1),
                direct_timing("input_embedding", 1),
                direct_timing("prefill_layer", 3),
                direct_timing("prefill", 3),
                direct_timing("decode_layer", 2),
                direct_timing("decode", 2),
                direct_timing("output_finalize", 1),
            ],
            aux_waves: Vec::new(),
        };

        let summary = render_infer_timing_summary(&timings);

        assert!(summary.contains("direct-infer timing:"));
        assert!(summary.contains("model load: 1ms"));
        assert!(summary.contains("phase records: 6"));
        assert!(summary.contains("accounted phase time: 9ms"));
        assert!(!summary.contains("validation"));
        assert!(!summary.contains("stages:"));
        assert!(!summary.contains("input synthesis:"));
        assert!(!summary.contains("encode:"));
    }

    #[test]
    fn token_trace_labels_special_terminal_byte_unknown_and_text_tokens() {
        let decoder = TokenDecoder {
            tokens: std::collections::BTreeMap::from([
                (1, String::from("<eos>")),
                (2, String::from("<bos>")),
                (3, String::from("late")),
                (10, String::from("<0xC3>")),
                (150917, String::from("Ciao")),
            ]),
            special_ids: std::collections::BTreeSet::from([1, 2]),
            terminal_ids: std::collections::BTreeSet::from([1]),
        };
        let result = staged_infer::InferenceResult {
            generated_token_count: 6,
            generated_token_ids: vec![2, 150917, 10, 999, 1, 3],
            generated_token_ids_sha256: String::from("sha"),
            generated_text: String::from("Ciao"),
            stop_reason: String::from("eos"),
            ..Default::default()
        };

        let trace = render_token_trace(&result, &decoder);

        assert!(trace.contains(r#"0: id=2 token="<bos>" kind=special rendered=<skipped>"#));
        assert!(trace.contains(r#"1: id=150917 token="Ciao" kind=text rendered="Ciao""#));
        assert!(trace.contains(r#"2: id=10 token="<0xC3>" kind=byte rendered=<byte 0xC3>"#));
        assert!(trace.contains("3: id=999 token=<unknown> kind=unknown rendered=<unknown>"));
        assert!(trace.contains(r#"4: id=1 token="<eos>" kind=terminal rendered=<stops text>"#));
        assert!(trace.contains(r#"5: id=3 token="late" kind=after_stop rendered=<skipped>"#));
    }

    fn direct_timing(stage: &str, millis: u64) -> staged_infer::InferStageTiming {
        let duration = std::time::Duration::from_millis(millis);
        staged_infer::InferStageTiming {
            stage: stage.to_string(),
            routine: String::from("direct-infer"),
            input_synthesis_duration: std::time::Duration::ZERO,
            input_load_duration: std::time::Duration::ZERO,
            kernel_duration: duration,
            encode_duration: std::time::Duration::ZERO,
            total_duration: duration,
        }
    }
}
