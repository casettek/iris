//! A run's inputs: its text and what the text was built from, resolved from
//! a specification and a case, and the files its stages load.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use inference_manifest::ModelManifest;
use inference_prompt::{
    claimed_ranges, prompt_text, split_prompt, tokenizer_profile, BoundRange, MarkedRange,
};
use input_embedding::input::ClaimedOutput;
use procedure_definition::{Case, PreparedPrompt, Specification};
use prompt_check::input::{PromptText, RunConfig, RunSchedule, TextDigest};
use raster::List;

pub mod pipeline_run;
pub mod run_inputs;
pub mod source;

/// One inference's prompt. `case_dir` is the directory the case's paths are
/// relative to. `relayed` is the result of the inference whose output this
/// one's prompt holds, for an inference a relay gives its prompt; for any
/// other the prompt is the run's own.
pub fn prepare_prompt(
    tokenizer: &serde_json::Value,
    specification: &Specification,
    case_dir: &Path,
    case: &Case,
    model: &ModelManifest,
    relayed: Option<&procedure_definition::InferenceResult>,
) -> Result<PreparedPrompt> {
    let (prompt, source, forms) = match relayed {
        Some(earlier) => {
            let run = relayed_run(specification, earlier)?;
            (run.text, Some(run.record), specification.forms())
        }
        None => resolve_prompt(specification, case_dir, case)?,
    };
    rendered_prompt(tokenizer, specification, model, prompt, source, forms)
}

/// A prompt of this text as a model's tokenizer prepares it, with the record
/// of what the text was built from and the forms its output is read by.
pub fn rendered_prompt(
    tokenizer: &serde_json::Value,
    specification: &Specification,
    model: &ModelManifest,
    prompt: String,
    source: Option<source::SourceRecord>,
    forms: Vec<source::Form>,
) -> Result<PreparedPrompt> {
    // The template is the tokenizer profile's, which is what the prompt stages
    // render the text with, whatever the model's family.
    let profile = tokenizer_profile(tokenizer)?;
    let rendered_prompt = if !specification.raw_prompt && profile.templated() {
        format!(
            "{}{}{}",
            profile.template_prefix,
            prompt.trim(),
            profile.template_suffix
        )
    } else {
        prompt.clone()
    };
    let initial_pieces = split_prompt(&rendered_prompt, tokenizer)?;

    Ok(PreparedPrompt {
        resolved_prompt: prompt,
        rendered_prompt,
        initial_pieces,
        eos_token_ids: model.eos_token_ids.clone(),
        source,
        forms,
    })
}

/// The prompt built around an earlier inference's output, and the record the
/// relay stage states of it. A relay with a fault gives no prompt: native
/// inference is the gate, and no claim is built on such a run.
pub fn relayed_run(
    specification: &Specification,
    earlier: &procedure_definition::InferenceResult,
) -> Result<source_phase::relay::RelayRun> {
    use source_phase::relay::{relay_fault_description, run_relay_steps};
    let relay = specification.relay.as_ref().context(
        "the specification has no `[relay]` table to build a prompt around an earlier output with",
    )?;
    let hash = |text: &str, what: &str| {
        hex::decode(text).with_context(|| format!("the earlier inference's {what} is not hex"))
    };
    let run = run_relay_steps(
        &relay.recipe(),
        &earlier.generated_text,
        &hash(&earlier.statement_sha256, "statement")?,
        &hash(&earlier.output_text_sha256, "output text hash")?,
        earlier.valid,
    );
    anyhow::ensure!(
        run.record.fault == source::FAULT_NONE,
        "the prompt built around the earlier inference's output has fault {} at {}: {}",
        run.record.fault,
        run.record.fault_position,
        relay_fault_description(run.record.fault)
    );
    Ok(run)
}

/// Writes the `claim-result` stage's forms for a run.
pub fn write_result_forms(forms: &[source::Form], dir: &Path) -> Result<RunFile> {
    write_run_file(
        &result_phase::ResultForms {
            forms: forms.to_vec().into(),
        },
        dir,
        "result_forms",
    )
}

/// The source record as the first bind takes it: the run's, or the default
/// for a run whose text is taken as given.
pub fn bind_source(record: Option<&source::SourceRecord>) -> prompt_check::input::SourceRecord {
    match record {
        Some(record) => prompt_check::input::SourceRecord {
            present: record.present,
            text_sha256: record.text_sha256.clone(),
            text_total: record.text_total,
            recipe_sha256: record.recipe_sha256.clone(),
            documents: record.documents,
            documents_sha256: record.documents_sha256.clone(),
            signer: record.signer.clone(),
            fault: record.fault,
            fault_position: record.fault_position,
        },
        None => Default::default(),
    }
}

/// The digest a statement names for a run's source: 64 zeros for none.
pub fn source_id(record: Option<&source::SourceRecord>) -> String {
    prompt_check::input::source_id(&bind_source(record))
}

/// Writes the `prompt-source` stage's inputs for a run: the signer's key, the
/// recipe, and the documents in chunks with their attestations.
pub fn write_source_inputs(
    source: &source::LoadedSource,
    dir: &Path,
) -> Result<procedure_definition::PreparedSourceInputs> {
    let file = |file: RunFile| procedure_definition::RunFileRef {
        path: file.path,
        index_path: file.index_path,
        commitment: file.commitment,
    };
    Ok(procedure_definition::PreparedSourceInputs {
        signer: file(write_run_file(
            &source.specification.signer,
            dir,
            "source_signer",
        )?),
        recipe: file(write_run_file(
            &source.specification.recipe(),
            dir,
            "source_recipe",
        )?),
        documents: file(write_run_file(
            &source.documents(),
            dir,
            "source_documents",
        )?),
    })
}

/// One run-specific external: its data file, index and commitment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunFile {
    pub path: std::path::PathBuf,
    pub index_path: std::path::PathBuf,
    pub commitment: String,
}

fn write_run_file<T: serde::Serialize>(value: &T, dir: &Path, name: &str) -> Result<RunFile> {
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let path = dir.join(format!("{name}.rastered"));
    let index_path = dir.join(format!("{name}.rindex"));
    let commitment = raster::write_raster_files(value, &path, &index_path)
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok(RunFile {
        path,
        index_path,
        commitment,
    })
}

/// One of a run's inputs as a stage is given it: the value the stage's
/// program reads, which is what its Raster file is written from.
///
/// A claim posts its run's inputs as values
/// ([`procedure_definition::run_inputs::PostedRun`]), and anyone writes the
/// file a stage loads from one. The token marks and the claimed ids are
/// posted as plain numbers and given to a stage as the stage's own type.
fn with_run_input<R>(
    input: procedure_definition::run_inputs::RunInput<'_>,
    encode: &mut dyn FnMut(&dyn ErasedRunValue) -> Result<R>,
) -> Result<R> {
    use procedure_definition::run_inputs::RunInput;
    match input {
        RunInput::Signer(value) => encode(value),
        RunInput::Recipe(value) => encode(value),
        RunInput::Documents(value) => encode(value),
        RunInput::Forms(value) => encode(value),
        RunInput::Config(value) => encode(value),
        RunInput::Schedule(value) => encode(value),
        RunInput::TextDigest(value) => encode(value),
        RunInput::Text(value) => encode(value),
        RunInput::Marks(starts) => encode(&prompt_check::check::TokenMarks {
            starts: List::from(starts.to_vec()),
        }),
        RunInput::Claimed(ids) => encode(&ClaimedOutput {
            token_ids: List::from(ids.to_vec()),
        }),
        RunInput::RelayRecipe(value) => encode(value),
    }
}

/// A run value of any of the types a stage is given.
trait ErasedRunValue {
    fn commitment(&self) -> Result<String>;
    fn write(&self, dir: &Path, name: &str) -> Result<RunFile>;
}

impl<T: serde::Serialize> ErasedRunValue for T {
    fn commitment(&self) -> Result<String> {
        let (_, _, commitment) =
            raster::encode_raster_value(self).map_err(|error| anyhow::anyhow!("{error}"))?;
        Ok(commitment)
    }

    fn write(&self, dir: &Path, name: &str) -> Result<RunFile> {
        write_run_file(self, dir, name)
    }
}

/// The commitment a stage's input list names this run input by.
pub fn run_input_commitment(
    input: procedure_definition::run_inputs::RunInput<'_>,
) -> Result<String> {
    with_run_input(input, &mut |value| value.commitment())
}

/// Writes a run input as the files a stage loads, named `name`.
pub fn write_run_input(
    input: procedure_definition::run_inputs::RunInput<'_>,
    dir: &Path,
    name: &str,
) -> Result<RunFile> {
    with_run_input(input, &mut |value| value.write(dir, name))
}

/// Writes each range's marks as `marks_{j}`.
pub fn write_range_marks(ranges: &[MarkedRange], dir: &Path) -> Result<Vec<RunFile>> {
    write_marks(
        &ranges
            .iter()
            .map(|range| range.marks.clone())
            .collect::<Vec<_>>(),
        dir,
    )
}

/// Writes these marks as `marks_{j}`.
pub fn write_marks(marks: &[Vec<u32>], dir: &Path) -> Result<Vec<RunFile>> {
    marks
        .iter()
        .enumerate()
        .map(|(index, starts)| {
            write_run_file(
                &prompt_check::check::TokenMarks {
                    starts: List::from(starts.clone()),
                },
                dir,
                &format!("marks_{index}"),
            )
        })
        .collect()
}

/// Writes these slices of the original prompt as `text_{j}`.
pub fn write_text_slices(slices: &[PromptText], dir: &Path) -> Result<Vec<RunFile>> {
    slices
        .iter()
        .enumerate()
        .map(|(index, slice)| write_run_file(slice, dir, &format!("text_{index}")))
        .collect()
}

/// Writes each range's slice of the original prompt as `text_{j}`.
pub fn write_bound_text(ranges: &[BoundRange], dir: &Path) -> Result<Vec<RunFile>> {
    ranges
        .iter()
        .enumerate()
        .map(|(index, range)| write_run_file(&range.text, dir, &format!("text_{index}")))
        .collect()
}

/// Writes what the run claims of the whole text as `text_digest`.
pub fn write_text_digest(text_digest: &TextDigest, dir: &Path) -> Result<RunFile> {
    write_run_file(text_digest, dir, "text_digest")
}

/// Writes the original prompt as `text`.
pub fn write_prompt_text(prompt: &str, dir: &Path) -> Result<RunFile> {
    write_run_file(&prompt_text(prompt), dir, "text")
}

/// Writes the run's configuration as `config`.
pub fn write_run_config(config: &RunConfig, dir: &Path) -> Result<RunFile> {
    write_run_file(config, dir, "config")
}

/// Writes the run's stage schedule as `schedule`, from the counts the pipeline
/// manifest is filled in with.
pub fn write_run_schedule(
    prompt_ranges: usize,
    claimed_ranges: usize,
    dir: &Path,
) -> Result<RunFile> {
    let schedule = RunSchedule {
        prompt_ranges: u32::try_from(prompt_ranges).context("prompt range count exceeds u32")?,
        claimed_ranges: u32::try_from(claimed_ranges).context("claimed range count exceeds u32")?,
    };
    write_run_file(&schedule, dir, "schedule")
}

/// Writes each claimed range as `claimed_{k}`.
pub fn write_claimed_ranges(
    token_ids: &[u32],
    range_size: usize,
    dir: &Path,
) -> Result<Vec<RunFile>> {
    claimed_ranges(token_ids, range_size)?
        .iter()
        .enumerate()
        .map(|(index, range)| write_run_file(range, dir, &format!("claimed_{index}")))
        .collect()
}

/// The commitments `write_claimed_ranges` would bind for these ids.
pub fn claimed_range_commitments(token_ids: &[u32], range_size: usize) -> Result<Vec<String>> {
    claimed_ranges(token_ids, range_size)?
        .iter()
        .map(|range| {
            let (_, _, commitment) =
                raster::encode_raster_value(range).map_err(|error| anyhow::anyhow!("{error}"))?;
            Ok(commitment)
        })
        .collect()
}

/// The run's prompt text, and for a prompt built from signed documents the
/// source record and the specification's forms.
fn resolve_prompt(
    specification: &Specification,
    case_dir: &Path,
    case: &Case,
) -> Result<(String, Option<source::SourceRecord>, Vec<source::Form>)> {
    if let Some(source) = source::load_source(specification, case_dir, case)? {
        let run = source::sourced_run(&source)?;
        return Ok((run.text, Some(run.record), source.forms()));
    }
    if let Some(prompt) = case.prompt.as_ref() {
        return Ok((prompt.clone(), None, Vec::new()));
    }
    let prompt_file = case
        .prompt_file
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("a case must set `prompt` or `prompt_file`"))?;
    let path = if prompt_file.is_absolute() {
        prompt_file.clone()
    } else {
        case_dir.join(prompt_file)
    };
    let prompt = fs::read_to_string(&path)
        .with_context(|| format!("failed to read prompt file {}", path.display()))?;
    Ok((prompt, None, Vec::new()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_manifest::{NativeInferBundle, NativeInferShape};
    use inference_prompt::*;
    use prompt_check::input::BpePiece;
    use std::path::PathBuf;

    #[test]
    fn raw_prompt_bypasses_turn_rendering() {
        let tokenizer = gemma_tokenizer();
        let (specification, case) = specification("h", true);
        let prepared = prepare_prompt(
            &tokenizer,
            &specification,
            Path::new("."),
            &case,
            &model_manifest(),
            None,
        )
        .unwrap();

        assert_eq!(prepared.rendered_prompt, "h");
        assert_eq!(
            prepared
                .initial_pieces
                .iter()
                .map(|p| p.text.as_str())
                .collect::<Vec<_>>(),
            vec!["h"]
        );
    }

    #[test]
    fn special_tokens_stay_whole_in_initial_pieces() {
        let tokenizer = gemma_tokenizer();
        let (specification, case) = specification("hello", false);
        let prepared = prepare_prompt(
            &tokenizer,
            &specification,
            Path::new("."),
            &case,
            &model_manifest(),
            None,
        )
        .unwrap();

        assert!(prepared.initial_pieces.iter().any(|p| p.text == "<bos>"));
        assert!(prepared.initial_pieces.iter().any(|p| p.text == "<|turn>"));
    }

    #[test]
    fn the_bind_stages_split_the_prompt_as_run_preparation_does() {
        let tokenizer = chat_tokenizer();
        let prompt_tokenizer = prompt_tokenizer(&tokenizer).unwrap();
        for (prompt, raw_prompt) in PROMPTS {
            let (specification, case) = specification(prompt, raw_prompt);
            let prepared = prepare_prompt(
                &tokenizer,
                &specification,
                Path::new("."),
                &case,
                &model_manifest(),
                None,
            )
            .unwrap();
            let whole_pieces = prepared
                .initial_pieces
                .iter()
                .map(|piece| BpePiece {
                    text: piece.text.clone(),
                    segment: piece.segment,
                })
                .collect::<Vec<_>>();
            if whole_pieces.is_empty() {
                // Nothing to claim over: an empty raw prompt has no pieces.
                assert!(bound_ranges(prompt, raw_prompt, 4, &prompt_tokenizer).is_err());
                continue;
            }
            let whole_ids: Vec<u32> = inference_kernels::kernels::prompt_check::merged_tokens(
                &whole_pieces,
                &prompt_tokenizer,
            )
            .unwrap()
            .iter()
            .map(|token| token.id)
            .collect();
            let config = RunConfig {
                raw_prompt,
                ..RunConfig::default()
            };
            for range_size in [1, 2, 3, 5, 64] {
                let label = format!("{prompt:?} raw {raw_prompt} range size {range_size}");
                let bound =
                    bound_ranges(prompt, raw_prompt, range_size, &prompt_tokenizer).unwrap();
                assert_eq!(
                    bound.ranges.len(),
                    whole_ids.len().div_ceil(range_size),
                    "{label}"
                );
                // The slices are the text, in order, and only a later range is empty.
                let text: String = bound
                    .ranges
                    .iter()
                    .flat_map(|range| range.text.chunks.iter().cloned())
                    .collect();
                assert_eq!(text, prompt, "{label}");
                assert!(!bound.ranges[0].text.chunks.is_empty(), "{label}");
                let (pieces, ids, last) = run_prompt_stages(&prompt_tokenizer, &bound, &config);
                assert_eq!(pieces, whole_pieces, "{label}");
                assert_eq!(ids, whole_ids, "{label}");
                assert_eq!(last.context.fault, 0, "{label}");
                assert!(last.context.complete, "{label}");
                assert!(last.rest.is_empty(), "{label}");
                assert_eq!(last.context.prompt_tokens as usize, whole_ids.len());
            }
        }
    }

    #[test]
    fn a_run_with_a_source_is_held_to_the_text_its_source_gives() {
        use prompt_check::input::{FAULT_SOURCE, FAULT_TEXT_SOURCE};
        let tokenizer = prompt_tokenizer(&chat_tokenizer()).unwrap();
        let (prompt, raw_prompt) = PROMPTS[0];
        let config = RunConfig {
            raw_prompt,
            ..RunConfig::default()
        };
        let bound = bound_ranges(prompt, raw_prompt, 3, &tokenizer).unwrap();
        assert!(bound.ranges.len() > 1);
        let honest = source::SourceRecord {
            present: true,
            text_sha256: bound.text_digest.input_text_sha256.clone(),
            text_total: bound.text_digest.total,
            recipe_sha256: "aa".repeat(32),
            documents: 2,
            documents_sha256: "bb".repeat(32),
            signer: "02ab".to_string(),
            fault: 0,
            fault_position: 0,
        };
        let last = |record: Option<&source::SourceRecord>| {
            run_sourced_prompt_stages(&tokenizer, &bound, &config, &bind_source(record))
                .2
                .context
        };
        let edited = |edit: &dyn Fn(&mut source::SourceRecord)| {
            let mut record = honest.clone();
            edit(&mut record);
            record
        };

        let none = last(None);
        assert_eq!((none.fault, none.complete), (FAULT_NONE, true));
        assert_eq!(none.source_sha256, "00".repeat(32));
        assert_eq!(source_id(None), "00".repeat(32));

        // Every range carries the record's digest, and the statement names it.
        let sourced = last(Some(&honest));
        assert_eq!((sourced.fault, sourced.complete), (FAULT_NONE, true));
        assert_eq!(sourced.source_sha256, source_id(Some(&honest)));
        assert_ne!(sourced.source_sha256, none.source_sha256);
        assert_eq!(sourced.input_text_sha256, none.input_text_sha256);

        // Another text than the source's, by hash or by length.
        let other = edited(&|record| record.text_sha256 = "00".repeat(32));
        assert_eq!(last(Some(&other)).fault, FAULT_TEXT_SOURCE);
        let other = edited(&|record| record.text_total += 1);
        assert_eq!(last(Some(&other)).fault, FAULT_TEXT_SOURCE);

        // A source with a fault of its own, which comes first.
        let faulty = edited(&|record| {
            record.fault = 6;
            record.fault_position = 2;
            record.text_sha256 = "00".repeat(32);
        });
        let context = last(Some(&faulty));
        assert_eq!((context.fault, context.fault_position), (FAULT_SOURCE, 2));

        // A record that is not present is no source, whatever else it says.
        let absent = edited(&|record| {
            record.present = false;
            record.fault = 6;
        });
        let context = last(Some(&absent));
        assert_eq!(context.fault, FAULT_NONE);
        assert_eq!(context.source_sha256, "00".repeat(32));

        // The digest is over every field.
        let stated = source_id(Some(&honest));
        for other in [
            edited(&|record| record.text_sha256 = "11".repeat(32)),
            edited(&|record| record.text_total += 1),
            edited(&|record| record.recipe_sha256 = "11".repeat(32)),
            edited(&|record| record.documents += 1),
            edited(&|record| record.documents_sha256 = "11".repeat(32)),
            edited(&|record| record.signer.push('0')),
            edited(&|record| record.fault = 1),
            edited(&|record| record.fault_position = 1),
        ] {
            assert_ne!(source_id(Some(&other)), stated);
        }
    }

    #[test]
    fn the_source_record_is_one_value_under_both_of_its_types() {
        // The source step states `source_phase::SourceRecord` and the bind
        // reads `SourceRecord` of the shared prompt source. A pipeline links
        // them by structure, so the two must commit alike.
        let record = source::SourceRecord {
            present: true,
            text_sha256: "11".repeat(32),
            text_total: 2388,
            recipe_sha256: "22".repeat(32),
            documents: 3,
            documents_sha256: "33".repeat(32),
            signer: "02ab".to_string(),
            fault: 4,
            fault_position: 7,
        };
        for record in [record, source::SourceRecord::default()] {
            assert_eq!(
                raster::core::tree::value_root(&record).unwrap(),
                raster::core::tree::value_root(&bind_source(Some(&record))).unwrap()
            );
        }
    }

    const ONE: i32 = 1 << 16;

    /// A tokenizer with a template, a special, merges and byte fallback.
    fn chat_tokenizer() -> serde_json::Value {
        serde_json::json!({
            "model": {
                "type": "BPE",
                "vocab": {
                    "<bos>": 0, "<|turn>": 1, "<turn|>": 2, "\n": 3, "u": 4, "s": 5, "e": 6,
                    "r": 7, "m": 8, "o": 9, "d": 10, "l": 11, "h": 12, "▁": 13, "<0xC3>": 14,
                    "<0xA9>": 15, "<unk>": 16, "he": 17, "ld": 18, "old": 19
                },
                "merges": ["h e", "l d", "o ld"],
                "byte_fallback": true,
                "unk_token": "<unk>",
                "fuse_unk": true
            },
            "normalizer": {"type":"Replace","pattern":{"String":" "},"content":"▁"},
            "added_tokens": [
                { "id": 0, "content": "<bos>", "special": true },
                { "id": 1, "content": "<|turn>", "special": true },
                { "id": 2, "content": "<turn|>", "special": true }
            ]
        })
    }

    const PROMPTS: [(&str, bool); 6] = [
        ("  he é<turn|>hold\u{4e16}\u{754c} he hold  ", false),
        ("  he é<turn|>hold\u{4e16}\u{754c} he hold  ", true),
        ("   ", false),
        ("", false),
        ("hold", true),
        ("<<|<|turn>|<", true),
    ];

    /// The pipeline's prompt stages, run natively over what `bound_ranges` cut.
    fn run_prompt_stages(
        tokenizer: &PromptTokenizer,
        bound: &BoundPrompt,
        config: &RunConfig,
    ) -> (Vec<BpePiece>, Vec<u32>, prompt_check::input::PromptState) {
        run_sourced_prompt_stages(tokenizer, bound, config, &Default::default())
    }

    /// The same, with the source record the first bind is given.
    fn run_sourced_prompt_stages(
        tokenizer: &PromptTokenizer,
        bound: &BoundPrompt,
        config: &RunConfig,
        source: &prompt_check::input::SourceRecord,
    ) -> (Vec<BpePiece>, Vec<u32>, prompt_check::input::PromptState) {
        use inference_kernels::kernels::prompt_check_stage::{run_prompt_check, Trusted};
        use inference_kernels::kernels::prompt_prepare::{run_prompt_bind, PromptBindInputs};
        use prompt_check::input::PromptState;
        let root = raster::BytesPage::__from_parts(0, 0, Vec::new());
        let schedule = RunSchedule {
            prompt_ranges: bound.ranges.len() as u32,
            claimed_ranges: 1,
        };
        let mut binding = PromptState::default();
        let mut checked = PromptState::default();
        let (mut pieces, mut ids) = (Vec::new(), Vec::new());
        for range in &bound.ranges {
            binding = run_prompt_bind(
                PromptBindInputs {
                    tokenizer,
                    root: &root,
                    text: &range.text,
                    config,
                    descriptor: &[],
                    schedule: &schedule,
                    text_digest: &bound.text_digest,
                    source,
                    previous: &binding,
                },
                &mut Trusted,
            )
            .unwrap();
            pieces.extend(binding.pieces.iter().cloned());
            checked = run_prompt_check(
                tokenizer,
                &root,
                &binding,
                &checked,
                &range.marks,
                &mut Trusted,
            )
            .unwrap();
            ids.extend(checked.token_ids.iter().copied());
        }
        (pieces, ids, checked)
    }

    fn specification(prompt: &str, raw_prompt: bool) -> (Specification, Case) {
        let specification = Specification {
            procedure_kind: Default::default(),
            raw_prompt,
            max_new_tokens: 1,
            models: vec![procedure_definition::SpecificationModel {
                manifest: PathBuf::from("model-artifacts/manifest.json"),
            }],
            source: None,
            forms: Vec::new(),
            relay: None,
        };
        (specification, Case::of_prompt(prompt.to_string()))
    }

    fn model_manifest() -> ModelManifest {
        ModelManifest {
            version: inference_manifest::MODEL_MANIFEST_VERSION,
            family: inference_manifest::ModelFamily::GemmaDense
                .name()
                .to_string(),
            model_id: inference_manifest::model_id_of(&[]),
            model_commitments: Vec::new(),
            range_size: 512,
            bundle: NativeInferBundle {
                model_detwgt_path: PathBuf::from("model.detwgt"),
                model_detwgt_sha256: String::new(),
                config_path: PathBuf::from("config.json"),
                config_sha256: String::new(),
                tokenizer_path: PathBuf::from("tokenizer.json"),
                tokenizer_sha256: String::new(),
            },
            shape: NativeInferShape {
                hidden_size: 1,
                num_hidden_layers: 0,
                num_attention_heads: 1,
                num_key_value_heads: 1,
                head_dim: 1,
                global_head_dim: 1,
                vocab_size: 1,
                hidden_size_per_layer_input: 1,
                sliding_window: 0,
                layer_types: Vec::new(),
                num_kv_shared_layers: 0,
                norm_eps: 0,
                rope_base_sliding: 10_000_i64 << 32,
                rope_base_full: 1_000_000_i64 << 32,
                full_partial_rotary_factor_q16: ONE,
                embedding_scale: ONE,
                ple_embedding_scale: ONE,
                ple_projection_scalar: ONE,
                ple_input_scale: ONE,
                final_logit_softcap: 0,
            }
            .into(),
            eos_token_ids: vec![1, 2],
            provenance: None,
        }
    }

    fn gemma_tokenizer() -> serde_json::Value {
        serde_json::json!({
            "model": {
                "type": "BPE",
                "vocab": {
                    "<bos>": 0,
                    "<|turn>": 1,
                    "<turn|>": 2,
                    "\n": 3,
                    "hello": 4,
                    "user": 5,
                    "model": 6,
                    "h": 7, "e": 8, "l": 9, "o": 10, "u": 11, "s": 12, "r": 13, "m": 14, "d": 15
                },
                "merges": []
            },
            "added_tokens": [
                { "id": 0, "content": "<bos>", "special": true },
                { "id": 1, "content": "<|turn>", "special": true },
                { "id": 2, "content": "<turn|>", "special": true }
            ]
        })
    }
}
