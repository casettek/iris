//! Preparing a staged run: from a specification and a case, or from what a
//! claim fixed or posted, to the files each stage loads and the chain the
//! stages are run from.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use model_manifest::{file_sha256 as sha256_file, ModelManifest};
use procedure_definition::run_inputs::PostedRun;
use procedure_definition::{
    read_json, write_json, Case, PreparedRun, Specification, PREPARED_RUN_JSON,
};
use sha2::{Digest, Sha256};

/// A prepared staged run: the frozen record of what it runs on, and the
/// chain its stages are run from.
pub struct PreparedClaimRun {
    pub prepared_run: PreparedRun,
    pub manifest_path: PathBuf,
    /// The ids written into the run's `claimed_ids` externals: the first
    /// inference's, which are all of them in a procedure kind of one. Kept for the
    /// claimer's own tooling only: the claim posts no copy of them, because its
    /// output is the chain's final record.
    pub claimed_token_ids: Vec<u32>,
}

/// A claim's source as a challenger is given it: the record, and the files
/// the source stage ran on.
pub type FrozenSource<'a> = (
    &'a crate::source::SourceRecord,
    &'a procedure_definition::PreparedSourceInputs,
);

/// What a claim fixed of its run, for a challenger to rebuild it under.
pub struct Frozen<'a> {
    pub source: Option<FrozenSource<'a>>,
    pub forms: &'a [crate::source::Form],
    pub ids: &'a [Vec<u32>],
}

/// One inference of a run being prepared: its model, and the prompt as that
/// model's tokenizer prepares it.
struct RunModel {
    manifest_path: PathBuf,
    manifest: ModelManifest,
    template_path: PathBuf,
    /// The model's segment of a chain, as its import wrote it.
    segment: String,
    tokenizer: serde_json::Value,
    prompt: procedure_definition::PreparedPrompt,
}

/// The files written for one inference of a run.
struct WrittenInference {
    label: String,
    files: crate::run_inputs::InferenceRunFiles,
}

/// What a staged run is prepared from.
pub struct StagedRun<'a> {
    /// Where the run's directory is made, under `target/iris/runs`.
    pub base_dir: &'a Path,
    pub specification_path: PathBuf,
    pub specification: &'a Specification,
    /// The file the case was read from, when it is one. A case's paths are
    /// relative to it.
    pub case_path: Option<PathBuf>,
    pub case: &'a Case,
    /// What a claim fixed of its run, for a challenger rebuilding it.
    pub frozen: Option<Frozen<'a>>,
    /// The inputs a claim posted, for a challenger of that claim: they are
    /// the run's inputs, in place of the ones this would prepare.
    pub given: Option<&'a PostedRun>,
    /// The output ids of the run's one inference, when they are given.
    pub given_ids: Option<&'a Vec<u32>>,
    /// Whether an output is held to the run's token budget here.
    pub check_claimed_length: bool,
    /// Finds the output of an inference whose ids no one gave: the direct
    /// run, which this crate does not make.
    pub discover: &'a dyn Fn(usize) -> Result<Vec<u32>>,
}

/// `case` is the case, with the file it was read from when it is one.
/// `given` is the inputs a claim posted, for a challenger of that claim:
/// they are the run's inputs, in place of the ones this would prepare.
pub fn prepare_run(run: StagedRun<'_>) -> Result<PreparedClaimRun> {
    let StagedRun {
        base_dir,
        specification_path,
        specification,
        case_path,
        case,
        frozen,
        given,
        given_ids,
        check_claimed_length,
        discover,
    } = run;
    use procedure_definition::run_inputs::{PostedInference, PostedSourceInputs};
    let specification_dir = specification_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("specification has no parent directory"))?;
    // A case's paths are relative to its file. A case that is no file names
    // none.
    let case_dir = match &case_path {
        Some(path) => path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("case has no parent directory"))?,
        None => specification_dir,
    };
    let procedure_kind = specification.procedure_kind;
    // Ids given for the run are one inference's.
    anyhow::ensure!(
        given_ids.is_none() || procedure_kind.inferences() == 1,
        "output ids can be given for a procedure kind of one inference, and the `{procedure_kind}` procedure kind \
         makes {}",
        procedure_kind.inferences()
    );
    if let Some(frozen) = &frozen {
        anyhow::ensure!(
            frozen.ids.len() == procedure_kind.inferences(),
            "the claim holds the ids of {} inference(s), and the `{procedure_kind}` procedure kind makes {}",
            frozen.ids.len(),
            procedure_kind.inferences()
        );
    }

    // Each inference's model, and the run's prompt as its tokenizer prepares
    // it. The text, the source record and the forms are the same in all.
    let mut models = Vec::new();
    for manifest in specification.model_manifests() {
        let manifest_path = absolute(specification_dir, manifest);
        let manifest = ModelManifest::load_verified(&manifest_path)?;
        let (template_path, segment) = manifest.verified_raster_template(&manifest_path)?;
        let model_manifest_dir = manifest_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("model manifest has no parent directory"))?;
        let tokenizer_path = absolute(model_manifest_dir, &manifest.bundle.tokenizer_path);
        let tokenizer: serde_json::Value = read_json(&tokenizer_path)?;
        let mut prompt =
            crate::prepare_prompt(&tokenizer, specification, case_dir, case, &manifest)?;
        if let Some(frozen) = &frozen {
            if let Some((record, _)) = frozen.source {
                prompt.source = Some(record.clone());
            }
            prompt.forms = frozen.forms.to_vec();
        }
        models.push(RunModel {
            manifest_path,
            manifest,
            template_path,
            segment,
            tokenizer,
            prompt,
        });
    }
    let first = models.first().context("the run names no model")?;

    let run_dir = base_dir
        .join("target")
        .join("iris")
        .join("runs")
        .join(run_id(
            &specification_path,
            &first.prompt.rendered_prompt,
            specification.max_new_tokens,
        ));
    let prompt_dir = run_dir.join("prompt");
    if let Some(given) = given {
        anyhow::ensure!(
            given.inferences.len() == procedure_kind.inferences(),
            "the claim posted the inputs of {} inference(s), and the `{procedure_kind}` procedure kind makes {}",
            given.inferences.len(),
            procedure_kind.inferences()
        );
    }
    // The source stage's inputs: the claim's own for a challenger, or written
    // from the run's source. A run without a source has no such stage. The
    // values are kept beside the files: they are what a claim posts. A
    // challenger given only the files of a claim built here posts nothing.
    let frozen_source = frozen.as_ref().and_then(|frozen| frozen.source);
    let source_values = match given {
        Some(given) => given.source.clone(),
        None => crate::source::load_source(specification, case_dir, case)?.map(|loaded| {
            PostedSourceInputs {
                signer: loaded.specification.signer.clone(),
                recipe: loaded.specification.recipe(),
                documents: loaded.documents(),
            }
        }),
    };
    let source_inputs = match (&source_values, frozen_source) {
        (Some(source), _) => Some(crate::run_inputs::write_source(source, &prompt_dir)?),
        (None, Some((_, inputs))) if given.is_none() => Some(inputs.clone()),
        _ => None,
    };
    let postable = given.is_some() || source_values.is_some() == source_inputs.is_some();
    // The imports wrote each model's segment of a chain. The procedure kind makes the
    // chain's template from them.
    let segments: Vec<&str> = models.iter().map(|model| model.segment.as_str()).collect();
    let template = procedure_kind.chain_template(&segments)?;
    let template_dir = first
        .template_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("manifest template has no parent directory"))?;
    let form_values = match given {
        Some(given) => given.forms.clone(),
        None => result_core::ResultForms {
            forms: first.prompt.forms.to_vec().into(),
        },
    };
    let forms = crate::write_run_input(
        procedure_definition::run_inputs::RunInput::Forms(&form_values),
        &prompt_dir,
        "result_forms",
    )?;

    let mut written = Vec::new();
    let mut values = Vec::new();
    let mut claimed_ids = Vec::new();
    for (inference, model) in models.iter().enumerate() {
        let label = procedure_kind.label(inference);
        // A procedure kind of one inference keeps its run files where they have
        // always been. Each inference of several has a directory of its own.
        let dir = if label.is_empty() {
            prompt_dir.clone()
        } else {
            prompt_dir.join(label.trim_end_matches('_'))
        };
        if let Some(given) = given {
            // The claim's own values: nothing is prepared.
            let posted = given.inferences[inference].clone();
            claimed_ids.push(posted.claimed.concat());
            written.push(WrittenInference {
                files: crate::run_inputs::write_inference(&posted, &dir)?,
                label,
            });
            values.push(posted);
            continue;
        }
        let ids = match (&frozen, given_ids) {
            (Some(frozen), _) => frozen.ids[inference].clone(),
            (None, Some(ids)) => ids.clone(),
            // Discovery is the only autoregression left: the claim posts what
            // direct decode produced and the chain checks it.
            (None, None) => discover(inference)?,
        };
        if check_claimed_length {
            prompt_prep::validate_claimed_length(
                &ids,
                specification.max_new_tokens,
                &model.prompt.eos_token_ids,
            )?;
        }
        let range_size = model.manifest.range_size;
        let bound = prompt_prep::bound_ranges(
            &model.prompt.resolved_prompt,
            specification.raw_prompt,
            range_size as usize,
            &prompt_prep::prompt_tokenizer(&model.tokenizer)?,
        )?;
        // A claim's ids are cut as `claimed_ranges` cuts them, which also
        // refuses a claim of no ids.
        prompt_prep::claimed_ranges(&ids, range_size as usize)?;
        let claimed: Vec<Vec<u32>> = ids
            .chunks(range_size as usize)
            .map(<[u32]>::to_vec)
            .collect();
        // The values a stage is given, which are what the claim posts. Each
        // file is written from its value, so the two cannot differ.
        let posted = PostedInference {
            config: prompt_core::RunConfig {
                model_id: model.manifest.model_id.clone(),
                raw_prompt: specification.raw_prompt,
                max_new_tokens: specification.max_new_tokens,
                range_size,
            },
            schedule: prompt_core::RunSchedule {
                prompt_ranges: u32::try_from(bound.ranges.len())
                    .context("prompt range count exceeds u32")?,
                claimed_ranges: u32::try_from(claimed.len())
                    .context("claimed range count exceeds u32")?,
            },
            text_digest: crate::run_inputs::recast(&bound.text_digest)?,
            text: bound
                .ranges
                .iter()
                .map(|range| crate::run_inputs::recast(&range.text))
                .collect::<Result<_>>()?,
            marks: bound
                .ranges
                .iter()
                .map(|range| range.marks.clone())
                .collect(),
            claimed,
        };
        written.push(WrittenInference {
            files: crate::run_inputs::write_inference(&posted, &dir)?,
            label,
        });
        values.push(posted);
        claimed_ids.push(ids);
    }
    let inferences: Vec<InferenceFiles<'_>> = written
        .iter()
        .map(|written| InferenceFiles {
            label: &written.label,
            text: &written.files.text,
            config: &written.files.config,
            schedule: &written.files.schedule,
            text_digest: &written.files.text_digest,
            marks: &written.files.marks,
            claimed: &written.files.claimed,
        })
        .collect();
    let run_manifest = render_chain(
        &template,
        template_dir,
        &inferences,
        source_inputs.as_ref(),
        &forms,
    )?;
    let manifest_path = run_dir.join("Raster.toml");
    fs::create_dir_all(&run_dir)
        .with_context(|| format!("failed to create {}", run_dir.display()))?;
    fs::write(&manifest_path, &run_manifest)
        .with_context(|| format!("failed to write {}", manifest_path.display()))?;
    let run_manifest_sha256 = format!("{:x}", Sha256::digest(run_manifest.as_bytes()));
    let prepared_run = PreparedRun {
        version: procedure_definition::PREPARED_RUN_VERSION,
        procedure_kind,
        specification_path,
        case_path,
        inferences: models
            .into_iter()
            .map(|model| {
                Ok(procedure_definition::PreparedInference {
                    model_manifest_sha256: sha256_file(&model.manifest_path)?,
                    model_manifest_path: model.manifest_path,
                    model_id: model.manifest.model_id,
                    prompt: model.prompt,
                })
            })
            .collect::<Result<Vec<_>>>()?,
        raw_prompt: specification.raw_prompt,
        max_new_tokens: specification.max_new_tokens,
        run_manifest_path: Some(manifest_path.clone()),
        run_manifest_sha256: Some(run_manifest_sha256),
        source_inputs,
    };
    write_json(&run_dir.join(PREPARED_RUN_JSON), &prepared_run)?;
    // The run's inputs as values, for the claim to post.
    if postable {
        crate::run_inputs::save(
            &PostedRun {
                source: source_values,
                forms: form_values,
                inferences: values,
            },
            &run_dir,
        )?;
    }

    Ok(PreparedClaimRun {
        prepared_run,
        manifest_path,
        claimed_token_ids: claimed_ids.into_iter().next().unwrap_or_default(),
    })
}

const RUN_PLACEHOLDER_DIR: &str = "target/iris/run-prompt-placeholder/";

/// Marks where the template names the last prompt range.
const LAST_PROMPT_RANGE: &str = "@LAST_PROMPT_RANGE@";

/// Marks where the template names the last claimed range.
const LAST_CLAIMED_RANGE: &str = "@LAST_CLAIMED_RANGE@";

/// The chain of a run whose text is taken as given: the template without its
/// source stage, and each inference's first bind given the constant its later
/// binds are.
pub fn without_source_stage(template: &str) -> Result<String> {
    let lines: Vec<&str> = template.lines().collect();
    let takes_record = |line: &str| line.trim() == "inputs.source = { from = \"source\" }";
    let mut out = Vec::new();
    let (mut dropped, mut rebound) = (0, 0);
    let mut index = 0;
    while index < lines.len() {
        let line = lines[index];
        let is_source_stage = line.trim() == "[[chain.stage]]"
            && lines
                .get(index + 1)
                .is_some_and(|next| next.trim() == "name = \"source\"");
        if is_source_stage {
            index += 1;
            while lines
                .get(index)
                .is_some_and(|next| !next.trim_start().starts_with('['))
            {
                index += 1;
            }
            dropped += 1;
            continue;
        }
        if takes_record(line) {
            // The constant is the one the same inference's later binds are
            // given: theirs is the next bind stage in the template.
            let constant = lines[index..]
                .iter()
                .map(|line| line.trim_start())
                .find(|line| line.starts_with("inputs.source = { external"))
                .context("chain template gives no later bind a constant source")?;
            out.push(constant.to_string());
            rebound += 1;
        } else {
            out.push(line.to_string());
        }
        index += 1;
    }
    anyhow::ensure!(
        dropped == 1 && rebound >= 1,
        "incompatible chain template: expected one source stage and a first bind that takes its record; found {dropped} and {rebound}"
    );
    out.push(String::new());
    Ok(out.join("\n"))
}

/// The run files of one inference, and the label its names carry in the
/// template: empty in a chain of one inference.
pub struct InferenceFiles<'a> {
    pub label: &'a str,
    /// One slice of the text per prompt range.
    pub text: &'a [crate::RunFile],
    pub config: &'a crate::RunFile,
    pub schedule: &'a crate::RunFile,
    pub text_digest: &'a crate::RunFile,
    /// One set of token marks per prompt range.
    pub marks: &'a [crate::RunFile],
    /// One set of claimed ids per claimed range.
    pub claimed: &'a [crate::RunFile],
}

/// The inference a stage, a repeat or a chain input belongs to, and its name
/// without the label. In a chain of one inference every name is that one's.
fn inference_name<'n>(
    name: &'n str,
    inferences: &[InferenceFiles<'_>],
) -> Option<(usize, &'n str)> {
    if let [only] = inferences {
        if only.label.is_empty() {
            return Some((0, name));
        }
    }
    inferences.iter().enumerate().find_map(|(index, files)| {
        name.strip_prefix(files.label)
            .filter(|_| !files.label.is_empty())
            .map(|rest| (index, rest))
    })
}

/// Fills the template of a chain of one inference for one run. See
/// [`render_chain`].
#[allow(clippy::too_many_arguments)]
pub fn render_ranged_manifest(
    template: &str,
    template_dir: &Path,
    text: &[crate::RunFile],
    config: &crate::RunFile,
    schedule: &crate::RunFile,
    text_digest: &crate::RunFile,
    source: Option<&procedure_definition::PreparedSourceInputs>,
    forms: &crate::RunFile,
    prompt: &[crate::RunFile],
    claimed: &[crate::RunFile],
) -> Result<String> {
    render_chain(
        template,
        template_dir,
        &[InferenceFiles {
            label: "",
            text,
            config,
            schedule,
            text_digest,
            marks: prompt,
            claimed,
        }],
        source,
        forms,
    )
}

/// Fills a chain template for one run. For each inference: the procedure,
/// the stage schedule and the text digest, the range counts, the last prompt
/// and claimed ranges' indexes, and each range's committed text, marks or
/// claimed ids. For the run: the source stage's inputs and the result stage's
/// forms. A run without a source gets a chain without the source stage.
pub fn render_chain(
    template: &str,
    template_dir: &Path,
    inferences: &[InferenceFiles<'_>],
    source: Option<&procedure_definition::PreparedSourceInputs>,
    forms: &crate::RunFile,
) -> Result<String> {
    anyhow::ensure!(!inferences.is_empty(), "a run needs at least one inference");
    for files in inferences {
        anyhow::ensure!(
            !files.marks.is_empty() && !files.claimed.is_empty(),
            "a run needs at least one prompt range and one claimed range"
        );
        anyhow::ensure!(
            files.text.len() == files.marks.len(),
            "a run needs one slice of text for each prompt range"
        );
    }
    let template = model_manifest::resolve_raster_manifest_paths(template, template_dir)?;
    let template = match source {
        Some(_) => template,
        None => without_source_stage(&template)?,
    };
    let mut table = String::new();
    let mut name = String::new();
    let mut fan_index = String::new();
    // Where the table being read starts its index. Range 0 of the prompt and
    // of the checks is written out as plain stages, so those repeats and the
    // prompt ranges' layer fans start at 1 and run one time fewer.
    let mut start = 0_usize;
    // For each inference: prompt repeats, claimed repeats, prompt externals,
    // claimed externals, per-layer prompt fans, per-layer claimed fans, text
    // externals, procedure bindings, schedule bindings, text digest
    // bindings.
    let mut replaced = vec![[0_usize; 10]; inferences.len()];
    // For the run: the source stage's inputs, the result stage's forms.
    let (mut source_inputs, mut result_forms) = (0_usize, 0_usize);
    let mut out = Vec::new();
    for line in template.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            table = trimmed.to_string();
            name.clear();
            fan_index.clear();
            start = 0;
        }
        if trimmed.starts_with("start =") {
            let value: toml::Value = toml::from_str(trimmed)?;
            start = value["start"]
                .as_integer()
                .and_then(|start| usize::try_from(start).ok())
                .context("invalid repeat start")?;
        }
        if trimmed.starts_with("name =") {
            let value: toml::Value = toml::from_str(trimmed)?;
            name = value["name"]
                .as_str()
                .context("invalid stage/repeat name")?
                .to_string();
        }
        if trimmed.starts_with("index =") {
            let value: toml::Value = toml::from_str(trimmed)?;
            fan_index = value["index"]
                .as_str()
                .context("invalid repeat index")?
                .to_string();
        }
        // The inference the stage or repeat being read belongs to.
        let named = inference_name(&name, inferences);
        // Every bind stage is given its inference's three statements: range
        // 0's, and the template of the ranges after it.
        if let Some((inference, base)) = named {
            if matches!(
                (table.as_str(), base),
                ("[[chain.stage]]", "prompt_bind_p0")
                    | ("[[chain.repeat.stage]]", "prompt_bind_p{j}")
            ) {
                let files = &inferences[inference];
                let binding = if trimmed.starts_with("inputs.config =") {
                    Some(("config", files.config, 7))
                } else if trimmed.starts_with("inputs.schedule =") {
                    Some(("schedule", files.schedule, 8))
                } else if trimmed.starts_with("inputs.text_digest =") {
                    Some(("text_digest", files.text_digest, 9))
                } else {
                    None
                };
                if let Some((param, file, slot)) = binding {
                    replaced[inference][slot] += 1;
                    out.push(
                        RunExternal::new(&file.path, &file.index_path, &file.commitment)
                            .binding(param),
                    );
                    continue;
                }
            }
        }
        if let (("[[chain.stage]]", "source"), Some(source)) =
            ((table.as_str(), name.as_str()), source)
        {
            let binding = [
                ("signer", &source.signer),
                ("recipe", &source.recipe),
                ("documents", &source.documents),
            ]
            .into_iter()
            .find(|(param, _)| trimmed.starts_with(&format!("inputs.{param} =")));
            if let Some((param, file)) = binding {
                source_inputs += 1;
                out.push(
                    RunExternal::new(&file.path, &file.index_path, &file.commitment).binding(param),
                );
                continue;
            }
        }
        if (table.as_str(), name.as_str()) == ("[[chain.stage]]", "result")
            && trimmed.starts_with("inputs.forms =")
        {
            result_forms += 1;
            out.push(
                RunExternal::new(&forms.path, &forms.index_path, &forms.commitment)
                    .binding("forms"),
            );
            continue;
        }
        // A chain input's table is named for the inference it belongs to.
        let input = table
            .strip_prefix("[chain.input.")
            .and_then(|input| input.strip_suffix(']'))
            .and_then(|input| inference_name(input, inferences));
        let files = input.and_then(|(inference, base)| {
            let files = &inferences[inference];
            match base {
                "prompt_text" => Some((inference, files.text, 6)),
                "prompt_marks" => Some((inference, files.marks, 2)),
                "claimed_ids" => Some((inference, files.claimed, 3)),
                _ => None,
            }
        });
        if let Some((inference, files, slot)) = files {
            if let Some(key) = ["index_path", "path"]
                .into_iter()
                .find(|key| trimmed.starts_with(&format!("{key} =")))
            {
                // Keep the indexed file name (`marks_{j}.rastered`), move it
                // from the placeholder directory to this run's.
                let value: toml::Value = toml::from_str(trimmed)?;
                let placeholder = value[key].as_str().context("invalid input path")?;
                anyhow::ensure!(
                    placeholder.contains(RUN_PLACEHOLDER_DIR),
                    "input path {placeholder} is not a run placeholder"
                );
                let file_name = Path::new(placeholder)
                    .file_name()
                    .context("input path has no file name")?;
                let dir = files[0]
                    .path
                    .parent()
                    .context("run file has no directory")?;
                out.push(format!("{key} = {:?}", dir.join(file_name)));
                continue;
            }
            if trimmed.starts_with("commitments =") {
                replaced[inference][slot] += 1;
                out.push(String::from("commitments = ["));
                out.extend(
                    files
                        .iter()
                        .enumerate()
                        .map(|(index, file)| format!("  {:?}, # {index}", file.commitment)),
                );
                out.push(String::from("]"));
                continue;
            }
        }
        if table == "[[chain.repeat]]" && trimmed.starts_with("count =") {
            let (inference, base) =
                named.with_context(|| format!("unexpected repeat `{name}` in chain template"))?;
            let files = &inferences[inference];
            let range_count = match base {
                "prompt" => Some((0, files.marks.len())),
                "claimed_embed" | "claimed" => Some((1, files.claimed.len())),
                // Layer blocks keep their model-fixed counts.
                "layers" | "shared_layers" => None,
                _ => anyhow::bail!("unexpected repeat `{name}` in chain template"),
            };
            if let Some((slot, count)) = range_count {
                replaced[inference][slot] += 1;
                out.push(format!("count = {}", iterations(count, start)?));
                continue;
            }
        }
        // A layer block's inner fans run once per prompt or claimed range.
        if table == "[[chain.repeat.stage]]" && trimmed.starts_with("count =") {
            if let Some((inference, _)) = named {
                let files = &inferences[inference];
                let range_count = match fan_index.as_str() {
                    "j" => Some((4, files.marks.len())),
                    "k" => Some((5, files.claimed.len())),
                    _ => None,
                };
                if let Some((slot, count)) = range_count {
                    replaced[inference][slot] += 1;
                    out.push(format!("count = {}", iterations(count, start)?));
                    continue;
                }
            }
        }
        // Where the template names an inference's last prompt or claimed
        // range, by that inference's label.
        let mut line = line.to_string();
        for files in inferences {
            let label = files.label;
            line = line
                .replace(
                    &format!("@{label}{}@", LAST_PROMPT_RANGE.trim_matches('@')),
                    &(files.marks.len() - 1).to_string(),
                )
                .replace(
                    &format!("@{label}{}@", LAST_CLAIMED_RANGE.trim_matches('@')),
                    &(files.claimed.len() - 1).to_string(),
                );
        }
        out.push(line);
    }
    let filled = replaced.iter().all(|replaced| {
        replaced[..4] == [1, 2, 1, 1]
            && replaced[4] != 0
            && replaced[4] == replaced[5]
            && replaced[6..10] == [1, 2, 2, 2]
    });
    if !filled || source_inputs != if source.is_some() { 3 } else { 0 } || result_forms != 1 {
        anyhow::bail!("incompatible chain template: expected of each inference [prompt repeats, claimed repeats, prompt commitments, claimed commitments] = [1, 2, 1, 1], matching per-layer range fans, one set of text commitments, and a procedure, a schedule and a text digest binding for range 0's bind and for the later ranges'; and of the run three inputs for the source stage when it has a source, and the result stage's forms; found {replaced:?}, {source_inputs} and {result_forms}");
    }
    out.push(String::new());
    Ok(out.join("\n"))
}

/// How many times a repeat over `ranges` ranges runs when its index starts
/// at `start`.
fn iterations(ranges: usize, start: usize) -> Result<usize> {
    ranges
        .checked_sub(start)
        .with_context(|| format!("a repeat starts at range {start} of {ranges}"))
}

/// One run-specific external: the files and commitment a template
/// placeholder is replaced with.
pub struct RunExternal<'a> {
    path: &'a Path,
    index_path: &'a Path,
    commitment: &'a str,
}

impl<'a> RunExternal<'a> {
    pub fn new(path: &'a Path, index_path: &'a Path, commitment: &'a str) -> Self {
        Self {
            path,
            index_path,
            commitment,
        }
    }

    pub fn binding(&self, param: &str) -> String {
        format!(
            "inputs.{param} = {{ external = {{ path = {:?}, index_path = {:?}, commitment = {:?} }} }}",
            self.path, self.index_path, self.commitment,
        )
    }
}

fn absolute(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn run_id(specification_path: &Path, rendered_prompt: &str, max_new_tokens: u32) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let digest = Sha256::digest(format!(
        "{}\0{}\0{}\0{}",
        specification_path.display(),
        rendered_prompt,
        max_new_tokens,
        nanos
    ));
    hex::encode(digest)[..16].to_string()
}
