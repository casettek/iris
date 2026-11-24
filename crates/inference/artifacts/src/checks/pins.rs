//! What a contract is configured with before any claim arrives: the programs
//! of a procedure kind and where each of their inputs comes from, the models, and
//! the parameters a claim must have run under. Read from a chain template
//! and the stage projects' locks.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use ::model_identity::context::parse_id;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::direct::{model_id_of, ModelCommitment, ModelManifest};
use crate::procedure_kind::ProcedureKind;

/// Written beside a model import's `manifest.json`.
pub const CONTRACT_PINS_JSON: &str = "contract_pins.json";

pub const CONTRACT_PINS_VERSION: u32 = 6;

/// What a contract pins before any claim arrives: the procedure kind it accepts
/// claims of, that procedure kind's stage programs, and the model of each inference
/// the procedure kind makes, in order.
///
/// A model import writes the pins of the `single` procedure kind over that model.
/// `iris claim pins` writes those of any procedure kind over the models a specification
/// names.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ContractPins {
    pub version: u32,
    pub procedure_kind: ProcedureKind,
    pub programs: ProgramPins,
    pub models: Vec<ModelPins>,
}

/// What is true of the stage programs whatever model they run: which
/// programs a claim may name, and which of their inputs must be the output of
/// an earlier stage.
///
/// The program ids are the ones locked in each stage's `Raster.lock` at
/// import time, so re-locking a stage means importing again.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProgramPins {
    /// The model family the stage programs belong to.
    pub family: String,
    /// The approved stage programs. A set: which stage runs where is the
    /// schedule's to enforce, inside the programs.
    pub stage_programs: Vec<StageProgram>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StageProgram {
    /// The stage project's directory name.
    pub stage: String,
    /// The program's identity from its `Raster.lock`, in lowercase hex.
    pub program_commitment: String,
    /// The inputs this program takes from earlier stages. Every other input
    /// is model data or a run input, which the program checks itself.
    pub chained_inputs: Vec<ChainedInput>,
    /// The inputs a run gives this program: the prompt, the claimed tokens,
    /// the run's configuration and schedule, the source and the forms. Each
    /// is made from the run's new information, the procedure and the
    /// model, so a challenger makes its own. One a challenger cannot make the
    /// same is a matter for a later step, in which the claimer posts it on
    /// request.
    #[serde(default)]
    pub run_inputs: Vec<String>,
    /// The inputs this program is given from fixed data: the model's, and
    /// whatever else an import writes once. The program checks what it loads
    /// against a pinned identity as it runs. A check inside a stage holds
    /// only for an input a challenger also has, so a checkpoint may name
    /// nothing for such an input but a commitment listed here.
    #[serde(default)]
    pub pinned_inputs: Vec<PinnedInput>,
}

/// One input a stage program is given from fixed data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PinnedInput {
    pub name: String,
    /// Every commitment the chain template binds the input to: one for an
    /// input every stage of the program takes the same way, and one per
    /// layer for an input that differs by layer.
    pub values: Vec<String>,
}

/// One input a stage program takes from an earlier stage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChainedInput {
    pub name: String,
    /// The stage programs whose output this input may be, by `stage` name.
    /// It keeps an output of the wrong shape out of a stage, which the stage
    /// could not decode and so could not record a fault about.
    pub producers: Vec<String>,
    /// The commitments of the empty values this input may hold in place of an
    /// earlier output, where the chain starts. The stages accept them only
    /// there.
    pub empty_values: Vec<String>,
}

/// What is true of one model: its identity and the range size it is imported
/// with.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelPins {
    /// The model id, in lowercase hex.
    pub model_id: String,
    /// What the model id is a hash of: every committed model input by role.
    pub model_commitments: Vec<ModelCommitment>,
    pub range_size: u32,
}

/// The inference parameters a contract requires besides the model's own. They
/// are the contract's choice, so no import records them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequiredParameters {
    pub raw_prompt: bool,
    pub max_new_tokens: u32,
}

impl ContractPins {
    /// The pins of an import: its model, and the procedure kind its chain template
    /// describes, with the program id each stage project has locked.
    pub fn from_import(manifest: &ModelManifest, segment: &str) -> Result<Self> {
        Self::of(ProcedureKind::Single, &[(manifest, segment)])
    }

    /// The pins of a procedure kind over imported models: each model as its import
    /// identifies it, and the stage programs of the procedure kind's chain over the
    /// imports' segments, with the program id each stage project has locked now.
    pub fn of(procedure_kind: ProcedureKind, imports: &[(&ModelManifest, &str)]) -> Result<Self> {
        let segments: Vec<&str> = imports.iter().map(|(_, segment)| *segment).collect();
        let template = procedure_kind.chain_template(&segments)?;
        let mut families: Vec<&str> = Vec::new();
        for (manifest, _) in imports {
            if !families.contains(&manifest.family.as_str()) {
                families.push(&manifest.family);
            }
        }
        Ok(Self {
            version: CONTRACT_PINS_VERSION,
            procedure_kind,
            programs: ProgramPins::from_template(families.join("+"), &template)?,
            models: imports
                .iter()
                .map(|(manifest, _)| ModelPins::of(manifest))
                .collect(),
        })
    }

    /// Why these are not the pins of this procedure kind over these models, if they
    /// are not: another version, another procedure kind, or other models.
    pub fn mismatch(
        &self,
        procedure_kind: ProcedureKind,
        models: &[&ModelManifest],
    ) -> Option<String> {
        if self.version != CONTRACT_PINS_VERSION {
            return Some(format!(
                "they are version {}, and this build reads version {CONTRACT_PINS_VERSION}",
                self.version
            ));
        }
        if self.procedure_kind != procedure_kind {
            return Some(format!(
                "they are of the `{}` procedure kind, not `{procedure_kind}`",
                self.procedure_kind
            ));
        }
        let same = self.models.len() == models.len()
            && self.models.iter().zip(models).all(|(pinned, manifest)| {
                pinned.model_id == manifest.model_id && pinned.range_size == manifest.range_size
            });
        (!same).then(|| String::from("they pin other models, or the same in another order"))
    }
}

impl ModelPins {
    /// What a contract pins of an imported model.
    pub fn of(manifest: &ModelManifest) -> Self {
        Self {
            model_id: manifest.model_id.clone(),
            model_commitments: manifest.model_commitments.clone(),
            range_size: manifest.range_size,
        }
    }
}

impl ModelPins {
    /// The model id is the hash of the commitments it is pinned with.
    pub fn is_consistent(&self) -> bool {
        parse_id(&self.model_id).is_some() && model_id_of(&self.model_commitments) == self.model_id
    }
}

impl ProgramPins {
    /// The stage programs a chain template names, with the program id each
    /// stage project has locked and the inputs each takes from earlier
    /// stages. `family` names the model families whose programs these are.
    pub fn from_template(family: String, template: &str) -> Result<Self> {
        let mut run_inputs = chain_run_inputs(template)?;
        let mut pinned_inputs = chain_pinned_inputs(template)?;
        let stage_programs = chain_wiring(template)?
            .into_iter()
            .map(|(project, chained_inputs)| {
                // A chained input's fixed values are its empty values.
                let pinned_inputs = pinned_inputs
                    .remove(&project)
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|(name, _)| !chained_inputs.iter().any(|input| input.name == *name))
                    .map(|(name, values)| PinnedInput {
                        name,
                        values: values.into_iter().collect(),
                    })
                    .collect();
                Ok(StageProgram {
                    stage: project_name(&project)?,
                    program_commitment: locked_program_commitment(&project)?,
                    chained_inputs,
                    run_inputs: run_inputs
                        .remove(&project)
                        .unwrap_or_default()
                        .into_iter()
                        .collect(),
                    pinned_inputs,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            family,
            stage_programs,
        })
    }

    /// Whether this program id is the approved program of this stage
    /// project.
    pub(super) fn runs(&self, program_commitment: &str, stage: &str) -> bool {
        self.program(program_commitment)
            .is_some_and(|program| program.stage == stage)
    }

    /// The approved program with this id, in lowercase hex.
    pub fn program(&self, program_commitment: &str) -> Option<&StageProgram> {
        self.stage_programs
            .iter()
            .find(|approved| approved.program_commitment == program_commitment)
    }

    pub fn approves(&self, program_commitment: &str) -> bool {
        self.program(program_commitment).is_some()
    }
}

impl StageProgram {
    pub fn chained_input(&self, name: &str) -> Option<&ChainedInput> {
        self.chained_inputs.iter().find(|input| input.name == name)
    }

    pub fn pinned_input(&self, name: &str) -> Option<&PinnedInput> {
        self.pinned_inputs.iter().find(|input| input.name == name)
    }
}

fn project_name(project: &Path) -> Result<String> {
    Ok(project
        .file_name()
        .and_then(|name| name.to_str())
        .with_context(|| format!("stage project {} has no name", project.display()))?
        .to_string())
}

/// A stage name with its indexes blanked, so `range_p3_l{l-1}` and
/// `range_p{j}_l{l}` name the same kind of stage. The label of the inference
/// a stage belongs to is kept as it is: two inferences of one chain may run
/// different programs for stages of one name.
fn stage_shape(name: &str) -> String {
    let (mut shape, name) = match crate::procedure_kind::labelled_name(name) {
        Some((inference, name)) => (crate::procedure_kind::inference_label(inference), name),
        None => (String::new(), name),
    };
    let mut chars = name.chars().peekable();
    while let Some(c) = chars.next() {
        let close = match c {
            '{' => Some('}'),
            '@' => Some('@'),
            _ => None,
        };
        if let Some(close) = close {
            for inner in chars.by_ref() {
                if inner == close {
                    break;
                }
            }
            shape.push('#');
        } else if c.is_ascii_digit() {
            while chars.peek().is_some_and(char::is_ascii_digit) {
                chars.next();
            }
            shape.push('#');
        } else {
            shape.push(c);
        }
    }
    shape
}

/// Each stage project a chain template names, with the inputs it takes from
/// earlier stages: for each, the projects that may produce it and the empty
/// values it holds where the chain starts. Projects are in the order the
/// template first names them.
///
/// An input is chained when any stage of the project binds it `from` another
/// stage. Where another stage of the same project binds it to a fixed
/// external instead, that external is the input's empty value.
pub fn chain_wiring(template: &str) -> Result<Vec<(PathBuf, Vec<ChainedInput>)>> {
    let document: toml::Value =
        toml::from_str(template).context("the chain template is not valid TOML")?;
    let chain = document
        .get("chain")
        .context("the chain template has no [chain] table")?;
    let stages_of = |table: &toml::Value| -> Vec<toml::Value> {
        ["stage", "stages"]
            .iter()
            .filter_map(|key| table.get(*key).and_then(toml::Value::as_array))
            .flatten()
            .cloned()
            .collect()
    };
    let mut stages = stages_of(chain);
    for repeat in chain
        .get("repeat")
        .and_then(toml::Value::as_array)
        .into_iter()
        .flatten()
    {
        stages.extend(stages_of(repeat));
    }

    let text = |stage: &toml::Value, key: &str| -> Result<String> {
        Ok(stage
            .get(key)
            .and_then(toml::Value::as_str)
            .with_context(|| format!("a chain stage has no `{key}`"))?
            .to_string())
    };
    // Which project runs each kind of stage.
    let mut project_of_shape = BTreeMap::new();
    let mut projects = Vec::new();
    for stage in &stages {
        let project = PathBuf::from(text(stage, "project")?);
        if !projects.contains(&project) {
            projects.push(project.clone());
        }
        project_of_shape.insert(stage_shape(&text(stage, "name")?), project);
    }

    let mut producers: BTreeMap<(PathBuf, String), BTreeSet<String>> = BTreeMap::new();
    let mut externals: BTreeMap<(PathBuf, String), BTreeSet<String>> = BTreeMap::new();
    for stage in &stages {
        let project = PathBuf::from(text(stage, "project")?);
        let inputs = stage.get("inputs").and_then(toml::Value::as_table);
        for (param, binding) in inputs.into_iter().flatten() {
            let key = (project.clone(), param.clone());
            for source in ["from", "first"] {
                let Some(producer) = binding.get(source).and_then(toml::Value::as_str) else {
                    continue;
                };
                let producer_project =
                    project_of_shape
                        .get(&stage_shape(producer))
                        .with_context(|| {
                            format!("the chain template binds `{param}` from `{producer}`, which is no stage of it")
                        })?;
                producers
                    .entry(key.clone())
                    .or_default()
                    .insert(project_name(producer_project)?);
            }
            if let Some(commitment) = binding
                .get("external")
                .and_then(|external| external.get("commitment"))
                .and_then(toml::Value::as_str)
            {
                externals
                    .entry(key)
                    .or_default()
                    .insert(commitment.to_string());
            }
        }
    }

    Ok(projects
        .into_iter()
        .map(|project| {
            let chained_inputs = producers
                .iter()
                .filter(|((of, _), _)| *of == project)
                .map(|(key, producers)| ChainedInput {
                    name: key.1.clone(),
                    producers: producers.iter().cloned().collect(),
                    empty_values: externals
                        .get(key)
                        .map(|values| values.iter().cloned().collect())
                        .unwrap_or_default(),
                })
                .collect();
            (project, chained_inputs)
        })
        .collect())
}

/// Where a model import's chain template points the files a run writes:
/// a run's chain points them at the run's own.
pub const RUN_PLACEHOLDER_DIR: &str = "run-prompt-placeholder/";

/// The inputs a run gives each stage project a chain template names: those
/// bound `{ run = ... }`, and those bound to an external or a chain input whose
/// file is a run's placeholder (see [`RUN_PLACEHOLDER_DIR`]). Everything else
/// a stage loads is an earlier output or model data.
pub fn chain_run_inputs(template: &str) -> Result<BTreeMap<PathBuf, BTreeSet<String>>> {
    let document: toml::Value =
        toml::from_str(template).context("the chain template is not valid TOML")?;
    let chain = document
        .get("chain")
        .context("the chain template has no [chain] table")?;
    let placeholder = |value: Option<&toml::Value>| {
        value
            .and_then(toml::Value::as_str)
            .is_some_and(|path| path.contains(RUN_PLACEHOLDER_DIR))
    };
    // The chain inputs whose files a run writes, by name.
    let run_chain_inputs: Vec<&String> = chain
        .get("input")
        .and_then(toml::Value::as_table)
        .into_iter()
        .flatten()
        .filter(|(_, input)| placeholder(input.get("path")))
        .map(|(name, _)| name)
        .collect();
    let stages_of = |table: &toml::Value| -> Vec<toml::Value> {
        ["stage", "stages"]
            .iter()
            .filter_map(|key| table.get(*key).and_then(toml::Value::as_array))
            .flatten()
            .cloned()
            .collect()
    };
    let mut stages = stages_of(chain);
    for repeat in chain
        .get("repeat")
        .and_then(toml::Value::as_array)
        .into_iter()
        .flatten()
    {
        stages.extend(stages_of(repeat));
    }
    let mut run_inputs: BTreeMap<PathBuf, BTreeSet<String>> = BTreeMap::new();
    for stage in &stages {
        let project = PathBuf::from(
            stage
                .get("project")
                .and_then(toml::Value::as_str)
                .context("a chain stage has no `project`")?,
        );
        let inputs = stage.get("inputs").and_then(toml::Value::as_table);
        for (param, binding) in inputs.into_iter().flatten() {
            let from_run = binding.get("run").is_some()
                || placeholder(
                    binding
                        .get("external")
                        .and_then(|external| external.get("path")),
                )
                || binding
                    .get("input")
                    .and_then(toml::Value::as_str)
                    .is_some_and(|name| {
                        run_chain_inputs.iter().any(|input| {
                            name.strip_prefix(input.as_str())
                                .is_some_and(|rest| rest.is_empty() || rest.starts_with('_'))
                        })
                    });
            if from_run {
                run_inputs
                    .entry(project.clone())
                    .or_default()
                    .insert(param.clone());
            }
        }
    }
    Ok(run_inputs)
}

/// The inputs each stage project a chain template names is given from fixed
/// data, with every commitment the template binds each to: an external whose
/// file is not a run's, or a chain input whose files are not a run's, which
/// lists a commitment per layer. These are the model's data and whatever
/// else an import writes once.
pub fn chain_pinned_inputs(
    template: &str,
) -> Result<BTreeMap<PathBuf, BTreeMap<String, BTreeSet<String>>>> {
    let document: toml::Value =
        toml::from_str(template).context("the chain template is not valid TOML")?;
    let chain = document
        .get("chain")
        .context("the chain template has no [chain] table")?;
    let from_run = |value: Option<&toml::Value>| {
        value
            .and_then(toml::Value::as_str)
            .is_some_and(|path| path.contains(RUN_PLACEHOLDER_DIR))
    };
    // The chain inputs whose files are fixed, each with its commitments.
    let fixed_chain_inputs: Vec<(&String, Vec<String>)> = chain
        .get("input")
        .and_then(toml::Value::as_table)
        .into_iter()
        .flatten()
        .filter(|(_, input)| !from_run(input.get("path")))
        .map(|(name, input)| {
            let commitments = input
                .get("commitments")
                .and_then(toml::Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(toml::Value::as_str)
                .map(String::from)
                .collect();
            (name, commitments)
        })
        .collect();
    let stages_of = |table: &toml::Value| -> Vec<toml::Value> {
        ["stage", "stages"]
            .iter()
            .filter_map(|key| table.get(*key).and_then(toml::Value::as_array))
            .flatten()
            .cloned()
            .collect()
    };
    let mut stages = stages_of(chain);
    for repeat in chain
        .get("repeat")
        .and_then(toml::Value::as_array)
        .into_iter()
        .flatten()
    {
        stages.extend(stages_of(repeat));
    }
    let mut pinned: BTreeMap<PathBuf, BTreeMap<String, BTreeSet<String>>> = BTreeMap::new();
    for stage in &stages {
        let project = PathBuf::from(
            stage
                .get("project")
                .and_then(toml::Value::as_str)
                .context("a chain stage has no `project`")?,
        );
        let inputs = stage.get("inputs").and_then(toml::Value::as_table);
        for (param, binding) in inputs.into_iter().flatten() {
            let mut values = Vec::new();
            if let Some(external) = binding.get("external") {
                if !from_run(external.get("path")) {
                    values.extend(
                        external
                            .get("commitment")
                            .and_then(toml::Value::as_str)
                            .map(String::from),
                    );
                }
            }
            if let Some(name) = binding.get("input").and_then(toml::Value::as_str) {
                for (input, commitments) in &fixed_chain_inputs {
                    let named = name
                        .strip_prefix(input.as_str())
                        .is_some_and(|rest| rest.is_empty() || rest.starts_with('_'));
                    if named {
                        values.extend(commitments.iter().cloned());
                    }
                }
            }
            if !values.is_empty() {
                pinned
                    .entry(project.clone())
                    .or_default()
                    .entry(param.clone())
                    .or_default()
                    .extend(values);
            }
        }
    }
    Ok(pinned)
}

/// The program identity a stage project's `Raster.lock` records.
pub fn locked_program_commitment(stage_project: &Path) -> Result<String> {
    #[derive(Deserialize)]
    struct Lock {
        program_commitment: String,
    }
    let path = stage_project.join("Raster.lock");
    let lock: Lock = serde_json::from_slice(
        &fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))?;
    anyhow::ensure!(
        parse_id(&lock.program_commitment).is_some(),
        "{} does not record a 32-byte program commitment",
        path.display()
    );
    Ok(lock.program_commitment)
}
