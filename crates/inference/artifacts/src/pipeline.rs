//! The pipelines iris runs.
//!
//! A pipeline is a named arrangement of the parts a claim is made of: an
//! optional source, one or more inferences, and the stage program that turns
//! their outputs into a result. A run spec names its pipeline, and so every
//! command that takes a run spec follows it: direct inference, claim build,
//! claim check and challenge build. A claim records the name in its frozen
//! run and in its submission, so a challenger and a contract know which
//! arrangement they are looking at.
//!
//! The name alone decides nothing a claimer could misstate. What a contract
//! holds is the shape: a claim of a pipeline must end in that pipeline's
//! result program, which is in the approved set of no other.
//!
//! A model import writes one inference's part of a chain: its chain inputs
//! and its stages, from the first bind to the last check, with the first
//! bind given the constant source. [`Pipeline::chain_template`] makes the
//! chain template of a run from one part per model: the `source` stage in
//! front, each part's first bind taking its record, and the pipeline's
//! result stage behind.
//!
//! See `docs/internals/pipelines.md`.

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

/// The stage that states what the prompt text was built from.
pub const SOURCE_STAGE: &str = "source";
/// The stage that states a pipeline's result.
pub const RESULT_STAGE: &str = "result";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Pipeline {
    /// One inference on one model, and the code of the form its output is.
    /// What a run spec that names no pipeline runs.
    #[default]
    Single,
    /// The same inference on two models, one after the other, and whether
    /// their outputs agree.
    Consensus,
}

impl Pipeline {
    /// Every pipeline, in the order they are listed to a user.
    pub const ALL: &'static [Pipeline] = &[Pipeline::Single, Pipeline::Consensus];

    /// The name a run spec, a prepared run and a submission use.
    pub fn name(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Consensus => "consensus",
        }
    }

    /// The stage program that ends a claim of this pipeline and states its
    /// result: the stage project's directory name.
    pub fn result_program(self) -> &'static str {
        match self {
            Self::Single => "claim-result",
            Self::Consensus => "consensus-result",
        }
    }

    /// The inputs of the result stage that are an inference's last check, in
    /// the order of the inferences.
    pub fn result_inputs(self) -> &'static [&'static str] {
        match self {
            Self::Single => &["check"],
            Self::Consensus => &["first", "second"],
        }
    }

    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// How many inferences a run of this pipeline makes: one per model.
    pub fn inferences(self) -> usize {
        match self {
            Self::Single => 1,
            Self::Consensus => 2,
        }
    }

    /// What marks a stage or a chain input as this inference's. Nothing in a
    /// pipeline of one inference, whose stages are named as an import names
    /// them.
    pub fn label(self, inference: usize) -> String {
        if self.inferences() == 1 {
            String::new()
        } else {
            inference_label(inference)
        }
    }

    /// The chain template of this pipeline over model imports: `parts` are
    /// the templates the imports wrote, one per inference, in order.
    ///
    /// The template always has the source stage. A run whose text is taken as
    /// given removes it when it renders its own chain.
    pub fn chain_template(self, parts: &[&str]) -> Result<String> {
        ensure!(
            parts.len() == self.inferences(),
            "the `{self}` pipeline makes {} inference(s), and {} model(s) were given",
            self.inferences(),
            parts.len()
        );
        let labels: Vec<String> = (0..parts.len()).map(|n| self.label(n)).collect();
        let stages_dir = stage_projects_dir(parts[0])?;
        // What the result stage reads of each inference: its last check.
        let checks: String = self
            .result_inputs()
            .iter()
            .zip(&labels)
            .map(|(input, label)| {
                format!("inputs.{input} = {{ from = \"{}\" }}\n", last_check(label))
            })
            .collect();
        let result = format!(
            "\n[[chain.stage]]\nname = \"{RESULT_STAGE}\"\n{}{checks}{}",
            project_line(&stages_dir, self.result_program()),
            run_input("forms", "result_forms"),
        );
        let source = format!(
            "\n[[chain.stage]]\nname = \"{SOURCE_STAGE}\"\n{}{}{}{}",
            project_line(&stages_dir, "prompt-source"),
            run_input("signer", "source_signer"),
            run_input("recipe", "source_recipe"),
            run_input("documents", "source_documents"),
        );
        assemble(parts, &labels, &source, &result)
    }
}

/// The label of inference `n` in a pipeline of several: `m0_`, `m1_`.
pub fn inference_label(inference: usize) -> String {
    format!("m{inference}_")
}

/// A stage name without the label of the inference it belongs to, so
/// `m1_check_c0` and `check_c0` name the same kind of stage.
pub fn unlabelled(stage: &str) -> &str {
    labelled_name(stage).map_or(stage, |(_, name)| name)
}

/// The inference a labelled name belongs to, and the name without its label.
/// `None` for a name with no label.
pub fn labelled_name(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix('m')?;
    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
    let inference = rest[..digits].parse().ok()?;
    Some((inference, rest[digits..].strip_prefix('_')?))
}

/// The last check of the inference with this label, as a template names it
/// before a run knows how many claimed ranges it has.
fn last_check(label: &str) -> String {
    format!("{label}check_c@{label}LAST_CLAIMED_RANGE@")
}

/// An input a run fills in when it renders its chain. Not a binding Raster
/// reads: a template that still holds one is not a chain yet.
fn run_input(param: &str, file: &str) -> String {
    format!("inputs.{param} = {{ run = \"{file}\" }}\n")
}

fn project_line(stages_dir: &str, program: &str) -> String {
    let project = if stages_dir.is_empty() {
        program.to_string()
    } else {
        format!("{}/{program}", stages_dir.trim_end_matches('/'))
    };
    format!("project = {project:?}\n")
}

/// Where the stage projects are, read from the first stage a part names:
/// they are side by side, and a pipeline's own stages are beside a model's.
fn stage_projects_dir(part: &str) -> Result<String> {
    let line = part
        .lines()
        .map(str::trim)
        .find_map(|line| line.strip_prefix("project ="))
        .context("the chain template of a model import names no stage project")?;
    let project: String = serde_json::from_str(line.trim())
        .with_context(|| format!("invalid stage project in a chain template: {line}"))?;
    Ok(project
        .rsplit_once('/')
        .map(|(dir, _)| dir.to_string())
        .unwrap_or_default())
}

/// One part with every name that is its own given the label: its chain
/// inputs, its stages and repeats, what its stages take from each other, and
/// the marks a run replaces with its range counts.
fn labelled(part: &str, label: &str) -> String {
    if label.is_empty() {
        return part.to_string();
    }
    let mut table = "";
    let mut out = Vec::new();
    for line in part.lines() {
        let trimmed = line.trim_start();
        let indent = &line[..line.len() - trimmed.len()];
        if trimmed.starts_with('[') {
            table = trimmed.trim_end();
        }
        let mut line = if let Some(name) = trimmed.strip_prefix("[chain.input.") {
            format!("{indent}[chain.input.{label}{name}")
        } else if table != "[chain]" && trimmed.starts_with("name = \"") {
            format!(
                "{indent}{}",
                trimmed.replacen("name = \"", &format!("name = \"{label}"), 1)
            )
        } else if trimmed.starts_with("inputs.") {
            let mut line = line.to_string();
            for key in ["from = \"", "first = \"", "input = \""] {
                line = line.replace(key, &format!("{key}{label}"));
            }
            line
        } else {
            line.to_string()
        };
        for mark in ["LAST_PROMPT_RANGE", "LAST_CLAIMED_RANGE"] {
            line = line.replace(&format!("@{mark}@"), &format!("@{label}{mark}@"));
        }
        out.push(line);
    }
    out.push(String::new());
    out.join("\n")
}

/// The whole template: the `[chain]` table, every part's chain inputs, the
/// source stage, every part's stages with its first bind taking the source
/// stage's record, and the result stage.
fn assemble(parts: &[&str], labels: &[String], source: &str, result: &str) -> Result<String> {
    let mut chain: Option<String> = None;
    let mut inputs = String::new();
    let mut stages = String::new();
    for (part, label) in parts.iter().zip(labels) {
        let part = labelled(part, label);
        // A part is its tables up to the first stage or repeat, then those.
        let cut = part
            .find("\n[[chain.")
            .context("the chain template of a model import has no stages")?;
        let (head, body) = part.split_at(cut);
        let at = head.find("\n[chain.input.").unwrap_or(head.len());
        let (table, part_inputs) = head.split_at(at);
        // The chain is named as the first import names it.
        if chain.is_none() {
            chain = Some(table.to_string());
        }
        inputs.push_str(part_inputs);

        let first_bind = format!("name = \"{label}prompt_bind_p0\"");
        let (mut in_first_bind, mut rebound) = (false, 0);
        for line in body.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('[') {
                in_first_bind = false;
            } else if trimmed == first_bind {
                in_first_bind = true;
            }
            if in_first_bind && trimmed.starts_with("inputs.source =") {
                rebound += 1;
                stages.push_str(&format!("inputs.source = {{ from = \"{SOURCE_STAGE}\" }}"));
            } else {
                stages.push_str(line);
            }
            stages.push('\n');
        }
        ensure!(
            rebound == 1,
            "the chain template of a model import gives its first bind {rebound} sources"
        );
    }
    Ok(format!(
        "{}{inputs}{source}{stages}{result}",
        chain.unwrap_or_default()
    ))
}

impl std::fmt::Display for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What an import writes, cut down: one chain input, range 0's bind and
    /// check, the repeat over later ranges, and the checks.
    const PART: &str = r#"[chain]
name = "iris"
version = "0.1.0"

[chain.input.prompt_text]
index = "j"
path = "target/iris/run-prompt-placeholder/text_{j}.rastered"
commitments = []

[[chain.stage]]
name = "prompt_bind_p0"
project = "raster-stages/prompt-bind"
inputs.text = { input = "prompt_text_0" }
inputs.source = { external = { path = "model/no_source.rastered", index_path = "model/no_source.rindex", commitment = "n0" } }

[[chain.repeat]]
name = "prompt"
index = "j"
start = 1
count = 0

  [[chain.repeat.stage]]
name = "prompt_bind_p{j}"
project = "raster-stages/prompt-bind"
inputs.text = { input = "prompt_text_{j}" }
inputs.source = { external = { path = "model/no_source.rastered", index_path = "model/no_source.rindex", commitment = "n0" } }
inputs.previous = { from = "prompt_bind_p{j-1}", first = "prompt_bind_p0" }

[[chain.stage]]
name = "check_c0"
project = "raster-stages/sequence-check"
inputs.prompt = { from = "prompt_bind_p@LAST_PROMPT_RANGE@" }
"#;

    #[test]
    fn the_single_pipeline_puts_a_source_before_its_inference_and_a_result_after() {
        let template = Pipeline::Single.chain_template(&[PART]).unwrap();
        let expected = PART
            .replacen(
                "\n[[chain.stage]]\nname = \"prompt_bind_p0\"",
                "\n[[chain.stage]]\nname = \"source\"\nproject = \"raster-stages/prompt-source\"\n\
                 inputs.signer = { run = \"source_signer\" }\n\
                 inputs.recipe = { run = \"source_recipe\" }\n\
                 inputs.documents = { run = \"source_documents\" }\n\
                 \n[[chain.stage]]\nname = \"prompt_bind_p0\"",
                1,
            )
            // Only range 0's bind takes the record: the later ones keep the
            // constant.
            .replacen(
                "inputs.source = { external = { path = \"model/no_source.rastered\", index_path = \"model/no_source.rindex\", commitment = \"n0\" } }",
                "inputs.source = { from = \"source\" }",
                1,
            )
            + "\n[[chain.stage]]\nname = \"result\"\nproject = \"raster-stages/claim-result\"\n\
               inputs.check = { from = \"check_c@LAST_CLAIMED_RANGE@\" }\n\
               inputs.forms = { run = \"result_forms\" }\n";
        assert_eq!(template, expected);
        // The stages a run does not fill in are a chain Raster's own reader
        // would take: the template is TOML.
        toml::from_str::<toml::Value>(&template).unwrap();

        assert!(Pipeline::Single.chain_template(&[]).is_err());
        assert!(Pipeline::Single.chain_template(&[PART, PART]).is_err());
        let no_bind = PART.replace("prompt_bind_p0", "bind_0");
        assert!(Pipeline::Single.chain_template(&[&no_bind]).is_err());
    }

    #[test]
    fn two_inferences_of_one_chain_keep_their_names_apart() {
        let labels = [inference_label(0), inference_label(1)];
        let template = assemble(
            &[PART, PART],
            &labels,
            "\n[[chain.stage]]\nname = \"source\"\n",
            "\n[[chain.stage]]\nname = \"result\"\n",
        )
        .unwrap();
        let chain: toml::Value = toml::from_str(&template).unwrap();
        // One `[chain]` table, and each inference's own inputs.
        assert_eq!(template.matches("[chain]\n").count(), 1);
        assert_eq!(chain["chain"]["name"].as_str(), Some("iris"));
        let inputs = chain["chain"]["input"].as_table().unwrap();
        assert_eq!(
            inputs.keys().collect::<Vec<_>>(),
            ["m0_prompt_text", "m1_prompt_text"]
        );

        // Every stage and repeat has its own name, and whatever a stage takes
        // from the chain is of its own inference, or the source.
        let stages = chain["chain"]["stage"].as_array().unwrap();
        let repeats = chain["chain"]["repeat"].as_array().unwrap();
        let mut names: Vec<&str> = stages
            .iter()
            .chain(repeats)
            .chain(
                repeats
                    .iter()
                    .flat_map(|repeat| repeat["stage"].as_array().unwrap()),
            )
            .map(|table| table["name"].as_str().unwrap())
            .collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "m0_check_c0",
                "m0_prompt",
                "m0_prompt_bind_p0",
                "m0_prompt_bind_p{j}",
                "m1_check_c0",
                "m1_prompt",
                "m1_prompt_bind_p0",
                "m1_prompt_bind_p{j}",
                "result",
                "source",
            ]
        );
        for label in &labels {
            let bind = stages
                .iter()
                .find(|stage| stage["name"].as_str() == Some(&format!("{label}prompt_bind_p0")))
                .unwrap();
            assert_eq!(bind["inputs"]["source"]["from"].as_str(), Some("source"));
            assert_eq!(
                bind["inputs"]["text"]["input"].as_str(),
                Some(format!("{label}prompt_text_0").as_str())
            );
            assert!(template.contains(&format!(
                "inputs.previous = {{ from = \"{label}prompt_bind_p{{j-1}}\", first = \"{label}prompt_bind_p0\" }}"
            )));
            assert!(template.contains(&format!(
                "inputs.prompt = {{ from = \"{label}prompt_bind_p@{label}LAST_PROMPT_RANGE@\" }}"
            )));
        }
        // The first inference's stages all come before the second's.
        assert!(
            template.find("m0_check_c0").unwrap() < template.find("m1_prompt_bind_p0").unwrap()
        );
    }

    #[test]
    fn a_label_is_taken_off_a_stage_name() {
        assert_eq!(Pipeline::Single.label(0), "");
        for (stage, name) in [
            ("m0_check_c0", "check_c0"),
            ("m12_range_p3_l{l}", "range_p3_l{l}"),
            ("check_c0", "check_c0"),
            ("source", "source"),
            ("m_check_c0", "m_check_c0"),
            ("m1check_c0", "m1check_c0"),
            ("embed_m1_x", "embed_m1_x"),
        ] {
            assert_eq!(unlabelled(stage), name);
        }
        assert_eq!(
            unlabelled(&format!("{}result", inference_label(3))),
            "result"
        );
        assert_eq!(labelled_name("m12_range_p3_l0"), Some((12, "range_p3_l0")));
        assert_eq!(labelled_name("range_p3_l0"), None);
    }

    #[test]
    fn a_pipeline_is_written_by_its_name() {
        for pipeline in Pipeline::ALL {
            let written = serde_json::to_string(pipeline).unwrap();
            assert_eq!(written, format!("{:?}", pipeline.name()));
            assert_eq!(
                serde_json::from_str::<Pipeline>(&written).unwrap(),
                *pipeline
            );
            assert_eq!(pipeline.to_string(), pipeline.name());
        }
        assert_eq!(Pipeline::default(), Pipeline::Single);
        // A name that is no pipeline is refused where it is read, and the
        // refusal says which names are.
        let error = serde_json::from_str::<Pipeline>("\"committee\"").unwrap_err();
        assert!(
            error.to_string().contains("single") && error.to_string().contains("consensus"),
            "{error}"
        );
    }

    #[test]
    fn the_consensus_pipeline_runs_two_inferences_into_one_result() {
        assert_eq!(Pipeline::Consensus.inferences(), 2);
        assert_eq!(
            [Pipeline::Consensus.label(0), Pipeline::Consensus.label(1)],
            ["m0_", "m1_"]
        );
        let template = Pipeline::Consensus.chain_template(&[PART, PART]).unwrap();
        let chain: toml::Value = toml::from_str(&template).unwrap();
        let stages = chain["chain"]["stage"].as_array().unwrap();
        let names: Vec<&str> = stages
            .iter()
            .map(|stage| stage["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            [
                "source",
                "m0_prompt_bind_p0",
                "m0_check_c0",
                "m1_prompt_bind_p0",
                "m1_check_c0",
                "result"
            ]
        );
        let result = stages.last().unwrap();
        assert_eq!(
            result["project"].as_str(),
            Some("raster-stages/consensus-result")
        );
        assert_eq!(
            result["inputs"]["first"]["from"].as_str(),
            Some("m0_check_c@m0_LAST_CLAIMED_RANGE@")
        );
        assert_eq!(
            result["inputs"]["second"]["from"].as_str(),
            Some("m1_check_c@m1_LAST_CLAIMED_RANGE@")
        );
        assert_eq!(
            result["inputs"]["forms"]["run"].as_str(),
            Some("result_forms")
        );
        // Both inferences are held to the one source.
        for bind in [&stages[1], &stages[3]] {
            assert_eq!(bind["inputs"]["source"]["from"].as_str(), Some("source"));
        }
        assert!(Pipeline::Consensus.chain_template(&[PART]).is_err());
    }
}
