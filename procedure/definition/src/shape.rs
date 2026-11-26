//! A procedure's shape: its steps in order.
//!
//! Each step is a use of a segment kind ([`crate::segment_kind`]), named, with
//! the stage program it runs where its kind has several, and the earlier step
//! each of its inputs comes from. A shape is data: a file in `procedures/`,
//! compiled in. It is held to what the segment kinds declare when it is read,
//! so a shape a pipeline could not be made of is refused there and not at a
//! run.
//!
//! [`Shape::pipeline_template`] makes the pipeline of a shape over model
//! imports. Nothing else arranges stages.

use std::collections::BTreeMap;

use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;

use crate::procedure_kind::inference_label;
use crate::segment_kind::{ResultKind, SegmentKind, StageProgram, INFERENCE_TAKES};

/// A shape as its file writes it.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ShapeFile {
    #[serde(rename = "step")]
    steps: Vec<StepFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StepFile {
    name: String,
    segment: String,
    #[serde(default)]
    program: Option<String>,
    #[serde(default)]
    inputs: BTreeMap<String, String>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Shape {
    /// The procedure kind's name: its file's, without the extension.
    pub name: String,
    pub steps: Vec<Step>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Step {
    /// The step's name. A step of one stage gives its stage this name.
    pub name: String,
    pub segment: SegmentKind,
    /// The stage program, for a kind that declares its programs.
    pub program: Option<&'static StageProgram>,
    /// Each input taken from an earlier step, in the order its program takes
    /// them: the input, and the position of the step that gives it.
    pub inputs: Vec<(&'static str, usize)>,
}

impl Step {
    /// The position of the step this input comes from.
    pub fn input(&self, name: &str) -> Option<usize> {
        self.inputs
            .iter()
            .find(|(input, _)| *input == name)
            .map(|(_, from)| *from)
    }
}

impl Shape {
    /// Read a shape and hold it to what its segment kinds declare.
    pub fn parse(name: &str, text: &str) -> Result<Self> {
        ensure!(
            !name.is_empty()
                && name
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-'),
            "a procedure kind is named in lower-case letters, digits and `-`: `{name}`"
        );
        let file: ShapeFile =
            toml::from_str(text).with_context(|| format!("the `{name}` shape does not parse"))?;
        let mut steps: Vec<Step> = Vec::new();
        for step in file.steps {
            let at = format!("step `{}` of the `{name}` shape", step.name);
            ensure!(
                !step.name.is_empty()
                    && step
                        .name
                        .bytes()
                        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                "{at} is named in lower-case letters, digits and `_`"
            );
            ensure!(
                steps.iter().all(|earlier| earlier.name != step.name),
                "{at} has the name of an earlier step"
            );
            let segment = SegmentKind::named(&step.segment).with_context(|| {
                format!(
                    "{at} is of the segment kind `{}`, and the kinds are {}",
                    step.segment,
                    listed(SegmentKind::ALL.iter().map(|kind| kind.name()))
                )
            })?;
            let programs = segment.programs();
            let program = match (&step.program, programs) {
                (Some(named), []) => bail!(
                    "{at} names the program `{named}`, and the programs of the `{}` segment are \
                     its model's",
                    segment.name()
                ),
                (None, []) => None,
                (None, [only]) => Some(only),
                (None, _) => bail!(
                    "{at} names no program, and the `{}` segment has {}",
                    segment.name(),
                    listed(programs.iter().map(|program| program.name))
                ),
                (Some(named), _) => Some(segment.program(named).with_context(|| {
                    format!(
                        "{at} names the program `{named}`, and the `{}` segment has {}",
                        segment.name(),
                        listed(programs.iter().map(|program| program.name))
                    )
                })?),
            };
            // An inference may leave its source out. A program takes every
            // input it declares.
            let (takes, all) = match program {
                Some(program) => (program.takes, true),
                None => (INFERENCE_TAKES, false),
            };
            for given in step.inputs.keys() {
                ensure!(
                    takes.iter().any(|(input, _)| *input == given.as_str()),
                    "{at} is given the input `{given}`, and it takes {}",
                    listed(takes.iter().map(|(input, _)| *input))
                );
            }
            let mut inputs = Vec::new();
            for (input, wanted) in takes {
                let Some(from) = step.inputs.get(*input) else {
                    ensure!(!all, "{at} is not given its input `{input}`");
                    continue;
                };
                let position = steps
                    .iter()
                    .position(|earlier| earlier.name == *from)
                    .with_context(|| {
                        format!("{at} takes `{input}` from `{from}`, which is no earlier step")
                    })?;
                let given = steps[position].segment.gives();
                ensure!(
                    given == *wanted,
                    "{at} takes `{input}` from `{from}`, which gives {} and not {}",
                    given.name(),
                    wanted.name()
                );
                inputs.push((*input, position));
            }
            steps.push(Step {
                name: step.name,
                segment,
                program,
                inputs,
            });
        }
        let shape = Self {
            name: name.to_string(),
            steps,
        };
        shape.hold_arrangement()?;
        Ok(shape)
    }

    /// What every shape holds besides its wiring.
    fn hold_arrangement(&self) -> Result<()> {
        let name = &self.name;
        let of = |kind: SegmentKind| {
            self.steps
                .iter()
                .enumerate()
                .filter(move |(_, step)| step.segment == kind)
        };
        ensure!(
            of(SegmentKind::Inference).count() >= 1,
            "the `{name}` shape has no inference"
        );
        // A claim's last checkpoint is its result's, and its stage is found by
        // this name.
        let results: Vec<(usize, &Step)> = of(SegmentKind::Result).collect();
        ensure!(
            matches!(results.as_slice(), [(at, step)]
                if *at == self.steps.len() - 1 && step.name == crate::procedure_kind::RESULT_STAGE),
            "the `{name}` shape ends in its one result step, named `{}`",
            crate::procedure_kind::RESULT_STAGE
        );
        // A run prepares one source's inputs, and a run without a source
        // removes the stage by this name.
        let (relays, sources): (Vec<_>, Vec<_>) =
            of(SegmentKind::Source).partition(|(_, step)| !step.inputs.is_empty());
        ensure!(
            matches!(sources.as_slice(), [] | [(0, _)])
                && sources
                    .iter()
                    .all(|(_, step)| step.name == crate::procedure_kind::SOURCE_STAGE),
            "the `{name}` shape has at most one source step that takes nothing from an earlier \
             step, first and named `{}`",
            crate::procedure_kind::SOURCE_STAGE
        );
        // A run prepares one relay recipe, and the stage is found by this name.
        ensure!(
            relays.len() <= 1
                && relays
                    .iter()
                    .all(|(_, step)| step.name == crate::procedure_kind::RELAY_STAGE),
            "the `{name}` shape has at most one source step that takes an earlier step's \
             output, named `{}`",
            crate::procedure_kind::RELAY_STAGE
        );
        Ok(())
    }

    /// Whether a step of the shape runs this stage program.
    pub fn runs(&self, program: &str) -> bool {
        self.steps
            .iter()
            .any(|step| step.program.is_some_and(|run| run.name == program))
    }

    /// The inference whose output this inference's prompt holds: its source
    /// is a step that takes that inference's last check. `None` for an
    /// inference whose prompt is the run's own.
    pub fn relayed_from(&self, inference: usize) -> Option<usize> {
        let (_, step) = self.inference_steps().nth(inference)?;
        self.steps[step.input("source")?]
            .inputs
            .iter()
            .find_map(|(_, from)| self.inference_at(*from))
    }

    /// The steps that are inferences, in the order they run.
    pub fn inference_steps(&self) -> impl Iterator<Item = (usize, &Step)> {
        self.steps
            .iter()
            .enumerate()
            .filter(|(_, step)| step.segment == SegmentKind::Inference)
    }

    /// How many inferences a run makes: one per model.
    pub fn inferences(&self) -> usize {
        self.inference_steps().count()
    }

    /// Which inference a step is, counting the inferences in order.
    fn inference_at(&self, position: usize) -> Option<usize> {
        self.inference_steps().position(|(at, _)| at == position)
    }

    /// What marks a stage or a pipeline input as this inference's. Nothing in
    /// a shape of one inference, whose stages are named as an import names
    /// them.
    pub fn label(&self, inference: usize) -> String {
        if self.inferences() == 1 {
            String::new()
        } else {
            inference_label(inference)
        }
    }

    /// The last step: the result.
    pub fn result_step(&self) -> &Step {
        self.steps
            .last()
            .expect("a shape ends in its result step: it was held to that when read")
    }

    /// The result step's program.
    pub fn result_program(&self) -> &'static StageProgram {
        self.result_step()
            .program
            .expect("a result step has a program: it was held to that when read")
    }

    /// What the result states.
    pub fn result(&self) -> ResultKind {
        ResultKind::of_program(self.result_program().name)
            .expect("every program of the result segment is a result kind")
    }

    /// The result stage's inputs that are an inference's last check, each
    /// with the inference it reads, in the order the stage takes them.
    pub fn result_reads(&self) -> Vec<(&'static str, usize)> {
        self.result_step()
            .inputs
            .iter()
            .filter_map(|(input, from)| Some((*input, self.inference_at(*from)?)))
            .collect()
    }

    /// What a later stage names a step's output by, in a template.
    fn output_of(&self, position: usize) -> String {
        match self.inference_at(position) {
            Some(inference) => last_check(&self.label(inference)),
            None => self.steps[position].name.clone(),
        }
    }

    /// The pipeline template of this shape over model imports: `segments` are
    /// the templates the imports wrote, one per inference, in order.
    ///
    /// The template has every step. A run whose text is taken as given
    /// removes the source stage when it renders its own pipeline.
    pub fn pipeline_template(&self, segments: &[&str]) -> Result<String> {
        ensure!(
            segments.len() == self.inferences(),
            "the `{}` procedure kind makes {} inference(s), and {} model(s) were given",
            self.name,
            self.inferences(),
            segments.len()
        );
        let stages_dir = stage_projects_dir(segments[0])?;
        let mut chain: Option<String> = None;
        let mut inputs = String::new();
        let mut stages = String::new();
        for (position, step) in self.steps.iter().enumerate() {
            let Some(inference) = self.inference_at(position) else {
                stages.push_str(&self.stage_of(position, step, &stages_dir));
                continue;
            };
            let label = self.label(inference);
            let segment = labelled(segments[inference], &label);
            // A segment is its tables up to the first stage or repeat, then those.
            let cut = segment
                .find("\n[[chain.")
                .context("the chain template of a model import has no stages")?;
            let (head, body) = segment.split_at(cut);
            let at = head.find("\n[chain.input.").unwrap_or(head.len());
            let (table, segment_inputs) = head.split_at(at);
            // The pipeline is named as the first import names it.
            if chain.is_none() {
                chain = Some(table.to_string());
            }
            inputs.push_str(segment_inputs);

            // The first bind takes the record of the step this inference's
            // source is. An inference with none keeps its import's constant.
            let Some(source) = step.input("source") else {
                stages.push_str(body);
                if !body.ends_with('\n') {
                    stages.push('\n');
                }
                continue;
            };
            let source = self.output_of(source);
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
                    stages.push_str(&format!("inputs.source = {{ from = \"{source}\" }}"));
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
        Ok(format!("{}{inputs}{stages}", chain.unwrap_or_default()))
    }

    /// The one stage of a step whose kind declares its programs: its project,
    /// what it takes from earlier steps, and what a run fills in.
    fn stage_of(&self, position: usize, step: &Step, stages_dir: &str) -> String {
        let program = step
            .program
            .expect("a step that is no inference has a program: it was held to that when read");
        let mut stage = format!(
            "\n[[chain.stage]]\nname = \"{}\"\n{}",
            self.steps[position].name,
            project_line(stages_dir, program.name)
        );
        for (input, from) in &step.inputs {
            stage.push_str(&format!(
                "inputs.{input} = {{ from = \"{}\" }}\n",
                self.output_of(*from)
            ));
        }
        for (param, file) in program.run_inputs {
            stage.push_str(&run_input(param, file));
        }
        stage
    }
}

fn listed<'a>(names: impl Iterator<Item = &'a str>) -> String {
    names
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ")
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

/// Where the stage projects are, read from the first stage a segment names:
/// they are side by side, and a shape's own stages are beside a model's.
fn stage_projects_dir(segment: &str) -> Result<String> {
    let line = segment
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

/// One segment with every name that is its own given the label: its chain
/// inputs, its stages and repeats, what its stages take from each other, and
/// the marks a run replaces with its range counts.
fn labelled(segment: &str, label: &str) -> String {
    if label.is_empty() {
        return segment.to_string();
    }
    let mut table = "";
    let mut out = Vec::new();
    for line in segment.lines() {
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// What an import writes, cut down: one chain input, range 0's bind and
    /// check, the repeat over later ranges, and the checks.
    pub(crate) const PART: &str = r#"[chain]
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

    const SOURCE: &str =
        "[[step]]\nname = \"source\"\nsegment = \"source\"\nprogram = \"prompt-source\"\n";
    const RELAY: &str = "[[step]]\nname = \"relay\"\nsegment = \"source\"\n\
                         program = \"prompt-relay\"\ninputs.check = \"only\"\n";
    const RESULT: &str = "[[step]]\nname = \"result\"\nsegment = \"result\"\n\
                          program = \"claim-result\"\ninputs.check = \"only\"\n";

    fn inference(name: &str, source: bool) -> String {
        format!(
            "[[step]]\nname = \"{name}\"\nsegment = \"inference\"\n{}",
            if source {
                "inputs.source = \"source\"\n"
            } else {
                ""
            }
        )
    }

    fn refusal(text: &str) -> String {
        format!("{:#}", Shape::parse("tried", text).unwrap_err())
    }

    #[test]
    fn a_shape_is_its_steps_with_each_input_from_an_earlier_one() {
        let text = format!("{SOURCE}{}{RESULT}", inference("only", true));
        let shape = Shape::parse("tried", &text).unwrap();
        assert_eq!(shape.name, "tried");
        assert_eq!(shape.inferences(), 1);
        assert_eq!(shape.label(0), "");
        assert_eq!(shape.result(), ResultKind::OneOutput);
        assert_eq!(shape.result_reads(), [("check", 0)]);
        assert_eq!(shape.steps[0].program.unwrap().name, "prompt-source");
        assert!(shape.runs("prompt-source") && !shape.runs("prompt-relay"));
        assert_eq!(shape.relayed_from(0), None);
        assert_eq!(shape.steps[1].program, None);
        assert_eq!(shape.steps[1].input("source"), Some(0));
        assert_eq!(shape.steps[2].inputs, [("check", 1)]);
    }

    #[test]
    fn an_inference_with_no_source_keeps_the_constant_its_import_gave() {
        let text = format!("{}{RESULT}", inference("only", false));
        let shape = Shape::parse("tried", &text).unwrap();
        let template = shape.pipeline_template(&[PART]).unwrap();
        let expected = PART.to_string()
            + "\n[[chain.stage]]\nname = \"result\"\nproject = \"raster-stages/claim-result\"\n\
               inputs.check = { from = \"check_c@LAST_CLAIMED_RANGE@\" }\n\
               inputs.forms = { run = \"result_forms\" }\n";
        assert_eq!(template, expected);
    }

    #[test]
    fn a_shape_a_pipeline_cannot_be_made_of_is_refused_when_read() {
        let one = inference("only", true);
        let cases: Vec<(String, &str)> = vec![
            // No inference.
            (format!("{SOURCE}{RESULT}"), "no earlier step"),
            (SOURCE.to_string(), "has no inference"),
            // An input from a later step, or from a step of the wrong kind.
            (
                format!("{}{SOURCE}{RESULT}", inference("only", true)),
                "which is no earlier step",
            ),
            (
                format!("{SOURCE}{one}{}", RESULT.replace("\"only\"", "\"source\"")),
                "which gives a source record and not an inference's last check",
            ),
            // A program no kind has, none where several are, one where none is.
            (
                format!(
                    "{SOURCE}{one}{}",
                    RESULT.replace("claim-result", "majority-result")
                ),
                "names the program `majority-result`",
            ),
            (
                format!(
                    "{SOURCE}{one}{}",
                    RESULT.replace("program = \"claim-result\"\n", "")
                ),
                "names no program",
            ),
            (
                format!("{SOURCE}{one}program = \"prompt-bind\"\n{RESULT}"),
                "are its model's",
            ),
            // Three inferences into the result of two, and one input missing.
            (
                format!(
                    "{SOURCE}{one}{}{}{}",
                    inference("b", true),
                    inference("c", true),
                    RESULT.replace("claim-result", "consensus-result").replace(
                        "inputs.check = \"only\"",
                        "inputs.first = \"only\"\ninputs.second = \"b\"\ninputs.third = \"c\""
                    )
                ),
                "is given the input `third`",
            ),
            (
                format!(
                    "{SOURCE}{one}{}",
                    RESULT
                        .replace("claim-result", "consensus-result")
                        .replace("inputs.check = \"only\"", "inputs.first = \"only\"")
                ),
                "is not given its input `second`",
            ),
            // The arrangement: one result, last; one source, first.
            (format!("{SOURCE}{one}"), "ends in its one result step"),
            (
                format!(
                    "{SOURCE}{one}{}",
                    RESULT.replace("\"result\"\nsegment", "\"verdict\"\nsegment")
                ),
                "ends in its one result step",
            ),
            (
                format!("{one}{SOURCE}{RESULT}").replace("inputs.source = \"source\"\n", ""),
                "at most one source step that takes nothing from an earlier step, first",
            ),
            // A relay: one, named for what it is, and after the inference it
            // reads.
            (
                format!(
                    "{SOURCE}{one}{RELAY}{}{RESULT}",
                    RELAY.replace("\"relay\"", "\"relay_again\"")
                ),
                "at most one source step that takes an earlier step's output, named `relay`",
            ),
            (
                format!("{SOURCE}{RELAY}{one}{RESULT}"),
                "takes `check` from `only`, which is no earlier step",
            ),
            (
                format!(
                    "{SOURCE}{one}{}{RESULT}",
                    RELAY.replace("inputs.check = \"only\"\n", "")
                ),
                "is not given its input `check`",
            ),
            (
                format!(
                    "{SOURCE}{one}{}{RESULT}",
                    RELAY.replace("inputs.check = \"only\"", "inputs.check = \"source\"")
                ),
                "which gives a source record and not an inference's last check",
            ),
            // Names: a kind that is none, a step named twice, a key no step has.
            (
                format!("{SOURCE}{one}{RESULT}")
                    .replace("segment = \"source\"", "segment = \"evidence\""),
                "the kinds are `source`, `inference`, `result`",
            ),
            (
                format!("{SOURCE}{one}{}{RESULT}", inference("only", true)),
                "has the name of an earlier step",
            ),
            (
                format!("{SOURCE}{one}{RESULT}repeat = 2\n"),
                "does not parse",
            ),
        ];
        for (text, expected) in cases {
            let refusal = refusal(&text);
            assert!(
                refusal.contains(expected),
                "{expected:?} not in {refusal:?}"
            );
        }
        assert!(Shape::parse("Tried", &format!("{SOURCE}{one}{RESULT}")).is_err());
    }

    #[test]
    fn two_inferences_of_one_chain_keep_their_names_apart() {
        let text = format!(
            "{SOURCE}{}{}{}",
            inference("first", true),
            inference("second", true),
            RESULT.replace("claim-result", "consensus-result").replace(
                "inputs.check = \"only\"",
                "inputs.first = \"first\"\ninputs.second = \"second\""
            )
        );
        let shape = Shape::parse("tried", &text).unwrap();
        let labels = [shape.label(0), shape.label(1)];
        assert_eq!(labels, ["m0_", "m1_"]);
        let template = shape.pipeline_template(&[PART, PART]).unwrap();
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
    fn a_relay_puts_one_inferences_output_in_the_next_ones_prompt() {
        let text = format!(
            "{SOURCE}{}{}{}{}",
            inference("first", true),
            RELAY.replace("\"only\"", "\"first\""),
            inference("second", true)
                .replace("inputs.source = \"source\"", "inputs.source = \"relay\""),
            RESULT.replace("\"only\"", "\"second\"")
        );
        let shape = Shape::parse("tried", &text).unwrap();
        assert_eq!(shape.inferences(), 2);
        assert!(shape.runs("prompt-relay"));
        assert_eq!(
            [shape.relayed_from(0), shape.relayed_from(1)],
            [None, Some(0)]
        );
        // The result reads the second inference only.
        assert_eq!(shape.result_reads(), [("check", 1)]);

        let template = shape.pipeline_template(&[PART, PART]).unwrap();
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
                "relay",
                "m1_prompt_bind_p0",
                "m1_check_c0",
                "result"
            ]
        );
        let relay = &stages[3];
        assert_eq!(
            relay["project"].as_str(),
            Some("raster-stages/prompt-relay")
        );
        assert_eq!(
            relay["inputs"]["check"]["from"].as_str(),
            Some("m0_check_c@m0_LAST_CLAIMED_RANGE@")
        );
        assert_eq!(
            relay["inputs"]["recipe"]["run"].as_str(),
            Some("relay_recipe")
        );
        // The first inference's first bind takes the run's source, and the
        // second's takes the relay's record.
        assert_eq!(
            stages[1]["inputs"]["source"]["from"].as_str(),
            Some("source")
        );
        assert_eq!(
            stages[4]["inputs"]["source"]["from"].as_str(),
            Some("relay")
        );
        assert_eq!(
            stages[6]["inputs"]["check"]["from"].as_str(),
            Some("m1_check_c@m1_LAST_CLAIMED_RANGE@")
        );
    }
}
