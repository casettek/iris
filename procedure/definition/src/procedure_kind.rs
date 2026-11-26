//! The procedure kinds IWEBS runs.
//!
//! A procedure kind is a named arrangement of the segments a claim is made of: an
//! optional source, one or more inferences, and the stage program that turns
//! their outputs into a result. A specification names its procedure kind, and so every
//! command that takes a specification follows it: the native run, claim build,
//! claim check and challenge build. A claim records the name in its frozen
//! run and in its submission, so a challenger and a contract know which
//! arrangement they are looking at.
//!
//! The name alone decides nothing a claimer could misstate. What a contract
//! holds is the shape: a claim of a procedure kind must end in that procedure kind's
//! result program, which is in the approved set of no other.
//!
//! A kind is not written in code. Each is a file of `procedures/`, its shape
//! ([`crate::shape::Shape`]), compiled in, and a [`ProcedureKind`] is one of
//! them by name. Everything that depends on the arrangement asks the shape.
//!
//! A model import writes one inference's segment of a pipeline: its pipeline inputs
//! and its stages, from the first bind to the last check, with the first
//! bind given the constant source. [`ProcedureKind::pipeline_template`] makes the
//! pipeline template of a run from one segment per model, by the kind's shape.
//!
//! See `docs/protocol/procedure-kinds.md`.

use std::sync::OnceLock;

use anyhow::{Context, Result};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::segment_kind::ResultKind;
use crate::shape::Shape;

include!(concat!(env!("OUT_DIR"), "/catalog.rs"));

/// The stage that states what the prompt text was built from.
pub const SOURCE_STAGE: &str = "source";
/// The stage that states a procedure kind's result.
pub const RESULT_STAGE: &str = "result";
/// The stage that states what a prompt built around an earlier inference's
/// output is made of.
pub const RELAY_STAGE: &str = "relay";
/// The stage program of that stage.
pub const RELAY_PROGRAM: &str = "prompt-relay";
/// The kind a specification that names none is of.
pub const DEFAULT_KIND: &str = "single";

/// The shapes of `procedures/`, in name order. A file that is no shape stops
/// the program where the kinds are first asked for: the files are compiled
/// in, and the unit tests read every one.
fn catalog() -> &'static [Shape] {
    static SHAPES: OnceLock<Vec<Shape>> = OnceLock::new();
    SHAPES.get_or_init(|| {
        CATALOG
            .iter()
            .map(|(name, text)| {
                Shape::parse(name, text)
                    .unwrap_or_else(|error| panic!("procedures/{name}.toml: {error:#}"))
            })
            .collect()
    })
}

/// One of the procedure kinds of `procedures/`.
#[derive(Clone, Copy)]
pub struct ProcedureKind(&'static Shape);

impl ProcedureKind {
    /// Every procedure kind, in name order.
    pub fn all() -> Vec<ProcedureKind> {
        catalog().iter().map(ProcedureKind).collect()
    }

    /// The procedure kind of this name.
    pub fn named(name: &str) -> Result<Self> {
        catalog()
            .iter()
            .find(|shape| shape.name == name)
            .map(ProcedureKind)
            .with_context(|| {
                let names: Vec<String> = catalog()
                    .iter()
                    .map(|shape| format!("`{}`", shape.name))
                    .collect();
                format!(
                    "unknown procedure kind `{name}`, expected one of {}",
                    names.join(", ")
                )
            })
    }

    /// The kind a specification that names none is of.
    pub fn single() -> Self {
        Self::named(DEFAULT_KIND).expect("procedures/single.toml is compiled in")
    }

    /// The kind's steps in order.
    pub fn shape(self) -> &'static Shape {
        self.0
    }

    /// The name a specification, a prepared run and a submission use.
    pub fn name(self) -> &'static str {
        &self.0.name
    }

    /// What this procedure kind's result is.
    pub fn result(self) -> ResultKind {
        self.0.result()
    }

    /// The stage program that ends a claim of this procedure kind and states its
    /// result: the stage project's directory name.
    pub fn result_program(self) -> &'static str {
        self.0.result_program().name
    }

    /// The input of the result stage that is this inference's last check, or
    /// `None` for an inference the result stage does not read.
    pub fn result_input_of(self, inference: usize) -> Option<&'static str> {
        self.0
            .result_reads()
            .into_iter()
            .find(|(_, read)| *read == inference)
            .map(|(input, _)| input)
    }

    /// The inference whose output this inference's prompt holds, or `None`
    /// for an inference whose prompt is the run's own.
    pub fn relayed_from(self, inference: usize) -> Option<usize> {
        self.0.relayed_from(inference)
    }

    /// The inference whose output is the procedure's answer: the first the
    /// result stage reads.
    pub fn answer(self) -> usize {
        self.0
            .result_reads()
            .first()
            .map_or(0, |(_, inference)| *inference)
    }

    /// Whether a run of this kind has a relay recipe.
    pub fn relays(self) -> bool {
        self.0.runs(RELAY_PROGRAM)
    }

    pub fn is_default(&self) -> bool {
        self.name() == DEFAULT_KIND
    }

    /// How many inferences a run of this procedure kind makes: one per model.
    pub fn inferences(self) -> usize {
        self.0.inferences()
    }

    /// What marks a stage or a pipeline input as this inference's. Nothing in a
    /// procedure kind of one inference, whose stages are named as an import names
    /// them.
    pub fn label(self, inference: usize) -> String {
        self.0.label(inference)
    }

    /// The pipeline template of this procedure kind over model imports: `segments` are
    /// the templates the imports wrote, one per inference, in order.
    ///
    /// The template always has the source stage. A run whose text is taken as
    /// given removes it when it renders its own pipeline.
    pub fn pipeline_template(self, segments: &[&str]) -> Result<String> {
        self.0.pipeline_template(segments)
    }
}

impl Default for ProcedureKind {
    fn default() -> Self {
        Self::single()
    }
}

impl PartialEq for ProcedureKind {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl Eq for ProcedureKind {}

impl std::hash::Hash for ProcedureKind {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name().hash(state);
    }
}

impl std::fmt::Debug for ProcedureKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ProcedureKind({})", self.name())
    }
}

impl std::fmt::Display for ProcedureKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A kind is written by its name.
impl Serialize for ProcedureKind {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name())
    }
}

/// A name that is no procedure kind is refused where it is read, and the
/// refusal says which names are.
impl<'de> Deserialize<'de> for ProcedureKind {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Self::named(&name).map_err(serde::de::Error::custom)
    }
}

/// The label of inference `n` in a procedure kind of several: `m0_`, `m1_`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::tests::PART;

    fn consensus() -> ProcedureKind {
        ProcedureKind::named("consensus").unwrap()
    }

    #[test]
    fn every_file_of_the_catalog_is_a_shape() {
        // Reading the catalog parses every file, and stops on one that is no
        // shape.
        let names: Vec<&str> = ProcedureKind::all()
            .iter()
            .map(|kind| kind.name())
            .collect();
        assert!(names.contains(&DEFAULT_KIND), "{names:?}");
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(names, sorted);
        for kind in ProcedureKind::all() {
            assert_eq!(kind.shape().name, kind.name());
            assert_eq!(ProcedureKind::named(kind.name()).unwrap(), kind);
        }
    }

    #[test]
    fn the_single_procedure_kind_puts_a_source_before_its_inference_and_a_result_after() {
        let single = ProcedureKind::single();
        let template = single.pipeline_template(&[PART]).unwrap();
        let expected = PART
            .replacen(
                "\n[[pipeline.stage]]\nname = \"prompt_bind_p0\"",
                "\n[[pipeline.stage]]\nname = \"source\"\nproject = \"raster-stages/prompt-source\"\n\
                 inputs.signer = { run = \"source_signer\" }\n\
                 inputs.recipe = { run = \"source_recipe\" }\n\
                 inputs.documents = { run = \"source_documents\" }\n\
                 \n[[pipeline.stage]]\nname = \"prompt_bind_p0\"",
                1,
            )
            // Only range 0's bind takes the record: the later ones keep the
            // constant.
            .replacen(
                "inputs.source = { external = { path = \"model/no_source.rastered\", index_path = \"model/no_source.rindex\", commitment = \"n0\" } }",
                "inputs.source = { from = \"source\" }",
                1,
            )
            + "\n[[pipeline.stage]]\nname = \"result\"\nproject = \"raster-stages/claim-result\"\n\
               inputs.check = { from = \"check_c@LAST_CLAIMED_RANGE@\" }\n\
               inputs.forms = { run = \"result_forms\" }\n";
        assert_eq!(template, expected);
        // The stages a run does not fill in are a pipeline Raster's own reader
        // would take: the template is TOML.
        toml::from_str::<toml::Value>(&template).unwrap();

        assert!(single.pipeline_template(&[]).is_err());
        assert!(single.pipeline_template(&[PART, PART]).is_err());
        let no_bind = PART.replace("prompt_bind_p0", "bind_0");
        assert!(single.pipeline_template(&[&no_bind]).is_err());

        assert_eq!(single.inferences(), 1);
        assert_eq!(single.result(), ResultKind::OneOutput);
        assert_eq!(single.result_program(), "claim-result");
        assert_eq!(single.result_input_of(0), Some("check"));
        assert_eq!(single.result_input_of(1), None);
    }

    #[test]
    fn a_label_is_taken_off_a_stage_name() {
        assert_eq!(ProcedureKind::single().label(0), "");
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
    fn a_procedure_kind_is_written_by_its_name() {
        for procedure_kind in ProcedureKind::all() {
            let written = serde_json::to_string(&procedure_kind).unwrap();
            assert_eq!(written, format!("{:?}", procedure_kind.name()));
            assert_eq!(
                serde_json::from_str::<ProcedureKind>(&written).unwrap(),
                procedure_kind
            );
            assert_eq!(procedure_kind.to_string(), procedure_kind.name());
        }
        assert_eq!(ProcedureKind::default(), ProcedureKind::single());
        assert!(ProcedureKind::default().is_default());
        assert!(!consensus().is_default());
        // A name that is no procedure kind is refused where it is read, and the
        // refusal says which names are.
        let error = serde_json::from_str::<ProcedureKind>("\"committee\"").unwrap_err();
        assert!(
            error.to_string().contains("single") && error.to_string().contains("consensus"),
            "{error}"
        );
    }

    #[test]
    fn the_consensus_procedure_kind_runs_two_inferences_into_one_result() {
        let consensus = consensus();
        assert_eq!(consensus.inferences(), 2);
        assert_eq!(consensus.result(), ResultKind::Agreement);
        assert_eq!([consensus.label(0), consensus.label(1)], ["m0_", "m1_"]);
        assert_eq!(
            [consensus.result_input_of(0), consensus.result_input_of(1)],
            [Some("first"), Some("second")]
        );
        let template = consensus.pipeline_template(&[PART, PART]).unwrap();
        let pipeline: toml::Value = toml::from_str(&template).unwrap();
        let stages = pipeline["pipeline"]["stage"].as_array().unwrap();
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
        assert!(consensus.pipeline_template(&[PART]).is_err());
        assert!(!consensus.relays());
        assert_eq!(consensus.relayed_from(1), None);
    }

    #[test]
    fn the_relay_procedure_kind_gives_the_second_inference_the_firsts_output() {
        let relay = ProcedureKind::named("relay").unwrap();
        assert_eq!(relay.inferences(), 2);
        assert!(relay.relays());
        // Its answer is the second output, and the other kinds' the first.
        assert_eq!(relay.answer(), 1);
        assert_eq!(ProcedureKind::single().answer(), 0);
        assert_eq!(ProcedureKind::named("consensus").unwrap().answer(), 0);
        assert_eq!(
            [relay.relayed_from(0), relay.relayed_from(1)],
            [None, Some(0)]
        );
        // Its result is of the second output alone.
        assert_eq!(relay.result(), ResultKind::OneOutput);
        assert_eq!(
            [relay.result_input_of(0), relay.result_input_of(1)],
            [None, Some("check")]
        );
        let template = relay.pipeline_template(&[PART, PART]).unwrap();
        let at = |stage: &str| {
            template
                .find(&format!("name = \"{stage}\""))
                .unwrap_or_else(|| panic!("no stage `{stage}`"))
        };
        assert!(at("source") < at("m0_prompt_bind_p0"));
        assert!(at("m0_check_c0") < at(RELAY_STAGE));
        assert!(at(RELAY_STAGE) < at("m1_prompt_bind_p0"));
        assert!(at("m1_check_c0") < at(RESULT_STAGE));
    }
}
