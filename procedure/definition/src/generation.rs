//! The procedures that find an output.
//!
//! A generation procedure runs each model forward and takes each token as
//! the model's own choice. It is not the procedure that verifies an output
//! stated beforehand ([`crate::registry::Procedure`]): the two are different
//! logic, they have different kinds, told apart by name, and each has its own
//! id. Nothing here pairs one with the other. Whoever needs an output
//! verified runs a generation procedure first and a verification procedure
//! on what it found.
//!
//! A generation procedure runs natively and has no stage programs. Its id is
//! a hash over its kind, its models, its parameters and what it requires of a
//! run's source and result, with the version of the generating logic.

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

use crate::encoding::Encoder;
use crate::pins::{RequiredOrigin, RequiredParameters, RequiredSource};
use crate::procedure_kind::ProcedureKind;

/// The version of a generation procedure: of this format, and of the logic
/// that finds an output. It goes up when either changes, so an id never names
/// two logics.
pub const GENERATION_PROCEDURE_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GenerationProcedure {
    pub version: u32,
    /// A kind that generates.
    pub procedure_kind: ProcedureKind,
    /// One model per generation, in the order the kind runs them: its id.
    pub models: Vec<String>,
    pub raw_prompt: bool,
    pub max_new_tokens: u32,
    /// The signer a source's documents must be signed by, or `None` for a
    /// prompt taken as given.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<RequiredSource>,
    /// The texts an output is read by, in order. May be empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub forms: Vec<result_phase::Form>,
    /// The recipe a relay builds its prompt with, or `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relay_recipe_sha256: Option<String>,
}

/// A generation procedure with its id, as a record holds one.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RegisteredGeneration {
    pub id: String,
    pub procedure: GenerationProcedure,
}

impl GenerationProcedure {
    /// The generation procedure of a kind over models, by their ids.
    pub fn of(
        procedure_kind: ProcedureKind,
        models: Vec<String>,
        required: RequiredParameters,
        origin: RequiredOrigin,
    ) -> Result<Self> {
        ensure!(
            procedure_kind.generates(),
            "the `{procedure_kind}` procedure kind verifies a stated output: its procedure is \
             not a generation procedure"
        );
        ensure!(
            models.len() == procedure_kind.inferences(),
            "the `{procedure_kind}` procedure kind runs {} model(s), and {} were given",
            procedure_kind.inferences(),
            models.len()
        );
        Ok(Self {
            version: GENERATION_PROCEDURE_VERSION,
            procedure_kind,
            models,
            raw_prompt: required.raw_prompt,
            max_new_tokens: required.max_new_tokens,
            source: origin.source,
            forms: origin.forms,
            relay_recipe_sha256: origin.relay_recipe_sha256,
        })
    }

    /// The procedure's id: a hash over all of it, in lowercase hex. Every
    /// field is hashed in the order it is declared, each list after its
    /// count.
    pub fn id(&self) -> String {
        let mut encoder = Encoder::new(b"iwebs/generation-procedure/v1\0");
        encoder
            .number(self.version)
            .text(self.procedure_kind.name())
            .count(self.models.len());
        for model in &self.models {
            encoder.hash(model);
        }
        encoder
            .byte(u8::from(self.raw_prompt))
            .number(self.max_new_tokens);
        match &self.source {
            None => {
                encoder.byte(0);
            }
            Some(source) => {
                encoder.byte(1).text(&source.signer);
                match &source.recipe_sha256 {
                    None => encoder.byte(0),
                    Some(recipe) => encoder.byte(1).hash(recipe),
                };
            }
        }
        encoder.count(self.forms.len());
        for form in &self.forms {
            encoder.text(&form.text).number(form.code);
        }
        match &self.relay_recipe_sha256 {
            None => encoder.byte(0),
            Some(recipe) => encoder.byte(1).hash(recipe),
        };
        hex::encode(encoder.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn procedure(kind: &str, models: &[&str]) -> Result<GenerationProcedure> {
        GenerationProcedure::of(
            ProcedureKind::named(kind).unwrap(),
            models.iter().map(|model| model.repeat(32)).collect(),
            RequiredParameters {
                raw_prompt: false,
                max_new_tokens: 3,
            },
            RequiredOrigin::default(),
        )
    }

    #[test]
    fn a_generation_procedure_is_named_by_its_kind_its_models_and_its_terms() {
        let one = procedure("generate-single", &["ab"]).unwrap();
        assert_eq!(
            one.id(),
            procedure("generate-single", &["ab"]).unwrap().id()
        );
        assert_eq!(one.id().len(), 64);

        // Another model, another budget or another answer is another procedure.
        assert_ne!(
            one.id(),
            procedure("generate-single", &["cd"]).unwrap().id()
        );
        let mut longer = one.clone();
        longer.max_new_tokens = 4;
        assert_ne!(one.id(), longer.id());
        let mut answered = one.clone();
        answered.forms.push(result_phase::Form {
            text: String::from("yes"),
            code: 1,
        });
        assert_ne!(one.id(), answered.id());

        // So is another kind over the same models.
        let two = procedure("generate-consensus", &["ab", "ab"]).unwrap();
        let relayed = procedure("generate-relay", &["ab", "ab"]).unwrap();
        assert_ne!(two.id(), relayed.id());

        // It reads back as written.
        let text = serde_json::to_string(&one).unwrap();
        assert_eq!(
            serde_json::from_str::<GenerationProcedure>(&text).unwrap(),
            one
        );
    }

    #[test]
    fn a_kind_that_verifies_has_no_generation_procedure() {
        let error = procedure("verify-single", &["ab"]).unwrap_err();
        assert!(
            error.to_string().contains("verifies a stated output"),
            "{error}"
        );
        // A kind runs one model per generation.
        let error = procedure("generate-consensus", &["ab"]).unwrap_err();
        assert!(error.to_string().contains("runs 2 model(s)"), "{error}");
    }

    #[test]
    fn every_kind_that_verifies_is_beside_one_that_generates() {
        // The two sets are told apart by name, and nothing pairs them in the
        // code. This holds only that both sets are there.
        let kinds = ProcedureKind::all();
        for kind in &kinds {
            let name = kind.name();
            assert_eq!(
                kind.generates(),
                name.starts_with("generate-"),
                "`{name}` is not named for what it does"
            );
            assert!(
                kind.generates() || name.starts_with("verify-"),
                "`{name}` is not named for what it does"
            );
        }
        assert!(kinds.iter().any(|kind| kind.generates()));
        assert!(kinds.iter().any(|kind| !kind.generates()));
    }
}
