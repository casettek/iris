//! The procedures claims are made of, each registered before any claim
//! arrives and named by its id.
//!
//! A procedure is everything a contract holds a claim to: the procedure kind
//! and its approved stage programs, the model of each inference, the
//! parameters, and what it requires of the claim's source and result. A claim
//! names the id of one ([`crate::checks::ClaimSubmission::procedure`]),
//! and a contract judges it by that one. Registering one is a choice made in
//! advance; no claim can bring its own.
//!
//! What a procedure does not pin yet: the base prompt and the domains a
//! source's documents may come from. A claim's statement still names the
//! recipe it used (through its source record), so it commits to both; a
//! procedure that pins them can come later.

use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::checks::{
    ContractPins, ModelPins, ProgramPins, RequiredOrigin, RequiredParameters, RequiredSource,
};
use procedure_definition::encoding::Encoder;
use procedure_definition::procedure_kind::ProcedureKind;

pub const PROCEDURE_VERSION: u32 = 4;
pub const REGISTRY_VERSION: u32 = 4;

/// How a procedure arranges its inferences into its result: what a contract
/// reads of the procedure kind, in place of its name.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Arrangement {
    /// The stage program that states the result.
    pub result_program: String,
    /// For each inference, the result stage's input that reads its last
    /// check. Empty for an inference the result does not read.
    pub result_inputs: Vec<String>,
    /// Whether the result states an agreement of its inferences.
    pub agreement: bool,
    /// For each inference, one more than the inference whose output its
    /// prompt holds. Zero for an inference whose prompt is the claim's own.
    pub relayed_from: Vec<u32>,
}

impl Arrangement {
    /// The arrangement of a procedure kind, from its shape.
    pub fn of(procedure_kind: ProcedureKind) -> Self {
        let inferences = 0..procedure_kind.inferences();
        Self {
            result_program: procedure_kind.result_program().to_string(),
            result_inputs: inferences
                .clone()
                .map(|inference| {
                    procedure_kind
                        .result_input_of(inference)
                        .unwrap_or_default()
                        .to_string()
                })
                .collect(),
            agreement: procedure_kind.result().states_agreement(),
            relayed_from: inferences
                .map(|inference| {
                    procedure_kind
                        .relayed_from(inference)
                        .map_or(0, |earlier| earlier as u32 + 1)
                })
                .collect(),
        }
    }
}
/// What `iris registry add` writes when given no other file.
pub const REGISTRY_JSON: &str = "registry.json";

/// One procedure a contract may accept claims of.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Procedure {
    pub version: u32,
    pub procedure_kind: ProcedureKind,
    /// The kind's arrangement, stated: a contract reads this, and the name
    /// is a label.
    pub arrangement: Arrangement,
    pub programs: ProgramPins,
    /// One model per inference, in the order the procedure kind runs them.
    pub models: Vec<ModelPins>,
    pub raw_prompt: bool,
    pub max_new_tokens: u32,
    /// The signer a source's documents must be signed by, or `None` for a
    /// prompt taken as given.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<RequiredSource>,
    /// The texts an output is read by, in order. May be empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub forms: Vec<result_core::Form>,
    /// The recipe a relay must build its prompt with, or `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relay_recipe_sha256: Option<String>,
}

impl Procedure {
    pub fn of(pins: ContractPins, required: RequiredParameters, origin: RequiredOrigin) -> Self {
        Self {
            version: PROCEDURE_VERSION,
            procedure_kind: pins.procedure_kind,
            arrangement: Arrangement::of(pins.procedure_kind),
            programs: pins.programs,
            models: pins.models,
            raw_prompt: required.raw_prompt,
            max_new_tokens: required.max_new_tokens,
            source: origin.source,
            forms: origin.forms,
            relay_recipe_sha256: origin.relay_recipe_sha256,
        }
    }

    /// The procedure's id: a hash over all of it, in lowercase hex, in
    /// the encoding a contract recomputes when one is registered
    /// (`procedure_definition::encoding`). Every field is hashed in the order it is
    /// declared, each list after its count.
    pub fn id(&self) -> String {
        let mut encoder = Encoder::new(b"iris/procedure/v4\0");
        encoder
            .number(self.version)
            .text(self.procedure_kind.name())
            .text(&self.programs.family)
            .count(self.programs.stage_programs.len());
        for program in &self.programs.stage_programs {
            encoder
                .text(&program.stage)
                .hash(&program.program_commitment)
                .count(program.piped_inputs.len());
            for input in &program.piped_inputs {
                encoder.text(&input.name).count(input.producers.len());
                for producer in &input.producers {
                    encoder.text(producer);
                }
                encoder.count(input.empty_values.len());
                for value in &input.empty_values {
                    encoder.hash(value);
                }
            }
            encoder.count(program.run_inputs.len());
            for input in &program.run_inputs {
                encoder.text(input);
            }
            encoder.count(program.pinned_inputs.len());
            for input in &program.pinned_inputs {
                encoder.text(&input.name).count(input.values.len());
                for value in &input.values {
                    encoder.hash(value);
                }
            }
        }
        encoder.count(self.models.len());
        for model in &self.models {
            encoder
                .hash(&model.model_id)
                .count(model.model_commitments.len());
            for commitment in &model.model_commitments {
                encoder.text(&commitment.role).hash(&commitment.commitment);
            }
            encoder.number(model.range_size);
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
        let arrangement = &self.arrangement;
        encoder
            .text(&arrangement.result_program)
            .count(arrangement.result_inputs.len());
        for input in &arrangement.result_inputs {
            encoder.text(input);
        }
        encoder
            .byte(u8::from(arrangement.agreement))
            .count(arrangement.relayed_from.len());
        for earlier in &arrangement.relayed_from {
            encoder.number(*earlier);
        }
        match &self.relay_recipe_sha256 {
            None => encoder.byte(0),
            Some(recipe) => encoder.byte(1).hash(recipe),
        };
        hex::encode(encoder.finish())
    }

    pub fn required(&self) -> RequiredParameters {
        RequiredParameters {
            raw_prompt: self.raw_prompt,
            max_new_tokens: self.max_new_tokens,
        }
    }

    pub fn origin(&self) -> RequiredOrigin {
        RequiredOrigin {
            source: self.source.clone(),
            forms: self.forms.clone(),
            relay_recipe_sha256: self.relay_recipe_sha256.clone(),
        }
    }
}

/// The registry: every procedure registered, by id. The stand-in for the
/// on-chain one is a JSON file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Registry {
    pub version: u32,
    pub procedures: Vec<Registered>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Registered {
    pub id: String,
    pub procedure: Procedure,
}

impl Default for Registry {
    fn default() -> Self {
        Self {
            version: REGISTRY_VERSION,
            procedures: Vec::new(),
        }
    }
}

impl Registry {
    /// The registry in this file, or an empty one when there is no file.
    pub fn read(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let registry: Self = serde_json::from_slice(
            &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", path.display()))?;
        if registry.version != REGISTRY_VERSION {
            bail!(
                "{} is registry version {}, and this build reads version {REGISTRY_VERSION}",
                path.display(),
                registry.version
            );
        }
        Ok(registry)
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        model_manifest::io::write_json(path, self)
    }

    /// Register a procedure and return its id. Registering one twice
    /// changes nothing.
    pub fn register(&mut self, procedure: Procedure) -> String {
        let id = procedure.id();
        if self.get(&id).is_none() {
            self.procedures.push(Registered {
                id: id.clone(),
                procedure,
            });
        }
        id
    }

    /// The procedure registered under this id. An entry whose
    /// procedure does not hash to its id is no entry. Neither is one whose
    /// arrangement is not its kind's in this build: the checks here follow
    /// the kind's shape, and a contract follows what was registered.
    pub fn get(&self, id: &str) -> Option<&Procedure> {
        self.procedures
            .iter()
            .find(|entry| {
                entry.id == id
                    && entry.procedure.id() == id
                    && entry.procedure.arrangement
                        == Arrangement::of(entry.procedure.procedure_kind)
            })
            .map(|entry| &entry.procedure)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn procedure(max_new_tokens: u32) -> Procedure {
        Procedure {
            version: PROCEDURE_VERSION,
            procedure_kind: ProcedureKind::single(),
            arrangement: Arrangement::of(ProcedureKind::single()),
            programs: ProgramPins {
                family: String::from("gemma-dense"),
                stage_programs: Vec::new(),
            },
            models: Vec::new(),
            raw_prompt: false,
            max_new_tokens,
            source: None,
            forms: Vec::new(),
            relay_recipe_sha256: None,
        }
    }

    #[test]
    fn an_arrangement_is_what_a_contract_reads_of_a_kind() {
        let single = Arrangement::of(ProcedureKind::single());
        assert_eq!(
            (single.result_program.as_str(), single.agreement),
            ("claim-result", false)
        );
        assert_eq!(single.result_inputs, ["check"]);
        assert_eq!(single.relayed_from, [0]);

        let consensus = Arrangement::of(ProcedureKind::named("consensus").unwrap());
        assert_eq!(consensus.result_program, "consensus-result");
        assert!(consensus.agreement);
        assert_eq!(consensus.result_inputs, ["first", "second"]);
        assert_eq!(consensus.relayed_from, [0, 0]);

        // A relay's result reads its second inference, whose prompt holds
        // the first one's output.
        let relay = Arrangement::of(ProcedureKind::named("relay").unwrap());
        assert_eq!(relay.result_program, "claim-result");
        assert!(!relay.agreement);
        assert_eq!(relay.result_inputs, ["", "check"]);
        assert_eq!(relay.relayed_from, [0, 1]);

        // The arrangement and the relay recipe are in the id, and an entry
        // with another arrangement than its kind's answers to nothing.
        let mut registry = Registry::default();
        let id = registry.register(procedure(8));
        let mut pinned = procedure(8);
        pinned.relay_recipe_sha256 = Some("ab".repeat(32));
        assert_ne!(pinned.id(), id);
        let mut rearranged = procedure(8);
        rearranged.arrangement = consensus;
        let other = registry.register(rearranged);
        assert_ne!(other, id);
        assert!(registry.get(&id).is_some());
        assert_eq!(registry.get(&other), None);
    }

    #[test]
    fn a_procedure_is_found_by_its_id_and_only_by_it() {
        let mut registry = Registry::default();
        let id = registry.register(procedure(8));
        assert_eq!(registry.register(procedure(8)), id);
        assert_eq!(registry.procedures.len(), 1);
        assert_eq!(registry.get(&id), Some(&procedure(8)));
        assert_ne!(procedure(9).id(), id);
        assert_eq!(registry.get(&procedure(9).id()), None);

        // An entry edited after it was registered no longer answers to the id.
        registry.procedures[0].procedure.max_new_tokens = 9;
        assert_eq!(registry.get(&id), None);
    }
}
