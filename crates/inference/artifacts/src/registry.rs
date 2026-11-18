//! The configurations claims are made of, each registered before any claim
//! arrives and named by its id.
//!
//! A configuration is everything a contract holds a claim to: the pipeline
//! and its approved stage programs, the model of each inference, the
//! parameters, and what it requires of the claim's source and result. A claim
//! names the id of one ([`crate::contract::ClaimSubmission::configuration`]),
//! and a contract judges it by that one. Registering one is a choice made in
//! advance; no claim can bring its own.
//!
//! What a configuration does not pin yet: the base prompt and the domains a
//! source's documents may come from. A claim's statement still names the
//! recipe it used (through its source record), so it commits to both; a
//! configuration that pins them can come later.

use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::contract::{
    ContractPins, ModelPins, PipelinePins, RequiredOrigin, RequiredParameters, RequiredSource,
};
use crate::pipeline::Pipeline;

pub const CONFIGURATION_VERSION: u32 = 1;
pub const REGISTRY_VERSION: u32 = 1;
/// What `iris registry add` writes when given no other file.
pub const REGISTRY_JSON: &str = "registry.json";

/// One configuration a contract may accept claims of.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Configuration {
    pub version: u32,
    pub pipeline: Pipeline,
    pub programs: PipelinePins,
    /// One model per inference, in the order the pipeline runs them.
    pub models: Vec<ModelPins>,
    pub raw_prompt: bool,
    pub max_new_tokens: u32,
    /// The signer a source's documents must be signed by, or `None` for a
    /// prompt taken as given.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<RequiredSource>,
    /// The texts an output is read by, in order. May be empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub forms: Vec<source_core::Form>,
}

impl Configuration {
    pub fn of(pins: ContractPins, required: RequiredParameters, origin: RequiredOrigin) -> Self {
        Self {
            version: CONFIGURATION_VERSION,
            pipeline: pins.pipeline,
            programs: pins.programs,
            models: pins.models,
            raw_prompt: required.raw_prompt,
            max_new_tokens: required.max_new_tokens,
            source: origin.source,
            forms: origin.forms,
        }
    }

    /// The configuration's id: a hash over all of it, in lowercase hex.
    pub fn id(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"iris/configuration/v1\0");
        hasher.update(serde_json::to_vec(self).expect("a configuration encodes"));
        hex::encode(hasher.finalize())
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
        }
    }
}

/// The registry: every configuration registered, by id. The stand-in for the
/// on-chain one is a JSON file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Registry {
    pub version: u32,
    pub configurations: Vec<Registered>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Registered {
    pub id: String,
    pub configuration: Configuration,
}

impl Default for Registry {
    fn default() -> Self {
        Self {
            version: REGISTRY_VERSION,
            configurations: Vec::new(),
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
        crate::io::write_json(path, self)
    }

    /// Register a configuration and return its id. Registering one twice
    /// changes nothing.
    pub fn register(&mut self, configuration: Configuration) -> String {
        let id = configuration.id();
        if self.get(&id).is_none() {
            self.configurations.push(Registered {
                id: id.clone(),
                configuration,
            });
        }
        id
    }

    /// The configuration registered under this id. An entry whose
    /// configuration does not hash to its id is no entry.
    pub fn get(&self, id: &str) -> Option<&Configuration> {
        self.configurations
            .iter()
            .find(|entry| entry.id == id && entry.configuration.id() == id)
            .map(|entry| &entry.configuration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn configuration(max_new_tokens: u32) -> Configuration {
        Configuration {
            version: CONFIGURATION_VERSION,
            pipeline: Pipeline::Single,
            programs: PipelinePins {
                family: String::from("gemma-dense"),
                stage_programs: Vec::new(),
            },
            models: Vec::new(),
            raw_prompt: false,
            max_new_tokens,
            source: None,
            forms: Vec::new(),
        }
    }

    #[test]
    fn a_configuration_is_found_by_its_id_and_only_by_it() {
        let mut registry = Registry::default();
        let id = registry.register(configuration(8));
        assert_eq!(registry.register(configuration(8)), id);
        assert_eq!(registry.configurations.len(), 1);
        assert_eq!(registry.get(&id), Some(&configuration(8)));
        assert_ne!(configuration(9).id(), id);
        assert_eq!(registry.get(&configuration(9).id()), None);

        // An entry edited after it was registered no longer answers to the id.
        registry.configurations[0].configuration.max_new_tokens = 9;
        assert_eq!(registry.get(&id), None);
    }
}
