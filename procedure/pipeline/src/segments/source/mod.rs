//! The native stages that say what a prompt's text was built from.

use std::path::Path;

use anyhow::Result;

use super::StageVisitor;

pub mod prompt_relay;
pub mod prompt_source;

/// The name a pipeline gives each of the segment's stages.
const SOURCE: &str = procedure_definition::procedure_kind::SOURCE_STAGE;
const RELAY: &str = procedure_definition::procedure_kind::RELAY_STAGE;

/// The source segment's stages.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Stage {
    /// What the prompt text was built from, for a run that has a source.
    PromptSource,
    /// What a prompt that holds an earlier inference's output was built
    /// from.
    PromptRelay,
}

impl Stage {
    /// The stage a pipeline names by this project and name, if it is this
    /// segment's.
    pub(super) fn from_spec(project: &str, name: &str) -> Result<Option<Self>> {
        Ok(match project {
            "prompt-source" if name == SOURCE => Some(Self::PromptSource),
            "prompt-relay" if name == RELAY => Some(Self::PromptRelay),
            _ => None,
        })
    }

    /// The stage of this name, if it is this segment's.
    pub(super) fn from_name(name: &str) -> Result<Option<Self>> {
        Ok(match name {
            SOURCE => Some(Self::PromptSource),
            RELAY => Some(Self::PromptRelay),
            _ => None,
        })
    }

    /// The stage its directory holds, where the name alone does not say.
    pub(super) fn told_by_inputs(self, _stage_dir: &Path) -> Self {
        self
    }

    pub fn instance(&self) -> Option<usize> {
        None
    }

    pub(super) fn visit<V: StageVisitor>(&self, visitor: V) -> V::Out {
        match self {
            Self::PromptSource => visitor.visit::<prompt_source::Stage>(),
            Self::PromptRelay => visitor.visit::<prompt_relay::Stage>(),
        }
    }
}
