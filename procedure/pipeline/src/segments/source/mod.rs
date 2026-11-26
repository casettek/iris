//! The native stage that says what a prompt's text was built from.

use std::path::Path;

use anyhow::Result;

use super::StageVisitor;

pub mod prompt_source;

/// The source segment's stages.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Stage {
    /// What the prompt text was built from, for a run that has a source.
    PromptSource,
}

impl Stage {
    /// The stage a pipeline names by this project and name, if it is this
    /// segment's.
    pub(super) fn from_spec(project: &str, name: &str) -> Result<Option<Self>> {
        Ok(match project {
            "prompt-source" if name == "source" => Some(Self::PromptSource),
            _ => None,
        })
    }

    /// The stage of this name, if it is this segment's.
    pub(super) fn from_name(name: &str) -> Result<Option<Self>> {
        Ok((name == "source").then_some(Self::PromptSource))
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
        }
    }
}
