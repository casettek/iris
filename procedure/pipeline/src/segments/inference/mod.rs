//! The native stages that verify one inference, and what only they
//! need: the model's weights, its embedding and head tables, and the
//! tokenizer's identity.

use std::path::Path;

use anyhow::{Context, Result};

use super::StageVisitor;

pub mod input_embedding;
pub mod layer_mistral;
pub mod prefill_prepare_aux;
pub mod prefill_range;
pub mod prompt_bind;
pub mod prompt_check;
pub mod sequence_check;

pub(crate) mod embedding;
pub(crate) mod head;
pub(crate) mod ple;
pub(crate) mod prompt_identity;
pub(crate) mod verified_weights;
#[cfg(test)]
mod weight_tests;
pub mod weights;

/// The inference segment's stages.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Stage {
    PromptBind,
    PromptCheck,
    InputEmbedding,
    PrefillPrepareAux {
        layer: usize,
    },
    PrefillRange {
        layer: usize,
    },
    /// A `mistral-dense` layer over one range. Its stages are named as
    /// `PrefillRange`'s are: the project tells the two apart.
    LayerMistral {
        layer: usize,
    },
    SequenceCheck,
}

impl Stage {
    /// The stage a pipeline names by this project and name, if it is this
    /// segment's.
    pub(super) fn from_spec(project: &str, name: &str) -> Result<Option<Self>> {
        Ok(match project {
            "prompt-bind" if name.starts_with("prompt_bind_p") => Some(Self::PromptBind),
            "prompt-check" if name.starts_with("prompt_check_p") => Some(Self::PromptCheck),
            "input-embedding" if name.starts_with("embed_") => Some(Self::InputEmbedding),
            "prefill-prepare-aux" => Some(Self::PrefillPrepareAux {
                layer: parse_trailing_layer_index(name)?,
            }),
            "prefill-range" => Some(Self::PrefillRange {
                layer: parse_trailing_layer_index(name)?,
            }),
            "layer-mistral" => Some(Self::LayerMistral {
                layer: parse_trailing_layer_index(name)?,
            }),
            "sequence-check" if name == "sequence_check" || name.starts_with("check_") => {
                Some(Self::SequenceCheck)
            }
            _ => None,
        })
    }

    /// The stage of this name, if it is this segment's. A layer stage's name
    /// does not say which family's program it runs:
    /// [`Stage::told_by_inputs`] does.
    pub(super) fn from_name(name: &str) -> Result<Option<Self>> {
        Ok(match name {
            _ if name.starts_with("prompt_bind_p") => Some(Self::PromptBind),
            _ if name.starts_with("prompt_check_p") => Some(Self::PromptCheck),
            _ if name.starts_with("embed_") => Some(Self::InputEmbedding),
            "sequence_check" => Some(Self::SequenceCheck),
            _ if name.starts_with("check_") => Some(Self::SequenceCheck),
            _ if name.starts_with("aux_") => Some(Self::PrefillPrepareAux {
                layer: parse_trailing_layer_index(name)?,
            }),
            _ if name.starts_with("range_p") || name.starts_with("range_c") => {
                Some(Self::PrefillRange {
                    layer: parse_trailing_layer_index(name)?,
                })
            }
            _ => None,
        })
    }

    /// The stage its directory holds, where the name alone does not say: only
    /// `prefill-range` takes a per-layer input.
    pub(super) fn told_by_inputs(self, stage_dir: &Path) -> Self {
        if let Self::PrefillRange { layer } = self {
            // A directory whose inputs cannot be read is left to the name; the
            // stage's own loading reports what is wrong with them.
            if super::stage_inputs(stage_dir)
                .is_some_and(|inputs| inputs.is_object() && inputs.get("ple").is_none())
            {
                return Self::LayerMistral { layer };
            }
        }
        self
    }

    /// The layer a stage is of, for a stage that is of one.
    pub fn instance(&self) -> Option<usize> {
        match self {
            Self::PrefillPrepareAux { layer }
            | Self::PrefillRange { layer }
            | Self::LayerMistral { layer } => Some(*layer),
            Self::PromptBind | Self::PromptCheck | Self::InputEmbedding | Self::SequenceCheck => {
                None
            }
        }
    }

    pub(super) fn visit<V: StageVisitor>(&self, visitor: V) -> V::Out {
        match self {
            Self::PromptBind => visitor.visit::<prompt_bind::Stage>(),
            Self::PromptCheck => visitor.visit::<prompt_check::Stage>(),
            Self::InputEmbedding => visitor.visit::<input_embedding::Stage>(),
            Self::PrefillPrepareAux { .. } => visitor.visit::<prefill_prepare_aux::Stage>(),
            Self::PrefillRange { .. } => visitor.visit::<prefill_range::Stage>(),
            Self::LayerMistral { .. } => visitor.visit::<layer_mistral::Stage>(),
            Self::SequenceCheck => visitor.visit::<sequence_check::Stage>(),
        }
    }
}

fn parse_trailing_layer_index(name: &str) -> Result<usize> {
    let raw = name
        .rsplit_once("_l")
        .map(|(_, layer)| layer)
        .ok_or_else(|| anyhow::anyhow!("stage `{name}` has no trailing `_lN` layer index"))?;
    raw.parse::<usize>()
        .with_context(|| format!("failed to parse layer index from `{name}`"))
}
