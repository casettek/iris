//! The native stages that state a procedure's result: what one
//! inference's output stands for, and whether two inferences agree.

use std::path::Path;

use anyhow::Result;

use super::StageVisitor;

pub mod claim_result;
pub mod consensus_result;

/// The result segment's stages. Each is named `result` in its pipeline: the
/// project tells them apart.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Stage {
    /// What the claim's output stands for: the last stage of a chain of one
    /// inference.
    ClaimResult,
    /// Whether two inferences agree: the last stage of a consensus chain.
    ConsensusResult,
}

impl Stage {
    /// The stage a pipeline names by this project and name, if it is this
    /// segment's.
    pub(super) fn from_spec(project: &str, name: &str) -> Result<Option<Self>> {
        Ok(match project {
            "claim-result" if name == "result" => Some(Self::ClaimResult),
            "consensus-result" if name == "result" => Some(Self::ConsensusResult),
            _ => None,
        })
    }

    /// The stage of this name, if it is this segment's. The name does not say
    /// which result it is: [`Stage::told_by_inputs`] does.
    pub(super) fn from_name(name: &str) -> Result<Option<Self>> {
        Ok((name == "result").then_some(Self::ClaimResult))
    }

    /// The stage its directory holds, where the name alone does not say: only
    /// a consensus takes two checks.
    pub(super) fn told_by_inputs(self, stage_dir: &Path) -> Self {
        if self == Self::ClaimResult
            && super::stage_inputs(stage_dir).is_some_and(|inputs| inputs.get("second").is_some())
        {
            return Self::ConsensusResult;
        }
        self
    }

    pub fn instance(&self) -> Option<usize> {
        None
    }

    pub(super) fn visit<V: StageVisitor>(&self, visitor: V) -> V::Out {
        match self {
            Self::ClaimResult => visitor.visit::<claim_result::Stage>(),
            Self::ConsensusResult => visitor.visit::<consensus_result::Stage>(),
        }
    }
}
