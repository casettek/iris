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
//! See `docs/internals/pipelines.md`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Pipeline {
    /// One inference on one model, and the code of the form its output is.
    /// What a run spec that names no pipeline runs.
    #[default]
    Single,
}

impl Pipeline {
    /// Every pipeline, in the order they are listed to a user.
    pub const ALL: &'static [Pipeline] = &[Pipeline::Single];

    /// The name a run spec, a prepared run and a submission use.
    pub fn name(self) -> &'static str {
        match self {
            Self::Single => "single",
        }
    }

    /// The stage program that ends a claim of this pipeline and states its
    /// result: the stage project's directory name.
    pub fn result_program(self) -> &'static str {
        match self {
            Self::Single => "claim-result",
        }
    }

    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

impl std::fmt::Display for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let error = serde_json::from_str::<Pipeline>("\"consensus\"").unwrap_err();
        assert!(error.to_string().contains("single"), "{error}");
    }
}
