//! A run's inputs as values: what its stages are given that neither an
//! earlier stage nor the procedure fixes.

use serde::{Deserialize, Serialize};

/// A run's inputs as a claimer posts them: each value a stage is given that
/// is neither an earlier stage's output nor the procedure's. A stage
/// loads each as a Raster file, which anyone writes from the value, and its
/// input list names it by that file's commitment.
///
/// The text is posted once, as the slices the binds are given: they
/// concatenate to the prompt. A source's documents are posted once, as the
/// chunks the source stage is given.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PostedRun {
    /// The source stage's inputs, in a run that has a source.
    pub source: Option<PostedSourceInputs>,
    /// The result stage's forms.
    pub forms: result_phase::ResultForms,
    /// One per inference, in the order the procedure kind runs them.
    pub inferences: Vec<PostedInference>,
    /// The relay stage's recipe, in a run of a procedure kind that relays an
    /// output.
    pub relay: Option<source_phase::relay::RelayRecipe>,
}

/// What the source stage is given: the key attestations are checked against,
/// the recipe, and the documents with their attestations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PostedSourceInputs {
    pub signer: source_phase::SignerKey,
    pub recipe: source_phase::Recipe,
    pub documents: source_phase::SourceDocuments,
}

/// What one inference's stages are given.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PostedInference {
    pub config: inference_tokenizer::RunConfig,
    pub schedule: inference_tokenizer::RunSchedule,
    /// What the run claims of the whole text.
    pub text_digest: inference_tokenizer::TextDigest,
    /// One slice of the text per prompt range.
    pub text: Vec<inference_tokenizer::PromptText>,
    /// One set of token marks per prompt range.
    pub marks: Vec<Vec<u32>>,
    /// One set of claimed output ids per claimed range.
    pub claimed: Vec<Vec<u32>>,
}

/// One of a run's inputs, with the stage input it is given as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunInput<'a> {
    Signer(&'a source_phase::SignerKey),
    Recipe(&'a source_phase::Recipe),
    Documents(&'a source_phase::SourceDocuments),
    Forms(&'a result_phase::ResultForms),
    Config(&'a inference_tokenizer::RunConfig),
    Schedule(&'a inference_tokenizer::RunSchedule),
    TextDigest(&'a inference_tokenizer::TextDigest),
    Text(&'a inference_tokenizer::PromptText),
    Marks(&'a [u32]),
    Claimed(&'a [u32]),
    RelayRecipe(&'a source_phase::relay::RelayRecipe),
}

impl RunInput<'_> {
    /// The name of the stage input this is given as.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Signer(_) => "signer",
            Self::Recipe(_) => "recipe",
            Self::Documents(_) => "documents",
            Self::Forms(_) => "forms",
            Self::Config(_) => "config",
            Self::Schedule(_) => "schedule",
            Self::TextDigest(_) => "text_digest",
            Self::Text(_) => "text",
            Self::Marks(_) => "marks",
            Self::Claimed(_) => "claimed",
            Self::RelayRecipe(_) => "recipe",
        }
    }
}

impl PostedRun {
    /// Every input of the run, in one order: the source's, the forms, then
    /// each inference's procedure, schedule, text digest, text slices,
    /// marks and claimed ids, then the relay's recipe.
    pub fn values(&self) -> Vec<RunInput<'_>> {
        let mut values = Vec::new();
        if let Some(source) = &self.source {
            values.push(RunInput::Signer(&source.signer));
            values.push(RunInput::Recipe(&source.recipe));
            values.push(RunInput::Documents(&source.documents));
        }
        values.push(RunInput::Forms(&self.forms));
        for inference in &self.inferences {
            values.push(RunInput::Config(&inference.config));
            values.push(RunInput::Schedule(&inference.schedule));
            values.push(RunInput::TextDigest(&inference.text_digest));
            values.extend(inference.text.iter().map(RunInput::Text));
            values.extend(inference.marks.iter().map(|marks| RunInput::Marks(marks)));
            values.extend(inference.claimed.iter().map(|ids| RunInput::Claimed(ids)));
        }
        if let Some(relay) = &self.relay {
            values.push(RunInput::RelayRecipe(relay));
        }
        values
    }
}
