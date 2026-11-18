//! The consensus result step: whether two inferences agree.
//!
//! A consensus run makes the same inference on two models. The step reads
//! each inference's output by the run's forms, as the single result step
//! does, and states whether the two stand for the same thing.
//!
//! The two outputs agree when both are the same form. A run that names no
//! forms has nothing to read an output by, and there they agree when they are
//! the same text once the whitespace around each is removed. The run's answer
//! is `agree`: the outputs agree, both claim records are valid, and both
//! inferences name the same input text and source. Its parts are stated beside
//! it, so a reader can tell which of them failed.
//!
//! Plain functions over a small state, one per step a stage takes, so direct
//! inference, the native stage and the tiles run one copy.
//!
//! See `docs/internals/pipelines.md`.

use alloc::string::String;
use serde::{Deserialize, Serialize};

use crate::forms::{begin_form_match, close_result, match_form, Form, FormMatch, ResultRecord};

/// What the consensus step carries from form to form.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ConsensusMatch {
    pub first: FormMatch,
    pub second: FormMatch,
    /// How many forms the outputs have been compared with.
    pub forms: u32,
    /// The two outputs are the same text without the whitespace around each.
    pub same_text: bool,
    /// The two inferences name the same input text and the same source.
    pub same_input: bool,
}

/// What the consensus step states: each inference's result, and whether the
/// two agree.
///
/// Small, so a tile may take it whole.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ConsensusRecord {
    /// The first inference's claim record and the code of its output.
    pub first: ResultRecord,
    /// The second inference's.
    pub second: ResultRecord,
    /// The two inferences name the same input text and the same source.
    pub same_input: bool,
    /// The run's answer: both records are valid, they name the same input,
    /// and the outputs agree.
    pub agree: bool,
    /// The code both outputs have, when they agree and the run names forms.
    /// Zero otherwise.
    pub code: u32,
}

/// Whether two inferences were given the same thing: `input` is the hash of
/// the input text and `source` the digest of its source, as each inference's
/// output record names them.
pub fn same_input(
    first_input: &str,
    first_source: &str,
    second_input: &str,
    second_source: &str,
) -> bool {
    first_input == second_input && first_source == second_source
}

pub fn begin_consensus(
    first_output: &str,
    second_output: &str,
    same_input: bool,
) -> ConsensusMatch {
    ConsensusMatch {
        first: begin_form_match(),
        second: begin_form_match(),
        forms: 0,
        same_text: first_output.trim() == second_output.trim(),
        same_input,
    }
}

/// Compares both outputs with one form.
pub fn match_both(
    mut matched: ConsensusMatch,
    form: &Form,
    first_output: &str,
    second_output: &str,
) -> ConsensusMatch {
    matched.first = match_form(matched.first, form, first_output);
    matched.second = match_form(matched.second, form, second_output);
    matched.forms = matched.forms.saturating_add(1);
    matched
}

/// Whether the two outputs stand for the same thing.
pub fn outputs_agree(matched: &ConsensusMatch) -> bool {
    if matched.forms == 0 {
        matched.same_text
    } else {
        matched.first.code != 0 && matched.first.code == matched.second.code
    }
}

/// One inference's claim record, as its last check stated it.
pub struct Claimed<'a> {
    pub statement: &'a [u8],
    pub output_text_sha256: &'a [u8],
    pub valid: bool,
}

/// States each inference's result and whether the two agree.
pub fn close_consensus(
    matched: ConsensusMatch,
    first: Claimed<'_>,
    second: Claimed<'_>,
) -> ConsensusRecord {
    let agree = first.valid && second.valid && matched.same_input && outputs_agree(&matched);
    let code = if agree { matched.first.code } else { 0 };
    ConsensusRecord {
        same_input: matched.same_input,
        agree,
        code,
        first: close_result(
            matched.first,
            first.statement,
            first.output_text_sha256,
            first.valid,
        ),
        second: close_result(
            matched.second,
            second.statement,
            second.output_text_sha256,
            second.valid,
        ),
    }
}

/// Every form, in order, as the stage compares them.
pub fn run_consensus<'a>(
    forms: impl IntoIterator<Item = &'a Form>,
    first_output: &str,
    second_output: &str,
    same_input: bool,
) -> ConsensusMatch {
    let mut matched = begin_consensus(first_output, second_output, same_input);
    for form in forms {
        matched = match_both(matched, form, first_output, second_output);
    }
    matched
}

/// The text a reader is shown for a record's answer.
pub fn consensus_description(record: &ConsensusRecord) -> String {
    String::from(if record.agree {
        "agree"
    } else if !record.first.valid || !record.second.valid {
        "no consensus: a claim record is not valid"
    } else if !record.same_input {
        "no consensus: the inferences name different inputs"
    } else {
        "disagree"
    })
}
