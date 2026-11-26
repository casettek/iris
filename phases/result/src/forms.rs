//! The result step: what a claim's output text stands for.
//!
//! The step compares the output with a list of forms, each a text and a code,
//! and states the code of the first form the output is. It states it beside
//! the claim record it was given, so a reader can tell which claim the code
//! belongs to.
//!
//! Plain functions over a small state, one per step a stage takes, so native
//! inference, the native stage and the tiles run one copy. This file stands
//! alone: the result stage compiles it without the source step's.
//!
//! See `docs/proposals/authenticated-prompt.md`.

use alloc::string::String;
use alloc::vec::Vec;
use raster::{BytesPage, List};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The longest form text, in bytes. A longer one matches nothing.
pub const MAX_FORM: usize = 1024;

/// An output text the result step recognizes, and what it stands for.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct Form {
    pub text: String,
    /// Not zero: zero is what an unrecognized output gets.
    pub code: u32,
}

/// The forms a run's output is compared with, in order. May be empty.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ResultForms {
    pub forms: List<Form>,
}

/// What the result step carries from form to form.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct FormMatch {
    pub code: u32,
    /// A hash chained over every form so far: 32 bytes.
    pub forms: BytesPage,
}

/// What the result step states: the claim record it was given, the code of
/// the first form the output is, and which forms it was compared with.
///
/// Small, so a tile may take it whole.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ResultRecord {
    /// The statement of the claim record the step was given: 32 bytes.
    pub statement: BytesPage,
    /// That record's commitment to the output text: 32 bytes.
    pub output_text_sha256: BytesPage,
    /// That record's verdict.
    pub valid: bool,
    /// The code of the first form the output is, or zero.
    pub code: u32,
    /// A hash over every form the output was compared with, in order, in
    /// lowercase hex.
    pub forms_sha256: String,
}

fn page(bytes: Vec<u8>) -> BytesPage {
    BytesPage::__from_parts(0, 0, bytes)
}

fn lower_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 15) as usize] as char);
    }
    out
}

pub fn begin_form_match() -> FormMatch {
    FormMatch {
        code: 0,
        forms: page([0_u8; 32].to_vec()),
    }
}

/// Compares the output with one form. The output is compared without the
/// whitespace around it. The first form that matches gives the code.
pub fn match_form(mut matched: FormMatch, form: &Form, output: &str) -> FormMatch {
    if matched.code == 0 && form.text.len() <= MAX_FORM && form.text == output.trim() {
        matched.code = form.code;
    }
    let mut hasher = Sha256::new();
    hasher.update(matched.forms.as_slice());
    hasher.update(form.code.to_le_bytes());
    hasher.update((form.text.len() as u32).to_le_bytes());
    hasher.update(form.text.as_bytes());
    let chained: [u8; 32] = hasher.finalize().into();
    matched.forms = page(chained.to_vec());
    matched
}

/// The hash over the forms compared so far, in lowercase hex.
pub fn forms_id(matched: &FormMatch) -> String {
    lower_hex(matched.forms.as_slice())
}

/// States the result beside the claim record's three values.
pub fn close_result(
    matched: FormMatch,
    statement: &[u8],
    output_text_sha256: &[u8],
    valid: bool,
) -> ResultRecord {
    ResultRecord {
        statement: page(statement.to_vec()),
        output_text_sha256: page(output_text_sha256.to_vec()),
        valid,
        code: matched.code,
        forms_sha256: forms_id(&matched),
    }
}

/// Every form, in order, as the stage compares them.
pub fn run_forms<'a>(forms: impl IntoIterator<Item = &'a Form>, output: &str) -> FormMatch {
    let mut matched = begin_form_match();
    for form in forms {
        matched = match_form(matched, form, output);
    }
    matched
}
