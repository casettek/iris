//! The routines of the stages that verify one inference, and what only they
//! need: the model's weights, its embedding and head tables, and the
//! tokenizer's identity.

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
