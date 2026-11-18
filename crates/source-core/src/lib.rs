//! The source and result steps' shared source, as a crate of its own so its
//! logic can be tested without a stage. See `source.rs`, `forms.rs` and
//! `consensus.rs`.

#![no_std]

extern crate alloc;

mod consensus;
mod forms;
mod source;
pub use consensus::*;
pub use forms::*;
pub use source::*;

#[cfg(test)]
mod tests;
