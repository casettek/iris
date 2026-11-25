//! The result steps' shared source, as a crate of its own so its logic can
//! be tested without a stage. See `forms.rs` and `consensus.rs`.

#![no_std]

extern crate alloc;

mod consensus;
mod forms;
pub use consensus::*;
pub use forms::*;

#[cfg(test)]
mod tests;
