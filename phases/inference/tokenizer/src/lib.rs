//! The prompt stages' shared source, as a crate of its own so its logic can be
//! tested without a stage. See `prompt.rs`.

#![no_std]

extern crate alloc;

mod prompt;
pub use prompt::*;

#[cfg(test)]
mod tests;
