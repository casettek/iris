//! The source step's shared source, as a crate of its own so its logic can
//! be tested without a stage. See `source.rs`, and `relay.rs` for the step
//! that builds a prompt around an earlier inference's output.

#![no_std]

extern crate alloc;

pub mod relay;
mod source;
pub use source::*;

#[cfg(test)]
mod relay_tests;
#[cfg(test)]
mod tests;
