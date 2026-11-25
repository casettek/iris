//! The source step's shared source, as a crate of its own so its logic can
//! be tested without a stage. See `source.rs`.

#![no_std]

extern crate alloc;

mod source;
pub use source::*;

#[cfg(test)]
mod tests;
