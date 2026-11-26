//! What a claimer, a challenger and a defender do on the host for a claim
//! that is settled on blockchains: build a claim and post it, fetch and audit
//! one, build a challenge, prove a fault, and the calls to the pipelines.

pub mod arrangement;
pub mod audit;
pub mod blockchain;
pub mod challenge;
pub mod challenge_profile;
pub mod claim;
pub mod claim_check;
pub mod fault;
pub mod measure;
pub mod posted;
pub mod posting;
pub mod raster_json;
pub use run_prep::run_inputs;
#[cfg(test)]
mod test_support;
