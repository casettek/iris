//! The rules a contract holds a claim to, in Rust.
//!
//! The contracts in `blockchain/contracts/src` make these same checks on
//! pipeline. This module is what they are written from and held to: the pipeline
//! tier (`just test-blockchain`) asks both about the same claims and challenges
//! and fails when they differ. A rule changes here and in the contract, and
//! nowhere else.
//!
//! By part:
//!
//! - `pins`: what a contract is configured with before any claim arrives
//!   ([`ContractPins`], [`RequiredParameters`]), read from a pipeline template
//!   and the stage projects' locks.
//! - `submission`: what a claimer submits ([`ClaimSubmission`]) and the
//!   checks made when it arrives ([`check_submission`], [`check_origin`],
//!   [`check_stated_records`]).
//! - `checkpoints`: what is read from the claim's own checkpoint list with
//!   no replay ([`CheckpointHashes::admit_stage_challenge`],
//!   [`CheckpointHashes::check_link`]): a stage that names an unapproved
//!   program, or an input that is not an earlier stage's output, a pinned
//!   commitment or a posted value, voids the claim on the spot.
//! - `dispute`: a dispute about one stage's execution: the challenge
//!   ([`ChallengeSubmission`], [`stage_challenge`]) and when a fault proof
//!   answers it ([`check_fault_proof`]).
//!
//! A claim commits to one checkpoint per stage. A checkpoint is the boundary a
//! Raster run of that stage is bound to: the program, the input manifest and
//! the output. So a dispute is about one stage, started at the claimer's own
//! boundary, and no pipeline-wide Raster commitment is needed.
//! [`raster_boundary_mismatch`] checks that correspondence against a Raster
//! pipeline run of the same claim, and [`dispute_commitment`] is the one-stage
//! Raster commitment a checkpoint stands as in such a dispute.
//!
//! Stages keep their own store, so nothing inside a stage can tell where an
//! input came from: a made-up input can carry the same labels a real one
//! does. The link check is what holds a stage's inputs to something a
//! challenger also holds. Which of those belongs at which stage is the stage
//! schedule's to enforce, inside the programs.
//!
//! Absent here and in the contracts: bonds, and what follows a claim nobody
//! challenged or a challenge nobody answered. Data availability is
//! [`crate::availability`]'s.

mod checkpoints;
mod dispute;
mod pins;
mod submission;

pub use checkpoints::*;
pub use dispute::*;
pub use pins::*;
pub use submission::*;

#[cfg(test)]
mod tests;
