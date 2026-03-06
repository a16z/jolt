//! ZK layer for Jolt sumcheck via committed round polynomials.
//!
//! This crate provides the committed-mode counterparts to jolt-sumcheck's
//! cleartext handlers:
//!
//! - [`CommittedRoundHandler`] implements
//!   [`RoundHandler`](jolt_sumcheck::RoundHandler): instead of appending
//!   polynomial coefficients to the transcript, it commits via any
//!   [`JoltCommitment`](jolt_crypto::JoltCommitment) scheme and appends the
//!   commitment.
//!
//! - [`CommittedRoundVerifier`] implements
//!   [`RoundVerifier`](jolt_sumcheck::RoundVerifier): absorbs commitments into
//!   the transcript and defers consistency checks to the BlindFold protocol.
//!
//! All types are generic over `JoltCommitment` — not hardcoded to Pedersen.
//! Hash-based or lattice-based commitment schemes can be substituted.
//!
//! # Crate structure
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`handler`] | [`CommittedRoundHandler`], [`CommittedRoundVerifier`] |
//! | [`proof`] | [`CommittedSumcheckProof`], [`CommittedRoundData`] |
//! | [`accumulator`] | [`BlindFoldAccumulator`] — collects ZK stage data |

pub mod accumulator;
pub mod handler;
pub mod proof;

pub use accumulator::BlindFoldAccumulator;
pub use handler::{CommittedRoundHandler, CommittedRoundVerifier};
pub use proof::{CommittedRoundData, CommittedSumcheckOutput, CommittedSumcheckProof};
