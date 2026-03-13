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
//! | [`verifier_r1cs`] | Verifier R1CS encoding deferred sumcheck checks |
//! | [`folding`] | Nova folding — commitment-agnostic relaxed R1CS folding |

pub mod accumulator;
pub mod error;
pub mod folding;
pub mod handler;
pub mod proof;
pub mod protocol;
pub mod verifier_r1cs;

pub use accumulator::BlindFoldAccumulator;
pub use error::BlindFoldError;
pub use folding::{
    check_relaxed_satisfaction, compute_cross_term, fold_instances, fold_scalar, fold_witnesses,
    sample_random_witness, RelaxedInstance, RelaxedWitness,
};
pub use handler::{CommittedRoundHandler, CommittedRoundVerifier};
pub use proof::{
    BlindFoldProof, CommittedRoundData, CommittedSumcheckOutput, CommittedSumcheckProof,
};
pub use protocol::{BlindFoldProver, BlindFoldVerifier};
pub use verifier_r1cs::{BakedPublicInputs, StageConfig};
