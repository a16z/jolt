//! Whiteout, the lattice zero-knowledge back end.
//!
//! This module holds the ring-level building blocks moved from Akita `dev`'s
//! `akita-zk-verifier` and `akita-zk-prover` crates (LayerZero-Labs/akita#120),
//! and the Σ-protocol's segment table and parameters
//! (`specs/whiteout-sigma.md`). It exposes no zero-knowledge mode yet. The verifier-side pieces build under
//! the `whiteout` feature; the prover-side sampler additionally needs
//! `whiteout-prover`, so a verifier that enables `whiteout` never compiles it.
//! Randomness here comes from a caller-supplied cryptographic RNG.

mod challenge;
mod error;
mod labels;
mod lift;
mod linear;
mod pad_commitment;
mod params;
mod placement;
mod response;
mod ring;
#[cfg(feature = "whiteout-prover")]
mod sample;
mod segments;
mod sigma;

pub use challenge::{challenge_ring, challenge_times_i64, challenge_times_ring, DenseChallenge};
pub use error::WhiteoutError;
pub use labels::RowLabels;
pub use lift::{
    constant_ring, AggregatedRelation, ChallengeRing, EvaluationClaims, LiftedRelation, LinearParts,
};
pub use linear::{GadgetTerm, LRow, LinearBlock, RingLinearMap};
pub use pad_commitment::{LatticePadCommitment, LatticePadOpening, RANDOMNESS_LOG_BASIS};
pub use params::{GaussianBlock, SigmaParams, ALPHA, EPSILON_BITS, RESPONSE_LOG_BASIS};
pub use placement::{
    PadPlace, PadPlacement, OPERAND_LEVELS, OPERAND_LOG_BASIS, PAD_LEVELS, PAD_LOG_BASIS,
};
pub use response::{
    ResponseLayout, ResponseProver, ResponseSegment, ResponseStatement, ResponseWitness,
    FIELD_LEVELS, THETA_COEFFICIENTS,
};
pub use ring::LiftedFunctional;
#[cfg(feature = "whiteout-prover")]
pub use sample::sample_zero_constant_term;
pub use segments::{Segment, SegmentMask, SegmentRole, SegmentShape, SegmentTable};
#[cfg(feature = "whiteout-prover")]
pub use sigma::prove_sigma;
pub use sigma::{verify_sigma, SigmaSetup};
