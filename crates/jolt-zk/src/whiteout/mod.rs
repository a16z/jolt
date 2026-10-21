//! Whiteout, the lattice zero-knowledge back end.
//!
//! This module holds the ring-level building blocks moved from Akita `dev`'s
//! `akita-zk-verifier` and `akita-zk-prover` crates (LayerZero-Labs/akita#120),
//! and the Σ-protocol of `specs/whiteout-sigma.md`: the lattice pad
//! commitment, the constant-term lift of the residual relation, the prover
//! and verifier, and the response statement. It exposes no zero-knowledge
//! mode yet (`ZkBackend for Whiteout` is W6). The verifier-side pieces build
//! under the `whiteout` feature; the prover-side sampler additionally needs
//! `whiteout-prover`, so a verifier that enables `whiteout` never compiles it.
//! Randomness here comes from a caller-supplied cryptographic RNG.

mod challenge;
#[cfg(feature = "whiteout-diagnostics")]
pub mod diagnostics;
#[cfg(feature = "whiteout-prover")]
mod dit;
mod error;
#[cfg(any(
    all(test, feature = "whiteout-prover"),
    feature = "whiteout-diagnostics"
))]
mod fixture;
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

pub use challenge::{challenge_times_i64, DenseChallenge};
pub use error::WhiteoutError;
pub use lift::{ChallengeRing, EvaluationClaims, LiftedRelation};
pub use linear::{ResidualBlock, ResidualGadget};
pub use pad_commitment::LatticePadCommitment;
#[cfg(feature = "whiteout-prover")]
pub use pad_commitment::LatticePadOpening;
pub use params::{GaussianBlock, SigmaParams, ALPHA, EPSILON_BITS, RESPONSE_LOG_BASIS};
pub use placement::{
    PadPlace, PadPlacement, OPERAND_LEVELS, OPERAND_LOG_BASIS, PAD_LEVELS, PAD_LOG_BASIS,
};
pub use response::{
    ResponseLayout, ResponseProver, ResponseSegment, ResponseStatement, ResponseWitness,
    FIELD_LEVELS,
};
pub use ring::LiftedFunctional;
#[cfg(feature = "whiteout-prover")]
pub use sample::sample_zero_constant_term;
pub use segments::{balanced_range, Segment, SegmentMask, SegmentRole, SegmentShape, SegmentTable};
#[cfg(feature = "whiteout-prover")]
pub use sigma::prove_sigma;
pub use sigma::{verify_sigma, SigmaSetup};

#[cfg(all(test, feature = "whiteout-prover"))]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests should fail loudly"
)]
mod tests;
