//! Whiteout, the lattice zero-knowledge back end.
//!
//! This module holds the ring-level building blocks (the constant-term lift
//! and the zero-constant-term sampler, first written in Akita for
//! LayerZero-Labs/akita#120), and the Σ-protocol of `specs/whiteout-sigma.md`: the lattice pad
//! commitment, the constant-term lift of the residual relation, the prover
//! and verifier, and the response statement. It exposes no zero-knowledge
//! mode yet (`ZkBackend for Whiteout` is W6). The verifier-side pieces build
//! under the `whiteout` feature; the prover-side sampler additionally needs
//! `whiteout-prover`. WARNING: an Akita verifier build compiles it anyway,
//! because jolt-akita's `zk` enables `whiteout-prover` for its prover half;
//! splitting jolt-akita's feature is a follow-up
//! (`specs/whiteout-jolt-integration.md`, prototype deltas).
//! Randomness here comes from a caller-supplied cryptographic RNG.

mod challenge;
mod committed_pads;
mod compression;
#[cfg(feature = "whiteout-diagnostics")]
pub mod diagnostics;
#[cfg(feature = "whiteout-prover")]
mod dit;
mod error;
#[cfg(any(
    all(test, feature = "whiteout-prover"),
    feature = "whiteout-diagnostics",
    feature = "whiteout-test-support"
))]
mod fixture;
mod labels;
mod lift;
mod linear;
mod linear_factored;
mod opening;
mod pad_commitment;
mod params;
mod placement;
mod response;
#[cfg(feature = "whiteout-prover")]
mod sample;
mod segments;
mod sigma;
#[cfg(any(
    all(test, feature = "whiteout-prover"),
    feature = "whiteout-test-support"
))]
pub mod test_support;
mod transport;
mod weights;

pub use challenge::DenseChallenge;
#[cfg(feature = "whiteout-prover")]
pub use committed_pads::{commit_lattice_pads, WhiteoutPadOpening};
pub use committed_pads::{receive_lattice_pads, LatticePadPayload};
pub use compression::{CompressedImage, ImageCompression};
pub use error::WhiteoutError;
pub use lift::{ChallengeRing, EvaluationClaims, LiftedRelation};
pub use linear::{BlockTarget, ResidualBlock, ResidualGadget};
pub use opening::{
    DenseSource, MaskedGroupClaim, MaskedGroupOpening, MaskedOpeningShape, PrecommittedShape,
};
#[cfg(feature = "whiteout-prover")]
pub use pad_commitment::LatticePadOpening;
pub use pad_commitment::{LatticePadCommitment, PAD_RANDOMNESS_LOG_BASIS};
pub use params::{
    binding_collision_radius, GaussianBlock, SigmaParams, ALPHA, BINDING_RESPONSE_DIGITS,
    EPSILON_BITS, MAX_RESPONSE_NUM_VARS, RESPONSE_LOG_BASIS,
};
pub use placement::PadPlacement;
pub use response::{
    CompressedTarget, ResponseLayout, ResponseProver, ResponseRowShape, ResponseRows,
    ResponseSegment, ResponseStatement, ResponseVerifier, ResponseWeights, ResponseWitness,
};
#[cfg(feature = "whiteout-prover")]
pub use sample::sample_zero_constant_term;
pub use segments::{Segment, SegmentMask, SegmentRole, SegmentShape, SegmentTable};
#[cfg(feature = "whiteout-prover")]
pub use sigma::prove_sigma;
#[cfg(feature = "whiteout-test-support")]
pub use sigma::prove_sigma_unchecked;
pub use sigma::{verify_sigma, SigmaSetup};

#[cfg(all(test, feature = "whiteout-prover"))]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests should fail loudly"
)]
mod tests;
