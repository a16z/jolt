//! Whiteout, the lattice zero-knowledge back end: the Σ-protocol of
//! `specs/whiteout-sigma.md` over a lattice pad commitment.
//!
//! The prover commits the pads (with the mask `g` and the operand pads) in
//! one Ajtai level ([`LatticePadCommitment`], sent by [`commit_lattice_pads`]
//! through the response prover's [`ImageCompression`]). After the masked
//! protocol, the residual relation is lifted to its constant term over the
//! challenge ring ([`LiftedRelation`], built by [`SigmaSetup`]); the
//! Σ-protocol ([`prove_sigma`], [`verify_sigma`]) masks the private vector
//! with discrete Gaussian noise and reduces the relation to the response
//! statement ([`ResponseStatement`]), which a [`ResponseProver`] proves.
//! Akita's lattice response prover implements it.
//!
//! No zero-knowledge mode selects this yet. The verifier-side pieces build
//! under the `whiteout` feature; the prover side (the Σ prover, the samplers,
//! the witness alphabet check) additionally needs `whiteout-prover`.
//! Randomness comes from a caller-supplied cryptographic RNG; the prover's
//! secret-dependent work runs in `jolt-dit`'s data-independent-timing pool.

mod challenge;
mod committed_pads;
mod compression;
#[cfg(feature = "whiteout-prover")]
mod dit;
mod error;
#[cfg(any(
    all(test, feature = "whiteout-prover"),
    feature = "whiteout-test-support"
))]
mod fixture;
mod labels;
mod lift;
mod linear;
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
    ResponseSegment, ResponseStatement, ResponseWeights, ResponseWitness,
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
