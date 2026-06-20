//! Typed outputs produced by stage 7 verification.

use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
use jolt_field::Field;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use crate::stages::relations::OpeningClaim;
use crate::stages::zk::outputs::CommittedOutputClaimOutput;

use super::inputs::Stage7OutputClaims;

/// Final opening of a precommitted polynomial, resolved from whichever stage
/// completed its claim reduction (stage 6b cycle phase or stage 7 address
/// phase). Stage 8 consumes these as anchors and batch members of the final
/// PCS opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrecommittedFinalOpening<F: Field> {
    pub polynomial: JoltCommittedPolynomial,
    pub point: Vec<F>,
    /// `None` in ZK mode, where opening claims stay committed.
    pub opening_claim: Option<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7ClearOutput<F: Field> {
    /// The produced stage-7 openings paired with their points (point + value) via
    /// the `OpeningClaim` cell.
    pub output_claims: Stage7OutputClaims<OpeningClaim<F>>,
    /// The hamming-weight reduction's opening point — the own point of the one-hot
    /// `Ra` polynomials, shared by all reduced RA openings. Stored contiguously so
    /// stage 8 can borrow it directly (the per-family RA opening cells can be empty
    /// for a missing family, so it cannot always be read off a cell).
    pub hamming_weight_opening_point: Vec<F>,
    pub precommitted_final_openings: Vec<PrecommittedFinalOpening<F>>,
}

/// ZK counterpart of [`Stage7ClearOutput`]. The produced opening *values* stay
/// committed (in `batch_output_claims`); BlindFold recomputes every per-relation
/// sumcheck point and public it needs from `batch_consistency`, so only the data
/// stage 8 consumes is carried in the clear: the shared hamming-weight opening
/// point and the precommitted final openings (point-only, claims committed).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7ZkOutput<F: Field, C> {
    pub public: Stage7PublicOutput<F>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    pub hamming_weight_opening_point: Vec<F>,
    pub precommitted_final_openings: Vec<PrecommittedFinalOpening<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage7Output<F: Field, C> {
    Clear(Stage7ClearOutput<F>),
    Zk(Stage7ZkOutput<F, C>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7PublicOutput<F: Field> {
    pub hamming_gamma: F,
}
