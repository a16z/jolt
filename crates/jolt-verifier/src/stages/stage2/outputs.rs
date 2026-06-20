//! Typed outputs produced by stage 2 verification.

use jolt_field::Field;
use jolt_poly::{Point, HIGH_TO_LOW};
use jolt_sumcheck::{BatchedCommittedSumcheckConsistency, CommittedSumcheckConsistency};

use crate::stages::relations::OpeningClaim;
use crate::stages::zk::outputs::CommittedOutputClaimOutput;

use super::inputs::Stage2BatchOutputClaims;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2PublicOutput<F: Field> {
    pub challenges: Vec<F>,
    pub batching_coefficients: Vec<F>,
    pub product_uniskip_challenge: F,
    pub product_tau_low: Vec<F>,
    pub product_tau_high: F,
    pub ram_read_write_gamma: F,
    pub instruction_gamma: F,
    pub output_address_challenges: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2ClearOutput<F: Field> {
    pub public: Stage2PublicOutput<F>,
    /// The produced batch openings paired with their points (point + value) via the
    /// `OpeningClaim` cell. The opening points are derived from each relation's
    /// sumcheck point; later stages read them through the
    /// `*_point()` accessors and read values through `.value`, instead of joining a
    /// separately-tracked `VerifiedStage2Batch` with the wire values.
    pub output_claims: Stage2BatchOutputClaims<OpeningClaim<F>>,
    pub product_uniskip: VerifiedProductUniSkip<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2ZkOutput<F: Field, C> {
    pub public: Stage2PublicOutput<F>,
    pub product_uniskip_consistency: CommittedSumcheckConsistency<F, C>,
    pub product_uniskip_output_claims: CommittedOutputClaimOutput<C>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    /// The produced batch opening points (point-only cell), the ZK counterpart of
    /// the clear path's `output_claims`. Later stages read them through the same
    /// `*_point()` accessors via [`GetPoint`](crate::stages::relations::GetPoint).
    pub output_points: Stage2BatchOutputClaims<Vec<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage2Output<F: Field, C> {
    Clear(Stage2ClearOutput<F>),
    Zk(Stage2ZkOutput<F, C>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedProductUniSkip<F: Field> {
    pub tau_low: Vec<F>,
    pub tau_high: F,
    pub sumcheck_point: Point<HIGH_TO_LOW, F>,
}
