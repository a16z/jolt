//! Typed outputs produced by stage 4 verification.

use jolt_field::Field;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use crate::stages::relations::OpeningClaim;
use crate::stages::zk::outputs::CommittedOutputClaimOutput;

use super::inputs::Stage4OutputClaims;
use super::ram_val_check::RamValCheckInitialEvaluation;

/// The Fiat-Shamir challenges the verifier draws during stage 4: the two
/// per-relation batching gammas. (The batch's own sumcheck point and batching
/// coefficients are stage-local verification artifacts and are not propagated to
/// later stages.)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4Challenges<F: Field> {
    pub registers_gamma: F,
    pub ram_val_check_gamma: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4ClearOutput<F: Field> {
    pub challenges: Stage4Challenges<F>,
    /// The produced stage-4 openings paired with their points (point + value)
    /// via the `OpeningClaim` cell. The opening points are derived from the
    /// batch's sumcheck point; pairing them with the values here lets later stages
    /// consume a ready `OpeningClaim` instead of re-joining a value with a
    /// separately-tracked point.
    pub output_claims: Stage4OutputClaims<OpeningClaim<F>>,
    pub ram_val_check_init: RamValCheckInitialEvaluation<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4ZkOutput<F: Field, C> {
    pub challenges: Stage4Challenges<F>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    pub ram_val_check_public_eval: F,
    /// The produced opening points (point-only cell), the ZK counterpart of the
    /// clear path's `output_claims`. Read through the same `*_point()` accessors.
    /// The advice / program-image leaves are absent in ZK (BlindFold carries those
    /// openings), so only the register and RAM value-check points are populated.
    pub output_points: Stage4OutputClaims<Vec<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage4Output<F: Field, C> {
    Clear(Stage4ClearOutput<F>),
    Zk(Stage4ZkOutput<F, C>),
}
