//! Typed outputs produced by stage 3 verification.

use jolt_field::Field;
use jolt_poly::{Point, HIGH_TO_LOW};
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use crate::stages::zk::outputs::CommittedOutputClaimOutput;

use super::inputs::Stage3Claims;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3PublicOutput<F: Field> {
    pub challenges: Vec<F>,
    pub batching_coefficients: Vec<F>,
    pub shift_gamma: F,
    pub instruction_gamma: F,
    pub registers_gamma: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3ClearOutput<F: Field> {
    pub public: Stage3PublicOutput<F>,
    pub output_claims: Stage3Claims<F>,
    pub batch: VerifiedStage3Batch<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3ZkOutput<F: Field, C> {
    pub public: Stage3PublicOutput<F>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage3Output<F: Field, C> {
    Clear(Stage3ClearOutput<F>),
    Zk(Stage3ZkOutput<F, C>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStage3Batch<F: Field> {
    pub batching_coefficients: Vec<F>,
    pub sumcheck_point: Point<HIGH_TO_LOW, F>,
    pub sumcheck_final_claim: F,
    pub expected_final_claim: F,
    pub shift: VerifiedStage3Sumcheck<F>,
    pub instruction_input: VerifiedStage3Sumcheck<F>,
    pub registers_claim_reduction: VerifiedStage3Sumcheck<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStage3Sumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub expected_output_claim: F,
}
