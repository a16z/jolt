//! Typed outputs produced by stage 5 verification.

use jolt_field::Field;
use jolt_poly::{Point, HIGH_TO_LOW};
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use crate::stages::zk::outputs::CommittedOutputClaimOutput;

use super::inputs::Stage5OutputClaims;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5PublicOutput<F: Field> {
    pub challenges: Vec<F>,
    pub batching_coefficients: Vec<F>,
    pub instruction_gamma: F,
    pub ram_gamma: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5ClearOutput<F: Field> {
    pub public: Stage5PublicOutput<F>,
    pub output_claims: Stage5OutputClaims<F>,
    pub batch: VerifiedStage5Batch<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5ZkOutput<F: Field, C> {
    pub public: Stage5PublicOutput<F>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    pub instruction_read_raf: InstructionReadRafPublicOutput<F>,
    pub ram_ra_claim_reduction: Stage5SumcheckPublicOutput<F>,
    pub registers_val_evaluation: Stage5SumcheckPublicOutput<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage5Output<F: Field, C> {
    Clear(Stage5ClearOutput<F>),
    Zk(Stage5ZkOutput<F, C>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStage5Batch<F: Field> {
    pub batching_coefficients: Vec<F>,
    pub sumcheck_point: Point<HIGH_TO_LOW, F>,
    pub sumcheck_final_claim: F,
    pub expected_final_claim: F,
    pub instruction_read_raf: VerifiedInstructionReadRafSumcheck<F>,
    pub ram_ra_claim_reduction: VerifiedStage5Sumcheck<F>,
    pub registers_val_evaluation: VerifiedStage5Sumcheck<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedInstructionReadRafSumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub r_address: Vec<F>,
    pub r_cycle: Vec<F>,
    pub full_opening_point: Vec<F>,
    pub lookup_table_flag_opening_point: Vec<F>,
    pub instruction_ra_opening_points: Vec<Vec<F>>,
    pub instruction_raf_flag_opening_point: Vec<F>,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstructionReadRafPublicOutput<F: Field> {
    pub sumcheck_point: Vec<F>,
    pub r_address: Vec<F>,
    pub r_cycle: Vec<F>,
    pub full_opening_point: Vec<F>,
    pub lookup_table_flag_opening_point: Vec<F>,
    pub instruction_ra_opening_points: Vec<Vec<F>>,
    pub instruction_raf_flag_opening_point: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStage5Sumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5SumcheckPublicOutput<F: Field> {
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
}
