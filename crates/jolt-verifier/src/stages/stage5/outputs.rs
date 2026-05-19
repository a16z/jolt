//! Typed outputs produced by stage 5 verification.

use jolt_field::Field;

use super::inputs::Stage5Claims;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5Output<F: Field> {
    pub challenges: Vec<F>,
    pub output_claims: Stage5Claims<F>,
    pub batch: VerifiedStage5Batch<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStage5Batch<F: Field> {
    pub batching_coefficients: Vec<F>,
    pub sumcheck_point: jolt_poly::Point<F>,
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
pub struct VerifiedStage5Sumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub expected_output_claim: F,
}
