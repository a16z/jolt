//! Typed outputs produced by stage 3 verification.

use jolt_field::Field;

use super::inputs::Stage3Claims;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3Output<F: Field> {
    pub challenges: Vec<F>,
    pub output_claims: Stage3Claims<F>,
    pub batch: VerifiedStage3Batch<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStage3Batch<F: Field> {
    pub batching_coefficients: Vec<F>,
    pub sumcheck_point: jolt_poly::Point<F>,
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
