//! Typed outputs produced by stage 2 verification.

use jolt_field::Field;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2Output<F: Field> {
    pub challenges: Vec<F>,
    pub product_uniskip_challenge: F,
    pub product_uniskip: VerifiedProductUniSkip<F>,
    pub batch: VerifiedStage2Batch<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedProductUniSkip<F: Field> {
    pub tau_low: Vec<F>,
    pub tau_high: F,
    pub input_claim: F,
    pub sumcheck_point: jolt_poly::Point<F>,
    pub sumcheck_final_claim: F,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStage2Batch<F: Field> {
    pub batching_coefficients: Vec<F>,
    pub sumcheck_point: jolt_poly::Point<F>,
    pub sumcheck_final_claim: F,
    pub expected_final_claim: F,
    pub ram_read_write: VerifiedStage2Sumcheck<F>,
    pub product_remainder: VerifiedStage2Sumcheck<F>,
    pub instruction_claim_reduction: VerifiedStage2Sumcheck<F>,
    pub ram_raf_evaluation: VerifiedStage2Sumcheck<F>,
    pub ram_output_check: VerifiedStage2Sumcheck<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStage2Sumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub expected_output_claim: F,
}
