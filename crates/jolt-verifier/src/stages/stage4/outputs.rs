//! Typed outputs produced by stage 4 verification.

use jolt_claims::protocols::jolt::JoltAdviceKind;
use jolt_field::Field;

use super::inputs::Stage4Claims;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4Output<F: Field> {
    pub challenges: Vec<F>,
    pub output_claims: Stage4Claims<F>,
    pub batch: VerifiedStage4Batch<F>,
    pub ram_val_check_init: RamValCheckInitialEvaluation<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStage4Batch<F: Field> {
    pub batching_coefficients: Vec<F>,
    pub sumcheck_point: jolt_poly::Point<F>,
    pub sumcheck_final_claim: F,
    pub expected_final_claim: F,
    pub registers_read_write: VerifiedStage4Sumcheck<F>,
    pub ram_val_check: VerifiedStage4Sumcheck<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStage4Sumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamValCheckInitialEvaluation<F: Field> {
    pub public_eval: F,
    pub advice_contributions: Vec<VerifiedRamValCheckAdviceContribution<F>>,
    pub full_eval: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedRamValCheckAdviceContribution<F: Field> {
    pub kind: JoltAdviceKind,
    pub selector: F,
    pub opening_claim: F,
    pub opening_point: Vec<F>,
}
