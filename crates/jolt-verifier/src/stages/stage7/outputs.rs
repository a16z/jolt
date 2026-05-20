//! Typed outputs produced by stage 7 verification.

use jolt_claims::protocols::jolt::JoltAdviceKind;
use jolt_field::Field;

use super::inputs::Stage7Claims;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7Output<F: Field> {
    pub challenges: Vec<F>,
    pub output_claims: Stage7Claims<F>,
    pub batch: VerifiedStage7Batch<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStage7Batch<F: Field> {
    pub batching_coefficients: Vec<F>,
    pub sumcheck_point: jolt_poly::Point<F>,
    pub sumcheck_final_claim: F,
    pub expected_final_claim: F,
    pub hamming_weight_claim_reduction: VerifiedHammingWeightClaimReductionSumcheck<F>,
    pub trusted_advice_address_phase: Option<VerifiedAdviceAddressPhaseSumcheck<F>>,
    pub untrusted_advice_address_phase: Option<VerifiedAdviceAddressPhaseSumcheck<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedHammingWeightClaimReductionSumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub instruction_ra_opening_points: Vec<Vec<F>>,
    pub bytecode_ra_opening_points: Vec<Vec<F>>,
    pub ram_ra_opening_points: Vec<Vec<F>>,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedAdviceAddressPhaseSumcheck<F: Field> {
    pub kind: JoltAdviceKind,
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub expected_output_claim: F,
}
