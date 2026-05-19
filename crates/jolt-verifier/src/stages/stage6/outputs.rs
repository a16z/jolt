//! Typed outputs produced by stage 6 verification.

use jolt_claims::protocols::jolt::JoltAdviceKind;
use jolt_field::Field;

use super::inputs::Stage6Claims;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6Output<F: Field> {
    pub challenges: Vec<F>,
    pub output_claims: Stage6Claims<F>,
    pub batch: VerifiedStage6Batch<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStage6Batch<F: Field> {
    pub batching_coefficients: Vec<F>,
    pub sumcheck_point: jolt_poly::Point<F>,
    pub sumcheck_final_claim: F,
    pub expected_final_claim: F,
    pub bytecode_read_raf: VerifiedBytecodeReadRafSumcheck<F>,
    pub booleanity: VerifiedBooleanitySumcheck<F>,
    pub ram_hamming_booleanity: VerifiedStage6Sumcheck<F>,
    pub ram_ra_virtualization: VerifiedRamRaVirtualizationSumcheck<F>,
    pub instruction_ra_virtualization: VerifiedInstructionRaVirtualizationSumcheck<F>,
    pub inc_claim_reduction: VerifiedStage6Sumcheck<F>,
    pub trusted_advice_cycle_phase: Option<VerifiedAdviceCyclePhaseSumcheck<F>>,
    pub untrusted_advice_cycle_phase: Option<VerifiedAdviceCyclePhaseSumcheck<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedBytecodeReadRafSumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub r_address: Vec<F>,
    pub r_cycle: Vec<F>,
    pub full_opening_point: Vec<F>,
    pub bytecode_ra_opening_points: Vec<Vec<F>>,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedBooleanitySumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub r_address: Vec<F>,
    pub r_cycle: Vec<F>,
    pub opening_point: Vec<F>,
    pub reference_address: Vec<F>,
    pub reference_cycle: Vec<F>,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedRamRaVirtualizationSumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub ram_ra_opening_points: Vec<Vec<F>>,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedInstructionRaVirtualizationSumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub instruction_ra_opening_points: Vec<Vec<F>>,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStage6Sumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedAdviceCyclePhaseSumcheck<F: Field> {
    pub kind: JoltAdviceKind,
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub cycle_phase_variables: Vec<F>,
    pub expected_output_claim: F,
}
