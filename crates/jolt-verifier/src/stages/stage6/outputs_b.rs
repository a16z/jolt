//! Typed outputs produced by stage 6b verification.

use jolt_claims::protocols::jolt::JoltAdviceKind;
use jolt_field::Field;

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
pub struct BytecodeReadRafPublicOutput<F: Field> {
    pub sumcheck_point: Vec<F>,
    pub r_address: Vec<F>,
    pub r_cycle: Vec<F>,
    pub full_opening_point: Vec<F>,
    pub bytecode_ra_opening_points: Vec<Vec<F>>,
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
pub struct BooleanityPublicOutput<F: Field> {
    pub sumcheck_point: Vec<F>,
    pub r_address: Vec<F>,
    pub r_cycle: Vec<F>,
    pub opening_point: Vec<F>,
    pub reference_address: Vec<F>,
    pub reference_cycle: Vec<F>,
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
pub struct RamRaVirtualizationPublicOutput<F: Field> {
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub ram_ra_opening_points: Vec<Vec<F>>,
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
pub struct InstructionRaVirtualizationPublicOutput<F: Field> {
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub instruction_ra_opening_points: Vec<Vec<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStage6Sumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6SumcheckPublicOutput<F: Field> {
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdviceCyclePhasePublicOutput<F: Field> {
    pub kind: JoltAdviceKind,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub cycle_phase_variables: Vec<F>,
}
