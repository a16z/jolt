//! Typed clear-mode inputs consumed by stage 6.

use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::stages::{
    stage1::Stage1ClearOutput, stage2::Stage2ClearOutput, stage3::Stage3ClearOutput,
    stage4::Stage4ClearOutput, stage5::Stage5ClearOutput,
};

#[derive(Clone, Copy)]
pub struct Deps<'a, F: Field> {
    pub stage1: &'a Stage1ClearOutput<F>,
    pub stage2: &'a Stage2ClearOutput<F>,
    pub stage3: &'a Stage3ClearOutput<F>,
    pub stage4: &'a Stage4ClearOutput<F>,
    pub stage5: &'a Stage5ClearOutput<F>,
}

pub fn deps<'a, F: Field>(
    stage1: &'a Stage1ClearOutput<F>,
    stage2: &'a Stage2ClearOutput<F>,
    stage3: &'a Stage3ClearOutput<F>,
    stage4: &'a Stage4ClearOutput<F>,
    stage5: &'a Stage5ClearOutput<F>,
) -> Deps<'a, F> {
    Deps {
        stage1,
        stage2,
        stage3,
        stage4,
        stage5,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Stage6Claims<F: Field> {
    pub bytecode_read_raf: BytecodeReadRafOutputOpeningClaims<F>,
    pub booleanity: BooleanityOutputOpeningClaims<F>,
    pub ram_hamming_booleanity: RamHammingBooleanityOutputOpeningClaims<F>,
    pub ram_ra_virtualization: RamRaVirtualizationOutputOpeningClaims<F>,
    pub instruction_ra_virtualization: InstructionRaVirtualizationOutputOpeningClaims<F>,
    pub inc_claim_reduction: IncClaimReductionOutputOpeningClaims<F>,
    pub advice_cycle_phase: Stage6AdviceCyclePhaseClaims<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct BytecodeReadRafOutputOpeningClaims<F: Field> {
    pub bytecode_ra: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct BooleanityOutputOpeningClaims<F: Field> {
    pub instruction_ra: Vec<F>,
    pub bytecode_ra: Vec<F>,
    pub ram_ra: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RamHammingBooleanityOutputOpeningClaims<F: Field> {
    pub ram_hamming_weight: F,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RamRaVirtualizationOutputOpeningClaims<F: Field> {
    pub ram_ra: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct InstructionRaVirtualizationOutputOpeningClaims<F: Field> {
    pub committed_instruction_ra: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct IncClaimReductionOutputOpeningClaims<F: Field> {
    pub ram_inc: F,
    pub rd_inc: F,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Stage6AdviceCyclePhaseClaims<F: Field> {
    pub trusted: Option<AdviceCyclePhaseOutputClaim<F>>,
    pub untrusted: Option<AdviceCyclePhaseOutputClaim<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct AdviceCyclePhaseOutputClaim<F: Field> {
    pub opening_claim: F,
}
