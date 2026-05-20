//! Typed clear-mode inputs consumed by stage 7.

use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::stages::{
    stage1::Stage1ClearOutput, stage2::Stage2ClearOutput, stage3::Stage3Output,
    stage4::Stage4Output, stage5::Stage5Output, stage6::Stage6Output,
};

#[derive(Clone, Copy)]
pub struct Deps<'a, F: Field> {
    pub stage4: &'a Stage4Output<F>,
    pub stage6: &'a Stage6Output<F>,
}

pub fn deps<'a, F: Field>(
    _stage1: &'a Stage1ClearOutput<F>,
    _stage2: &'a Stage2ClearOutput<F>,
    _stage3: &'a Stage3Output<F>,
    stage4: &'a Stage4Output<F>,
    _stage5: &'a Stage5Output<F>,
    stage6: &'a Stage6Output<F>,
) -> Deps<'a, F> {
    Deps { stage4, stage6 }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Stage7Claims<F: Field> {
    pub hamming_weight_claim_reduction: HammingWeightClaimReductionOutputOpeningClaims<F>,
    pub advice_address_phase: Stage7AdviceAddressPhaseClaims<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HammingWeightClaimReductionOutputOpeningClaims<F: Field> {
    pub instruction_ra: Vec<F>,
    pub bytecode_ra: Vec<F>,
    pub ram_ra: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Stage7AdviceAddressPhaseClaims<F: Field> {
    pub trusted: Option<AdviceAddressPhaseOutputClaim<F>>,
    pub untrusted: Option<AdviceAddressPhaseOutputClaim<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct AdviceAddressPhaseOutputClaim<F: Field> {
    pub opening_claim: F,
}
