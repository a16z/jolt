//! Typed inputs consumed by stage 6.

use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::stages::{
    stage1::{Stage1ClearOutput, Stage1Output},
    stage2::{Stage2ClearOutput, Stage2Output},
    stage3::{Stage3ClearOutput, Stage3Output},
    stage4::{Stage4ClearOutput, Stage4Output},
    stage5::{Stage5ClearOutput, Stage5Output, Stage5ZkOutput},
};

#[derive(Clone, Copy)]
pub enum Deps<'a, F: Field, C> {
    Clear {
        stage1: &'a Stage1ClearOutput<F>,
        stage2: &'a Stage2ClearOutput<F>,
        stage3: &'a Stage3ClearOutput<F>,
        stage4: &'a Stage4ClearOutput<F>,
        stage5: &'a Stage5ClearOutput<F>,
    },
    Zk {
        stage5: &'a Stage5ZkOutput<F, C>,
    },
}

pub fn deps<'a, F: Field, C>(
    stage1: &'a Stage1Output<F, C>,
    stage2: &'a Stage2Output<F, C>,
    stage3: &'a Stage3Output<F, C>,
    stage4: &'a Stage4Output<F, C>,
    stage5: &'a Stage5Output<F, C>,
) -> Result<Deps<'a, F, C>, crate::VerifierError> {
    match (stage1, stage2, stage3, stage4, stage5) {
        (
            Stage1Output::Clear(stage1),
            Stage2Output::Clear(stage2),
            Stage3Output::Clear(stage3),
            Stage4Output::Clear(stage4),
            Stage5Output::Clear(stage5),
        ) => Ok(Deps::Clear {
            stage1,
            stage2,
            stage3,
            stage4,
            stage5,
        }),
        (
            Stage1Output::Zk(_),
            Stage2Output::Zk(_),
            Stage3Output::Zk(_),
            Stage4Output::Zk(_),
            Stage5Output::Zk(stage5),
        ) => Ok(Deps::Zk { stage5 }),
        (_, _, _, _, Stage5Output::Clear(_)) => {
            Err(crate::VerifierError::ExpectedClearProof { field: "stage5" })
        }
        (_, _, _, _, Stage5Output::Zk(_)) => {
            Err(crate::VerifierError::ExpectedCommittedProof { field: "stage5" })
        }
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
    #[cfg(feature = "field-inline")]
    pub field_inline: FieldInlineStage6Claims<F>,
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

#[cfg(feature = "field-inline")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FieldInlineStage6Claims<F: Field> {
    pub field_registers_inc_claim_reduction: FieldRegistersIncClaimReductionOutputOpeningClaims<F>,
}

#[cfg(feature = "field-inline")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FieldRegistersIncClaimReductionOutputOpeningClaims<F: Field> {
    pub field_rd_inc: F,
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
