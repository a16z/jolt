//! Typed inputs consumed by stage 5.

use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::stages::{
    stage2::{Stage2ClearOutput, Stage2Output, Stage2ZkOutput},
    stage4::{Stage4ClearOutput, Stage4Output, Stage4ZkOutput},
};

#[derive(Clone, Copy)]
pub enum Deps<'a, F: Field, C> {
    Clear {
        stage2: &'a Stage2ClearOutput<F>,
        stage4: &'a Stage4ClearOutput<F>,
    },
    Zk {
        stage2: &'a Stage2ZkOutput<F, C>,
        stage4: &'a Stage4ZkOutput<F, C>,
    },
}

pub fn deps<'a, F: Field, C>(
    stage2: &'a Stage2Output<F, C>,
    stage4: &'a Stage4Output<F, C>,
) -> Result<Deps<'a, F, C>, crate::VerifierError> {
    match (stage2, stage4) {
        (Stage2Output::Clear(stage2), Stage4Output::Clear(stage4)) => {
            Ok(Deps::Clear { stage2, stage4 })
        }
        (Stage2Output::Zk(stage2), Stage4Output::Zk(stage4)) => Ok(Deps::Zk { stage2, stage4 }),
        (Stage2Output::Clear(_), Stage4Output::Zk(_)) => {
            Err(crate::VerifierError::ExpectedClearProof { field: "stage4" })
        }
        (Stage2Output::Zk(_), Stage4Output::Clear(_)) => {
            Err(crate::VerifierError::ExpectedCommittedProof { field: "stage4" })
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct Stage5Claims<F: Field> {
    pub instruction_read_raf: InstructionReadRafOutputOpeningClaims<F>,
    pub ram_ra_claim_reduction: RamRaClaimReductionOutputOpeningClaims<F>,
    pub registers_val_evaluation: RegistersValEvaluationOutputOpeningClaims<F>,
    #[cfg(feature = "field-inline")]
    pub field_inline: FieldInlineStage5Claims<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct InstructionReadRafOutputOpeningClaims<F: Field> {
    pub lookup_table_flags: Vec<F>,
    pub instruction_ra: Vec<F>,
    pub instruction_raf_flag: F,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct RamRaClaimReductionOutputOpeningClaims<F: Field> {
    pub ram_ra: F,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct RegistersValEvaluationOutputOpeningClaims<F: Field> {
    pub rd_inc: F,
    pub rd_wa: F,
}

#[cfg(feature = "field-inline")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct FieldInlineStage5Claims<F: Field> {
    pub field_registers_val_evaluation: FieldRegistersValEvaluationOutputOpeningClaims<F>,
}

#[cfg(feature = "field-inline")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct FieldRegistersValEvaluationOutputOpeningClaims<F: Field> {
    pub field_rd_inc: F,
    pub field_rd_wa: F,
}
