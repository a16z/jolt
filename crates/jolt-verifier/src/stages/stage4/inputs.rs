//! Typed inputs consumed by stage 4.

use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::stages::{
    stage2::{Stage2ClearOutput, Stage2Output, Stage2ZkOutput},
    stage3::{Stage3ClearOutput, Stage3Output, Stage3ZkOutput},
};

#[derive(Clone, Copy)]
pub enum Deps<'a, F: Field, C> {
    Clear {
        stage2: &'a Stage2ClearOutput<F>,
        stage3: &'a Stage3ClearOutput<F>,
    },
    Zk {
        stage2: &'a Stage2ZkOutput<F, C>,
        stage3: &'a Stage3ZkOutput<F, C>,
    },
}

pub fn deps<'a, F: Field, C>(
    stage2: &'a Stage2Output<F, C>,
    stage3: &'a Stage3Output<F, C>,
) -> Result<Deps<'a, F, C>, crate::VerifierError> {
    match (stage2, stage3) {
        (Stage2Output::Clear(stage2), Stage3Output::Clear(stage3)) => {
            Ok(Deps::Clear { stage2, stage3 })
        }
        (Stage2Output::Zk(stage2), Stage3Output::Zk(stage3)) => Ok(Deps::Zk { stage2, stage3 }),
        (Stage2Output::Clear(_), Stage3Output::Zk(_)) => {
            Err(crate::VerifierError::ExpectedClearProof { field: "stage3" })
        }
        (Stage2Output::Zk(_), Stage3Output::Clear(_)) => {
            Err(crate::VerifierError::ExpectedCommittedProof { field: "stage3" })
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Stage4Claims<F: Field> {
    pub advice: RamValCheckAdviceOpeningClaims<F>,
    pub registers_read_write: RegistersReadWriteOutputOpeningClaims<F>,
    #[cfg(feature = "field-inline")]
    pub field_inline: FieldInlineStage4Claims<F>,
    pub ram_val_check: RamValCheckOutputOpeningClaims<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RamValCheckAdviceOpeningClaims<F: Field> {
    pub untrusted: Option<F>,
    pub trusted: Option<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RegistersReadWriteOutputOpeningClaims<F: Field> {
    pub registers_val: F,
    pub rs1_ra: F,
    pub rs2_ra: F,
    pub rd_wa: F,
    pub rd_inc: F,
}

#[cfg(feature = "field-inline")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FieldInlineStage4Claims<F: Field> {
    pub field_registers_read_write: FieldRegistersReadWriteOutputOpeningClaims<F>,
}

#[cfg(feature = "field-inline")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FieldRegistersReadWriteOutputOpeningClaims<F: Field> {
    pub field_registers_val: F,
    pub field_rs1_ra: F,
    pub field_rs2_ra: F,
    pub field_rd_wa: F,
    pub field_rd_inc: F,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RamValCheckOutputOpeningClaims<F: Field> {
    pub ram_ra: F,
    pub ram_inc: F,
}
