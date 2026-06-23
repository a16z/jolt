//! Typed inputs consumed by stage 3.

use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::stages::{
    stage1::{Stage1ClearOutput, Stage1Output, Stage1ZkOutput},
    stage2::{Stage2ClearOutput, Stage2Output, Stage2ZkOutput},
};

#[derive(Clone, Copy)]
pub enum Deps<'a, F: Field, C> {
    Clear {
        stage1: &'a Stage1ClearOutput<F>,
        stage2: &'a Stage2ClearOutput<F>,
    },
    Zk {
        stage1: &'a Stage1ZkOutput<F, C>,
        stage2: &'a Stage2ZkOutput<F, C>,
    },
}

pub fn deps<'a, F: Field, C>(
    stage1: &'a Stage1Output<F, C>,
    stage2: &'a Stage2Output<F, C>,
) -> Result<Deps<'a, F, C>, crate::VerifierError> {
    match (stage1, stage2) {
        (Stage1Output::Clear(stage1), Stage2Output::Clear(stage2)) => {
            Ok(Deps::Clear { stage1, stage2 })
        }
        (Stage1Output::Zk(stage1), Stage2Output::Zk(stage2)) => Ok(Deps::Zk { stage1, stage2 }),
        (Stage1Output::Clear(_), Stage2Output::Zk(_)) => {
            Err(crate::VerifierError::ExpectedClearProof { field: "stage2" })
        }
        (Stage1Output::Zk(_), Stage2Output::Clear(_)) => {
            Err(crate::VerifierError::ExpectedCommittedProof { field: "stage2" })
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct Stage3Claims<F: Field> {
    pub shift: SpartanShiftOutputOpeningClaims<F>,
    pub instruction_input: InstructionInputOutputOpeningClaims<F>,
    pub registers_claim_reduction: RegistersClaimReductionOutputOpeningClaims<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct SpartanShiftOutputOpeningClaims<F: Field> {
    pub unexpanded_pc: F,
    pub pc: F,
    pub is_virtual: F,
    pub is_first_in_sequence: F,
    pub is_noop: F,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct InstructionInputOutputOpeningClaims<F: Field> {
    pub left_operand_is_rs1: F,
    pub rs1_value: F,
    pub left_operand_is_pc: F,
    pub unexpanded_pc: F,
    pub right_operand_is_rs2: F,
    pub rs2_value: F,
    pub right_operand_is_imm: F,
    pub imm: F,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct RegistersClaimReductionOutputOpeningClaims<F: Field> {
    pub rd_write_value: F,
    pub rs1_value: F,
    pub rs2_value: F,
}
