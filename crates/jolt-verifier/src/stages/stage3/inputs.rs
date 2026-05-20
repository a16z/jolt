//! Typed clear-mode inputs consumed by stage 3.

use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::stages::{stage1::Stage1ClearOutput, stage2::Stage2Output};

#[derive(Clone, Copy)]
pub struct Deps<'a, F: Field> {
    pub stage1: &'a Stage1ClearOutput<F>,
    pub stage2: &'a Stage2Output<F>,
}

pub fn deps<'a, F: Field>(
    stage1: &'a Stage1ClearOutput<F>,
    stage2: &'a Stage2Output<F>,
) -> Deps<'a, F> {
    Deps { stage1, stage2 }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Stage3Claims<F: Field> {
    pub shift: SpartanShiftOutputOpeningClaims<F>,
    pub instruction_input: InstructionInputOutputOpeningClaims<F>,
    pub registers_claim_reduction: RegistersClaimReductionOutputOpeningClaims<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SpartanShiftOutputOpeningClaims<F: Field> {
    pub unexpanded_pc: F,
    pub pc: F,
    pub is_virtual: F,
    pub is_first_in_sequence: F,
    pub is_noop: F,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
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
#[serde(bound = "")]
pub struct RegistersClaimReductionOutputOpeningClaims<F: Field> {
    pub rd_write_value: F,
    pub rs1_value: F,
    pub rs2_value: F,
}
