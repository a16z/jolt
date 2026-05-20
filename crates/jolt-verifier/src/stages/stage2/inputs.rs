//! Typed clear-mode inputs consumed by stage 2.

use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::stages::stage1::Stage1ClearOutput;

#[derive(Clone, Copy)]
pub struct Deps<'a, F: Field> {
    pub stage1: &'a Stage1ClearOutput<F>,
}

pub fn deps<F: Field>(stage1: &Stage1ClearOutput<F>) -> Deps<'_, F> {
    Deps { stage1 }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Stage2Claims<F: Field> {
    pub product_uniskip_output_claim: F,
    pub batch_outputs: Stage2BatchOutputOpeningClaims<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Stage2BatchOutputOpeningClaims<F: Field> {
    pub ram_read_write: RamReadWriteOutputOpeningClaims<F>,
    pub product_remainder: ProductRemainderOutputOpeningClaims<F>,
    pub instruction_claim_reduction: InstructionClaimReductionOutputOpeningClaims<F>,
    pub ram_raf_evaluation: F,
    pub ram_output_check: F,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RamReadWriteOutputOpeningClaims<F: Field> {
    pub val: F,
    pub ra: F,
    pub inc: F,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProductRemainderOutputOpeningClaims<F: Field> {
    pub left_instruction_input: F,
    pub right_instruction_input: F,
    pub jump_flag: F,
    pub write_lookup_output_to_rd: F,
    pub lookup_output: F,
    pub branch_flag: F,
    pub next_is_noop: F,
    pub virtual_instruction: F,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct InstructionClaimReductionOutputOpeningClaims<F: Field> {
    pub lookup_output: Option<F>,
    pub left_lookup_operand: F,
    pub right_lookup_operand: F,
    pub left_instruction_input: Option<F>,
    pub right_instruction_input: Option<F>,
}
