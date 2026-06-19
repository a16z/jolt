//! Typed inputs consumed by stage 2.

use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::stages::stage1::{Stage1ClearOutput, Stage1Output, Stage1ZkOutput};

#[derive(Clone, Copy)]
pub enum Deps<'a, F: Field, C> {
    Clear { stage1: &'a Stage1ClearOutput<F> },
    Zk { stage1: &'a Stage1ZkOutput<F, C> },
}

pub fn deps<F: Field, C>(stage1: &Stage1Output<F, C>) -> Deps<'_, F, C> {
    match stage1 {
        Stage1Output::Clear(stage1) => Deps::Clear { stage1 },
        Stage1Output::Zk(stage1) => Deps::Zk { stage1 },
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct Stage2Claims<F: Field> {
    pub product_uniskip_output_claim: F,
    pub batch_outputs: Stage2BatchOutputOpeningClaims<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct Stage2BatchOutputOpeningClaims<F: Field> {
    pub ram_read_write: RamReadWriteOutputOpeningClaims<F>,
    pub product_remainder: ProductRemainderOutputOpeningClaims<F>,
    #[cfg(feature = "field-inline")]
    pub field_inline: FieldInlineStage2OutputOpeningClaims<F>,
    pub instruction_claim_reduction: InstructionClaimReductionOutputOpeningClaims<F>,
    pub ram_raf_evaluation: F,
    pub ram_output_check: F,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct RamReadWriteOutputOpeningClaims<F: Field> {
    pub val: F,
    pub ra: F,
    pub inc: F,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
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

#[cfg(feature = "field-inline")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct FieldInlineStage2OutputOpeningClaims<F: Field> {
    pub product: FieldInlineProductOutputOpeningClaims<F>,
}

#[cfg(feature = "field-inline")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct FieldInlineProductOutputOpeningClaims<F: Field> {
    pub field_rs1_value: F,
    pub field_rs2_value: F,
    pub field_rd_value: F,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct InstructionClaimReductionOutputOpeningClaims<F: Field> {
    pub lookup_output: Option<F>,
    pub left_lookup_operand: F,
    pub right_lookup_operand: F,
    pub left_instruction_input: Option<F>,
    pub right_instruction_input: Option<F>,
}
