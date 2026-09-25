use jolt_program::field_inline::{FieldInlineTraceData, FieldRegisterRead};
use jolt_riscv::FieldInlineOp;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::Cpu,
    instruction::{format::format_field_inline::FormatFieldInline, RISCVInstruction, RISCVTrace},
};

use super::{decode_field, FieldInlineCycleData, ProofField};

declare_riscv_instr!(
    name   = FIELD_ASSERT_EQ,
    mask   = FieldInlineOp::AssertEq.instruction_mask(),
    match  = FieldInlineOp::AssertEq.instruction_match(),
    format = FormatFieldInline,
    ram    = FieldInlineCycleData
);

impl FIELD_ASSERT_EQ {
    fn exec(
        &self,
        cpu: &mut Cpu,
        ram_access: &mut <FIELD_ASSERT_EQ as RISCVInstruction>::RAMAccess,
    ) {
        let rs1_register = self.operands.rs1.unwrap_or(0);
        let rs2_register = self.operands.rs2.unwrap_or(0);
        let rs1_value = cpu.field_registers.read(rs1_register);
        let rs2_value = cpu.field_registers.read(rs2_register);
        assert_eq!(
            decode_field::<ProofField>(rs1_value),
            decode_field::<ProofField>(rs2_value),
            "field-inline assert_eq failed"
        );
        *ram_access = FieldInlineTraceData {
            op: Some(FieldInlineOp::AssertEq),
            rs1: Some(FieldRegisterRead {
                register: rs1_register,
                value: rs1_value,
            }),
            rs2: Some(FieldRegisterRead {
                register: rs2_register,
                value: rs2_value,
            }),
            ..Default::default()
        }
        .into();
    }
}

impl RISCVTrace for FIELD_ASSERT_EQ {}
