use jolt_field::Ring;
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
    name   = FIELD_ASSERT_ZERO,
    mask   = FieldInlineOp::AssertZero.instruction_mask(),
    match  = FieldInlineOp::AssertZero.instruction_match(),
    format = FormatFieldInline,
    ram    = FieldInlineCycleData
);

impl FIELD_ASSERT_ZERO {
    fn exec(
        &self,
        cpu: &mut Cpu,
        ram_access: &mut <FIELD_ASSERT_ZERO as RISCVInstruction>::RAMAccess,
    ) {
        let register = self.operands.rs1.unwrap_or(0);
        let value = cpu.field_registers.read(register);
        assert_eq!(
            decode_field::<ProofField>(value),
            ProofField::from_u64(0),
            "FIELD_ASSERT_ZERO of nonzero field register {register} at pc 0x{:x}",
            cpu.read_pc(),
        );
        *ram_access = FieldInlineTraceData {
            op: Some(FieldInlineOp::AssertZero),
            rs1: Some(FieldRegisterRead { register, value }),
            ..Default::default()
        }
        .into();
    }
}

impl RISCVTrace for FIELD_ASSERT_ZERO {}
