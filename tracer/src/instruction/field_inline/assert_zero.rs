use jolt_field::Ring;
use jolt_riscv::FieldInlineOp;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::Cpu,
    instruction::{
        format::format_field_inline::FormatFieldInline,
        registers::field_inline::RegisterStateFieldInline, RISCVInstruction, RISCVTrace,
    },
};

use super::{decode_field, ProofField};

declare_riscv_instr!(
    name   = FIELD_ASSERT_ZERO,
    mask   = FieldInlineOp::AssertZero.instruction_mask(),
    match  = FieldInlineOp::AssertZero.instruction_match(),
    format = FormatFieldInline,
    registers = RegisterStateFieldInline,
    ram    = ()
);

impl FIELD_ASSERT_ZERO {
    fn exec(&self, cpu: &mut Cpu, _: &mut <FIELD_ASSERT_ZERO as RISCVInstruction>::RAMAccess) {
        let register = self.operands.rs1.unwrap_or(0);
        let value = cpu.field_registers.read(register);
        assert_eq!(
            decode_field::<ProofField>(value),
            ProofField::from_u64(0),
            "FIELD_ASSERT_ZERO of nonzero field register {register} at pc 0x{:x}",
            cpu.read_pc(),
        );
    }
}

impl RISCVTrace for FIELD_ASSERT_ZERO {}
