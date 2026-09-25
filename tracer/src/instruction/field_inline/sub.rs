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

use super::{execute_binary, ProofField};

declare_riscv_instr!(
    name   = FIELD_SUB,
    mask   = FieldInlineOp::Sub.instruction_mask(),
    match  = FieldInlineOp::Sub.instruction_match(),
    format = FormatFieldInline,
    registers = RegisterStateFieldInline,
    ram    = ()
);

impl FIELD_SUB {
    fn exec(&self, cpu: &mut Cpu, _: &mut <FIELD_SUB as RISCVInstruction>::RAMAccess) {
        execute_binary::<ProofField>(self.operands, cpu, |left, right| left - right);
    }
}

impl RISCVTrace for FIELD_SUB {}
