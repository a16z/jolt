use jolt_riscv::FieldInlineOp;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::Cpu,
    instruction::{format::format_field_inline::FormatFieldInline, RISCVInstruction, RISCVTrace},
};

use super::{execute_binary, FieldInlineCycleData, ProofField};

declare_riscv_instr!(
    name   = FIELD_ADD,
    mask   = FieldInlineOp::Add.instruction_mask(),
    match  = FieldInlineOp::Add.instruction_match(),
    format = FormatFieldInline,
    ram    = FieldInlineCycleData
);

impl FIELD_ADD {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <FIELD_ADD as RISCVInstruction>::RAMAccess) {
        *ram_access =
            execute_binary::<ProofField>(FieldInlineOp::Add, self.operands, cpu, |left, right| {
                left + right
            })
            .into();
    }
}

impl RISCVTrace for FIELD_ADD {}
