use jolt_riscv::FieldInlineOp;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::Cpu,
    instruction::{format::format_field_inline::FormatFieldInline, RISCVInstruction, RISCVTrace},
};

use super::{execute_binary, FieldInlineCycleData, ProofField};

declare_riscv_instr!(
    name   = FIELD_SUB,
    mask   = FieldInlineOp::Sub.instruction_mask(),
    match  = FieldInlineOp::Sub.instruction_match(),
    format = FormatFieldInline,
    ram    = FieldInlineCycleData
);

impl FIELD_SUB {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <FIELD_SUB as RISCVInstruction>::RAMAccess) {
        *ram_access =
            execute_binary::<ProofField>(FieldInlineOp::Sub, self.operands, cpu, |left, right| {
                left - right
            })
            .into();
    }
}

impl RISCVTrace for FIELD_SUB {}
