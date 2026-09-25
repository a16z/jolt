use jolt_riscv::FieldInlineOp;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::Cpu,
    instruction::{format::format_field_inline::FormatFieldInline, RISCVInstruction, RISCVTrace},
};

use super::{execute_binary, FieldInlineCycleData, ProofField};

declare_riscv_instr!(
    name   = FIELD_MUL,
    mask   = FieldInlineOp::Mul.instruction_mask(),
    match  = FieldInlineOp::Mul.instruction_match(),
    format = FormatFieldInline,
    ram    = FieldInlineCycleData
);

impl FIELD_MUL {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <FIELD_MUL as RISCVInstruction>::RAMAccess) {
        let mut trace =
            execute_binary::<ProofField>(FieldInlineOp::Mul, self.operands, cpu, |left, right| {
                left * right
            });
        trace.product = trace.rd.map(|write| write.post_value);
        *ram_access = trace.into();
    }
}

impl RISCVTrace for FIELD_MUL {}
