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

use super::{accumulate_word, ProofField};

declare_riscv_instr!(
    name   = FIELD_LOAD_ACCUMULATE_FROM_REGISTER,
    mask   = FieldInlineOp::LoadAccumulateFromRegister.instruction_mask(),
    match  = FieldInlineOp::LoadAccumulateFromRegister.instruction_match(),
    format = FormatFieldInline,
    registers = RegisterStateFieldInline,
    ram    = ()
);

impl FIELD_LOAD_ACCUMULATE_FROM_REGISTER {
    fn exec(
        &self,
        cpu: &mut Cpu,
        _: &mut <FIELD_LOAD_ACCUMULATE_FROM_REGISTER as RISCVInstruction>::RAMAccess,
    ) {
        let x_register = self.operands.rs1.unwrap_or(0);
        let rd_register = self.operands.rd.unwrap_or(0);
        let x_value = cpu.read_register(x_register) as u64;
        let pre_value = cpu.field_registers.read(rd_register);
        let field_value = accumulate_word::<ProofField>(pre_value, x_value);
        cpu.field_registers.write(rd_register, field_value);
    }
}

impl RISCVTrace for FIELD_LOAD_ACCUMULATE_FROM_REGISTER {}
