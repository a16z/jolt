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

use super::{decode_field, encode_field, ProofField};

declare_riscv_instr!(
    name   = FIELD_ADD,
    mask   = FieldInlineOp::Add.instruction_mask(),
    match  = FieldInlineOp::Add.instruction_match(),
    format = FormatFieldInline,
    registers = RegisterStateFieldInline,
    ram    = ()
);

impl FIELD_ADD {
    fn exec(&self, cpu: &mut Cpu, _: &mut <FIELD_ADD as RISCVInstruction>::RAMAccess) {
        let rs1_register = self.operands.rs1.unwrap_or(0);
        let rs2_register = self.operands.rs2.unwrap_or(0);
        let rd_register = self.operands.rd.unwrap_or(0);
        let rs1_value = cpu.field_registers.read(rs1_register);
        let rs2_value = cpu.field_registers.read(rs2_register);
        let post_value = encode_field(
            decode_field::<ProofField>(rs1_value) + decode_field::<ProofField>(rs2_value),
        );
        cpu.field_registers.write(rd_register, post_value);
    }
}

impl RISCVTrace for FIELD_ADD {}
