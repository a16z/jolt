use jolt_program::field_inline::FieldEncodedValue;
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

declare_riscv_instr!(
    name   = FIELD_LOAD_IMM,
    mask   = FieldInlineOp::LoadImm.instruction_mask(),
    match  = FieldInlineOp::LoadImm.instruction_match(),
    format = FormatFieldInline,
    registers = RegisterStateFieldInline,
    ram    = ()
);

impl FIELD_LOAD_IMM {
    fn exec(&self, cpu: &mut Cpu, _: &mut <FIELD_LOAD_IMM as RISCVInstruction>::RAMAccess) {
        let rd_register = self.operands.rd.unwrap_or(0);
        // Decoded FIELD_LOAD_IMM immediates are zero-extended 12-bit values (0..=4095);
        // anything else can only arrive through synthetic instruction construction and
        // indicates a caller bug, so fail loudly instead of loading zero.
        let value = u64::try_from(self.operands.imm).map_or_else(
            |_| {
                panic!(
                    "FIELD_LOAD_IMM with out-of-range immediate {} at pc 0x{:x}: decoded \
                     field-inline immediates are zero-extended 12-bit values",
                    self.operands.imm,
                    cpu.read_pc(),
                )
            },
            FieldEncodedValue::from_u64,
        );
        cpu.field_registers.write(rd_register, value);
    }
}

impl RISCVTrace for FIELD_LOAD_IMM {}
