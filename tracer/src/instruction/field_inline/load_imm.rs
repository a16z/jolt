use jolt_program::field_inline::{FieldEncodedValue, FieldInlineTraceData, FieldRegisterWrite};
use jolt_riscv::FieldInlineOp;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::Cpu,
    instruction::{format::format_field_inline::FormatFieldInline, RISCVInstruction, RISCVTrace},
};

use super::FieldInlineCycleData;

declare_riscv_instr!(
    name   = FIELD_LOAD_IMM,
    mask   = FieldInlineOp::LoadImm.instruction_mask(),
    match  = FieldInlineOp::LoadImm.instruction_match(),
    format = FormatFieldInline,
    ram    = FieldInlineCycleData
);

impl FIELD_LOAD_IMM {
    fn exec(
        &self,
        cpu: &mut Cpu,
        ram_access: &mut <FIELD_LOAD_IMM as RISCVInstruction>::RAMAccess,
    ) {
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
        let pre_value = cpu.field_registers.read(rd_register);
        cpu.field_registers.write(rd_register, value);
        *ram_access = FieldInlineTraceData {
            op: Some(FieldInlineOp::LoadImm),
            rd: Some(FieldRegisterWrite {
                register: rd_register,
                pre_value,
                post_value: value,
            }),
            ..Default::default()
        }
        .into();
    }
}

impl RISCVTrace for FIELD_LOAD_IMM {}
