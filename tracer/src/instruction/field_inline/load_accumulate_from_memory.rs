use jolt_program::field_inline::{
    FieldInlineBridge, FieldInlineTraceData, FieldRegisterRead, FieldRegisterWrite,
};
use jolt_riscv::FieldInlineOp;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::Cpu,
    instruction::{format::format_field_inline::FormatFieldInline, RISCVInstruction, RISCVTrace},
};

use super::{accumulate_word, FieldInlineCycleData, ProofField};

declare_riscv_instr!(
    name   = FIELD_LOAD_ACCUMULATE_FROM_MEMORY,
    mask   = FieldInlineOp::LoadAccumulateFromMemory.instruction_mask(),
    match  = FieldInlineOp::LoadAccumulateFromMemory.instruction_match(),
    format = FormatFieldInline,
    ram    = FieldInlineCycleData
);

impl FIELD_LOAD_ACCUMULATE_FROM_MEMORY {
    fn exec(
        &self,
        cpu: &mut Cpu,
        ram_access: &mut <FIELD_LOAD_ACCUMULATE_FROM_MEMORY as RISCVInstruction>::RAMAccess,
    ) {
        let x_base = self.operands.rs1.unwrap_or(0);
        let x_register = self.operands.rd.unwrap_or(0);
        let field_register = self.operands.rs2.unwrap_or(0);
        // The integer destination must take the write: x0 would drop it and the
        // load rows equate the rd write with the loaded word.
        assert!(
            x_register != 0,
            "field-inline LoadAccumulateFromMemory to x0 at pc 0x{:x}: the integer destination must be a real register",
            cpu.read_pc(),
        );
        let address = (cpu.read_register(x_base) as u64).wrapping_add(self.operands.imm as u64);
        let (word, ram_read) = cpu
            .get_mut_mmu()
            .load_doubleword(address)
            .unwrap_or_else(|_| panic!("MMU load error at pc 0x{:x}", cpu.read_pc()));
        cpu.write_register(x_register as usize, word as i64);
        let pre_value = cpu.field_registers.read(field_register);
        let value = accumulate_word::<ProofField>(pre_value, word);
        cpu.field_registers.write(field_register, value);
        *ram_access = FieldInlineCycleData {
            trace: Some(FieldInlineTraceData {
                op: Some(FieldInlineOp::LoadAccumulateFromMemory),
                rs1: Some(FieldRegisterRead {
                    register: field_register,
                    value: pre_value,
                }),
                rd: Some(FieldRegisterWrite {
                    register: field_register,
                    pre_value,
                    post_value: value,
                }),
                bridge: Some(FieldInlineBridge::LoadAccumulateFromMemory {
                    x_base,
                    x_register,
                    word,
                    field_value: value,
                }),
                ..Default::default()
            }),
            ram_read: Some(ram_read),
        };
    }
}

impl RISCVTrace for FIELD_LOAD_ACCUMULATE_FROM_MEMORY {}
