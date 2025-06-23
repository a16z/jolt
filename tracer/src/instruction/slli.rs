use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    instruction::virtual_muli::VirtualMULI,
};

use super::{
    format::{format_i::FormatI, InstructionFormat},
    RISCVInstruction, RISCVTrace, RV32IMInstruction, VirtualInstructionSequence,
};

declare_riscv_instr!(
    name   = SLLI,
    mask   = 0xfc00707f,
    match  = 0x00001013,
    format = FormatI,
    ram    = ()
);

impl SLLI {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SLLI as RISCVInstruction>::RAMAccess) {
        let mask = match cpu.xlen {
            Xlen::Bit32 => 0x1f,
            Xlen::Bit64 => 0x3f,
        };
        cpu.x[self.operands.rd] =
            cpu.sign_extend(cpu.x[self.operands.rs1].wrapping_shl(self.operands.imm as u32 & mask));
    }
}

impl RISCVTrace for SLLI {
    fn trace(&self, cpu: &mut Cpu) {
        let virtual_sequence = self.virtual_sequence();
        for instr in virtual_sequence {
            instr.trace(cpu);
        }
    }
}

impl VirtualInstructionSequence for SLLI {
    fn virtual_sequence(&self) -> Vec<RV32IMInstruction> {
        let virtual_sequence_remaining = self.virtual_sequence_remaining.unwrap_or(0);
        let mut sequence = vec![];

        let mul = RV32IMInstruction::VirtualMULI(VirtualMULI {
            address: self.address,
            operands: FormatI {
                rd: self.operands.rd,
                rs1: self.operands.rs1,
                // TODO: this only works for Xlen = 32
                imm: (1 << (self.operands.imm % 32)),
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
        });
        sequence.push(mul);

        sequence
    }
}
