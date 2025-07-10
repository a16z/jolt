use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    instruction::virtual_muli::VirtualMULI,
};

use super::{
    format::{format_i::FormatI, InstructionFormat},
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction, VirtualInstructionSequence,
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
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence(cpu.xlen == Xlen::Bit32);
        let mut trace = trace;
        for instr in virtual_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl VirtualInstructionSequence for SLLI {
    fn virtual_sequence(&self, is_32: bool) -> Vec<RV32IMInstruction> {
        let virtual_sequence_remaining = self.virtual_sequence_remaining.unwrap_or(0);
        let mut sequence = vec![];

        let mul = RV32IMInstruction::VirtualMULI(VirtualMULI {
            address: self.address,
            operands: FormatI {
                rd: self.operands.rd,
                rs1: self.operands.rs1,
                imm: if is_32 {
                    (1 << (self.operands.imm & 0x1F)) //low 5bits
                } else {
                    (1 << (self.operands.imm & 0x3F)) //low 6bits
                },
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
        });
        sequence.push(mul);

        sequence
    }
}
