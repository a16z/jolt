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
        let virtual_sequence = self.virtual_sequence();
        let mut trace = trace;
        for instr in virtual_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl VirtualInstructionSequence for SLLI {
    fn virtual_sequence(&self) -> Vec<RV32IMInstruction> {
        let virtual_sequence_remaining = self.virtual_sequence_remaining.unwrap_or(0);
        let mut sequence = vec![];

        // Determine word size based on immediate value and instruction encoding
        // For SLLI: RV32 uses 5-bit immediates (0-31), RV64 uses 6-bit immediates (0-63)
        let is_64bit = self.operands.imm > 31;
        let shift_mask = if is_64bit { 0x3f } else { 0x1f };
        let shift = self.operands.imm & shift_mask;
        let mul = RV32IMInstruction::VirtualMULI(VirtualMULI {
            address: self.address,
            operands: FormatI {
                rd: self.operands.rd,
                rs1: self.operands.rs1,
                imm: (1 << shift),
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
        });
        sequence.push(mul);

        sequence
    }
}
