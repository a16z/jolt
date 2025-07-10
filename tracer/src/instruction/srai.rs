use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    instruction::{
        format::format_virtual_right_shift_i::FormatVirtualRightShiftI, virtual_srai::VirtualSRAI,
    },
};

use super::{
    format::{format_i::FormatI, InstructionFormat},
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction, VirtualInstructionSequence,
};

declare_riscv_instr!(
    name   = SRAI,
    mask   = 0xfc00707f,
    match  = 0x40005013,
    format = FormatI,
    ram    = ()
);

impl SRAI {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SRAI as RISCVInstruction>::RAMAccess) {
        let mask = match cpu.xlen {
            Xlen::Bit32 => 0x1f,
            Xlen::Bit64 => 0x3f,
        };
        cpu.x[self.operands.rd] =
            cpu.sign_extend(cpu.x[self.operands.rs1].wrapping_shr(self.operands.imm as u32 & mask));
    }
}

impl RISCVTrace for SRAI {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence(cpu.xlen == Xlen::Bit32);
        let mut trace = trace;
        for instr in virtual_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl VirtualInstructionSequence for SRAI {
    fn virtual_sequence(&self, is_32: bool) -> Vec<RV32IMInstruction> {
        let virtual_sequence_remaining = self.virtual_sequence_remaining.unwrap_or(0);
        let mut sequence = vec![];

        let (shift, len) = if is_32 {
            (self.operands.imm & 0x1f, 32)
        } else {
            (self.operands.imm & 0x3f, 64)
        };
        let ones = (1u64 << (len - shift)) - 1;
        let bitmask = ones << shift;

        let sra = VirtualSRAI {
            address: self.address,
            operands: FormatVirtualRightShiftI {
                rd: self.operands.rd,
                rs1: self.operands.rs1,
                imm: bitmask,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
        };
        sequence.push(sra.into());

        sequence
    }
}
