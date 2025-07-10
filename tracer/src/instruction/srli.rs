use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    instruction::{
        format::format_virtual_right_shift_i::FormatVirtualRightShiftI, virtual_srli::VirtualSRLI,
    },
};

use super::{
    format::{format_i::FormatI, InstructionFormat},
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction, VirtualInstructionSequence,
};

declare_riscv_instr!(
    name   = SRLI,
    mask   = 0xfc00707f,
    match  = 0x00005013,
    format = FormatI,
    ram    = ()
);

impl SRLI {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SRLI as RISCVInstruction>::RAMAccess) {
        let mask = match cpu.xlen {
            Xlen::Bit32 => 0x1f,
            Xlen::Bit64 => 0x3f,
        };
        cpu.x[self.operands.rd] = cpu.sign_extend(
            cpu.unsigned_data(cpu.x[self.operands.rs1])
                .wrapping_shr(self.operands.imm as u32 & mask) as i64,
        );
    }
}

impl RISCVTrace for SRLI {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence();
        let mut trace = trace;
        for instr in virtual_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl VirtualInstructionSequence for SRLI {
    fn virtual_sequence(&self) -> Vec<RV32IMInstruction> {
        let virtual_sequence_remaining = self.virtual_sequence_remaining.unwrap_or(0);
        let mut sequence = vec![];

        // TODO: this only works for Xlen = 32
        let shift = self.operands.imm % 32;
        let ones = (1u64 << (32 - shift)) - 1;
        let bitmask = ones << shift;

        let srl = VirtualSRLI {
            address: self.address,
            operands: FormatVirtualRightShiftI {
                rd: self.operands.rd,
                rs1: self.operands.rs1,
                imm: bitmask,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
        };
        sequence.push(srl.into());

        sequence
    }
}
