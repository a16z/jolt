use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    instruction::virtual_muli::VirtualMULI,
};
use serde::{Deserialize, Serialize};

use super::virtual_sign_extend::VirtualSignExtend;
use super::{
    format::{format_i::FormatI, InstructionFormat},
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction, VirtualInstructionSequence,
};

declare_riscv_instr!(
    name   = SLLIW,
    mask   = 0xfc00707f,
    match  = 0x0000101b,
    format = FormatI,
    ram    = ()
);

impl SLLIW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SLLIW as RISCVInstruction>::RAMAccess) {
        // SLLIW, SRLIW, and SRAIW are RV64I-only instructions that are analogously defined but
        // operate on 32-bit values and sign-extend their 32-bit results to 64 bits. SLLIW, SRLIW,
        // and SRAIW encodings with imm[5] ≠ 0 are reserved.
        let shamt = (self.operands.imm & 0x1f) as u32;
        cpu.x[self.operands.rd] = ((cpu.x[self.operands.rs1] as u32) << shamt) as i32 as i64;
    }
}

impl RISCVTrace for SLLIW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence(cpu.xlen == Xlen::Bit32);
        let mut trace = trace;
        for instr in virtual_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl VirtualInstructionSequence for SLLIW {
    fn virtual_sequence(&self, is_32: bool) -> Vec<RV32IMInstruction> {
        let virtual_sequence_remaining = self.virtual_sequence_remaining.unwrap_or(0);
        let mut sequence = vec![];

        let mul = VirtualMULI {
            address: self.address,
            operands: FormatI {
                rd: self.operands.rd,
                rs1: self.operands.rs1,
                imm: if is_32 {
                    panic!("SLLIW is invalid in 32b mode")
                } else {
                    (1 << (self.operands.imm & 0x1F)) //low 5bits
                },
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
        };
        sequence.push(mul.into());

        let signext = VirtualSignExtend {
            address: self.address,
            operands: FormatI {
                rd: self.operands.rd,
                rs1: self.operands.rd,
                imm: 0,
            },
            virtual_sequence_remaining: Some(0),
        };
        sequence.push(signext.into());

        sequence
    }
}
