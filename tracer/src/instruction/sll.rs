use common::constants::virtual_register_index;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::{format_i::FormatI, format_r::FormatR, InstructionFormat},
    mul::MUL,
    virtual_pow2::VirtualPow2,
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction, VirtualInstructionSequence,
};

declare_riscv_instr!(
    name   = SLL,
    mask   = 0xfe00707f,
    match  = 0x00001033,
    format = FormatR,
    ram    = ()
);

impl SLL {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SLL as RISCVInstruction>::RAMAccess) {
        let mask = match cpu.xlen {
            Xlen::Bit32 => 0x1f,
            Xlen::Bit64 => 0x3f,
        };
        cpu.x[self.operands.rd] = cpu.sign_extend(
            cpu.x[self.operands.rs1].wrapping_shl(cpu.x[self.operands.rs2] as u32 & mask),
        );
    }
}

impl RISCVTrace for SLL {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence(cpu.xlen == Xlen::Bit32);
        let mut trace = trace;
        for instr in virtual_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl VirtualInstructionSequence for SLL {
    fn virtual_sequence(&self, is_32: bool) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_pow2 = virtual_register_index(6) as usize;

        let mut virtual_sequence_remaining = self.virtual_sequence_remaining.unwrap_or(1);
        let mut sequence = vec![];

        let pow2 = RV32IMInstruction::VirtualPow2(VirtualPow2 {
            address: self.address,
            operands: FormatI {
                rd: v_pow2,
                rs1: self.operands.rs2,
                imm: 0,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
        });
        sequence.push(pow2);
        virtual_sequence_remaining -= 1;

        let mul = RV32IMInstruction::MUL(MUL {
            address: self.address,
            operands: FormatR {
                rd: self.operands.rd,
                rs1: self.operands.rs1,
                rs2: v_pow2,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
        });
        sequence.push(mul);

        sequence
    }
}
