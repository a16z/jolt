use serde::{Deserialize, Serialize};

use crate::utils::virtual_registers::allocate_virtual_register;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::{format_i::FormatI, format_r::FormatR, InstructionFormat},
    mul::MUL,
    virtual_pow2::VirtualPow2,
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction,
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
        cpu.x[self.operands.rd as usize] = cpu.sign_extend(
            cpu.x[self.operands.rs1 as usize]
                .wrapping_shl(cpu.x[self.operands.rs2 as usize] as u32 & mask),
        );
    }
}

impl RISCVTrace for SLL {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let inline_sequence = self.inline_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(&self, _xlen: Xlen) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_pow2 = allocate_virtual_register();

        let mut sequence = vec![];
        let mut inline_sequence_remaining = self.inline_sequence_remaining.unwrap_or(1);

        let pow2 = RV32IMInstruction::VirtualPow2(VirtualPow2 {
            address: self.address,
            operands: FormatI {
                rd: *v_pow2,
                rs1: self.operands.rs2,
                imm: 0,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        });
        sequence.push(pow2);
        inline_sequence_remaining -= 1;

        let mul = RV32IMInstruction::MUL(MUL {
            address: self.address,
            operands: FormatR {
                rd: self.operands.rd,
                rs1: self.operands.rs1,
                rs2: *v_pow2,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        });
        sequence.push(mul);

        sequence
    }
}
