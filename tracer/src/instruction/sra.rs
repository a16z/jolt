use common::constants::virtual_register_index;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::{
        format_i::FormatI, format_r::FormatR,
        format_virtual_right_shift_r::FormatVirtualRightShiftR, InstructionFormat,
    },
    virtual_shift_right_bitmask::VirtualShiftRightBitmask,
    virtual_sra::VirtualSRA,
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction,
};

declare_riscv_instr!(
    name   = SRA,
    mask   = 0xfe00707f,
    match  = 0x40005033,
    format = FormatR,
    ram    = ()
);

impl SRA {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SRA as RISCVInstruction>::RAMAccess) {
        let mask = match cpu.xlen {
            Xlen::Bit32 => 0x1f,
            Xlen::Bit64 => 0x3f,
        };
        cpu.x[self.operands.rd as usize] = cpu.sign_extend(
            cpu.x[self.operands.rs1 as usize]
                .wrapping_shr(cpu.x[self.operands.rs2 as usize] as u32 & mask),
        );
    }
}

impl RISCVTrace for SRA {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let inline_sequence = self.inline_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(&self, _xlen: Xlen) -> Vec<RV32IMInstruction> {
        let v_bitmask = virtual_register_index(6);

        let mut sequence = vec![];
        let mut inline_sequence_remaining = self.inline_sequence_remaining.unwrap_or(1);

        let bitmask = VirtualShiftRightBitmask {
            address: self.address,
            operands: FormatI {
                rd: v_bitmask,
                rs1: self.operands.rs2,
                imm: 0,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(bitmask.into());
        inline_sequence_remaining -= 1;

        let sra = VirtualSRA {
            address: self.address,
            operands: FormatVirtualRightShiftR {
                rd: self.operands.rd,
                rs1: self.operands.rs1,
                rs2: v_bitmask,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(sra.into());

        sequence
    }
}
