use common::constants::virtual_register_index;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::ori::ORI;
use super::slli::SLLI;
use super::virtual_sign_extend::VirtualSignExtend;
use super::{
    format::{
        format_i::FormatI, format_r::FormatR,
        format_virtual_right_shift_r::FormatVirtualRightShiftR, InstructionFormat,
    },
    virtual_shift_right_bitmask::VirtualShiftRightBitmask,
    virtual_srl::VirtualSRL,
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction, VirtualInstructionSequence,
};

declare_riscv_instr!(
    name   = SRLW,
    mask   = 0xfe00707f,
    match  = 0x0000003b | (0b101 << 12),
    format = FormatR,
    ram    = ()
);

impl SRLW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SRLW as RISCVInstruction>::RAMAccess) {
        // SLLW, SRLW, and SRAW are RV64I-only instructions that are analogously defined but operate
        // on 32-bit values and sign-extend their 32-bit results to 64 bits. The shift amount is
        // given by rs2[4:0].
        let shamt = (cpu.x[self.operands.rs2 as usize] & 0x1f) as u32;
        cpu.x[self.operands.rd as usize] =
            ((cpu.x[self.operands.rs1 as usize] as u32) >> shamt) as i32 as i64;
    }
}

impl RISCVTrace for SRLW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in virtual_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl VirtualInstructionSequence for SRLW {
    fn virtual_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_bitmask = virtual_register_index(6);
        let v_rs1 = virtual_register_index(5);

        let mut sequence = vec![];

        let slli = SLLI {
            address: self.address,
            operands: FormatI {
                rd: v_rs1,
                rs1: self.operands.rs1,
                imm: 32,
            },
            virtual_sequence_remaining: Some(4),
            is_compressed: self.is_compressed,
        };
        sequence.extend(slli.virtual_sequence(xlen));

        let ori = ORI {
            address: self.address,
            operands: FormatI {
                rd: v_bitmask,
                rs1: self.operands.rs2,
                imm: 32,
            },
            virtual_sequence_remaining: Some(3),
            is_compressed: self.is_compressed,
        };
        sequence.push(ori.into());

        let bitmask = VirtualShiftRightBitmask {
            address: self.address,
            operands: FormatI {
                rd: v_bitmask,
                rs1: v_bitmask,
                imm: 0,
            },
            virtual_sequence_remaining: Some(2),
            is_compressed: self.is_compressed,
        };
        sequence.push(bitmask.into());

        let srl = VirtualSRL {
            address: self.address,
            operands: FormatVirtualRightShiftR {
                rd: self.operands.rd,
                rs1: v_rs1,
                rs2: v_bitmask,
            },
            virtual_sequence_remaining: Some(1),
            is_compressed: self.is_compressed,
        };
        sequence.push(srl.into());

        let signext = VirtualSignExtend {
            address: self.address,
            operands: FormatI {
                rd: self.operands.rd,
                rs1: self.operands.rd,
                imm: 0,
            },
            virtual_sequence_remaining: Some(0),
            is_compressed: self.is_compressed,
        };
        sequence.push(signext.into());

        sequence
    }
}
