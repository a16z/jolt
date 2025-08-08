use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    instruction::{
        format::format_virtual_right_shift_i::FormatVirtualRightShiftI, virtual_srai::VirtualSRAI,
    },
};

use super::virtual_sign_extend::VirtualSignExtend;
use super::{
    format::{format_i::FormatI, InstructionFormat},
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction,
};
use common::constants::virtual_register_index;

declare_riscv_instr!(
    name   = SRAIW,
    mask   = 0xfc00707f,
    match  = 0x4000501b,
    format = FormatI,
    ram    = ()
);

impl SRAIW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SRAIW as RISCVInstruction>::RAMAccess) {
        // SLLIW, SRLIW, and SRAIW are RV64I-only instructions that are analogously defined but
        // operate on 32-bit values and sign-extend their 32-bit results to 64 bits. SLLIW, SRLIW,
        // and SRAIW encodings with imm[5] ≠ 0 are reserved.
        let shamt = (self.operands.imm & 0x1f) as u32;
        cpu.x[self.operands.rd as usize] =
            ((cpu.x[self.operands.rs1 as usize] as i32) >> shamt) as i64;
    }
}

impl RISCVTrace for SRAIW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in virtual_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn virtual_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
        let v_rs1 = virtual_register_index(0);
        let mut sequence = vec![];

        let signext = VirtualSignExtend {
            address: self.address,
            operands: FormatI {
                rd: v_rs1,
                rs1: self.operands.rs1,
                imm: 0,
            },
            virtual_sequence_remaining: Some(2),
            is_compressed: self.is_compressed,
        };
        sequence.push(signext.into());

        let shift = self.operands.imm & 0x1f;
        let len = match xlen {
            Xlen::Bit32 => panic!("SRAIW is invalid in 32b mode"),
            Xlen::Bit64 => 64,
        };
        let ones = (1u128 << (len - shift)) - 1;
        let bitmask = (ones << shift) as u64;

        let sra = VirtualSRAI {
            address: self.address,
            operands: FormatVirtualRightShiftI {
                rd: self.operands.rd,
                rs1: v_rs1,
                imm: bitmask,
            },
            virtual_sequence_remaining: Some(1),
            is_compressed: self.is_compressed,
        };
        sequence.push(sra.into());

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
