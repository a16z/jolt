use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    instruction::{
        format::format_virtual_right_shift_i::FormatVirtualRightShiftI, virtual_srli::VirtualSRLI,
    },
};

use super::slli::SLLI;
use super::virtual_sign_extend::VirtualSignExtend;
use super::{
    format::{format_i::FormatI, InstructionFormat},
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction,
};
use crate::utils::virtual_registers::allocate_virtual_register;

declare_riscv_instr!(
    name   = SRLIW,
    mask   = 0xfc00707f,
    match  = 0x0000501b,
    format = FormatI,
    ram    = ()
);

impl SRLIW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SRLIW as RISCVInstruction>::RAMAccess) {
        // SLLIW, SRLIW, and SRAIW are RV64I-only instructions that are analogously defined but
        // operate on 32-bit values and sign-extend their 32-bit results to 64 bits. SLLIW, SRLIW,
        // and SRAIW encodings with imm[5] ≠ 0 are reserved.
        let shamt = (self.operands.imm & 0x1f) as u32;
        cpu.x[self.operands.rd as usize] =
            ((cpu.x[self.operands.rs1 as usize] as u32) >> shamt) as i32 as i64;
    }
}

impl RISCVTrace for SRLIW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let inline_sequence = self.inline_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
        let v_rs1 = allocate_virtual_register();
        let mut sequence = vec![];

        let slli = SLLI {
            address: self.address,
            operands: FormatI {
                rd: *v_rs1,
                rs1: self.operands.rs1,
                imm: 32,
            },
            inline_sequence_remaining: Some(2),
            is_compressed: self.is_compressed,
        };
        sequence.extend(slli.inline_sequence(xlen));

        let (shift, len) = match xlen {
            Xlen::Bit32 => panic!("SRLIW is invalid in 32b mode"),
            Xlen::Bit64 => ((self.operands.imm & 0x1f) + 32, 64),
        };
        let ones = (1u128 << (len - shift)) - 1;
        let bitmask = (ones << shift) as u64;

        let srl = VirtualSRLI {
            address: self.address,
            operands: FormatVirtualRightShiftI {
                rd: self.operands.rd,
                rs1: *v_rs1,
                imm: bitmask,
            },
            inline_sequence_remaining: Some(1),
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
            inline_sequence_remaining: Some(0),
            is_compressed: self.is_compressed,
        };
        sequence.push(signext.into());
        sequence
    }
}
