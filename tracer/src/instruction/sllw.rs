use crate::utils::virtual_registers::allocate_virtual_register;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::virtual_sign_extend::VirtualSignExtend;
use super::{
    format::{format_i::FormatI, format_r::FormatR, InstructionFormat},
    mul::MUL,
    virtual_pow2_w::VirtualPow2W,
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction,
};

declare_riscv_instr!(
    name   = SLLW,
    mask   = 0xfe00707f,
    match  = 0x0000003b | (0b001 << 12),
    format = FormatR,
    ram    = ()
);

impl SLLW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SLLW as RISCVInstruction>::RAMAccess) {
        // SLLW, SRLW, and SRAW are RV64I-only instructions that are analogously defined but operate
        // on 32-bit values and sign-extend their 32-bit results to 64 bits. The shift amount is
        // given by rs2[4:0].
        let shamt = (cpu.x[self.operands.rs2 as usize] & 0x1f) as u32;
        cpu.x[self.operands.rd as usize] =
            ((cpu.x[self.operands.rs1 as usize] as u32) << shamt) as i32 as i64;
    }
}

impl RISCVTrace for SLLW {
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

        let pow2w = VirtualPow2W {
            address: self.address,
            operands: FormatI {
                rd: *v_pow2,
                rs1: self.operands.rs2,
                imm: 0,
            },
            inline_sequence_remaining: Some(2),
            is_compressed: self.is_compressed,
        };
        sequence.push(pow2w.into());

        let mul = MUL {
            address: self.address,
            operands: FormatR {
                rd: self.operands.rd,
                rs1: self.operands.rs1,
                rs2: *v_pow2,
            },
            inline_sequence_remaining: Some(1),
            is_compressed: self.is_compressed,
        };
        sequence.push(mul.into());

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
