use crate::utils::virtual_registers::allocate_virtual_register;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::virtual_sign_extend::VirtualSignExtend;
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
    name   = SRAW,
    mask   = 0xfe00707f,
    match  = 0x4000003b | (0b101 << 12),
    format = FormatR,
    ram    = ()
);

impl SRAW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SRAW as RISCVInstruction>::RAMAccess) {
        // SLLW, SRLW, and SRAW are RV64I-only instructions that are analogously defined but operate
        // on 32-bit values and sign-extend their 32-bit results to 64 bits. The shift amount is
        // given by rs2[4:0].
        let shamt = (cpu.x[self.operands.rs2 as usize] & 0x1f) as u32;
        cpu.x[self.operands.rd as usize] =
            ((cpu.x[self.operands.rs1 as usize] as i32) >> shamt) as i64;
    }
}

impl RISCVTrace for SRAW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let inline_sequence = self.inline_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(&self, _xlen: Xlen) -> Vec<RV32IMInstruction> {
        let v_rs1 = allocate_virtual_register();
        let v_bitmask = allocate_virtual_register();

        let mut sequence = vec![];

        let signext = VirtualSignExtend {
            address: self.address,
            operands: FormatI {
                rd: *v_rs1,
                rs1: self.operands.rs1,
                imm: 0,
            },
            inline_sequence_remaining: Some(3),
            is_compressed: self.is_compressed,
        };
        sequence.push(signext.into());

        let bitmask = VirtualShiftRightBitmask {
            address: self.address,
            operands: FormatI {
                rd: *v_bitmask,
                rs1: self.operands.rs2,
                imm: 0,
            },
            inline_sequence_remaining: Some(2),
            is_compressed: self.is_compressed,
        };
        sequence.push(bitmask.into());

        let sra = VirtualSRA {
            address: self.address,
            operands: FormatVirtualRightShiftR {
                rd: self.operands.rd,
                rs1: *v_rs1,
                rs2: *v_bitmask,
            },
            inline_sequence_remaining: Some(1),
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
            inline_sequence_remaining: Some(0),
            is_compressed: self.is_compressed,
        };
        sequence.push(signext.into());

        sequence
    }
}
