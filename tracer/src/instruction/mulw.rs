use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::{format_i::FormatI, format_r::FormatR, InstructionFormat},
    mul::MUL,
    virtual_sign_extend::VirtualSignExtend,
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction,
};

declare_riscv_instr!(
    name   = MULW,
    mask   = 0xfe00707f,
    match  = 0x0200003b,
    format = FormatR,
    ram    = ()
);

impl MULW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <MULW as RISCVInstruction>::RAMAccess) {
        // MULW is an RV64 instruction that multiplies the lower 32 bits of the source registers,
        // placing the sign extension of the lower 32 bits of the result into the destination
        // register.
        let a = cpu.x[self.operands.rs1 as usize] as i32;
        let b = cpu.x[self.operands.rs2 as usize] as i32;
        cpu.x[self.operands.rd as usize] = a.wrapping_mul(b) as i64;
    }
}

impl RISCVTrace for MULW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let inline_sequence = self.inline_sequence(cpu.xlen);

        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(&self, _xlen: Xlen) -> Vec<RV32IMInstruction> {
        let mut sequence = vec![];

        let mul = MUL {
            address: self.address,
            operands: FormatR {
                rd: self.operands.rd,
                rs1: self.operands.rs1,
                rs2: self.operands.rs2,
            },
            inline_sequence_remaining: Some(1),
            is_compressed: self.is_compressed,
        };
        sequence.push(mul.into());

        let ext = VirtualSignExtend {
            address: self.address,
            operands: FormatI {
                rd: self.operands.rd,
                rs1: self.operands.rd,
                imm: 0,
            },
            inline_sequence_remaining: Some(0),
            is_compressed: self.is_compressed,
        };
        sequence.push(ext.into());

        sequence
    }
}
