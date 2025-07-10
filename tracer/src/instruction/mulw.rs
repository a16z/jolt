use common::constants::virtual_register_index;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::{
        format_b::FormatB, format_i::FormatI, format_j::FormatJ, format_r::FormatR,
        InstructionFormat,
    },
    mul::MUL,
    virtual_sign_extend::VirtualSignExtend,
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction, VirtualInstructionSequence,
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
        let a = cpu.x[self.operands.rs1] as i32;
        let b = cpu.x[self.operands.rs2] as i32;
        cpu.x[self.operands.rd] = a.wrapping_mul(b) as i64;
    }
}

impl RISCVTrace for MULW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence(cpu.xlen == Xlen::Bit32);

        let mut trace = trace;
        for instr in virtual_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl VirtualInstructionSequence for MULW {
    fn virtual_sequence(&self, _: bool) -> Vec<RV32IMInstruction> {
        let mut sequence = vec![];

        let mul = MUL {
            address: self.address,
            operands: FormatR {
                rd: self.operands.rd,
                rs1: self.operands.rs1,
                rs2: self.operands.rs2,
            },
            virtual_sequence_remaining: Some(1),
        };
        sequence.push(mul.into());

        let ext = VirtualSignExtend {
            address: self.address,
            operands: FormatI {
                rd: self.operands.rd,
                rs1: self.operands.rd,
                imm: 0,
            },
            virtual_sequence_remaining: Some(0),
        };
        sequence.push(ext.into());

        sequence
    }
}
