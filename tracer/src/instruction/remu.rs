use common::constants::virtual_register_index;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    add::ADD,
    format::{
        format_b::FormatB, format_i::FormatI, format_j::FormatJ, format_r::FormatR,
        InstructionFormat,
    },
    mul::MUL,
    virtual_advice::VirtualAdvice,
    virtual_assert_eq::VirtualAssertEQ,
    virtual_assert_lte::VirtualAssertLTE,
    virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder,
    virtual_move::VirtualMove,
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction, VirtualInstructionSequence,
};

declare_riscv_instr!(
    name   = REMU,
    mask   = 0xfe00707f,
    match  = 0x02007033,
    format = FormatR,
    ram    = ()
);

impl REMU {
    fn exec(&self, cpu: &mut Cpu, _: &mut <REMU as RISCVInstruction>::RAMAccess) {
        let dividend = cpu.unsigned_data(cpu.x[self.operands.rs1]);
        let divisor = cpu.unsigned_data(cpu.x[self.operands.rs2]);
        cpu.x[self.operands.rd] = match divisor {
            0 => cpu.sign_extend(dividend as i64),
            _ => cpu.sign_extend(dividend.wrapping_rem(divisor) as i64),
        };
    }
}

impl RISCVTrace for REMU {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let mut virtual_sequence = self.virtual_sequence(cpu.xlen == Xlen::Bit32);
        if let RV32IMInstruction::VirtualAdvice(instr) = &mut virtual_sequence[0] {
            instr.advice = if cpu.unsigned_data(cpu.x[self.operands.rs2]) == 0 {
                match cpu.xlen {
                    Xlen::Bit32 => u32::MAX as u64,
                    Xlen::Bit64 => u64::MAX,
                }
            } else {
                cpu.unsigned_data(cpu.x[self.operands.rs1])
                    / cpu.unsigned_data(cpu.x[self.operands.rs2])
            };
        } else {
            panic!("Expected Advice instruction");
        }
        if let RV32IMInstruction::VirtualAdvice(instr) = &mut virtual_sequence[1] {
            instr.advice = match cpu.unsigned_data(cpu.x[self.operands.rs2]) {
                0 => cpu.unsigned_data(cpu.x[self.operands.rs1]),
                divisor => {
                    let dividend = cpu.unsigned_data(cpu.x[self.operands.rs1]);
                    let quotient = dividend / divisor;
                    dividend - quotient * divisor
                }
            };
        } else {
            panic!("Expected Advice instruction");
        }

        let mut trace = trace;
        for instr in virtual_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl VirtualInstructionSequence for REMU {
    fn virtual_sequence(&self, _: bool) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_0 = virtual_register_index(0) as usize;
        let v_q = virtual_register_index(1) as usize;
        let v_r = virtual_register_index(2) as usize;
        let v_qy = virtual_register_index(3) as usize;

        let mut sequence = vec![];

        let advice = VirtualAdvice {
            address: self.address,
            operands: FormatJ { rd: v_q, imm: 0 },
            virtual_sequence_remaining: Some(6),
            advice: 0,
        };
        sequence.push(advice.into());

        let advice = VirtualAdvice {
            address: self.address,
            operands: FormatJ { rd: v_r, imm: 0 },
            virtual_sequence_remaining: Some(5),
            advice: 0,
        };
        sequence.push(advice.into());

        let mul = MUL {
            address: self.address,
            operands: FormatR {
                rd: v_qy,
                rs1: v_q,
                rs2: self.operands.rs2,
            },
            virtual_sequence_remaining: Some(4),
        };
        sequence.push(mul.into());

        let assert_remainder = VirtualAssertValidUnsignedRemainder {
            address: self.address,
            operands: FormatB {
                rs1: v_r,
                rs2: self.operands.rs2,
                imm: 0,
            },
            virtual_sequence_remaining: Some(3),
        };
        sequence.push(assert_remainder.into());

        let assert_lte = VirtualAssertLTE {
            address: self.address,
            operands: FormatB {
                rs1: v_qy,
                rs2: self.operands.rs1,
                imm: 0,
            },
            virtual_sequence_remaining: Some(2),
        };
        sequence.push(assert_lte.into());

        let add = ADD {
            address: self.address,
            operands: FormatR {
                rd: v_0,
                rs1: v_qy,
                rs2: v_r,
            },
            virtual_sequence_remaining: Some(1),
        };
        sequence.push(add.into());

        let assert_eq = VirtualAssertEQ {
            address: self.address,
            operands: FormatB {
                rs1: v_0,
                rs2: self.operands.rs1,
                imm: 0,
            },
            virtual_sequence_remaining: Some(1),
        };
        sequence.push(assert_eq.into());

        let virtual_move = VirtualMove {
            address: self.address,
            operands: FormatI {
                rd: self.operands.rd,
                rs1: v_r,
                imm: 0,
            },
            virtual_sequence_remaining: Some(0),
        };
        sequence.push(virtual_move.into());

        sequence
    }
}
