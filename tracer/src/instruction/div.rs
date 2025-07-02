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
    virtual_assert_valid_div0::VirtualAssertValidDiv0,
    virtual_assert_valid_signed_remainder::VirtualAssertValidSignedRemainder,
    virtual_move::VirtualMove,
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction, VirtualInstructionSequence,
};

declare_riscv_instr!(
    name   = DIV,
    mask   = 0xfe00707f,
    match  = 0x02004033,
    format = FormatR,
    ram    = ()
);

impl DIV {
    fn exec(&self, cpu: &mut Cpu, _: &mut <DIV as RISCVInstruction>::RAMAccess) {
        let dividend = cpu.x[self.operands.rs1];
        let divisor = cpu.x[self.operands.rs2];
        if divisor == 0 {
            cpu.x[self.operands.rd] = -1;
        } else if dividend == cpu.most_negative() && divisor == -1 {
            cpu.x[self.operands.rd] = dividend;
        } else {
            cpu.x[self.operands.rd] = cpu.sign_extend(dividend.wrapping_div(divisor))
        }
    }
}

impl RISCVTrace for DIV {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        // DIV operands
        let x = cpu.x[self.operands.rs1];
        let y = cpu.x[self.operands.rs2];

        let (quotient, remainder) = match cpu.xlen {
            Xlen::Bit32 => {
                if y == 0 {
                    (u32::MAX as u64, x as u64)
                } else {
                    let mut quotient = x as i32 / y as i32;
                    let mut remainder = x as i32 % y as i32;
                    if (remainder < 0 && (y as i32) > 0) || (remainder > 0 && (y as i32) < 0) {
                        remainder += y as i32;
                        quotient -= 1;
                    }
                    (quotient as u32 as u64, remainder as u32 as u64)
                }
            }
            Xlen::Bit64 => {
                if y == 0 {
                    (u64::MAX, x as u64)
                } else {
                    let mut quotient = x / y;
                    let mut remainder = x % y;
                    if (remainder < 0 && y > 0) || (remainder > 0 && y < 0) {
                        remainder += y;
                        quotient -= 1;
                    }
                    (quotient as u64, remainder as u64)
                }
            }
        };

        let mut virtual_sequence = self.virtual_sequence();
        if let RV32IMInstruction::VirtualAdvice(instr) = &mut virtual_sequence[0] {
            instr.advice = quotient;
        } else {
            panic!("Expected Advice instruction");
        }
        if let RV32IMInstruction::VirtualAdvice(instr) = &mut virtual_sequence[1] {
            instr.advice = remainder;
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

impl VirtualInstructionSequence for DIV {
    fn virtual_sequence(&self) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_0 = virtual_register_index(0) as usize;
        let v_q = virtual_register_index(1) as usize;
        let v_r = virtual_register_index(2) as usize;
        let v_qy = virtual_register_index(3) as usize;

        let mut sequence = vec![];

        let advice = VirtualAdvice {
            address: self.address,
            operands: FormatJ { rd: v_q, imm: 0 },
            virtual_sequence_remaining: Some(7),
            advice: 0,
        };
        sequence.push(advice.into());

        let advice = VirtualAdvice {
            address: self.address,
            operands: FormatJ { rd: v_r, imm: 0 },
            virtual_sequence_remaining: Some(6),
            advice: 0,
        };
        sequence.push(advice.into());

        let is_valid = VirtualAssertValidSignedRemainder {
            address: self.address,
            operands: FormatB {
                rs1: v_r,
                rs2: self.operands.rs2,
                imm: 0,
            },
            virtual_sequence_remaining: Some(5),
        };
        sequence.push(is_valid.into());

        let is_valid = VirtualAssertValidDiv0 {
            address: self.address,
            operands: FormatB {
                rs1: self.operands.rs2,
                rs2: v_q,
                imm: 0,
            },
            virtual_sequence_remaining: Some(4),
        };
        sequence.push(is_valid.into());

        let mul = MUL {
            address: self.address,
            operands: FormatR {
                rd: v_qy,
                rs1: v_q,
                rs2: self.operands.rs2,
            },
            virtual_sequence_remaining: Some(3),
        };
        sequence.push(mul.into());

        let add = ADD {
            address: self.address,
            operands: FormatR {
                rd: v_0,
                rs1: v_qy,
                rs2: v_r,
            },
            virtual_sequence_remaining: Some(2),
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
                rs1: v_q,
                imm: 0,
            },
            virtual_sequence_remaining: Some(0),
        };
        sequence.push(virtual_move.into());

        sequence
    }
}
