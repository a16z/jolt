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
    virtual_change_divisor::VirtualChangeDivisor,
    virtual_move::VirtualMove,
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction,
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
        let dividend = cpu.x[self.operands.rs1 as usize];
        let divisor = cpu.x[self.operands.rs2 as usize];
        if divisor == 0 {
            cpu.x[self.operands.rd as usize] = -1;
        } else if dividend == cpu.most_negative() && divisor == -1 {
            cpu.x[self.operands.rd as usize] = dividend;
        } else {
            cpu.x[self.operands.rd as usize] = cpu.sign_extend(dividend.wrapping_div(divisor))
        }
    }
}

impl RISCVTrace for DIV {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        // RISCV spec: For REM, the sign of a nonzero result equals the sign of the dividend.
        // DIV operands
        let x = cpu.x[self.operands.rs1 as usize];
        let y = cpu.x[self.operands.rs2 as usize];

        let (quotient, remainder) = match cpu.xlen {
            Xlen::Bit32 => {
                if y == 0 {
                    (u32::MAX as u64, x as u32 as u64)
                } else if x == cpu.most_negative() && y == -1 {
                    (x as u32 as u64, 0)
                } else {
                    let quotient = x as i32 / y as i32;
                    let remainder = x as i32 % y as i32;
                    (quotient as u32 as u64, remainder as u32 as u64)
                }
            }
            Xlen::Bit64 => {
                if y == 0 {
                    (u64::MAX, x as u64)
                } else if x == cpu.most_negative() && y == -1 {
                    (x as u64, 0)
                } else {
                    let quotient = x / y;
                    let remainder = x % y;
                    (quotient as u64, remainder as u64)
                }
            }
        };

        let mut virtual_sequence = self.virtual_sequence(cpu.xlen);
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

    fn virtual_sequence(&self, _xlen: Xlen) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_0 = virtual_register_index(0);
        let v_q = virtual_register_index(1);
        let v_r = virtual_register_index(2);
        let v_qy = virtual_register_index(3);
        let v_rs2 = virtual_register_index(4);

        let mut sequence = vec![];
        let mut virtual_sequence_remaining = self.virtual_sequence_remaining.unwrap_or(8);

        let advice = VirtualAdvice {
            address: self.address,
            operands: FormatJ { rd: v_q, imm: 0 },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
            advice: 0,
            is_compressed: self.is_compressed,
        };
        sequence.push(advice.into());
        virtual_sequence_remaining -= 1;

        let advice = VirtualAdvice {
            address: self.address,
            operands: FormatJ { rd: v_r, imm: 0 },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
            advice: 0,
            is_compressed: self.is_compressed,
        };
        sequence.push(advice.into());
        virtual_sequence_remaining -= 1;

        let change_divisor = VirtualChangeDivisor {
            address: self.address,
            operands: FormatR {
                rd: v_rs2,
                rs1: self.operands.rs1,
                rs2: self.operands.rs2,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(change_divisor.into());
        virtual_sequence_remaining -= 1;

        let is_valid = VirtualAssertValidSignedRemainder {
            address: self.address,
            operands: FormatB {
                rs1: v_r,
                rs2: self.operands.rs2,
                imm: 0,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(is_valid.into());
        virtual_sequence_remaining -= 1;

        let is_valid = VirtualAssertValidDiv0 {
            address: self.address,
            operands: FormatB {
                rs1: self.operands.rs2,
                rs2: v_q,
                imm: 0,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(is_valid.into());
        virtual_sequence_remaining -= 1;

        let mul = MUL {
            address: self.address,
            operands: FormatR {
                rd: v_qy,
                rs1: v_q,
                rs2: v_rs2,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(mul.into());
        virtual_sequence_remaining -= 1;

        let add = ADD {
            address: self.address,
            operands: FormatR {
                rd: v_0,
                rs1: v_qy,
                rs2: v_r,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(add.into());
        virtual_sequence_remaining -= 1;

        let assert_eq = VirtualAssertEQ {
            address: self.address,
            operands: FormatB {
                rs1: v_0,
                rs2: self.operands.rs1,
                imm: 0,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(assert_eq.into());
        virtual_sequence_remaining -= 1;

        let virtual_move = VirtualMove {
            address: self.address,
            operands: FormatI {
                rd: self.operands.rd,
                rs1: v_q,
                imm: 0,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(virtual_move.into());

        sequence
    }
}
