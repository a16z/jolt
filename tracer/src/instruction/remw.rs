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
    virtual_change_divisor_w::VirtualChangeDivisorW,
    virtual_move::VirtualMove,
    virtual_sign_extend::VirtualSignExtend,
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction, VirtualInstructionSequence,
};

declare_riscv_instr!(
    name   = REMW,
    mask   = 0xfe00707f,
    match  = 0x1e00003b,
    format = FormatR,
    ram    = ()
);

impl REMW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <REMW as RISCVInstruction>::RAMAccess) {
        // REMW and REMUW are RV64 instructions that provide the corresponding signed and unsigned
        // remainder operations. Both REMW and REMUW always sign-extend the 32-bit result to 64 bits,
        // including on a divide by zero.
        let dividend = cpu.x[self.operands.rs1] as i32;
        let divisor = cpu.x[self.operands.rs2] as i32;
        cpu.x[self.operands.rd] = (if divisor == 0 {
            dividend
        } else if dividend == i32::MIN && divisor == -1 {
            0
        } else {
            dividend.wrapping_rem(divisor)
        }) as i64;
    }
}

impl RISCVTrace for REMW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        // REMW operands
        let x = cpu.x[self.operands.rs1] as i32;
        let y = cpu.x[self.operands.rs2] as i32;

        let (quotient, remainder) = match cpu.xlen {
            Xlen::Bit32 => {
                panic!("REMW is invalid in 32b mode");
            }
            Xlen::Bit64 => {
                if y == 0 {
                    (-1i32, x)
                } else if y == -1 && x == i32::MIN {
                    (i32::MIN, 0) //overflow
                } else {
                    let mut quotient = x / y;
                    let mut remainder = x % y;
                    (quotient, remainder)
                }
            }
        };

        let mut virtual_sequence = self.virtual_sequence(cpu.xlen == Xlen::Bit32);
        if let RV32IMInstruction::VirtualAdvice(instr) = &mut virtual_sequence[0] {
            instr.advice = quotient as u64;
        } else {
            panic!("Expected Advice instruction");
        }
        if let RV32IMInstruction::VirtualAdvice(instr) = &mut virtual_sequence[1] {
            instr.advice = remainder as u64;
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

impl VirtualInstructionSequence for REMW {
    fn virtual_sequence(&self, _: bool) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_0 = virtual_register_index(0) as usize;
        let v_q = virtual_register_index(1) as usize;
        let v_r = virtual_register_index(2) as usize;
        let v_qy = virtual_register_index(3) as usize;
        let v_rs1 = virtual_register_index(4) as usize;
        let v_rs2 = virtual_register_index(5) as usize;

        let mut sequence = vec![];

        let advice = VirtualAdvice {
            address: self.address,
            operands: FormatJ { rd: v_q, imm: 0 },
            virtual_sequence_remaining: Some(12),
            advice: 0,
        };
        sequence.push(advice.into());

        let advice = VirtualAdvice {
            address: self.address,
            operands: FormatJ { rd: v_r, imm: 0 },
            virtual_sequence_remaining: Some(11),
            advice: 0,
        };
        sequence.push(advice.into());

        let ext = VirtualSignExtend {
            address: self.address,
            operands: FormatI {
                rd: v_rs1,
                rs1: self.operands.rs1,
                imm: 0,
            },
            virtual_sequence_remaining: Some(10),
        };
        sequence.push(ext.into());

        let ext = VirtualSignExtend {
            address: self.address,
            operands: FormatI {
                rd: v_rs2,
                rs1: self.operands.rs2,
                imm: 0,
            },
            virtual_sequence_remaining: Some(9),
        };
        sequence.push(ext.into());

        let ext = VirtualSignExtend {
            address: self.address,
            operands: FormatI {
                rd: v_q,
                rs1: v_q,
                imm: 0,
            },
            virtual_sequence_remaining: Some(8),
        };
        sequence.push(ext.into());

        let ext = VirtualSignExtend {
            address: self.address,
            operands: FormatI {
                rd: v_r,
                rs1: v_r,
                imm: 0,
            },
            virtual_sequence_remaining: Some(7),
        };
        sequence.push(ext.into());

        let change_divisor = VirtualChangeDivisorW {
            address: self.address,
            operands: FormatR {
                rd: v_rs2,
                rs1: v_rs1,
                rs2: v_rs2,
            },
            virtual_sequence_remaining: Some(6),
        };
        sequence.push(change_divisor.into());

        let is_valid = VirtualAssertValidSignedRemainder {
            address: self.address,
            operands: FormatB {
                rs1: v_r,
                rs2: v_rs2,
                imm: 0,
            },
            virtual_sequence_remaining: Some(5),
        };
        sequence.push(is_valid.into());

        let is_valid = VirtualAssertValidDiv0 {
            address: self.address,
            operands: FormatB {
                rs1: v_rs2,
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
                rs2: v_rs2,
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
                rs2: v_rs1,
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
