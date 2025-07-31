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
    virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder,
    virtual_change_divisor_w::VirtualChangeDivisorW,
    virtual_extend::VirtualExtend,
    virtual_sign_extend::VirtualSignExtend,
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction, VirtualInstructionSequence,
};

declare_riscv_instr!(
    name   = DIVUW,
    mask   = 0xfe00707f,
    match  = 0x1b00003b,
    format = FormatR,
    ram    = ()
);

impl DIVUW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <DIVUW as RISCVInstruction>::RAMAccess) {
        // DIVW and DIVUW are RV64 instructions that divide the lower 32 bits of rs1 by the lower
        // 32 bits of rs2, treating them as signed and unsigned integers, placing the 32-bit
        // quotient in rd, sign-extended to 64 bits.
        let dividend = cpu.x[self.operands.rs1] as u32;
        let divisor = cpu.x[self.operands.rs2] as u32;
        cpu.x[self.operands.rd] = (if divisor == 0 {
            u32::MAX
        } else {
            dividend.wrapping_div(divisor)
        }) as i32 as i64;
    }
}

impl RISCVTrace for DIVUW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        // DIVUW operands
        let x = cpu.x[self.operands.rs1] as u32;
        let y = cpu.x[self.operands.rs2] as u32;

        let (quotient, remainder) = match cpu.xlen {
            Xlen::Bit32 => {
                panic!("DIVUW is invalid in 32b mode");
            }
            Xlen::Bit64 => {
                if y == 0 {
                    (u64::MAX, x as u64)
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
}

impl VirtualInstructionSequence for DIVUW {
    fn virtual_sequence(&self, _xlen: Xlen) -> Vec<RV32IMInstruction> {
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
            virtual_sequence_remaining: Some(13),
            advice: 0,
            is_compressed: self.is_compressed,
        };
        sequence.push(advice.into());

        let advice = VirtualAdvice {
            address: self.address,
            operands: FormatJ { rd: v_r, imm: 0 },
            virtual_sequence_remaining: Some(12),
            advice: 0,
            is_compressed: self.is_compressed,
        };
        sequence.push(advice.into());

        let ext = VirtualExtend {
            address: self.address,
            operands: FormatI {
                rd: v_rs1,
                rs1: self.operands.rs1,
                imm: 0,
            },
            virtual_sequence_remaining: Some(11),
            is_compressed: self.is_compressed,
        };
        sequence.push(ext.into());

        let ext = VirtualExtend {
            address: self.address,
            operands: FormatI {
                rd: v_rs2,
                rs1: self.operands.rs2,
                imm: 0,
            },
            virtual_sequence_remaining: Some(10),
            is_compressed: self.is_compressed,
        };
        sequence.push(ext.into());

        let ext = VirtualExtend {
            address: self.address,
            operands: FormatI {
                rd: v_r,
                rs1: v_r,
                imm: 0,
            },
            virtual_sequence_remaining: Some(9),
            is_compressed: self.is_compressed,
        };
        sequence.push(ext.into());

        let change_divisor = VirtualChangeDivisorW {
            address: self.address,
            operands: FormatR {
                rd: v_rs2,
                rs1: v_rs1,
                rs2: v_rs2,
            },
            virtual_sequence_remaining: Some(8),
            is_compressed: self.is_compressed,
        };
        sequence.push(change_divisor.into());

        let is_valid = VirtualAssertValidUnsignedRemainder {
            address: self.address,
            operands: FormatB {
                rs1: v_r,
                rs2: v_rs2,
                imm: 0,
            },
            virtual_sequence_remaining: Some(7),
            is_compressed: self.is_compressed,
        };
        sequence.push(is_valid.into());

        let is_valid = VirtualAssertValidDiv0 {
            address: self.address,
            operands: FormatB {
                rs1: v_rs2,
                rs2: v_q,
                imm: 0,
            },
            virtual_sequence_remaining: Some(6),
            is_compressed: self.is_compressed,
        };
        sequence.push(is_valid.into());

        let ext = VirtualExtend {
            address: self.address,
            operands: FormatI {
                rd: v_q,
                rs1: v_q,
                imm: 0,
            },
            virtual_sequence_remaining: Some(5),
            is_compressed: self.is_compressed,
        };
        sequence.push(ext.into());

        let mul = MUL {
            address: self.address,
            operands: FormatR {
                rd: v_qy,
                rs1: v_q,
                rs2: v_rs2,
            },
            virtual_sequence_remaining: Some(4),
            is_compressed: self.is_compressed,
        };
        sequence.push(mul.into());

        let add = ADD {
            address: self.address,
            operands: FormatR {
                rd: v_0,
                rs1: v_qy,
                rs2: v_r,
            },
            virtual_sequence_remaining: Some(3),
            is_compressed: self.is_compressed,
        };
        sequence.push(add.into());

        let ext = VirtualExtend {
            address: self.address,
            operands: FormatI {
                rd: v_0,
                rs1: v_0,
                imm: 0,
            },
            virtual_sequence_remaining: Some(2),
            is_compressed: self.is_compressed,
        };
        sequence.push(ext.into());

        let assert_eq = VirtualAssertEQ {
            address: self.address,
            operands: FormatB {
                rs1: v_0,
                rs2: v_rs1,
                imm: 0,
            },
            virtual_sequence_remaining: Some(1),
            is_compressed: self.is_compressed,
        };
        sequence.push(assert_eq.into());

        let ext = VirtualSignExtend {
            address: self.address,
            operands: FormatI {
                rd: self.operands.rd,
                rs1: v_q,
                imm: 0,
            },
            virtual_sequence_remaining: Some(0),
            is_compressed: self.is_compressed,
        };
        sequence.push(ext.into());

        sequence
    }
}
