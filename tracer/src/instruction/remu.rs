use crate::utils::virtual_registers::allocate_virtual_register;
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
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction,
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
        let dividend = cpu.unsigned_data(cpu.x[self.operands.rs1 as usize]);
        let divisor = cpu.unsigned_data(cpu.x[self.operands.rs2 as usize]);
        cpu.x[self.operands.rd as usize] = match divisor {
            0 => cpu.sign_extend(dividend as i64),
            _ => cpu.sign_extend(dividend.wrapping_rem(divisor) as i64),
        };
    }
}

impl RISCVTrace for REMU {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let mut inline_sequence = self.inline_sequence(cpu.xlen);
        if let RV32IMInstruction::VirtualAdvice(instr) = &mut inline_sequence[0] {
            instr.advice = if cpu.unsigned_data(cpu.x[self.operands.rs2 as usize]) == 0 {
                match cpu.xlen {
                    Xlen::Bit32 => u32::MAX as u64,
                    Xlen::Bit64 => u64::MAX,
                }
            } else {
                cpu.unsigned_data(cpu.x[self.operands.rs1 as usize])
                    / cpu.unsigned_data(cpu.x[self.operands.rs2 as usize])
            };
        } else {
            panic!("Expected Advice instruction");
        }
        if let RV32IMInstruction::VirtualAdvice(instr) = &mut inline_sequence[1] {
            instr.advice = match cpu.unsigned_data(cpu.x[self.operands.rs2 as usize]) {
                0 => cpu.unsigned_data(cpu.x[self.operands.rs1 as usize]),
                divisor => {
                    let dividend = cpu.unsigned_data(cpu.x[self.operands.rs1 as usize]);
                    let quotient = dividend / divisor;
                    dividend - quotient * divisor
                }
            };
        } else {
            panic!("Expected Advice instruction");
        }

        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(&self, _xlen: Xlen) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_0 = allocate_virtual_register();
        let v_q = allocate_virtual_register();
        let v_r = allocate_virtual_register();
        let v_qy = allocate_virtual_register();

        let mut sequence = vec![];
        let mut inline_sequence_remaining = self.inline_sequence_remaining.unwrap_or(7);

        let advice = VirtualAdvice {
            address: self.address,
            operands: FormatJ { rd: *v_q, imm: 0 },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            advice: 0,
            is_compressed: self.is_compressed,
        };
        sequence.push(advice.into());
        inline_sequence_remaining -= 1;

        let advice = VirtualAdvice {
            address: self.address,
            operands: FormatJ { rd: *v_r, imm: 0 },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            advice: 0,
            is_compressed: self.is_compressed,
        };
        sequence.push(advice.into());
        inline_sequence_remaining -= 1;

        let mul = MUL {
            address: self.address,
            operands: FormatR {
                rd: *v_qy,
                rs1: *v_q,
                rs2: self.operands.rs2,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(mul.into());
        inline_sequence_remaining -= 1;

        let assert_remainder = VirtualAssertValidUnsignedRemainder {
            address: self.address,
            operands: FormatB {
                rs1: *v_r,
                rs2: self.operands.rs2,
                imm: 0,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(assert_remainder.into());
        inline_sequence_remaining -= 1;

        let assert_lte = VirtualAssertLTE {
            address: self.address,
            operands: FormatB {
                rs1: *v_qy,
                rs2: self.operands.rs1,
                imm: 0,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(assert_lte.into());
        inline_sequence_remaining -= 1;

        let add = ADD {
            address: self.address,
            operands: FormatR {
                rd: *v_0,
                rs1: *v_qy,
                rs2: *v_r,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(add.into());
        inline_sequence_remaining -= 1;

        let assert_eq = VirtualAssertEQ {
            address: self.address,
            operands: FormatB {
                rs1: *v_0,
                rs2: self.operands.rs1,
                imm: 0,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(assert_eq.into());
        inline_sequence_remaining -= 1;

        let virtual_move = VirtualMove {
            address: self.address,
            operands: FormatI {
                rd: self.operands.rd,
                rs1: *v_r,
                imm: 0,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(virtual_move.into());

        sequence
    }
}
