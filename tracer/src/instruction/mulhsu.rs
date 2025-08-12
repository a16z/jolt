use crate::utils::virtual_registers::allocate_virtual_register;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    add::ADD,
    andi::ANDI,
    format::{format_i::FormatI, format_r::FormatR, InstructionFormat},
    mul::MUL,
    mulhu::MULHU,
    sltu::SLTU,
    virtual_movsign::VirtualMovsign,
    xor::XOR,
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction,
};

declare_riscv_instr!(
    name   = MULHSU,
    mask   = 0xfe00707f,
    match  = 0x02002033,
    format = FormatR,
    ram    = ()
);

impl MULHSU {
    fn exec(&self, cpu: &mut Cpu, _: &mut <MULHSU as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd as usize] = match cpu.xlen {
            Xlen::Bit32 => cpu.sign_extend(
                cpu.x[self.operands.rs1 as usize]
                    .wrapping_mul(cpu.x[self.operands.rs2 as usize] as u32 as i64)
                    >> 32,
            ),
            Xlen::Bit64 => {
                ((cpu.x[self.operands.rs1 as usize] as u128)
                    .wrapping_mul(cpu.x[self.operands.rs2 as usize] as u64 as u128)
                    >> 64) as i64
            }
        };
    }
}

impl RISCVTrace for MULHSU {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let inline_sequence = self.inline_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(&self, _xlen: Xlen) -> Vec<RV32IMInstruction> {
        // MULHSU implements signed-unsigned multiplication: rs1 (signed) × rs2 (unsigned)
        //
        // For negative rs1, two's complement encoding means:
        // rs1_unsigned = rs1 + 2^32 (when rs1 < 0)
        //
        // Therefore:
        // MULHU(rs1_unsigned, rs2) = upper_bits((rs1 + 2^32) × rs2)
        //                          = upper_bits(rs1 × rs2 + 2^32 × rs2)
        //                          = upper_bits(rs1 × rs2) + rs2
        //                          = MULHSU(rs1, rs2) + rs2
        //
        // So: MULHSU(rs1, rs2) = MULHU(rs1_unsigned, rs2) - rs2

        // Virtual registers used in sequence
        let v_sx = allocate_virtual_register();
        let v_sx_0 = allocate_virtual_register();
        let v_rs1 = allocate_virtual_register();
        let v_hi = allocate_virtual_register();
        let v_lo = allocate_virtual_register();
        let v_tmp = allocate_virtual_register();
        let v_carry = allocate_virtual_register();

        let mut sequence = vec![];

        let movsign = VirtualMovsign {
            address: self.address,
            operands: FormatI {
                rd: *v_sx,
                rs1: self.operands.rs1,
                imm: 0,
            },
            inline_sequence_remaining: Some(10),
            is_compressed: self.is_compressed,
        };
        sequence.push(movsign.into());

        let take_lsb = ANDI {
            address: self.address,
            operands: FormatI {
                rd: *v_sx_0,
                rs1: *v_sx,
                imm: 1,
            },
            inline_sequence_remaining: Some(9),
            is_compressed: self.is_compressed,
        };
        sequence.push(take_lsb.into());

        let xor_0 = XOR {
            address: self.address,
            operands: FormatR {
                rd: *v_rs1,
                rs1: self.operands.rs1,
                rs2: *v_sx,
            },
            inline_sequence_remaining: Some(8),
            is_compressed: self.is_compressed,
        };
        sequence.push(xor_0.into());

        let add_0 = ADD {
            address: self.address,
            operands: FormatR {
                rd: *v_rs1,
                rs1: *v_rs1,
                rs2: *v_sx_0,
            },
            inline_sequence_remaining: Some(7),
            is_compressed: self.is_compressed,
        };
        sequence.push(add_0.into());

        let mulhu = MULHU {
            address: self.address,
            operands: FormatR {
                rd: *v_hi,
                rs1: *v_rs1,
                rs2: self.operands.rs2,
            },
            inline_sequence_remaining: Some(6),
            is_compressed: self.is_compressed,
        };
        sequence.push(mulhu.into());

        let mul = MUL {
            address: self.address,
            operands: FormatR {
                rd: *v_lo,
                rs1: *v_rs1,
                rs2: self.operands.rs2,
            },
            inline_sequence_remaining: Some(5),
            is_compressed: self.is_compressed,
        };
        sequence.push(mul.into());

        let xor_1 = XOR {
            address: self.address,
            operands: FormatR {
                rd: *v_hi,
                rs1: *v_hi,
                rs2: *v_sx,
            },
            inline_sequence_remaining: Some(4),
            is_compressed: self.is_compressed,
        };
        sequence.push(xor_1.into());

        let xor_2 = XOR {
            address: self.address,
            operands: FormatR {
                rd: *v_lo,
                rs1: *v_lo,
                rs2: *v_sx,
            },
            inline_sequence_remaining: Some(3),
            is_compressed: self.is_compressed,
        };
        sequence.push(xor_2.into());

        let add_1 = ADD {
            address: self.address,
            operands: FormatR {
                rd: *v_tmp,
                rs1: *v_lo,
                rs2: *v_sx_0,
            },
            inline_sequence_remaining: Some(2),
            is_compressed: self.is_compressed,
        };
        sequence.push(add_1.into());

        let sltu_0 = SLTU {
            address: self.address,
            operands: FormatR {
                rd: *v_carry,
                rs1: *v_tmp,
                rs2: *v_lo,
            },
            inline_sequence_remaining: Some(1),
            is_compressed: self.is_compressed,
        };
        sequence.push(sltu_0.into());

        let add_2 = ADD {
            address: self.address,
            operands: FormatR {
                rd: self.operands.rd,
                rs1: *v_hi,
                rs2: *v_carry,
            },
            inline_sequence_remaining: Some(0),
            is_compressed: self.is_compressed,
        };
        sequence.push(add_2.into());

        sequence
    }
}
