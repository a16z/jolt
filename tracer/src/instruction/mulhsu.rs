use common::constants::virtual_register_index;
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
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction, VirtualInstructionSequence,
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
        cpu.x[self.operands.rd] = match cpu.xlen {
            Xlen::Bit32 => cpu.sign_extend(
                cpu.x[self.operands.rs1].wrapping_mul(cpu.x[self.operands.rs2] as u32 as i64) >> 32,
            ),
            Xlen::Bit64 => {
                ((cpu.x[self.operands.rs1] as u128)
                    .wrapping_mul(cpu.x[self.operands.rs2] as u64 as u128)
                    >> 64) as i64
            }
        };
    }
}

impl RISCVTrace for MULHSU {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence(cpu.xlen == Xlen::Bit32);
        let mut trace = trace;
        for instr in virtual_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl VirtualInstructionSequence for MULHSU {
    fn virtual_sequence(&self, _: bool) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_sx = virtual_register_index(0) as usize;
        let v_sx_0 = virtual_register_index(1) as usize;
        let v_rs1 = virtual_register_index(2) as usize;
        let v_hi = virtual_register_index(3) as usize;
        let v_lo = virtual_register_index(4) as usize;
        let v_tmp = virtual_register_index(5) as usize;
        let v_carry = virtual_register_index(6) as usize;

        let mut sequence = vec![];

        let movsign = VirtualMovsign {
            address: self.address,
            operands: FormatI {
                rd: v_sx,
                rs1: self.operands.rs1,
                imm: 0,
            },
            virtual_sequence_remaining: Some(10),
        };
        sequence.push(movsign.into());

        let take_lsb = ANDI {
            address: self.address,
            operands: FormatI {
                rd: v_sx_0,
                rs1: v_sx,
                imm: 1,
            },
            virtual_sequence_remaining: Some(9),
        };
        sequence.push(take_lsb.into());

        let xor_0 = XOR {
            address: self.address,
            operands: FormatR {
                rd: v_rs1,
                rs1: self.operands.rs1,
                rs2: v_sx,
            },
            virtual_sequence_remaining: Some(8),
        };
        sequence.push(xor_0.into());

        let add_0 = ADD {
            address: self.address,
            operands: FormatR {
                rd: v_rs1,
                rs1: v_rs1,
                rs2: v_sx_0,
            },
            virtual_sequence_remaining: Some(7),
        };
        sequence.push(add_0.into());

        let mulhu = MULHU {
            address: self.address,
            operands: FormatR {
                rd: v_hi,
                rs1: v_rs1,
                rs2: self.operands.rs2,
            },
            virtual_sequence_remaining: Some(6),
        };
        sequence.push(mulhu.into());

        let mul = MUL {
            address: self.address,
            operands: FormatR {
                rd: v_lo,
                rs1: v_rs1,
                rs2: self.operands.rs2,
            },
            virtual_sequence_remaining: Some(5),
        };
        sequence.push(mul.into());

        let xor_1 = XOR {
            address: self.address,
            operands: FormatR {
                rd: v_hi,
                rs1: v_hi,
                rs2: v_sx,
            },
            virtual_sequence_remaining: Some(4),
        };
        sequence.push(xor_1.into());

        let xor_2 = XOR {
            address: self.address,
            operands: FormatR {
                rd: v_lo,
                rs1: v_lo,
                rs2: v_sx,
            },
            virtual_sequence_remaining: Some(3),
        };
        sequence.push(xor_2.into());

        let add_1 = ADD {
            address: self.address,
            operands: FormatR {
                rd: v_tmp,
                rs1: v_lo,
                rs2: v_sx_0,
            },
            virtual_sequence_remaining: Some(2),
        };
        sequence.push(add_1.into());

        let sltu_0 = SLTU {
            address: self.address,
            operands: FormatR {
                rd: v_carry,
                rs1: v_tmp,
                rs2: v_lo,
            },
            virtual_sequence_remaining: Some(1),
        };
        sequence.push(sltu_0.into());

        let add_2 = ADD {
            address: self.address,
            operands: FormatR {
                rd: self.operands.rd,
                rs1: v_hi,
                rs2: v_carry,
            },
            virtual_sequence_remaining: Some(0),
        };
        sequence.push(add_2.into());

        sequence
    }
}
