use common::constants::virtual_register_index;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    instruction::mul::MUL,
};

use super::{
    add::ADD,
    format::{format_i::FormatI, format_r::FormatR, InstructionFormat},
    mulhu::MULHU,
    virtual_movsign::VirtualMovsign,
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
        let virtual_sequence = self.virtual_sequence();
        let mut trace = trace;
        for instr in virtual_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl VirtualInstructionSequence for MULHSU {
    fn virtual_sequence(&self) -> Vec<RV32IMInstruction> {
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
        let v_sx = virtual_register_index(0) as usize;
        let v_1 = virtual_register_index(1) as usize;
        let v_2 = virtual_register_index(2) as usize;

        let mut sequence = vec![];

        let movsign = VirtualMovsign {
            address: self.address,
            operands: FormatI {
                rd: v_sx,
                rs1: self.operands.rs1,
                imm: 0,
            },
            virtual_sequence_remaining: Some(3),
        };
        sequence.push(movsign.into());

        let mulhu = MULHU {
            address: self.address,
            operands: FormatR {
                rd: v_1,
                rs1: self.operands.rs1,
                rs2: self.operands.rs2,
            },
            virtual_sequence_remaining: Some(2),
        };
        sequence.push(mulhu.into());

        let mulu = MUL {
            address: self.address,
            operands: FormatR {
                rd: v_2,
                rs1: v_sx,
                rs2: self.operands.rs2,
            },
            virtual_sequence_remaining: Some(1),
        };
        sequence.push(mulu.into());

        let add = ADD {
            address: self.address,
            operands: FormatR {
                rd: self.operands.rd,
                rs1: v_1,
                rs2: v_2,
            },
            virtual_sequence_remaining: Some(0),
        };
        sequence.push(add.into());

        sequence
    }
}
