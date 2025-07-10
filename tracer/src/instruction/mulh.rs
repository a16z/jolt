use common::constants::virtual_register_index;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    add::ADD,
    format::{format_i::FormatI, format_r::FormatR, InstructionFormat},
    mul::MUL,
    mulhu::MULHU,
    virtual_movsign::VirtualMovsign,
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction, VirtualInstructionSequence,
};

declare_riscv_instr!(
    name   = MULH,
    mask   = 0xfe00707f,
    match  = 0x02001033,
    format = FormatR,
    ram    = ()
);

impl MULH {
    fn exec(&self, cpu: &mut Cpu, _: &mut <MULH as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd] = match cpu.xlen {
            Xlen::Bit32 => {
                cpu.sign_extend((cpu.x[self.operands.rs1] * cpu.x[self.operands.rs2]) >> 32)
            }
            Xlen::Bit64 => {
                (((cpu.x[self.operands.rs1] as i128) * (cpu.x[self.operands.rs2] as i128)) >> 64)
                    as i64
            }
        };
    }
}

impl RISCVTrace for MULH {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence(cpu.xlen == Xlen::Bit32);
        let mut trace = trace;
        for instr in virtual_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl VirtualInstructionSequence for MULH {
    fn virtual_sequence(&self, is_32: bool) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_sx = virtual_register_index(0) as usize;
        let v_sy = virtual_register_index(1) as usize;
        let v_0 = virtual_register_index(2) as usize;
        let v_1 = virtual_register_index(3) as usize;
        let v_2 = virtual_register_index(4) as usize;
        let v_3 = virtual_register_index(5) as usize;

        let mut sequence = vec![];

        let movsign_x = VirtualMovsign {
            address: self.address,
            operands: FormatI {
                rd: v_sx,
                rs1: self.operands.rs1,
                imm: 0,
            },
            virtual_sequence_remaining: Some(6),
        };
        sequence.push(movsign_x.into());

        let movsign_y = VirtualMovsign {
            address: self.address,
            operands: FormatI {
                rd: v_sy,
                rs1: self.operands.rs2,
                imm: 0,
            },
            virtual_sequence_remaining: Some(5),
        };
        sequence.push(movsign_y.into());

        let mulhu = MULHU {
            address: self.address,
            operands: FormatR {
                rd: v_0,
                rs1: self.operands.rs1,
                rs2: self.operands.rs2,
            },
            virtual_sequence_remaining: Some(4),
        };
        sequence.push(mulhu.into());

        let mulu_sx_y = MUL {
            address: self.address,
            operands: FormatR {
                rd: v_1,
                rs1: v_sx,
                rs2: self.operands.rs2,
            },
            virtual_sequence_remaining: Some(3),
        };
        sequence.push(mulu_sx_y.into());

        let mulu_sy_x = MUL {
            address: self.address,
            operands: FormatR {
                rd: v_2,
                rs1: v_sy,
                rs2: self.operands.rs1,
            },
            virtual_sequence_remaining: Some(2),
        };
        sequence.push(mulu_sy_x.into());

        let add_1 = ADD {
            address: self.address,
            operands: FormatR {
                rd: v_3,
                rs1: v_0,
                rs2: v_1,
            },
            virtual_sequence_remaining: Some(1),
        };
        sequence.push(add_1.into());

        let add_2 = ADD {
            address: self.address,
            operands: FormatR {
                rd: self.operands.rd,
                rs1: v_3,
                rs2: v_2,
            },
            virtual_sequence_remaining: Some(0),
        };
        sequence.push(add_2.into());

        sequence
    }
}
