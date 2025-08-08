use serde::{Deserialize, Serialize};

use super::format::format_i::FormatI;
use super::format::format_s::FormatS;
use super::ld::LD;
use super::or::OR;
use super::sd::SD;
use super::virtual_move::VirtualMove;
use super::RV32IMInstruction;
use super::VirtualInstructionSequence;
use crate::instruction::format::format_load::FormatLoad;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};
use common::constants::virtual_register_index;

use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace, RV32IMCycle,
};

declare_riscv_instr!(
    name   = AMOORD,
    mask   = 0xf800707f,
    match  = 0x4000302f,
    format = FormatR,
    ram    = ()
);

impl AMOORD {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOORD as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let or_value = cpu.x[self.operands.rs2 as usize] as u64;

        // Load the original doubleword from memory
        let load_result = cpu.mmu.load_doubleword(address);
        let original_value = match load_result {
            Ok((doubleword, _)) => doubleword as i64,
            Err(_) => panic!("MMU load error"),
        };

        // OR the values and store back to memory
        let new_value = (original_value as u64) | or_value;
        cpu.mmu
            .store_doubleword(address, new_value)
            .expect("MMU store error");

        // Return the original value
        cpu.x[self.operands.rd as usize] = original_value;
    }
}

impl RISCVTrace for AMOORD {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in virtual_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl VirtualInstructionSequence for AMOORD {
    fn virtual_sequence(&self, _xlen: Xlen) -> Vec<RV32IMInstruction> {
        let v_rs2 = virtual_register_index(6);
        let v_rd = virtual_register_index(7);
        let mut sequence = vec![];

        let ld = LD {
            address: self.address,
            operands: FormatLoad {
                rd: v_rd,
                rs1: self.operands.rs1,
                imm: 0,
            },
            virtual_sequence_remaining: Some(3),
            is_compressed: self.is_compressed,
        };
        sequence.push(ld.into());

        let or = OR {
            address: self.address,
            operands: FormatR {
                rd: v_rs2,
                rs1: v_rd,
                rs2: self.operands.rs2,
            },
            virtual_sequence_remaining: Some(2),
            is_compressed: self.is_compressed,
        };
        sequence.push(or.into());

        let sd = SD {
            address: self.address,
            operands: FormatS {
                rs1: self.operands.rs1,
                rs2: v_rs2,
                imm: 0,
            },
            virtual_sequence_remaining: Some(1),
            is_compressed: self.is_compressed,
        };
        sequence.push(sd.into());

        let vmove = VirtualMove {
            address: self.address,
            operands: FormatI {
                rd: self.operands.rd,
                rs1: v_rd,
                imm: 0,
            },
            virtual_sequence_remaining: Some(0),
            is_compressed: self.is_compressed,
        };
        sequence.push(vmove.into());

        sequence
    }
}
