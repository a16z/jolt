use serde::{Deserialize, Serialize};

use super::amo::{amo_post32, amo_post64, amo_pre32, amo_pre64};
use super::and::AND;
use super::RV32IMInstruction;
use super::VirtualInstructionSequence;
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
    name   = AMOANDW,
    mask   = 0xf800707f,
    match  = 0x6000202f,
    format = FormatR,
    ram    = ()
);

impl AMOANDW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOANDW as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let and_value = cpu.x[self.operands.rs2 as usize] as u32;

        // Load the original word from memory
        let load_result = cpu.mmu.load_word(address);
        let original_value = match load_result {
            Ok((word, _)) => word as i32 as i64,
            Err(_) => panic!("MMU load error"),
        };

        // AND the values and store back to memory
        let new_value = (original_value as u32) & and_value;
        cpu.mmu
            .store_word(address, new_value)
            .expect("MMU store error");

        // Return the original value
        cpu.x[self.operands.rd as usize] = original_value;
    }
}

impl RISCVTrace for AMOANDW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in virtual_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl VirtualInstructionSequence for AMOANDW {
    fn virtual_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
        match xlen {
            Xlen::Bit32 => self.virtual_sequence_32(xlen),
            Xlen::Bit64 => self.virtual_sequence_64(xlen),
        }
    }
}

impl AMOANDW {
    fn virtual_sequence_32(&self, _xlen: Xlen) -> Vec<RV32IMInstruction> {
        let v_rd = virtual_register_index(7);
        let v_rs2 = virtual_register_index(8);

        let mut sequence = vec![];
        let mut remaining = 4;
        remaining = amo_pre32(
            &mut sequence,
            self.address,
            self.is_compressed,
            self.operands.rs1,
            v_rd,
            remaining,
        );

        let and = AND {
            address: self.address,
            operands: FormatR {
                rd: v_rs2,
                rs1: v_rd,
                rs2: self.operands.rs2,
            },
            virtual_sequence_remaining: Some(remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(and.into());
        remaining -= 1;

        amo_post32(
            &mut sequence,
            self.address,
            self.is_compressed,
            v_rs2,
            self.operands.rs1,
            self.operands.rd,
            v_rd,
            remaining,
        );

        sequence
    }

    fn virtual_sequence_64(&self, _xlen: Xlen) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_mask = virtual_register_index(10);
        let v_dword_address = virtual_register_index(11);
        let v_dword = virtual_register_index(12);
        let v_word = virtual_register_index(13);
        let v_shift = virtual_register_index(14);
        let v_rd = virtual_register_index(15);
        let v_rs2 = virtual_register_index(16);

        let mut sequence = vec![];
        let mut remaining = 17;
        remaining = amo_pre64(
            &mut sequence,
            self.address,
            self.is_compressed,
            self.operands.rs1,
            v_rd,
            v_dword_address,
            v_dword,
            v_shift,
            remaining,
        );
        let and = AND {
            address: self.address,
            operands: FormatR {
                rd: v_rs2,
                rs1: v_rd,
                rs2: self.operands.rs2,
            },
            virtual_sequence_remaining: Some(remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(and.into());
        remaining -= 1;
        amo_post64(
            &mut sequence,
            self.address,
            self.is_compressed,
            v_rs2,
            v_dword_address,
            v_dword,
            v_shift,
            v_mask,
            v_word,
            self.operands.rd,
            v_rd,
            remaining,
        );

        sequence
    }
}
