use serde::{Deserialize, Serialize};

use super::amo::{amo_post32, amo_post64, amo_pre32, amo_pre64};
use super::RV32IMInstruction;
use common::constants::virtual_register_index;

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace, RV32IMCycle,
};

declare_riscv_instr!(
    name   = AMOSWAPW,
    mask   = 0xf800707f,
    match  = 0x0800202f,
    format = FormatR,
    ram    = ()
);

impl AMOSWAPW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOSWAPW as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let new_value = cpu.x[self.operands.rs2 as usize] as u32;

        // Load the original word from memory
        let load_result = cpu.mmu.load_word(address);
        let original_value = match load_result {
            Ok((word, _)) => word as i32 as i64,
            Err(_) => panic!("MMU load error"),
        };

        // Store the new value to memory
        cpu.mmu
            .store_word(address, new_value)
            .expect("MMU store error");

        // Return the original value
        cpu.x[self.operands.rd as usize] = original_value;
    }
}

impl RISCVTrace for AMOSWAPW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let inline_sequence = self.inline_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
        match xlen {
            Xlen::Bit32 => self.inline_sequence_32(xlen),
            Xlen::Bit64 => self.inline_sequence_64(xlen),
        }
    }
}

impl AMOSWAPW {
    fn inline_sequence_32(&self, _xlen: Xlen) -> Vec<RV32IMInstruction> {
        let v_rd = virtual_register_index(7);

        let mut sequence = vec![];
        let mut remaining = 3;
        remaining = amo_pre32(
            &mut sequence,
            self.address,
            self.is_compressed,
            self.operands.rs1,
            v_rd,
            remaining,
        );

        amo_post32(
            &mut sequence,
            self.address,
            self.is_compressed,
            self.operands.rs2,
            self.operands.rs1,
            self.operands.rd,
            v_rd,
            remaining,
        );

        sequence
    }

    fn inline_sequence_64(&self, _xlen: Xlen) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_mask = virtual_register_index(10);
        let v_dword_address = virtual_register_index(11);
        let v_dword = virtual_register_index(12);
        let v_word = virtual_register_index(13);
        let v_shift = virtual_register_index(14);
        let v_rd = virtual_register_index(15);

        let mut sequence = vec![];
        let mut remaining = 16;
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
        amo_post64(
            &mut sequence,
            self.address,
            self.is_compressed,
            self.operands.rs2,
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
