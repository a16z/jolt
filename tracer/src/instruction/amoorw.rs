use serde::{Deserialize, Serialize};

use super::amo::{amo_post32, amo_post64, amo_pre32, amo_pre64};
use super::or::OR;
use super::RV32IMInstruction;
use crate::utils::virtual_registers::allocate_virtual_register;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace, RV32IMCycle,
};

declare_riscv_instr!(
    name   = AMOORW,
    mask   = 0xf800707f,
    match  = 0x4000202f,
    format = FormatR,
    ram    = ()
);

impl AMOORW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOORW as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let or_value = cpu.x[self.operands.rs2 as usize] as u32;

        // Load the original word from memory
        let load_result = cpu.mmu.load_word(address);
        let original_value = match load_result {
            Ok((word, _)) => word as i32 as i64,
            Err(_) => panic!("MMU load error"),
        };

        // OR the values and store back to memory
        let new_value = (original_value as u32) | or_value;
        cpu.mmu
            .store_word(address, new_value)
            .expect("MMU store error");

        // Return the original value
        cpu.x[self.operands.rd as usize] = original_value;
    }
}

impl RISCVTrace for AMOORW {
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

impl AMOORW {
    fn inline_sequence_32(&self, _xlen: Xlen) -> Vec<RV32IMInstruction> {
        let v_rd = allocate_virtual_register();
        let v_rs2 = allocate_virtual_register();

        let mut sequence = vec![];
        let mut inline_sequence_remaining = self.inline_sequence_remaining.unwrap_or(4);

        inline_sequence_remaining = amo_pre32(
            &mut sequence,
            self.address,
            self.is_compressed,
            self.operands.rs1,
            *v_rd,
            inline_sequence_remaining,
        );

        let or = OR {
            address: self.address,
            operands: FormatR {
                rd: *v_rs2,
                rs1: *v_rd,
                rs2: self.operands.rs2,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(or.into());
        inline_sequence_remaining -= 1;

        amo_post32(
            &mut sequence,
            self.address,
            self.is_compressed,
            *v_rs2,
            self.operands.rs1,
            self.operands.rd,
            *v_rd,
            inline_sequence_remaining,
        );

        sequence
    }

    fn inline_sequence_64(&self, _xlen: Xlen) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_mask = allocate_virtual_register();
        let v_dword_address = allocate_virtual_register();
        let v_dword = allocate_virtual_register();
        let v_word = allocate_virtual_register();
        let v_shift = allocate_virtual_register();
        let v_rd = allocate_virtual_register();
        let v_rs2 = allocate_virtual_register();

        let mut sequence = vec![];
        let mut remaining = 17;
        remaining = amo_pre64(
            &mut sequence,
            self.address,
            self.is_compressed,
            self.operands.rs1,
            *v_rd,
            *v_dword_address,
            *v_dword,
            *v_shift,
            remaining,
        );
        let or = OR {
            address: self.address,
            operands: FormatR {
                rd: *v_rs2,
                rs1: *v_rd,
                rs2: self.operands.rs2,
            },
            inline_sequence_remaining: Some(remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(or.into());
        remaining -= 1;
        amo_post64(
            &mut sequence,
            self.address,
            self.is_compressed,
            *v_rs2,
            *v_dword_address,
            *v_dword,
            *v_shift,
            *v_mask,
            *v_word,
            self.operands.rd,
            *v_rd,
            remaining,
        );

        sequence
    }
}
