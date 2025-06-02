use common::constants::virtual_register_index;
use serde::{Deserialize, Serialize};

use crate::declare_riscv_instr;
use crate::emulator::cpu::Cpu;
use crate::instruction::format::format_r::FormatR;
use crate::instruction::format::InstructionFormat;
use crate::instruction::inline_sha256::{
    execute_sha256_compression, Sha256SequenceBuilder, NEEDED_REGISTERS,
};
use crate::instruction::{
    RISCVInstruction, RISCVTrace, RV32IMInstruction, VirtualInstructionSequence,
};

declare_riscv_instr!(
    name   = SHA256,
    mask   = 0xfe00707f,  // Mask for funct7 + funct3 + opcode
    match  = 0x0000000b,  // funct7=0x00, funct3=0x0, opcode=0x0B (custom-0)
    format = FormatR,
    ram    = ()
);

impl SHA256 {
    fn exec(&self, cpu: &mut Cpu, _ram_access: &mut <SHA256 as RISCVInstruction>::RAMAccess) {
        // Load 16 input words from memory at rs1
        let mut input = [0u32; 16];
        for (i, word) in input.iter_mut().enumerate() {
            *word = cpu
                .mmu
                .load_word(cpu.x[self.operands.rs1].wrapping_add((i * 4) as i64) as u64)
                .expect("SHA256: Failed to load input word")
                .0;
        }

        // Load 8 initial state words from memory at rs2
        let mut iv = [0u32; 8];
        for (i, word) in iv.iter_mut().enumerate() {
            *word = cpu
                .mmu
                .load_word(cpu.x[self.operands.rs2].wrapping_add((i * 4) as i64) as u64)
                .expect("SHA256: Failed to load initial state")
                .0;
        }

        // Execute compression and store result at rs2
        let result = execute_sha256_compression(iv, input);
        for (i, &word) in result.iter().enumerate() {
            cpu.mmu
                .store_word(
                    cpu.x[self.operands.rs2].wrapping_add((i * 4) as i64) as u64,
                    word,
                )
                .expect("SHA256: Failed to store result");
        }
    }
}

impl RISCVTrace for SHA256 {
    fn trace(&self, cpu: &mut Cpu) {
        let virtual_sequence = self.virtual_sequence();

        for instr in virtual_sequence {
            instr.trace(cpu);
        }
    }
}

impl VirtualInstructionSequence for SHA256 {
    fn virtual_sequence(&self) -> Vec<RV32IMInstruction> {
        // Virtual registers used as a scratch space
        let mut vr = [0; NEEDED_REGISTERS];
        (0..NEEDED_REGISTERS).for_each(|i| {
            vr[i] = virtual_register_index(i as u64) as usize;
        });
        let builder = Sha256SequenceBuilder::new(
            self.address,
            vr,
            self.operands.rs1,
            self.operands.rs2,
            false, // not initial - uses custom IV from rs2
        );
        builder.build()
    }
}
