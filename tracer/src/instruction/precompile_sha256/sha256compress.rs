use common::constants::virtual_register_index;
use serde::{Deserialize, Serialize};

use crate::declare_riscv_instr;
use crate::emulator::cpu::Cpu;
use crate::instruction::format::format_r::FormatR;
use crate::instruction::format::InstructionFormat;
use crate::instruction::precompile_sha256::{
    execute_sha256_compression, Sha256SequenceBuilder, NEEDED_REGISTERS,
};
use crate::instruction::{
    RISCVInstruction, RISCVTrace, RV32IMInstruction, VirtualInstructionSequence,
};

declare_riscv_instr!(
    name   = SHA256COMPRESS,
    mask   = 0xfe00707f,  // Mask for funct7 + funct3 + opcode
    match  = 0x0000000b,  // funct7=0x00, funct3=0x0, opcode=0x0B (custom-0)
    format = FormatR,
    ram    = ()
);

impl SHA256COMPRESS {
    fn exec(
        &self,
        cpu: &mut Cpu,
        _ram_access: &mut <SHA256COMPRESS as RISCVInstruction>::RAMAccess,
    ) {
        let mut input = [0u32; 16];
        let mut initial_state = [0u32; 8];
        (0..16).for_each(|i| {
            input[i] = match cpu
                .mmu
                .load_word(cpu.x[self.operands.rs1].wrapping_add((i * 4) as i64) as u64)
            {
                Ok((word, _memory_read)) => {
                    // *ram_access = memory_read;
                    word as u32
                }
                Err(_) => panic!("MMU load error"),
            };
        });
        (0..8).for_each(|i| {
            initial_state[i] = match cpu
                .mmu
                .load_word(cpu.x[self.operands.rs2].wrapping_add((i * 4) as i64) as u64)
            {
                Ok((word, _memory_read)) => {
                    // *ram_access = memory_read;
                    word as u32
                }
                Err(_) => panic!("MMU load error"),
            };
        });
        let result = execute_sha256_compression(initial_state, input);
        result.into_iter().enumerate().for_each(|(i, r)| {
            match cpu
                .mmu
                // Write after input data
                .store_word(cpu.x[self.operands.rs2].wrapping_add(((i + 8) * 4) as i64) as u64, r)
            {
                Ok(_) => {
                    // *ram_access = memory_write;
                }
                Err(_) => panic!("MMU store error"),
            }
        })
    }
}

impl RISCVTrace for SHA256COMPRESS {
    fn trace(&self, cpu: &mut Cpu) {
        let virtual_sequence = self.virtual_sequence();

        for instr in virtual_sequence {
            instr.trace(cpu);
        }
    }
}

impl VirtualInstructionSequence for SHA256COMPRESS {
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
            Some(self.operands.rs2),
        );
        builder.build()
    }
}
