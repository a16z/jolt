use common::constants::virtual_register_index;
use serde::{Deserialize, Serialize};

use crate::declare_riscv_instr;
use crate::emulator::cpu::Cpu;
use crate::instruction::format::format_i::FormatI;
use crate::instruction::format::InstructionFormat;
use crate::instruction::precompile_sha256::{
    execute_sha256_compression_initial, Sha256SequenceBuilder, NEEDED_REGISTERS,
};
use crate::instruction::{
    RISCVInstruction, RISCVTrace, RV32IMInstruction, VirtualInstructionSequence,
};

declare_riscv_instr!(
    name   = SHA256COMPRESSI,
    mask   = 0xfe00707f,  // Mask for funct7 + funct3 + opcode
    match  = 0x0000100b,  // funct7=0x00, funct3=0x1, opcode=0x0B (custom-0)
    format = FormatI,
    ram    = ()
);

impl SHA256COMPRESSI {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SHA256COMPRESSI as RISCVInstruction>::RAMAccess) {
        let mut input = [0u32; 16];
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
        let result = execute_sha256_compression_initial(input);
        result.into_iter().enumerate().for_each(|(i, r)| {
            match cpu.mmu.store_word(
                cpu.x[self.operands.rs1].wrapping_add(((i + 16) * 4) as i64) as u64,
                r,
            ) {
                Ok(_) => {
                    // *ram_access = memory_write;
                }
                Err(_) => panic!("MMU store error"),
            }
        })
    }
}

impl RISCVTrace for SHA256COMPRESSI {
    fn trace(&self, cpu: &mut Cpu) {
        let virtual_sequence = self.virtual_sequence();

        for instr in virtual_sequence {
            instr.trace(cpu);
        }
    }
}

impl VirtualInstructionSequence for SHA256COMPRESSI {
    fn virtual_sequence(&self) -> Vec<RV32IMInstruction> {
        // Virtual registers used as a scratch space
        let mut vr = [0; NEEDED_REGISTERS];
        (0..NEEDED_REGISTERS).for_each(|i| {
            vr[i] = virtual_register_index(i as u64) as usize;
        });
        let builder = Sha256SequenceBuilder::new(self.address, vr, self.operands.rs1, None);
        builder.build()
    }
}
