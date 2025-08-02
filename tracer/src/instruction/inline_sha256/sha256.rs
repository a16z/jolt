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
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction, VirtualInstructionSequence,
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
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence();

        let mut trace = trace;
        for instr in virtual_sequence {
            instr.trace(cpu, trace.as_deref_mut());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emulator::{cpu::Cpu, default_terminal::DefaultTerminal, mmu::DRAM_BASE};
    use crate::instruction::format::format_r::FormatR;

    const TEST_MEMORY_CAPACITY: u64 = 1024 * 1024; // 1MB

    // SHA256 initial hash values (FIPS 180-4)
    const SHA256_INITIAL_STATE: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    // Expected state after processing "abc" (first block)
    // This is the result of applying SHA256 compression to the padded "abc" message
    const EXPECTED_STATE_AFTER_ABC: [u32; 8] = [
        0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223, 0xb00361a3, 0x96177a9c, 0xb410ff61,
        0xf20015ad,
    ];

    #[test]
    fn test_sha256_compression() {
        // Create padded message block for "abc"
        // SHA256 uses 512-bit blocks (16 32-bit words)
        let mut message_block = [0u32; 16];
        // "abc" = 0x61, 0x62, 0x63
        message_block[0] = 0x61626380; // "abc" + 0x80 padding bit (big-endian)
                                       // message_block[1..14] remain 0
        message_block[15] = 0x00000018; // bit length = 24 bits (3 bytes * 8)

        let instruction = SHA256 {
            address: 0,
            operands: FormatR {
                rs1: 10, // Points to message block
                rs2: 11, // Points to initial state
                rd: 0,
            },
            virtual_sequence_remaining: None,
        };

        // Set up CPU
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        let message_addr = DRAM_BASE;
        let state_addr = DRAM_BASE + 1024; // Separate address for state
        cpu.x[10] = message_addr as i64; // rs1 points to message
        cpu.x[11] = state_addr as i64; // rs2 points to state

        // Store message block (16 words) at rs1
        for (i, &word) in message_block.iter().enumerate() {
            cpu.mmu
                .store_word(message_addr + (i * 4) as u64, word)
                .expect("Failed to store message word");
        }

        // Store initial state (8 words) at rs2
        for (i, &word) in SHA256_INITIAL_STATE.iter().enumerate() {
            cpu.mmu
                .store_word(state_addr + (i * 4) as u64, word)
                .expect("Failed to store initial state");
        }

        // Execute the instruction
        instruction.exec(&mut cpu, &mut ());

        // Verify results (SHA256 compression outputs 8 words at rs2)
        let mut result = [0u32; 8];
        for i in 0..8 {
            let addr = state_addr + (i * 4) as u64;
            result[i] = cpu.mmu.load_word(addr).unwrap().0;
            assert_eq!(
                result[i], EXPECTED_STATE_AFTER_ABC[i],
                "Mismatch at word {}: got {:#010x}, expected {:#010x}",
                i, result[i], EXPECTED_STATE_AFTER_ABC[i]
            );
        }
    }
}
