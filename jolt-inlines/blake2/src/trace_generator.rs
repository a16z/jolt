//! This file contains Blake2b-specific logic to be used in the Blake2b inline:
//! 1) Prover: Blake2SequenceBuilder expands the inline to a list of RV instructions.
//! 2) Host: Rust reference implementation to be called by jolt-sdk.
//!
//! Blake2b is a cryptographic hash function operating on 64-bit words.
//! It uses a compression function that performs 12 rounds of mixing operations
//! on a 16-word working state derived from the hash state and message block.
//!
//! Glossary:
//!   - "Working state" = 16-word state array (v[0..15]) used during compression
//!   - "Hash state" = 8-word state array (h[0..7]) that holds the current hash value
//!   - "Message block" = 16-word input block (m[0..15]) to be compressed
//!   - "Round" = single application of G function mixing to the working state
//!   - "G function" = core mixing function that updates 4 state words using 2 message words

use tracer::emulator::cpu::Xlen;
use tracer::instruction::sub::SUB;
use tracer::instruction::addi::ADDI;
use tracer::instruction::lui::LUI;
use tracer::instruction::ld::LD;
use tracer::instruction::sd::SD;
use tracer::instruction::virtual_xor_rot::{VirtualXORROT16, VirtualXORROT24, VirtualXORROT32, VirtualXORROT63};
use tracer::instruction::RV32IMInstruction;
use tracer::utils::inline_helpers::{
    InstrAssembler,
    Value::Reg,
};
use tracer::utils::virtual_registers::allocate_virtual_register;

/// Blake2b initialization vector (IV)
#[rustfmt::skip]
const BLAKE2B_IV: [u64; 8] = [
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
];

/// Blake2b sigma permutation constants for message scheduling
/// Each round uses a different permutation of the input words
#[rustfmt::skip]
const SIGMA: [[usize; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

/// Layout of the 96 virtual registers (`vr`).
///
/// Jolt requires the total number of registers (physical + virtual) to be a power of two.
/// With 32 physical registers, we need 96 virtual registers to reach 128.
///
/// Virtual register layout:
/// - `vr[0..15]`:  Working state `v` (16 words)
/// - `vr[16..31]`: Message block `m` (16 words)
/// - `vr[32..39]`: Hash state `h` (8 words)
/// - `vr[40]`:     Counter value `t`
/// - `vr[41]`:     Final block flag
/// - `vr[42]`:     Temporary register
/// - `vr[43]`:     Zero constant
/// - `vr[44..95]`: Unused, allocated for padding to meet the power-of-two requirement
pub const NEEDED_REGISTERS: usize = 96;

// Memory layout constants for Blake2 virtual registers
const VR_WORKING_STATE_START: usize = 0;
const WORKING_STATE_SIZE: usize = 16;
const VR_MESSAGE_BLOCK_START: usize = 16;
pub const MESSAGE_BLOCK_SIZE: usize = 16;
const VR_HASH_STATE_START: usize = 32;
pub const HASH_STATE_SIZE: usize = 8;
const VR_T: usize = 40;
const VR_IS_FINAL: usize = 41;
const VR_TEMP: usize = 42;
const VR_ZERO: usize = 43;

// Rotation constants required in Blake2b
pub enum RotationAmount {
    ROT32,
    ROT24,
    ROT16,
    ROT63,
}

struct Blake2SequenceBuilder {
    asm: InstrAssembler,
    round: u32,
    vr: [u8; NEEDED_REGISTERS],
    operand_rs1: u8,
    operand_rs2: u8,
}

/// `Blake2SequenceBuilder` is a helper struct for constructing the virtual instruction
/// sequence required to emulate the Blake2b compression function within the RISC-V
/// instruction set. This builder is responsible for generating the correct sequence of
/// `RV32IMInstruction` instances that together perform the Blake2b compression,
/// using a set of virtual registers to hold intermediate state.
///
/// # Fields
/// - `asm`: Builder for the vector of generated instructions representing the Blake2b operation.
/// - `round`: The current round of the Blake2b compression (0..12).
/// - `vr`: An array of virtual register indices used for state and temporary values.
/// - `operand_rs1`: The source register index for the hash state pointer.
/// - `operand_rs2`: The source register index for the message block pointer.
///
/// # Usage
/// Typically, you construct a `Blake2SequenceBuilder` with the required register mapping
/// and operands, then call `.build()` to obtain the full instruction sequence for the
/// Blake2b operation. This is used to inline the Blake2b compression logic into the
/// RISC-V instruction stream for tracing or emulation purposes.
impl Blake2SequenceBuilder {
    fn new(
        address: u64,
        is_compressed: bool,
        xlen: Xlen,
        vr: [u8; NEEDED_REGISTERS],
        operand_rs1: u8,
        operand_rs2: u8,
    ) -> Self {
        Blake2SequenceBuilder {
            asm: InstrAssembler::new(address, is_compressed, xlen, true),
            round: 0,
            vr,
            operand_rs1,
            operand_rs2,
        }
    }

    fn build(mut self) -> Vec<RV32IMInstruction> {
        // Load inputs
        self.load_hash_state();
        self.load_message_blocks();
        self.load_counter_and_is_final_and_zero();

        // Initialize the working state v[0..15]
        self.initialize_working_state();

        // Cryptographic mixing for 12 rounds
        for round in 0..12 {
            self.round = round;
            self.blake2_round();
        }

        // Finalize the hash state
        self.finalize_state();

        // Store the final hash state back to memory
        self.store_state();

        self.asm.finalize()
    }

    fn load_hash_state(&mut self) {
        self.load_data_range(self.operand_rs1, 0, VR_HASH_STATE_START, HASH_STATE_SIZE);
    }

    fn load_message_blocks(&mut self) {
        self.load_data_range(
            self.operand_rs2,
            0,
            VR_MESSAGE_BLOCK_START,
            MESSAGE_BLOCK_SIZE,
        );
    }

    fn load_counter_and_is_final_and_zero(&mut self) {
        self.asm.emit_ld::<LD>(
            self.vr[VR_T],
            self.operand_rs2,
            MESSAGE_BLOCK_SIZE as i64 * 8,
        );
        self.asm.emit_ld::<LD>(
            self.vr[VR_IS_FINAL],
            self.operand_rs2,
            (MESSAGE_BLOCK_SIZE as i64 + 1) * 8,
        );
        // Load 0 into VR_ZERO register
        self.asm.emit_i::<ADDI>(self.vr[VR_ZERO], 0, 0);
    }

    // Initialize the working state v[0..15] according to Blake2b specification:
    fn initialize_working_state(&mut self) {
        // v[0..7] = h[0..7] (current hash state)
        for i in 0..HASH_STATE_SIZE {
            self.asm.xor(
                Reg(self.vr[VR_HASH_STATE_START + i]),
                Reg(self.vr[VR_ZERO]),
                self.vr[VR_WORKING_STATE_START + i],
            );
        }

        // v[8..15] = IV[0..7] (initialization vector)
        for i in 0..WORKING_STATE_SIZE - HASH_STATE_SIZE {
            // Load Blake2b IV constants
            // For now, we'll load from memory as a workaround for missing load_64bit_immediate
            // In a real implementation, these constants would be pre-loaded or generated with LUI/ADDI sequences
            let rd = self.vr[VR_WORKING_STATE_START + HASH_STATE_SIZE + i];
            self.asm.emit_u::<LUI>(rd, BLAKE2B_IV[i]);
        }

        // v[12] = v[12] ^ t (counter low)
        self.asm.xor(
            Reg(self.vr[VR_WORKING_STATE_START + 12]),
            Reg(self.vr[VR_T]),
            self.vr[VR_WORKING_STATE_START + 12],
        );

        // v[13] = IV[5] ^ (t >> 64) (counter high) - for 64-bit counter, high part is 0
        // Since we are using 64-bit counter, the high part is always 0, so v[13] remains unchanged

        // Handle final block flag: if is_final != 0, invert all bits of v[14]
        // We need to create a mask that is 0xFFFFFFFFFFFFFFFF if is_final != 0, or 0 if is_final == 0
        // Use the formula: mask = (0 - is_final) to convert 1 to 0xFFFFFFFFFFFFFFFF and 0 to 0
        let temp_mask = self.vr[VR_TEMP];
        // First, negate is_final (0 - is_final)
        self.asm.emit_r::<SUB>(temp_mask, self.vr[VR_ZERO], self.vr[VR_IS_FINAL]);
        // XOR v[14] with the mask: inverts all bits if is_final=1, leaves unchanged if is_final=0
        self.asm.xor(
            Reg(self.vr[VR_WORKING_STATE_START + 14]),
            Reg(temp_mask),
            self.vr[VR_WORKING_STATE_START + 14],
        );
    }

    /// Execute one round of Blake2b compression
    fn blake2_round(&mut self) {
        let sigma_round = &SIGMA[self.round as usize];

        // Column step: apply G function to columns
        self.g_function(0, 4, 8, 12, sigma_round[0], sigma_round[1]);
        self.g_function(1, 5, 9, 13, sigma_round[2], sigma_round[3]);
        self.g_function(2, 6, 10, 14, sigma_round[4], sigma_round[5]);
        self.g_function(3, 7, 11, 15, sigma_round[6], sigma_round[7]);

        // Diagonal step: apply G function to diagonals
        self.g_function(0, 5, 10, 15, sigma_round[8], sigma_round[9]);
        self.g_function(1, 6, 11, 12, sigma_round[10], sigma_round[11]);
        self.g_function(2, 7, 8, 13, sigma_round[12], sigma_round[13]);
        self.g_function(3, 4, 9, 14, sigma_round[14], sigma_round[15]);
    }

    /// Blake2b G function: core mixing function
    /// Updates v[a], v[b], v[c], v[d] using message words m[x], m[y]
    fn g_function(&mut self, a: usize, b: usize, c: usize, d: usize, x: usize, y: usize) {
        let va = self.vr[VR_WORKING_STATE_START + a];
        let vb = self.vr[VR_WORKING_STATE_START + b];
        let vc = self.vr[VR_WORKING_STATE_START + c];
        let vd = self.vr[VR_WORKING_STATE_START + d];
        let mx = self.vr[VR_MESSAGE_BLOCK_START + x];
        let my = self.vr[VR_MESSAGE_BLOCK_START + y];
        let temp1 = self.vr[VR_TEMP];

        // v[a] = v[a] + v[b] + m[x]
        self.asm.add(Reg(va), Reg(vb), temp1);
        self.asm.add(Reg(temp1), Reg(mx), va);

        // v[d] = rotr64(v[d] ^ v[a], 32)
        self.xor_rotate(vd, va, RotationAmount::ROT32, vd);

        // v[c] = v[c] + v[d]
        self.asm.add(Reg(vc), Reg(vd), vc);

        // v[b] = rotr64(v[b] ^ v[c], 24)
        self.xor_rotate(vb, vc, RotationAmount::ROT24, vb);

        // v[a] = v[a] + v[b] + m[y]
        self.asm.add(Reg(va), Reg(vb), temp1);
        self.asm.add(Reg(temp1), Reg(my), va);

        // v[d] = rotr64(v[d] ^ v[a], 16)
        self.xor_rotate(vd, va, RotationAmount::ROT16, vd);

        // v[c] = v[c] + v[d]
        self.asm.add(Reg(vc), Reg(vd), vc);

        // v[b] = rotr64(v[b] ^ v[c], 63)
        self.xor_rotate(vb, vc, RotationAmount::ROT63, vb);
    }

    /// Finalize the hash state according to Blake2b specification:
    /// For i in 0..8: h[i] = h[i] ^ v[i] ^ v[i+8]
    /// This produces the final 8-word hash output in the hash state registers.
    fn finalize_state(&mut self) {
        let temp = self.vr[VR_TEMP];

        for i in 0..HASH_STATE_SIZE {
            let hi = self.vr[VR_HASH_STATE_START + i];
            let vi = self.vr[VR_WORKING_STATE_START + i];
            let vi8 = self.vr[VR_WORKING_STATE_START + i + HASH_STATE_SIZE];

            // temp = v[i] ^ v[i+8]
            self.asm.xor(Reg(vi), Reg(vi8), temp);
            // h[i] = h[i] ^ temp
            self.asm.xor(Reg(hi), Reg(temp), hi);
        }
    }

    /// Store the final hash state (8 words) back to memory
    /// Blake2b compression outputs an 8-word hash state
    fn store_state(&mut self) {
        for i in 0..HASH_STATE_SIZE {
            self.asm.emit_s::<SD>(
                self.operand_rs1,
                self.vr[VR_HASH_STATE_START + i],
                (i * 8) as i64,
            );
        }
    }

    /// Load data from memory into virtual registers starting at a given offset
    fn load_data_range(
        &mut self,
        base_register: u8,
        memory_offset_start: usize,
        vr_start: usize,
        count: usize,
    ) {
        (0..count).for_each(|i| {
            self.asm.emit_ld::<LD>(
                self.vr[vr_start + i],
                base_register,
                ((memory_offset_start + i) * 8) as i64,
            );
        });
    }

    // XOR two registers, and then rotate right by the given amount using a single virtual instruction.
    fn xor_rotate(&mut self, rs1: u8, rs2: u8, amount: RotationAmount, rd: u8) {
        match amount {
            RotationAmount::ROT32 => {
                self.asm.emit_r::<VirtualXORROT32>(rd, rs1, rs2);
            }
            RotationAmount::ROT24 => {
                self.asm.emit_r::<VirtualXORROT24>(rd, rs1, rs2);
            }
            RotationAmount::ROT16 => {
                self.asm.emit_r::<VirtualXORROT16>(rd, rs1, rs2);
            }
            RotationAmount::ROT63 => {
                self.asm.emit_r::<VirtualXORROT63>(rd, rs1, rs2);
            }
        }
    }
}

/// Build Blake2b inline sequence for the RISC-V instruction stream
pub fn blake2b_inline_sequence_builder(
    address: u64,
    is_compressed: bool,
    xlen: Xlen,
    operand_rs1: u8,
    operand_rs2: u8,
    _operand_rs3: u8,
) -> Vec<RV32IMInstruction> {
    // Virtual registers used as a scratch space
    let guards: Vec<_> = (0..NEEDED_REGISTERS)
        .map(|_| allocate_virtual_register())
        .collect();
    let mut vr = [0; NEEDED_REGISTERS];
    for (i, guard) in guards.iter().enumerate() {
        vr[i] = **guard;
    }
    let builder =
        Blake2SequenceBuilder::new(address, is_compressed, xlen, vr, operand_rs1, operand_rs2);
    builder.build()
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use tracer::emulator::{cpu::Cpu, default_terminal::DefaultTerminal, mmu::DRAM_BASE};
//     use tracer::instruction::format::format_inline::FormatInline;
//     use tracer::instruction::{inline::INLINE, RISCVInstruction, RISCVTrace};
//     use crate::test_utils::store_words_to_memory;

//     const TEST_MEMORY_CAPACITY: u64 = 1024 * 1024; // 1MB

//     // Test constants from RFC 7693 Appendix A (Blake2b with "abc")
//     const INITIAL_STATE: [u64; HASH_STATE_SIZE] = [
//         0x6a09e667f3bcc908,
//         0xbb67ae8584caa73b,
//         0x3c6ef372fe94f82b,
//         0xa54ff53a5f1d36f1,
//         0x510e527fade682d1,
//         0x9b05688c2b3e6c1f,
//         0x1f83d9abfb41bd6b,
//         0x5be0cd19137e2179,
//     ];

//     const EXPECTED_STATE: [u64; HASH_STATE_SIZE] = [
//         0x0D4D1C983FA580BAu64, // BA 80 A5 3F 98 1C 4D 0D (little-endian)
//         0xE9F6129FB697276Au64, // 6A 27 97 B6 9F 12 F6 E9
//         0xB7C45A68142F214Cu64, // 4C 21 2F 14 68 5A C4 B7
//         0xD1A2FFDB6FBB124Bu64, // 4B 12 BB 6F DB FF A2 D1
//         0x2D79AB2A39C5877Du64, // 7D 87 C5 39 2A AB 79 2D
//         0x95CC3345DED552C2u64, // C2 52 D5 DE 45 33 CC 95
//         0x5A92F1DBA88AD318u64, // 18 D3 8A A8 DB F1 92 5A
//         0x239900D4ED8623B9u64, // B9 23 86 ED D4 00 99 23
//     ];

//     fn get_pre_post_states() -> ([u64; HASH_STATE_SIZE], [u64; HASH_STATE_SIZE]) {
//         (INITIAL_STATE, EXPECTED_STATE)
//     }

//     /// Test macro to reduce repetitive setup and verification
//     macro_rules! test_blake2 {
//         ($test_name:ident, $exec_block:expr) => {
//             #[test]
//             fn $test_name() {
//                 let (mut initial_state, expected_state) = get_pre_post_states();
//                 // Apply Blake2b parameter block: h[0] ^= 0x01010000 ^ (kk << 8) ^ nn
//                 initial_state[0] ^= 0x01010000 ^ (0u64 << 8) ^ 64u64;

//                 // Message block with "abc" in little-endian
//                 let mut message_block = [0u64; MESSAGE_BLOCK_SIZE];
//                 message_block[0] = 0x0000000000636261u64; // "abc"

//                 let (counter, is_final) = (3u64, true);

//                 let instruction = INLINE {
//                     address: 0,
//                     operands: FormatInline {
//                         rs1: 10, // Points to state
//                         rs2: 11, // Points to message block + counter + final flag
//                         rs3: 0,
//                     },
//                     // BLAKE2 inline opcode values (you may need to adjust these)
//                     opcode: 0x2B, // custom-1 opcode
//                     funct3: 0x00,
//                     funct7: 0x00, // Blake2 specific encoding
//                     inline_sequence_remaining: None,
//                     is_compressed: false,
//                 };

//                 // Set up CPU
//                 let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
//                 cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
//                 let state_addr = DRAM_BASE;
//                 let message_addr = DRAM_BASE + 1024; // Separate address for message block
//                 cpu.x[10] = state_addr as i64; // rs1 points to state
//                 cpu.x[11] = message_addr as i64; // rs2 points to message block

//                 // Store initial state (8 words) at rs1
//                 store_words_to_memory(&mut cpu, state_addr, &initial_state)
//                     .expect("Failed to store initial state");
//                 // Store message block (16 words) at rs2
//                 store_words_to_memory(&mut cpu, message_addr, &message_block)
//                     .expect("Failed to store message block");
//                 // Store counter after message block
//                 store_words_to_memory(&mut cpu, message_addr + 128, &[counter])
//                     .expect("Failed to store counter");
//                 // Store final flag after counter
//                 store_words_to_memory(
//                     &mut cpu,
//                     message_addr + 136,
//                     &[if is_final { 1 } else { 0 }],
//                 )
//                 .expect("Failed to store final flag");

//                 // Execute the instruction
//                 $exec_block(&instruction, &mut cpu);

//                 // Verify results (Blake2b compression outputs 8 words)
//                 let mut result = [0u64; HASH_STATE_SIZE];
//                 for i in 0..HASH_STATE_SIZE {
//                     let addr = state_addr + (i * 8) as u64;
//                     result[i] = cpu.mmu.load_doubleword(addr).unwrap().0;
//                     assert_eq!(
//                         result[i], expected_state[i],
//                         "Mismatch at word {}: got {:#x}, expected {:#x}",
//                         i, result[i], expected_state[i]
//                     );
//                 }
//             }
//         };
//     }

//     test_blake2!(
//         test_exec_correctness,
//         |instruction: &INLINE, cpu: &mut Cpu| {
//             instruction.execute(cpu, &mut ());
//         }
//     );

//     test_blake2!(
//         test_trace_correctness,
//         |instruction: &INLINE, cpu: &mut Cpu| {
//             instruction.trace(cpu, None);
//         }
//     );
// }