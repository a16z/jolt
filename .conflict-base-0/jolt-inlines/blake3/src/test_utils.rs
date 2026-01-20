use crate::{BLAKE3_FUNCT3, BLAKE3_FUNCT7, BLAKE3_KEYED64_FUNCT3, INLINE_OPCODE};
use tracer::emulator::cpu::Xlen;
use tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};

pub type ChainingValue = [u32; crate::CHAINING_VALUE_LEN];
pub type MessageBlock = [u32; crate::MSG_BLOCK_LEN];

pub fn create_blake3_harness() -> InlineTestHarness {
    // Blake3 needs message block (64 bytes) + params (16 bytes) contiguous at rs2
    // and state (32 bytes) at rs1
    let layout = InlineMemoryLayout::single_input(80, 32); // 80 bytes for message+params, 32-byte state
    InlineTestHarness::new(layout, Xlen::Bit64)
}

/// Create harness for Keyed64 instruction (Merkle tree merge)
/// ABI: rs1 = left, rs2 = right, rd = iv (in/out)
pub fn create_blake3_keyed64_harness() -> InlineTestHarness {
    // Keyed64 needs:
    // - rs1: left CV (32 bytes) -> input
    // - rs2: right CV (32 bytes) -> input2
    // - rd: IV (32 bytes, in/out) -> output
    let layout = InlineMemoryLayout::two_inputs(32, 32, 32); // left, right, iv
    InlineTestHarness::new(layout, Xlen::Bit64)
}

pub fn load_blake3_keyed64_data(
    harness: &mut InlineTestHarness,
    left: &ChainingValue,
    right: &ChainingValue,
    iv: &ChainingValue,
) {
    harness.setup_registers();
    // Load left to rs1 location (input)
    harness.load_input32(left);
    // Load right to rs2 location (input2)
    harness.load_input2_32(right);
    // Load IV to rd/rs3 location (output)
    harness.load_state32(iv);
}

pub fn keyed64_instruction() -> tracer::instruction::inline::INLINE {
    InlineTestHarness::create_default_instruction(
        INLINE_OPCODE,
        BLAKE3_KEYED64_FUNCT3,
        BLAKE3_FUNCT7,
    )
}

pub fn load_blake3_data(
    harness: &mut InlineTestHarness,
    chaining_value: &ChainingValue,
    message: &MessageBlock,
    counter: &[u32; 2],
    block_len: u32,
    flags: u32,
) {
    harness.setup_registers();
    // Load chaining value to output location (rs1 points here)
    harness.load_state32(chaining_value);

    // Blake3 expects message + parameters contiguously at rs2
    // Create combined input: message (16 u32s) + counter (2 u32s) + block_len (1 u32) + flags (1 u32)
    let mut combined_input = Vec::with_capacity(20);
    combined_input.extend_from_slice(message);
    combined_input.extend_from_slice(counter);
    combined_input.push(block_len);
    combined_input.push(flags);

    // Load the combined input
    harness.load_input32(&combined_input);
}

pub fn read_output(harness: &mut InlineTestHarness) -> ChainingValue {
    let vec = harness.read_output32(crate::CHAINING_VALUE_LEN);
    let mut output = [0u32; crate::CHAINING_VALUE_LEN];
    output.copy_from_slice(&vec);
    output
}

pub fn instruction() -> tracer::instruction::inline::INLINE {
    InlineTestHarness::create_default_instruction(INLINE_OPCODE, BLAKE3_FUNCT3, BLAKE3_FUNCT7)
}

#[cfg(test)]
pub mod helpers {
    pub fn generate_random_bytes(len: usize) -> Vec<u8> {
        use rand::rngs::StdRng;
        use rand::{RngCore, SeedableRng};

        let mut buf = vec![0u8; len];
        // Use a fixed seed for deterministic test results
        let mut rng = StdRng::seed_from_u64(12345);
        rng.fill_bytes(&mut buf);
        buf
    }

    pub fn bytes_to_u32_vec(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    pub fn compute_expected_result(input: &[u8]) -> [u8; crate::OUTPUT_SIZE_IN_BYTES] {
        blake3::hash(input).as_bytes()[0..crate::OUTPUT_SIZE_IN_BYTES]
            .try_into()
            .unwrap()
    }

    pub fn compute_keyed_expected_result(
        input: &[u8],
        key: [u32; crate::CHAINING_VALUE_LEN],
    ) -> [u8; crate::OUTPUT_SIZE_IN_BYTES] {
        let mut key_bytes = [0u8; 32];
        for (i, word) in key.iter().enumerate() {
            key_bytes[i * 4..(i + 1) * 4].copy_from_slice(&word.to_le_bytes());
        }
        blake3::keyed_hash(&key_bytes, input).as_bytes()[0..crate::OUTPUT_SIZE_IN_BYTES]
            .try_into()
            .unwrap()
    }
}
