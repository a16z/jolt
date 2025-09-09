use crate::{BLAKE3_FUNCT3, BLAKE3_FUNCT7, INLINE_OPCODE};
use tracer::emulator::cpu::Xlen;
use tracer::utils::inline_test_harness::{hash_helpers, InlineTestHarness};

pub type ChainingValue = [u32; crate::CHAINING_VALUE_LEN];
pub type MessageBlock = [u32; crate::MSG_BLOCK_LEN];

pub const RS1: u8 = 10;
pub const RS2: u8 = 11;

pub fn create_blake3_harness() -> InlineTestHarness {
    hash_helpers::blake3_harness(Xlen::Bit64)
}

pub fn load_blake3_data(
    harness: &mut InlineTestHarness,
    chaining_value: &ChainingValue,
    message: &MessageBlock,
    counter: &[u32; 2],
    block_len: u32,
    flags: u32,
) {
    harness.setup_registers(RS1, RS2, None);
    harness.load_state32(chaining_value);
    harness.load_input32(message);
    // Pack parameters: counter (2 u32s), block_len (1 u32), flags (1 u32)
    let mut params = Vec::with_capacity(4);
    params.extend_from_slice(counter);
    params.push(block_len);
    params.push(flags);
    harness.load_input2_32(&params);
}

pub fn read_output(harness: &mut InlineTestHarness) -> ChainingValue {
    let vec = harness.read_output32(crate::CHAINING_VALUE_LEN);
    let mut output = [0u32; crate::CHAINING_VALUE_LEN];
    output.copy_from_slice(&vec);
    output
}

pub fn instruction() -> tracer::instruction::inline::INLINE {
    InlineTestHarness::create_instruction(INLINE_OPCODE, BLAKE3_FUNCT3, BLAKE3_FUNCT7, RS1, RS2, 0)
}

#[cfg(test)]
pub mod helpers {
    pub fn generate_random_bytes(len: usize) -> Vec<u8> {
        let mut buf = vec![0u8; len];
        let mut rng = rand::thread_rng();
        use rand::RngCore;
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
