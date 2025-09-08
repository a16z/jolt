//! This file provides high-level API to use BLAKE3 compression, both in host and guest mode.

use crate::{
    BLOCK_INPUT_SIZE_IN_BYTES, CHAINING_VALUE_LEN, COUNTER_LEN, FLAG_CHUNK_END, FLAG_CHUNK_START,
    FLAG_KEYED_HASH, FLAG_ROOT, IV, MSG_BLOCK_LEN, OUTPUT_SIZE_IN_BYTES,
};

pub struct Blake3 {
    /// Hash state (8 x 32-bit words)
    h: [u32; CHAINING_VALUE_LEN],
    /// Buffer for incomplete blocks
    buffer: [u8; BLOCK_INPUT_SIZE_IN_BYTES],
    /// Current buffer length
    buffer_len: usize,
    /// Total bytes processed
    counter: u64,
    /// Whether it is in the keyed_hash mode
    is_keyed_hash: bool,
}

/// Note: Current implementation only supports hashing input of at most 64 bytes. Larger inputs are not supported yet.
impl Blake3 {
    pub fn new() -> Self {
        Self {
            h: IV,
            buffer: [0; BLOCK_INPUT_SIZE_IN_BYTES],
            buffer_len: 0,
            counter: 0,
            is_keyed_hash: false,
        }
    }

    pub fn new_keyed(key: [u32; CHAINING_VALUE_LEN]) -> Self {
        Self {
            h: key,
            buffer: [0; BLOCK_INPUT_SIZE_IN_BYTES],
            buffer_len: 0,
            counter: 0,
            is_keyed_hash: true
        }
    }

    /// Processes input data incrementally.
    ///
    /// # Panics
    /// Panics if the total input exceeds 64 bytes.
    pub fn update(&mut self, input: &[u8]) {
        if input.is_empty() {
            return;
        }
        for byte in input {
            if self.buffer_len == BLOCK_INPUT_SIZE_IN_BYTES {
                panic!("Buffer is full and cannot add any new data");
            }
            self.buffer[self.buffer_len] = *byte;
            self.buffer_len += 1;
        }
    }

    /// Finalizes the hash and returns the digest.
    pub fn finalize(mut self) -> [u8; OUTPUT_SIZE_IN_BYTES] {
        self.buffer[self.buffer_len..].fill(0);
        // Process final block
        compression_caller(
            &mut self.h,
            &self.buffer,
            self.counter,
            self.buffer_len as u32,
            FLAG_CHUNK_START | FLAG_CHUNK_END | FLAG_ROOT | (if self.is_keyed_hash { FLAG_KEYED_HASH } else { 0 }) // CHUNK_START | CHUNK_END | ROOT | KEYED_HASH
        );

        // Extract hash bytes
        let mut hash = [0u8; OUTPUT_SIZE_IN_BYTES];
        let state_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(self.h.as_ptr() as *const u8, CHAINING_VALUE_LEN * 4)
        };
        hash.copy_from_slice(&state_bytes[..OUTPUT_SIZE_IN_BYTES]);
        hash
    }

    pub fn digest(input: &[u8]) -> [u8; OUTPUT_SIZE_IN_BYTES] {
        let mut hasher = Self::new();
        hasher.update(input);
        hasher.finalize()
    }

    pub fn keyed_hash(input: &[u8], key: [u32; CHAINING_VALUE_LEN]) -> [u8; OUTPUT_SIZE_IN_BYTES] {
        let mut hasher = Self::new_keyed(key);
        hasher.update(input);
        hasher.finalize()
    }
}

fn compression_caller(
    hash_state: &mut [u32; CHAINING_VALUE_LEN],
    message_block: &[u8],
    counter: u64,
    input_bytes_num: u32,
    flags: u32
) {
    // Convert buffer to u32 words
    let mut message = [0u32; MSG_BLOCK_LEN + COUNTER_LEN + 2];
    for i in 0..MSG_BLOCK_LEN {
        message[i] = u32::from_le_bytes(message_block[i * 4..(i + 1) * 4].try_into().unwrap());
    }

    message[MSG_BLOCK_LEN] = counter as u32;
    message[MSG_BLOCK_LEN + 1] = (counter >> 32) as u32;
    message[MSG_BLOCK_LEN + COUNTER_LEN] = input_bytes_num;
    message[MSG_BLOCK_LEN + COUNTER_LEN + 1] = flags;

    unsafe {
        blake3_compress(hash_state.as_mut_ptr(), message.as_ptr());
    }
}

impl Default for Blake3 {
    fn default() -> Self {
        Self::new()
    }
}

/// BLAKE3 compression function - guest implementation.
///
/// # Safety
/// - `chaining_value` must be a valid pointer to 32 bytes of readable and writable memory.
/// - `message` must be a valid pointer to 64 bytes of readable memory.
/// - Both pointers must be properly aligned for u32 access (4-byte alignment).
#[cfg(not(feature = "host"))]
pub unsafe fn blake3_compress(chaining_value: *mut u32, message: *const u32) {
    use jolt_inlines_common::constants::{blake3 as blake3_consts, INLINE_OPCODE};
    // Memory layout for BLAKE3 instruction:
    // rs1: points to chaining value (32 bytes)
    // rs2: points to message block (64 bytes) + counter (8 bytes) + block_len (4 bytes) + flags (4 bytes)

    core::arch::asm!(
        ".insn r {opcode}, {funct3}, {funct7}, x0, {rs1}, {rs2}",
        opcode = const INLINE_OPCODE,
        funct3 = const blake3_consts::FUNCT3,
        funct7 = const blake3_consts::FUNCT7,
        rs1 = in(reg) chaining_value,
        rs2 = in(reg) message,
        options(nostack)
    );
}

/// BLAKE3 compression function - host implementation.
///
/// # Safety
/// - `chaining_value` must be a valid pointer to 32 bytes.
/// - `message` must be a valid pointer to 64 bytes.
#[cfg(feature = "host")]
pub unsafe fn blake3_compress(chaining_value: *mut u32, message: *const u32) {
    // Memory layout for BLAKE3 instruction:
    // message points to: message block (64 bytes / 16 u32s) + counter (8 bytes / 2 u32s) + block_len (4 bytes / 1 u32) + flags (4 bytes / 1 u32)

    // Extract the message block (first 16 u32s)
    let message_block = &*(message as *const [u32; 16]);

    // Extract counter (next 2 u32s at offset 16)
    let counter_ptr = message.add(16);
    let counter_low = *counter_ptr;
    let counter_high = *counter_ptr.add(1);
    let counter_array = [counter_low, counter_high];

    // Extract block_len (next u32 at offset 18)
    let block_len = *message.add(18);

    // Extract flags (next u32 at offset 19)
    let flags = *message.add(19);

    // On the host, we call our reference implementation from the exec module.
    crate::exec::execute_blake3_compression(
        &mut *(chaining_value as *mut [u32; 8]),
        message_block,
        &counter_array,
        block_len,
        flags,
    );
}

#[cfg(test)]
#[cfg(feature = "host")]
mod tests {
    use super::Blake3 as inline_blake3;
    use crate::{test_utils::helpers::*, BLOCK_INPUT_SIZE_IN_BYTES, CHAINING_VALUE_LEN};

    fn random_partition<'a>(data: &'a [u8]) -> Vec<&'a [u8]> {
        if data.is_empty() {
            return vec![&[]];
        }
        let len = data.len();
        let mut parts = Vec::new();
        let mut partitioned_num = 0usize;
        let mut rng = rand::thread_rng();
        use rand::Rng;
        while partitioned_num < len {
            let remaining = len - partitioned_num;
            let take = rng.gen_range(1..=remaining);
            parts.push(&data[partitioned_num..partitioned_num + take]);
            partitioned_num += take;
        }
        parts
    }

    #[test]
    fn digest_matches_standard() {
        for _ in 0..1000 {
            let len = rand::random::<usize>() % (BLOCK_INPUT_SIZE_IN_BYTES);
            let input = generate_random_bytes(len);
            let result = inline_blake3::digest(&input);
            let expected = compute_expected_result(&input);
            assert_eq!(result, expected, "digest mismatch for input={input:02x?}");
        }
    }

    #[test]
    fn keyed_digest_random_keys_match_standard() {
        for _ in 0..1000 {
            let len = rand::random::<usize>() % (BLOCK_INPUT_SIZE_IN_BYTES);
            let input = generate_random_bytes(len);
            let key_bytes = generate_random_bytes(CHAINING_VALUE_LEN * 4);
            let mut key = [0u32; CHAINING_VALUE_LEN];
            key.copy_from_slice(&bytes_to_u32_vec(&key_bytes));
            let result = inline_blake3::keyed_hash(&input, key);
            let expected = compute_keyed_expected_result(&input, key);
            assert_eq!(
                result,
                expected,
                "keyed digest mismatch for input={input:02x?} and random key={key:x?}"
            );
        }
    }

    #[test]
    fn streaming_update_finalize_matches_standard() {
        for _ in 0..1000 {
            let len = rand::random::<usize>() % (BLOCK_INPUT_SIZE_IN_BYTES + 1);
            let data = generate_random_bytes(len);
            let expected = compute_expected_result(&data);
            let parts = random_partition(&data);
            let mut hasher = inline_blake3::new();
            for p in &parts {
                hasher.update(p);
            }
            let got = hasher.finalize();
            assert_eq!(got, expected, "stream mismatch for input={data:02x?}");
        }
    }

    #[test]
    #[should_panic]
    fn panics_on_larger_than_block_input() {
        // 65 bytes triggers internal buffer overflow guard
        let input = vec![0u8; BLOCK_INPUT_SIZE_IN_BYTES + 1];
        let _ = inline_blake3::digest(&input);
    }

    #[test]
    fn edge_cases() {
        // Empty
        let empty: &[u8] = &[];
        assert_eq!(inline_blake3::digest(empty), compute_expected_result(empty));

        // Length 64 (block-size)
        let mut l64 = [0u8; 64];
        for (i, b) in l64.iter_mut().enumerate() {
            *b = 255 - i as u8;
        }
        assert_eq!(inline_blake3::digest(&l64), compute_expected_result(&l64));

        // All zeros (64 bytes)
        let zeros = [0u8; 64];
        assert_eq!(
            inline_blake3::digest(&zeros),
            compute_expected_result(&zeros)
        );

        // All 0xFF (64 bytes)
        let maxes = [0xFFu8; 64];
        assert_eq!(
            inline_blake3::digest(&maxes),
            compute_expected_result(&maxes)
        );
    }
}
