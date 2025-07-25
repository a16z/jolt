//! Blake2b hash function optimized for Jolt zkVM.
//!
//! This implementation provides:
//! - Custom RISC-V instruction support for guest execution
//! - Reference implementation for host execution  
//! - Streaming interface for large inputs

// Blake2b constants
const BLOCK_SIZE: usize = 128;
const BLOCK_SIZE_U64: usize = 16; // BLOCK_SIZE / 8
const STATE_SIZE: usize = 64;
const STATE_SIZE_U64: usize = 8; // STATE_SIZE / 8
const OUTPUT_SIZE: usize = 64;
const MAX_OUTPUT_SIZE: usize = 64;

/// Blake2b initialization vector (IV)
#[rustfmt::skip]
const BLAKE2B_IV: [u64; 8] = [
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
];

/// Blake2b hasher state for streaming operation.
pub struct Blake2b {
    /// Hash state (8 x 64-bit words)
    h: [u64; STATE_SIZE_U64],
    /// Buffer for incomplete blocks
    buffer: [u8; BLOCK_SIZE],
    /// Current buffer length
    buffer_len: usize,
    /// Total bytes processed
    counter: u64,
}

impl Blake2b {
    /// Creates a new Blake2b hasher with specified output length.
    ///
    /// # Panics
    /// Panics if `output_len` is 0 or greater than 64.
    pub fn new(output_len: usize) -> Self {
        assert!(
            output_len > 0 && output_len <= MAX_OUTPUT_SIZE,
            "Output length must be between 1 and {MAX_OUTPUT_SIZE} bytes, got {output_len}"
        );

        let mut h = BLAKE2B_IV;
        h[0] ^= 0x01010000 ^ (output_len as u64);

        Self {
            h,
            buffer: [0; BLOCK_SIZE],
            buffer_len: 0,
            counter: 0,
        }
    }

    /// Processes input data incrementally.
    pub fn update(&mut self, input: &[u8]) {
        if input.is_empty() {
            return;
        }
        for char in input {
            if self.buffer_len == 128 {
                self.counter += BLOCK_SIZE as u64;
                compression_caller(&mut self.h, &self.buffer, self.counter, false);
                self.buffer_len = 0;
            }
            self.buffer[self.buffer_len] = *char;
            self.buffer_len += 1;
        }
    }

    /// Finalizes the hash and returns the digest.
    pub fn finalize(mut self) -> [u8; OUTPUT_SIZE] {
        self.counter += self.buffer_len as u64;
        self.buffer[self.buffer_len..].fill(0);
        // Process final block
        compression_caller(&mut self.h, &self.buffer, self.counter, true);

        // Extract hash bytes
        let mut hash = [0u8; OUTPUT_SIZE];
        let state_bytes: &[u8] =
            unsafe { core::slice::from_raw_parts(self.h.as_ptr() as *const u8, STATE_SIZE) };
        hash.copy_from_slice(&state_bytes[..OUTPUT_SIZE]);
        hash
    }

    /// Computes Blake2b hash in one call.
    pub fn digest(input: &[u8]) -> [u8; OUTPUT_SIZE] {
        let mut hasher = Self::new(OUTPUT_SIZE);
        hasher.update(input);
        hasher.finalize()
    }
}

fn compression_caller(
    hash_state: &mut [u64; STATE_SIZE_U64],
    message_block: &[u8],
    counter: u64,
    is_final: bool,
) {
    // Convert buffer to u64 words
    let mut message = [0u64; BLOCK_SIZE_U64 + 2];
    for i in 0..BLOCK_SIZE_U64 {
        message[i] = u64::from_le_bytes(message_block[i * 8..(i + 1) * 8].try_into().unwrap());
    }

    message[16] = counter;
    message[17] = is_final as u64;

    unsafe {
        blake2b_compress(hash_state.as_mut_ptr(), message.as_ptr());
    }
}

impl Default for Blake2b {
    fn default() -> Self {
        Self::new(OUTPUT_SIZE)
    }
}

/// Blake2b compression function - guest implementation.
///
/// # Safety
/// - `state` must point to a valid array of 8 u64 values
/// - `message` must point to a valid array of 18 u64 values (16 message + counter + final flag)
/// - Both pointers must be properly aligned for u64 access
#[cfg(not(feature = "host"))]
pub unsafe fn blake2b_compress(state: *mut u64, message: *const u64) {
    // Memory layout for Blake2 instruction:
    // rs1: points to state (64 bytes)
    // rs2: points to message block (128 bytes) + counter (8 bytes) + final flag (8 bytes)

    // Call Blake2 instruction using funct7=0x02 to distinguish from Keccak (0x01) and SHA-256 (0x00)
    core::arch::asm!(
        ".insn r 0x0B, 0x0, 0x02, x0, {}, {}",
        in(reg) state,
        in(reg) message,
        options(nostack)
    );
}

/// Blake2b compression function - host implementation.
///
/// # Safety  
/// - `state` must point to a valid array of 8 u64 values
/// - `message` must point to a valid array of 18 u64 values
#[cfg(feature = "host")]
pub unsafe fn blake2b_compress(state: *mut u64, message: *const u64) {
    let state_slice = core::slice::from_raw_parts_mut(state, 8);
    let message_slice = core::slice::from_raw_parts(message, 18);

    // Convert to arrays for type safety
    let state_array: &mut [u64; 8] = state_slice
        .try_into()
        .expect("State pointer must reference exactly 8 u64 values");
    let message_array: [u64; 18] = message_slice
        .try_into()
        .expect("Message pointer must reference exactly 18 u64 values");

    tracer::instruction::inline_blake2::execute_blake2b_compression(state_array, &message_array);
}

#[cfg(test)]
mod digest_tests {
    use super::*;
    use hex_literal::hex;

    #[test]
    fn test_blake2b_digest() {
        let test_cases: [(&'static str, &[u8], [u8; 64]); 5] = [
            (
                "empty",
                b"",
                hex!("786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce")
            ),
            (
                "lt_128_bytes",
                b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456",
                hex!("2d95bd8dfdf8c4077f9bf54fe1a622e8bff985727a1f937f05c19608b93afbde331cc949d67cf29f3cbe081f2a853c13131b7f8f5d162810eec2e0001df9199f")
            ),
            (
                "exactly_128_bytes",
                b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                hex!("687222a8b7e18fe2351529741f9f377dbfe57ccc40ffacd7dad6457eb0f5434b308c25eeb85f2c434889877eae9cfcda86e2220bbedb5ddeeef1db1b76113997")
            ),
            (
                "gt_128_bytes",
                b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef456789abcdef0123456789abcdef0123456789abcdef",
                hex!("eec6581ca2d51e7f8bff0cb9e0742b454bad4d28bb5078737a6bce318bb29902ca6c2fd4c412d9ed6bb2940692b39012b69ab81ca33cca4d292f3a095cd84007")
            ),
            (
                "exactly_256_bytes",
                b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                hex!("342949a83f4809037dcb71d5d527ef9c8060c20cda8a7e4414bcca487e9bc5726e0d4646b7f869b3f3decb362508ec4672c3314ad345d1c36377fc1f3020585c")
            ),
        ];
        for (test_name, input, expected) in test_cases {
            let hash = Blake2b::digest(input);
            assert_eq!(
                hash,
                expected,
                "Blake2b test failed for case: {test_name} (input length: {} bytes)",
                input.len()
            );
        }
    }

    #[test]
    fn test_blake2b_against_reference_implementation() {
        for pattern_id in 0..4 {
            let (pattern_name, pattern_fn): (&str, fn(usize) -> u8) = match pattern_id {
                0 => ("sequential", |i| (i) as u8),
                1 => ("zeros", |_| 0u8),
                2 => ("ones", |_| 255u8),
                3 => ("random_pattern", |i| ((i * 7 + 13) % 256) as u8),
                _ => unreachable!(),
            };
            let mut input = [0u8; 1200];
            for i in 0..1200 {
                input[i] = pattern_fn(i);
            }
            for length in 0..=1200 {
                use blake2::Digest as RefDigest;
                assert_eq!(
                    Blake2b::digest(&input),
                    Into::<[u8; 64]>::into(blake2::Blake2b512::digest(&input)),
                    "Blake2b mismatch with {pattern_name} pattern at length {length}"
                );
            }
        }
    }

    #[test]
    fn test_blake2b_variable_input_lengths() {
        const MAX_LENGTH: usize = 1200;
        // Pre-generate a large input buffer with a repeating pattern
        let input_buffer: [u8; MAX_LENGTH] = std::array::from_fn(|i| {
            // Create a more complex pattern that changes based on position
            let base = (i % 256) as u8;
            let modifier = ((i / 256) * 17 + (i % 7) * 31) as u8;
            base.wrapping_add(modifier)
        });

        // Test every length from 0 to MAX_LENGTH
        for length in 0..=MAX_LENGTH {
            let input = &input_buffer[..length];
            use blake2::Digest as RefDigest;
            assert_eq!(
                Blake2b::digest(input),
                Into::<[u8; 64]>::into(blake2::Blake2b512::digest(input)),
                "Blake2b mismatch at input length {length}"
            );
        }
    }

    #[test]
    fn test_blake2b_edge_case_lengths() {
        use blake2::{Blake2b512, Digest as RefDigest};

        // Test specific edge case lengths that are important for Blake2b
        let critical_lengths = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, // Very small
            15, 16, 17, // Around 16-byte boundary
            31, 32, 33, // Around 32-byte boundary
            55, 56, 57, // Just before block size
            63, 64, 65, // Around 64-byte boundary
            111, 112, 113, // Random mid-range
            127, 128, 129, // Around 128-byte boundary (2 blocks)
            191, 192, 193, // 3 blocks
            255, 256, 257, // Around 256-byte boundary
            511, 512, 513, // Around 512-byte boundary
            1023, 1024, 1025, // Around 1024-byte boundary
            1199, 1200, // At max length
        ];

        const MAX_TEST_LENGTH: usize = 1200;
        let input_buffer: [u8; MAX_TEST_LENGTH] = std::array::from_fn(|i| {
            // Create a pseudo-random but deterministic pattern
            ((i * 213 + 17) % 256) as u8
        });

        for &length in &critical_lengths {
            if length <= MAX_TEST_LENGTH {
                let input = &input_buffer[..length];
                assert_eq!(
                    Blake2b::digest(input),
                    Into::<[u8; 64]>::into(Blake2b512::digest(input)),
                    "Blake2b mismatch at critical length {length}"
                );
            }
        }
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;
    use hex_literal::hex;

    const OUTPUT_LEN: usize = 64;
    #[test]
    fn test_blake2b_streaming_digest() {
        let test_cases: [(&'static str, &[u8], [u8; 64]); 5] = [
            (
                "empty",
                b"",
                hex!("786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce")
            ),
            (
                "lt_128_bytes",
                b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456",
                hex!("2d95bd8dfdf8c4077f9bf54fe1a622e8bff985727a1f937f05c19608b93afbde331cc949d67cf29f3cbe081f2a853c13131b7f8f5d162810eec2e0001df9199f")
            ),
            (
                "exactly_128_bytes",
                b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                hex!("687222a8b7e18fe2351529741f9f377dbfe57ccc40ffacd7dad6457eb0f5434b308c25eeb85f2c434889877eae9cfcda86e2220bbedb5ddeeef1db1b76113997")
            ),
            (
                "gt_128_bytes",
                b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef456789abcdef0123456789abcdef0123456789abcdef",
                hex!("eec6581ca2d51e7f8bff0cb9e0742b454bad4d28bb5078737a6bce318bb29902ca6c2fd4c412d9ed6bb2940692b39012b69ab81ca33cca4d292f3a095cd84007")
            ),
            (
                "exactly_256_bytes",
                b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                hex!("342949a83f4809037dcb71d5d527ef9c8060c20cda8a7e4414bcca487e9bc5726e0d4646b7f869b3f3decb362508ec4672c3314ad345d1c36377fc1f3020585c")
            ),
        ];

        for (test_name, input, expected) in test_cases {
            let mut hasher = Blake2b::new(OUTPUT_LEN);
            hasher.update(input);
            let hash = hasher.finalize();
            assert_eq!(
                hash,
                expected,
                "Blake2b streaming test failed for case: {test_name} (input length: {} bytes)",
                input.len()
            );
        }
    }

    #[test]
    fn test_blake2b_streaming_against_reference() {
        for pattern_id in 0..4 {
            let (pattern_name, pattern_fn): (&str, fn(usize) -> u8) = match pattern_id {
                0 => ("sequential", |i| i as u8),
                1 => ("zeros", |_| 0u8),
                2 => ("ones", |_| 255u8),
                3 => ("random_pattern", |i| ((i * 7 + 13) % 256) as u8),
                _ => unreachable!(),
            };

            let mut input = [0u8; 1200];
            for i in 0..1200 {
                input[i] = pattern_fn(i);
            }

            for length in 0..=1200 {
                let test_input = &input[..length];
                let mut hasher = Blake2b::new(OUTPUT_LEN);
                hasher.update(test_input);

                use blake2::Digest as RefDigest;
                assert_eq!(
                    hasher.finalize(),
                    Into::<[u8; 64]>::into(blake2::Blake2b512::digest(test_input)),
                    "Blake2b streaming mismatch with {pattern_name} pattern at length {length}"
                );
            }
        }
    }

    #[test]
    fn test_blake2b_streaming_variable_lengths() {
        const MAX_LENGTH: usize = 1200;

        let input_buffer: [u8; MAX_LENGTH] = std::array::from_fn(|i| {
            let base = (i % 256) as u8;
            let modifier = ((i / 256) * 17 + (i % 7) * 31) as u8;
            base.wrapping_add(modifier)
        });

        for length in 0..=MAX_LENGTH {
            let input = &input_buffer[..length];
            let mut hasher = Blake2b::new(OUTPUT_LEN);
            hasher.update(input);

            use blake2::Digest as RefDigest;
            assert_eq!(
                hasher.finalize(),
                Into::<[u8; 64]>::into(blake2::Blake2b512::digest(input)),
                "Blake2b streaming mismatch at input length {length}"
            );
        }
    }

    #[test]
    fn test_blake2b_streaming_incremental_updates() {
        use blake2::Digest as RefDigest;

        const MAX_LENGTH: usize = 512;
        let input_buffer: [u8; MAX_LENGTH] = std::array::from_fn(|i| ((i * 137 + 42) % 256) as u8);

        // Test different chunk sizes
        let chunk_sizes = [1, 3, 7, 16, 32, 63, 64, 65, 128];
        let test_lengths = [0, 1, 63, 64, 65, 127, 128, 129, 255, 256, 257, MAX_LENGTH];

        for &chunk_size in &chunk_sizes {
            for total_length in test_lengths {
                let input = &input_buffer[..total_length];
                // Feed data incrementally
                let mut hasher = Blake2b::new(OUTPUT_LEN);
                let mut expected_hasher = blake2::Blake2b512::new();
                let mut offset = 0;
                while offset < total_length {
                    let end = std::cmp::min(offset + chunk_size, total_length);
                    hasher.update(&input[offset..end]);
                    expected_hasher.update(&input[offset..end]);
                    offset = end;
                }
                assert_eq!(
                    hasher.finalize(),
                    Into::<[u8; 64]>::into(expected_hasher.finalize()),
                    "Incremental update mismatch: chunk_size={chunk_size}, total_length={total_length}"
                );
            }
        }
    }

    #[test]
    fn test_blake2b_streaming_edge_cases() {
        use blake2::Digest as RefDigest;

        let critical_lengths = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 31, 32, 33, 55, 56, 57, 63, 64, 65, 111, 112,
            113, 127, 128, 129, 191, 192, 193, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025,
            1199, 1200,
        ];

        const MAX_TEST_LENGTH: usize = 1200;
        let input_buffer: [u8; MAX_TEST_LENGTH] =
            std::array::from_fn(|i| ((i * 213 + 17) % 256) as u8);

        for &length in &critical_lengths {
            if length <= MAX_TEST_LENGTH {
                let input = &input_buffer[..length];
                let mut hasher = Blake2b::new(OUTPUT_LEN);
                hasher.update(input);
                assert_eq!(
                    hasher.finalize(),
                    Into::<[u8; 64]>::into(blake2::Blake2b512::digest(input)),
                    "Blake2b streaming mismatch at critical length {length}"
                );
            }
        }
    }

    #[test]
    fn test_blake2b_streaming_multiple_updates() {
        use blake2::Digest as RefDigest;

        // Test that multiple small updates produce the same result as one large update
        let data_parts: &[&[u8]] = &[
            b"Hello, ",
            b"this is ",
            b"a test of ",
            b"multiple ",
            b"updates to ",
            b"the Blake2b ",
            b"streaming ",
            b"interface!",
        ];

        // Pre-calculated: total is 73 bytes
        const TOTAL_LEN: usize = 77;
        let mut full_data = [0u8; TOTAL_LEN];
        let mut offset = 0;

        for part in data_parts {
            full_data[offset..offset + part.len()].copy_from_slice(part);
            offset += part.len();
        }

        // Our streaming implementation with multiple updates
        let mut hasher = Blake2b::new(OUTPUT_LEN);
        for part in data_parts {
            hasher.update(part);
        }
        assert_eq!(
            hasher.finalize(),
            Into::<[u8; 64]>::into(blake2::Blake2b512::digest(&full_data)),
            "Multiple updates should produce same result as single update"
        );
    }

    #[test]
    fn test_blake2b_streaming_empty_updates() {
        use blake2::Digest as RefDigest;

        let test_data = b"Some test data for empty update testing";

        // Test with empty updates mixed in
        let mut hasher = Blake2b::new(OUTPUT_LEN);
        hasher.update(b""); // Empty update at start
        hasher.update(&test_data[..10]);
        hasher.update(b""); // Empty update in middle
        hasher.update(&test_data[10..]);
        hasher.update(b""); // Empty update at end

        assert_eq!(
            hasher.finalize(),
            Into::<[u8; 64]>::into(blake2::Blake2b512::digest(test_data)),
            "Empty updates should not affect the result"
        );
    }
}
