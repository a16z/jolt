//! Blake2b hash function implementation optimized for Jolt zkVM.
//!
//! On the host, it calls the tracer implementation. On the guest (zkVM),
//! it uses a custom RISC-V instruction for efficient proving.

const BLOCK_SIZE: usize = 128; // Blake2b block size in bytes
const BLOCK_SIZE_U64: usize = BLOCK_SIZE / 8; // 16 words
const STATE_SIZE: usize = 64; // Blake2b state size in bytes
const STATE_SIZE_U64: usize = STATE_SIZE / 8; // 8 words
const OUTPUT_SIZE: usize = 64;

/// Blake2b initialization vector (IV)
#[rustfmt::skip]
const BLAKE2B_IV: [u64; 8] = [
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
];

/// Blake2b hasher state for variable output lengths.
pub struct Blake2b {
    /// The 8-word (64-byte) Blake2b hash state.
    h: [u64; STATE_SIZE_U64],
    /// Buffer for incomplete blocks.
    buffer: [u8; BLOCK_SIZE],
    /// Number of bytes in the buffer.
    buffer_len: usize,
    /// Total number of bytes processed.
    counter: u64,
}

impl Blake2b {
    /// Creates a new Blake2b hasher with specified output length.
    #[inline(always)]
    pub fn new(output_len: usize) -> Self {
        assert!(output_len > 0 && output_len <= 64, "Invalid output length");

        // Blake2b initialization vector
        let mut h = BLAKE2B_IV;

        // XOR h[0] with parameter block: 0x01010000 ^ (kk << 8) ^ nn
        // where kk=0 (unkeyed) and nn=output_len
        h[0] ^= 0x01010000 ^ (output_len as u64);
        Self {
            h,
            buffer: [0; BLOCK_SIZE],
            buffer_len: 0,
            counter: 0,
        }
    }

    /// Writes data to the hasher.
    #[inline(always)]
    pub fn update(&mut self, input: &[u8]) {
        if input.len() == 0 {
            return;
        }
        // If the buffer was filled to exactly BLOCK_SIZE in a previous update() call,
        // we deferred compression to determine if it was the final block. Since we're
        // receiving more data, we can now safely compress it as a non-final block.
        if self.buffer_len == BLOCK_SIZE {
            self.counter += BLOCK_SIZE as u64;
            compression_caller(&mut self.h, &self.buffer, self.counter, false);
            self.buffer_len = 0;
        }

        // Track the current position in the input data which is not compressed or stored to buffer
        let mut offset = 0;
        // If there is existing data in the buffer, fill it first before processing complete blocks
        if self.buffer_len > 0 {
            let space_available = BLOCK_SIZE - self.buffer_len;
            let bytes_to_copy = space_available.min(input.len());
            self.buffer[self.buffer_len..self.buffer_len + bytes_to_copy]
                .copy_from_slice(&input[..bytes_to_copy]);
            self.buffer_len += bytes_to_copy;
            offset = bytes_to_copy;
        }

        // If the buffer is now full and there is more input data to process,
        // compress the buffer as a non-final block since we know more data follows
        if self.buffer_len == BLOCK_SIZE && offset < input.len() {
            self.counter += BLOCK_SIZE as u64;
            compression_caller(&mut self.h, &self.buffer, self.counter, false);
            self.buffer_len = 0;
        }

        // Process complete blocks directly from the input data for efficiency
        while offset + BLOCK_SIZE < input.len() {
            self.counter += BLOCK_SIZE as u64;
            compression_caller(
                &mut self.h,
                &input[offset..offset + BLOCK_SIZE],
                self.counter,
                false,
            );
            offset += BLOCK_SIZE;
        }

        // Store any remaining bytes in the buffer for processing later
        let remaining = input.len() - offset;
        if remaining > 0 {
            self.buffer[..remaining].copy_from_slice(&input[offset..]);
            self.buffer_len = remaining;
        }
    }

    /// Reads hash digest and consumes the hasher.
    #[inline(always)]
    pub fn finalize(mut self) -> [u8; OUTPUT_SIZE] {
        // Add remaining bytes to counter
        self.counter += self.buffer_len as u64;
        // Pad buffer with zeros
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

    /// Computes Blake2b hash of the input data in one call.
    #[inline(always)]
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
    let mut message = [0u64; BLOCK_SIZE_U64];
    for i in 0..BLOCK_SIZE_U64 {
        message[i] = u64::from_le_bytes(message_block[i * 8..(i + 1) * 8].try_into().unwrap());
    }

    unsafe {
        blake2b_compress(
            hash_state.as_mut_ptr(),
            message.as_ptr(),
            counter,
            is_final as u64, // final
        );
    }
}

impl Default for Blake2b {
    fn default() -> Self {
        Self::new(64)
    }
}

/// Calls the Blake2b compression custom instruction.
///
/// # Arguments
/// * `state` - Pointer to the 8-word (64-byte) Blake2b state
/// * `message` - Pointer to the 16-word (128-byte) message block
/// * `counter` - Counter value (number of bytes processed)
/// * `is_final` - Final block flag (1 if final, 0 otherwise)
///
/// # Safety
/// - `state` must be a valid pointer to 64 bytes of readable and writable memory.
/// - `message` must be a valid pointer to 128 bytes of readable memory.
/// - Both pointers must be properly aligned for u64 access (8-byte alignment).
#[cfg(not(feature = "host"))]
pub unsafe fn blake2b_compress(state: *mut u64, message: *const u64, counter: u64, is_final: u64) {
    // Memory layout for Blake2 instruction:
    // rs1: points to state (64 bytes)
    // rs2: points to message block (128 bytes) + counter (8 bytes) + final flag (8 bytes)

    // We need to set up memory with the message block followed by counter and final flag
    let mut block_data = [0u64; 18]; // 16 words message + 1 word counter + 1 word final

    // Copy message
    core::ptr::copy_nonoverlapping(message, block_data.as_mut_ptr(), 16);
    // Set counter and final flag
    block_data[16] = counter;
    block_data[17] = is_final;

    // Call Blake2 instruction using funct7=0x02 to distinguish from Keccak (0x01) and SHA-256 (0x00)
    core::arch::asm!(
        ".insn r 0x0B, 0x0, 0x02, x0, {}, {}",
        in(reg) state,
        in(reg) block_data.as_ptr(),
        options(nostack)
    );
}

#[cfg(feature = "host")]
pub unsafe fn blake2b_compress(state: *mut u64, message: *const u64, counter: u64, is_final: u64) {
    // On the host, we call our reference implementation from the tracer crate.
    let state_slice = core::slice::from_raw_parts_mut(state, 8);
    let message_slice = core::slice::from_raw_parts(message, 16);
    let message_array: [u64; 16] = message_slice
        .try_into()
        .expect("Message slice was not 16 words");

    tracer::instruction::inline_blake2::execute_blake2b_compression(
        state_slice.try_into().expect("State slice was not 8 words"),
        &message_array,
        counter,
        is_final != 0,
    );
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
        // println!("Input buffer: {:?}", &input_buffer);
        // [42, 179, 60, 197, 78, 215, 96, 233, 114, 251, 132, 13, 150, 31, 168, 49, 186, 67, 204, 85, 222, 103, 240, 121, 2, 139, 20, 157, 38, 175, 56, 193, 74, 211, 92, 229, 110, 247, 128, 9, 146, 27, 164, 45, 182, 63, 200, 81, 218, 99, 236, 117, 254, 135, 16, 153, 34, 171, 52, 189, 70, 207, 88, 225, 106, 243, 124, 5, 142, 23, 160, 41, 178, 59, 196, 77, 214, 95, 232, 113, 250, 131, 12, 149, 30, 167, 48, 185, 66, 203, 84, 221, 102, 239, 120, 1, 138, 19, 156, 37, 174, 55, 192, 73, 210, 91, 228, 109, 246, 127, 8, 145, 26, 163, 44, 181, 62, 199, 80, 217, 98, 235, 116, 253, 134, 15, 152, 33, 170, 51, 188, 69, 206, 87, 224, 105, 242, 123, 4, 141, 22, 159, 40, 177, 58, 195, 76, 213, 94, 231, 112, 249, 130, 11, 148, 29, 166, 47, 184, 65, 202, 83, 220, 101, 238, 119, 0, 137, 18, 155, 36, 173, 54, 191, 72, 209, 90, 227, 108, 245, 126, 7, 144, 25, 162, 43, 180, 61, 198, 79, 216, 97, 234, 115, 252, 133, 14, 151, 32, 169, 50, 187, 68, 205, 86, 223, 104, 241, 122, 3, 140, 21, 158, 39, 176, 57, 194, 75, 212, 93, 230, 111, 248, 129, 10, 147, 28, 165, 46, 183, 64, 201, 82, 219, 100, 237, 118, 255, 136, 17, 154, 35, 172, 53, 190, 71, 208, 89, 226, 107, 244, 125, 6, 143, 24, 161, 42, 179, 60, 197, 78, 215, 96, 233, 114, 251, 132, 13, 150, 31, 168, 49, 186, 67, 204, 85, 222, 103, 240, 121, 2, 139, 20, 157, 38, 175, 56, 193, 74, 211, 92, 229, 110, 247, 128, 9, 146, 27, 164, 45, 182, 63, 200, 81, 218, 99, 236, 117, 254, 135, 16, 153, 34, 171, 52, 189, 70, 207, 88, 225, 106, 243, 124, 5, 142, 23, 160, 41, 178, 59, 196, 77, 214, 95, 232, 113, 250, 131, 12, 149, 30, 167, 48, 185, 66, 203, 84, 221, 102, 239, 120, 1, 138, 19, 156, 37, 174, 55, 192, 73, 210, 91, 228, 109, 246, 127, 8, 145, 26, 163, 44, 181, 62, 199, 80, 217, 98, 235, 116, 253, 134, 15, 152, 33, 170, 51, 188, 69, 206, 87, 224, 105, 242, 123, 4, 141, 22, 159, 40, 177, 58, 195, 76, 213, 94, 231, 112, 249, 130, 11, 148, 29, 166, 47, 184, 65, 202, 83, 220, 101, 238, 119, 0, 137, 18, 155, 36, 173, 54, 191, 72, 209, 90, 227, 108, 245, 126, 7, 144, 25, 162, 43, 180, 61, 198, 79, 216, 97, 234, 115, 252, 133, 14, 151, 32, 169, 50, 187, 68, 205, 86, 223, 104, 241, 122, 3, 140, 21, 158, 39, 176, 57, 194, 75, 212, 93, 230, 111, 248, 129, 10, 147, 28, 165, 46, 183, 64, 201, 82, 219, 100, 237, 118, 255, 136, 17, 154, 35, 172, 53, 190, 71, 208, 89, 226, 107, 244, 125, 6, 143, 24, 161]
        // [42, 179, 60, 197, 78, 215, 96, 233, 114, 251, 132, 13, 150, 31, 168, 49, 186, 67, 204, 85, 222, 103, 240, 121, 2, 139, 20, 157, 38, 175, 56, 193, 74, 211, 92, 229, 110, 247, 128, 9, 146, 27, 164, 45, 182, 63, 200, 81, 218, 99, 236, 117, 254, 135, 16, 153, 34, 171, 52, 189, 70, 207, 88, 225, 106, 243, 124, 5, 142, 23, 160, 41, 178, 59, 196, 77, 214, 95, 232, 113, 250, 131, 12, 149, 30, 167, 48, 185, 66, 203, 84, 221, 102, 239, 120, 1, 138, 19, 156, 37, 174, 55, 192, 73, 210, 91, 228, 109, 246, 127, 8, 145, 26, 163, 44, 181, 62, 199, 80, 217, 98, 235, 116, 253, 134, 15, 152, 33]

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
