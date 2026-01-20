//! High-level Blake2b hashing API for host and guest modes.
use crate::{BLOCK_INPUT_SIZE_IN_BYTES, IV, MSG_BLOCK_LEN, STATE_VECTOR_LEN};
const OUTPUT_SIZE: usize = 64;

pub struct Blake2b {
    /// Hash state (8 x 64-bit words)
    h: [u64; STATE_VECTOR_LEN],
    /// Buffer for incomplete blocks
    buffer: [u8; BLOCK_INPUT_SIZE_IN_BYTES],
    /// Current number of bytes in `buffer`.
    buffer_len: usize,
    /// Total number of bytes processed so far.
    counter: u64,
}

impl Blake2b {
    #[inline(always)]
    pub fn new() -> Self {
        let mut h = IV;
        h[0] ^= 0x01010000 ^ (OUTPUT_SIZE as u64);

        Self {
            h,
            buffer: [0; BLOCK_INPUT_SIZE_IN_BYTES],
            buffer_len: 0,
            counter: 0,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, input: &[u8]) {
        let input_len = input.len();
        if input_len == 0 {
            return;
        }

        let mut offset = 0;

        // Handle partial buffer first
        if self.buffer_len != 0 {
            let needed = BLOCK_INPUT_SIZE_IN_BYTES - self.buffer_len;
            let to_copy = needed.min(input_len);

            unsafe {
                core::ptr::copy_nonoverlapping(
                    input.as_ptr(),
                    self.buffer.as_mut_ptr().add(self.buffer_len),
                    to_copy,
                );
            }

            self.buffer_len += to_copy;
            offset = to_copy;

            // Only process if we have a complete block AND there's more data
            // (to ensure we don't process what might be the final block)
            if self.buffer_len == BLOCK_INPUT_SIZE_IN_BYTES && offset < input_len {
                self.counter += BLOCK_INPUT_SIZE_IN_BYTES as u64;
                compression_caller(&mut self.h, &self.buffer, self.counter, false);
                self.buffer_len = 0;
            }
        }

        // Process complete blocks directly from input
        // We need to keep at least one byte to ensure we don't process what might be the final block
        // This guarantees the final block is always processed in finalize() with is_final=true
        while offset + BLOCK_INPUT_SIZE_IN_BYTES < input_len {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    input.as_ptr().add(offset),
                    self.buffer.as_mut_ptr(),
                    BLOCK_INPUT_SIZE_IN_BYTES,
                );
            }

            self.counter += BLOCK_INPUT_SIZE_IN_BYTES as u64;
            compression_caller(&mut self.h, &self.buffer, self.counter, false);
            offset += BLOCK_INPUT_SIZE_IN_BYTES;
        }

        // Buffer any remaining bytes
        let final_bytes = input_len - offset;
        if final_bytes > 0 {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    input.as_ptr().add(offset),
                    self.buffer.as_mut_ptr().add(self.buffer_len),
                    final_bytes,
                );
            }
            self.buffer_len += final_bytes;
        }
    }

    #[inline(always)]
    pub fn finalize(mut self) -> [u8; OUTPUT_SIZE] {
        self.counter += self.buffer_len as u64;

        // Zero the remaining bytes using optimized pointer write
        if self.buffer_len < BLOCK_INPUT_SIZE_IN_BYTES {
            unsafe {
                core::ptr::write_bytes(
                    self.buffer.as_mut_ptr().add(self.buffer_len),
                    0,
                    BLOCK_INPUT_SIZE_IN_BYTES - self.buffer_len,
                );
            }
        }

        // Process the final block
        compression_caller(&mut self.h, &self.buffer, self.counter, true);

        #[cfg(target_endian = "little")]
        {
            // Safety: [u64; 8] and [u8; 64] have identical size (64 bytes)
            unsafe { core::mem::transmute::<[u64; STATE_VECTOR_LEN], [u8; OUTPUT_SIZE]>(self.h) }
        }

        #[cfg(target_endian = "big")]
        {
            // For big-endian, convert each u64 to little-endian bytes
            let mut hash = [0u8; OUTPUT_SIZE];
            for i in 0..STATE_VECTOR_LEN {
                let bytes = self.h[i].to_le_bytes();
                hash[i * 8..(i + 1) * 8].copy_from_slice(&bytes);
            }
            hash
        }
    }

    /// Computes BLAKE2b hash in one call.
    /// Optimized for virtual cycles by avoiding intermediate buffers for small inputs.
    #[inline(always)]
    pub fn digest(input: &[u8]) -> [u8; OUTPUT_SIZE] {
        let mut h = IV;
        h[0] ^= 0x01010000 ^ (OUTPUT_SIZE as u64);

        let len = input.len();

        // Empty input: direct compression
        if len == 0 {
            compress_direct(&mut h, &[], 0, true);
            return to_bytes(h);
        }

        // Single block (≤128 bytes): direct compression (no intermediate buffer)
        if len <= BLOCK_INPUT_SIZE_IN_BYTES {
            compress_direct(&mut h, input, len as u64, true);
            return to_bytes(h);
        }

        // Large input: process full blocks directly, then final block
        let full_blocks = len / BLOCK_INPUT_SIZE_IN_BYTES;
        let tail_len = len % BLOCK_INPUT_SIZE_IN_BYTES;
        let non_final_blocks = if tail_len == 0 {
            full_blocks - 1
        } else {
            full_blocks
        };

        // Process non-final blocks directly (no copy)
        for i in 0..non_final_blocks {
            let offset = i * BLOCK_INPUT_SIZE_IN_BYTES;
            let block = &input[offset..offset + BLOCK_INPUT_SIZE_IN_BYTES];
            compress(
                &mut h,
                block,
                ((i + 1) * BLOCK_INPUT_SIZE_IN_BYTES) as u64,
                false,
            );
        }

        // Final block
        if tail_len == 0 {
            // Last full block is final
            let offset = (full_blocks - 1) * BLOCK_INPUT_SIZE_IN_BYTES;
            let block = &input[offset..offset + BLOCK_INPUT_SIZE_IN_BYTES];
            compress(&mut h, block, len as u64, true);
        } else {
            // Partial final block: use direct compression
            let tail_offset = full_blocks * BLOCK_INPUT_SIZE_IN_BYTES;
            compress_direct(&mut h, &input[tail_offset..], len as u64, true);
        }

        to_bytes(h)
    }
}

/// Convert hash state to output bytes.
#[inline(always)]
fn to_bytes(h: [u64; STATE_VECTOR_LEN]) -> [u8; OUTPUT_SIZE] {
    #[cfg(target_endian = "little")]
    {
        unsafe { core::mem::transmute(h) }
    }

    #[cfg(target_endian = "big")]
    {
        let mut hash = [0u8; OUTPUT_SIZE];
        for i in 0..STATE_VECTOR_LEN {
            let bytes = h[i].to_le_bytes();
            hash[i * 8..(i + 1) * 8].copy_from_slice(&bytes);
        }
        hash
    }
}

#[inline(always)]
fn compression_caller(
    hash_state: &mut [u64; STATE_VECTOR_LEN],
    message_block: &[u8],
    counter: u64,
    is_final: bool,
) {
    let mut message = [0u64; MSG_BLOCK_LEN + 2];
    debug_assert_eq!(message_block.len(), BLOCK_INPUT_SIZE_IN_BYTES);

    #[cfg(target_endian = "little")]
    unsafe {
        core::ptr::copy_nonoverlapping(
            message_block.as_ptr() as *const u64,
            message.as_mut_ptr(),
            MSG_BLOCK_LEN,
        );
    }

    #[cfg(target_endian = "big")]
    {
        // For big-endian, we need to convert each u64
        for i in 0..MSG_BLOCK_LEN {
            let offset = i * 8;
            message[i] = u64::from_le_bytes([
                message_block[offset],
                message_block[offset + 1],
                message_block[offset + 2],
                message_block[offset + 3],
                message_block[offset + 4],
                message_block[offset + 5],
                message_block[offset + 6],
                message_block[offset + 7],
            ]);
        }
    }

    message[MSG_BLOCK_LEN] = counter;
    message[MSG_BLOCK_LEN + 1] = is_final as u64;

    unsafe {
        blake2b_compress(hash_state.as_mut_ptr(), message.as_ptr());
    }
}

/// Compress a 128-byte block.
#[inline(always)]
fn compress(hash_state: &mut [u64; STATE_VECTOR_LEN], block: &[u8], counter: u64, is_final: bool) {
    let mut message = [0u64; MSG_BLOCK_LEN + 2];

    #[cfg(target_endian = "little")]
    unsafe {
        core::ptr::copy_nonoverlapping(
            block.as_ptr(),
            message.as_mut_ptr() as *mut u8,
            BLOCK_INPUT_SIZE_IN_BYTES,
        );
    }

    #[cfg(target_endian = "big")]
    {
        for i in 0..MSG_BLOCK_LEN {
            let offset = i * 8;
            message[i] = u64::from_le_bytes([
                block[offset],
                block[offset + 1],
                block[offset + 2],
                block[offset + 3],
                block[offset + 4],
                block[offset + 5],
                block[offset + 6],
                block[offset + 7],
            ]);
        }
    }

    message[MSG_BLOCK_LEN] = counter;
    message[MSG_BLOCK_LEN + 1] = is_final as u64;

    unsafe {
        blake2b_compress(hash_state.as_mut_ptr(), message.as_ptr());
    }
}

/// Compress with direct copy to message array (no intermediate buffer).
/// Optimized for virtual cycles by avoiding double-copy for small inputs.
#[inline(always)]
fn compress_direct(
    hash_state: &mut [u64; STATE_VECTOR_LEN],
    input: &[u8],
    counter: u64,
    is_final: bool,
) {
    // Use MaybeUninit to avoid zeroing the full array
    let mut message: core::mem::MaybeUninit<[u64; MSG_BLOCK_LEN + 2]> =
        core::mem::MaybeUninit::uninit();
    let len = input.len();
    let message_ptr = message.as_mut_ptr() as *mut u8;

    #[cfg(target_endian = "little")]
    unsafe {
        // Copy input directly to message
        if len > 0 {
            core::ptr::copy_nonoverlapping(input.as_ptr(), message_ptr, len);
        }
        // Zero only the padding bytes (from input end to block end)
        if len < BLOCK_INPUT_SIZE_IN_BYTES {
            core::ptr::write_bytes(message_ptr.add(len), 0, BLOCK_INPUT_SIZE_IN_BYTES - len);
        }
    }

    #[cfg(target_endian = "big")]
    {
        let message_ref = unsafe { &mut *message.as_mut_ptr() };
        // Zero the message block first for big-endian
        for i in 0..MSG_BLOCK_LEN {
            message_ref[i] = 0;
        }

        // For big-endian, handle partial bytes carefully
        let full_words = len / 8;
        let remaining = len % 8;

        for i in 0..full_words {
            let offset = i * 8;
            message_ref[i] = u64::from_le_bytes([
                input[offset],
                input[offset + 1],
                input[offset + 2],
                input[offset + 3],
                input[offset + 4],
                input[offset + 5],
                input[offset + 6],
                input[offset + 7],
            ]);
        }

        if remaining > 0 {
            let mut bytes = [0u8; 8];
            let offset = full_words * 8;
            for j in 0..remaining {
                bytes[j] = input[offset + j];
            }
            message_ref[full_words] = u64::from_le_bytes(bytes);
        }
    }

    // Set counter and is_final, then compress
    unsafe {
        let message_ref = &mut *message.as_mut_ptr();
        message_ref[MSG_BLOCK_LEN] = counter;
        message_ref[MSG_BLOCK_LEN + 1] = is_final as u64;
        blake2b_compress(hash_state.as_mut_ptr(), message_ref.as_ptr());
    }
}

impl Default for Blake2b {
    fn default() -> Self {
        Self::new()
    }
}

/// BLAKE2b compression function - guest implementation.
///
/// # Safety
/// - `state` must point to a valid array of 8 u64 values
/// - `message` must point to a valid array of 18 u64 values (16 message + counter + final flag)
/// - Both pointers must be properly aligned for u64 access
#[cfg(not(feature = "host"))]
pub(crate) unsafe fn blake2b_compress(state: *mut u64, message: *const u64) {
    use crate::{BLAKE2_FUNCT3, BLAKE2_FUNCT7, INLINE_OPCODE};
    // Memory layout for Blake2 instruction:
    // rs1: points to state (64 bytes)
    // rs2: points to message block (128 bytes) + counter (8 bytes) + final flag (8 bytes)

    core::arch::asm!(
        ".insn r {opcode}, {funct3}, {funct7}, x0, {rs1}, {rs2}",
        opcode = const INLINE_OPCODE,
        funct3 = const BLAKE2_FUNCT3,
        funct7 = const BLAKE2_FUNCT7,
        rs1 = in(reg) state,
        rs2 = in(reg) message,
        options(nostack)
    );
}

/// BLAKE2b compression function - host implementation.
///
/// # Safety  
/// - `state` must point to a valid array of 8 u64 values
/// - `message` must point to a valid array of 18 u64 values
#[cfg(feature = "host")]
pub(crate) unsafe fn blake2b_compress(state: *mut u64, message: *const u64) {
    let state_slice = core::slice::from_raw_parts_mut(state, 8);
    let message_slice = core::slice::from_raw_parts(message, 18);

    // Convert to arrays for type safety
    let state_array: &mut [u64; 8] = state_slice
        .try_into()
        .expect("State pointer must reference exactly 8 u64 values");
    let message_array: [u64; 18] = message_slice
        .try_into()
        .expect("Message pointer must reference exactly 18 u64 values");

    crate::exec::execute_blake2b_compression(state_array, &message_array);
}

#[cfg(all(test, feature = "host"))]
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
            for (i, item) in input.iter_mut().enumerate() {
                *item = pattern_fn(i);
            }
            for length in 0..=1200 {
                use blake2::Digest as RefDigest;
                assert_eq!(
                    Blake2b::digest(&input),
                    Into::<[u8; 64]>::into(blake2::Blake2b512::digest(input)),
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

#[cfg(all(test, feature = "host"))]
mod streaming_tests {
    use super::*;
    use hex_literal::hex;

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
            let mut hasher = Blake2b::new();
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
            for (i, item) in input.iter_mut().enumerate() {
                *item = pattern_fn(i);
            }

            for length in 0..=1200 {
                let test_input = &input[..length];
                let mut hasher = Blake2b::new();
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
            let mut hasher = Blake2b::new();
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
                let mut hasher = Blake2b::new();
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
                let mut hasher = Blake2b::new();
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

        const TOTAL_LEN: usize = 77;
        let mut full_data = [0u8; TOTAL_LEN];
        let mut offset = 0;

        for part in data_parts {
            full_data[offset..offset + part.len()].copy_from_slice(part);
            offset += part.len();
        }

        let mut hasher = Blake2b::new();
        for part in data_parts {
            hasher.update(part);
        }
        assert_eq!(
            hasher.finalize(),
            Into::<[u8; 64]>::into(blake2::Blake2b512::digest(full_data)),
            "Multiple updates should produce same result as single update"
        );
    }

    #[test]
    fn test_blake2b_streaming_empty_updates() {
        use blake2::Digest as RefDigest;

        let test_data = b"Some test data for empty update testing";

        let mut hasher = Blake2b::new();
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

    #[test]
    fn test_blake2b_aligned_vs_unaligned() {
        // Test various sizes to cover different code paths
        let test_sizes = [
            0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128, 129, 256, 512, 1024, 2048,
        ];

        for &size in &test_sizes {
            // Create aligned buffer (array is naturally aligned)
            let aligned: Vec<u8> = (0..size).map(|i| (i * 37 + 11) as u8).collect();

            // Create unaligned buffer by adding 1-byte offset
            let mut unaligned_buf = vec![0u8; size + 1];
            unaligned_buf[1..].copy_from_slice(&aligned);
            let unaligned = &unaligned_buf[1..];

            // Verify alignment difference
            if size > 0 {
                assert_ne!(
                    aligned.as_ptr() as usize % 8,
                    unaligned.as_ptr() as usize % 8,
                    "Test setup error: pointers should have different alignment"
                );
            }

            // Both should produce identical results
            let aligned_result = Blake2b::digest(&aligned);
            let unaligned_result = Blake2b::digest(unaligned);

            assert_eq!(
                aligned_result, unaligned_result,
                "Blake2b: aligned vs unaligned mismatch at size {size}"
            );

            // Also verify against reference implementation
            use blake2::Digest as RefDigest;
            let expected: [u8; 64] = blake2::Blake2b512::digest(&aligned).into();
            assert_eq!(
                aligned_result, expected,
                "Blake2b: result doesn't match reference at size {size}"
            );
        }
    }
}
