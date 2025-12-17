//! Keccak-256 hash function implementation optimized for Jolt zkVM.
//!
//! This module provides an API similar to the `sha3` crate.
//! On the host

const RATE_IN_BYTES: usize = 136;
const RATE_IN_U64: usize = RATE_IN_BYTES / 8;
const HASH_LEN: usize = 32;

/// Keccak-256 hasher state.
pub struct Keccak256 {
    /// The 25-word (1600-bit) Keccak state.
    state: [u64; 25],
    /// Buffer for incomplete blocks.
    buffer: [u8; RATE_IN_BYTES],
    /// Number of bytes in the buffer.
    buffer_len: usize,
}

impl Keccak256 {
    /// Creates a new Keccak-256 hasher.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            state: [0; 25],
            buffer: [0; RATE_IN_BYTES],
            buffer_len: 0,
        }
    }

    /// Writes data to the hasher.
    #[inline(always)]
    pub fn update(&mut self, input: &[u8]) {
        if input.is_empty() {
            return;
        }

        let mut offset = 0;

        // Absorb input into the buffer
        if self.buffer_len > 0 {
            let needed = RATE_IN_BYTES - self.buffer_len;
            let to_copy = needed.min(input.len());

            unsafe {
                core::ptr::copy_nonoverlapping(
                    input.as_ptr(),
                    self.buffer.as_mut_ptr().add(self.buffer_len),
                    to_copy,
                );
            }

            self.buffer_len += to_copy;
            offset += to_copy;

            if self.buffer_len == RATE_IN_BYTES {
                self.absorb_buffer();
            }
        }

        // Process complete blocks
        while offset + RATE_IN_BYTES <= input.len() {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    input.as_ptr().add(offset),
                    self.buffer.as_mut_ptr(),
                    RATE_IN_BYTES,
                );
            }
            self.buffer_len = RATE_IN_BYTES;
            self.absorb_buffer();
            offset += RATE_IN_BYTES;
        }

        // Buffer any remaining input
        let remaining = input.len() - offset;
        if remaining > 0 {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    input.as_ptr().add(offset),
                    self.buffer.as_mut_ptr(),
                    remaining,
                );
            }
            self.buffer_len = remaining;
        }
    }

    /// Reads hash digest and consumes the hasher.
    #[inline(always)]
    pub fn finalize(mut self) -> [u8; HASH_LEN] {
        // Pad the message. Keccak uses `0x01` padding.
        // If buffer_len == RATE_IN_BYTES-1 both markers land in the same byte (0x01 | 0x80 = 0x81)
        self.buffer[self.buffer_len] = 0x01;

        // Zero the remaining bytes (including the last byte if needed)
        if self.buffer_len + 1 < RATE_IN_BYTES {
            unsafe {
                core::ptr::write_bytes(
                    self.buffer.as_mut_ptr().add(self.buffer_len + 1),
                    0,
                    RATE_IN_BYTES - self.buffer_len - 1,
                );
            }
        }
        self.buffer[RATE_IN_BYTES - 1] |= 0x80;

        self.absorb_buffer();

        let mut hash = [0u8; HASH_LEN];

        #[cfg(target_endian = "little")]
        {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    self.state.as_ptr() as *const u8,
                    hash.as_mut_ptr(),
                    HASH_LEN,
                );
            }
        }

        #[cfg(target_endian = "big")]
        {
            // For big-endian, convert each u64 to little-endian bytes
            for i in 0..HASH_LEN / 8 {
                let bytes = self.state[i].to_le_bytes();
                hash[i * 8..(i + 1) * 8].copy_from_slice(&bytes);
            }
        }

        hash
    }

    /// Computes Keccak-256 hash in one call.
    /// Optimized for virtual cycles by avoiding intermediate buffer for final block.
    #[inline(always)]
    pub fn digest(input: &[u8]) -> [u8; HASH_LEN] {
        let len = input.len();
        let mut state = [0u64; 25];

        // Process complete 136-byte blocks
        let full_blocks = len / RATE_IN_BYTES;
        let mut offset = 0;

        // Check alignment once, then use branch-free loop
        let is_aligned = input.as_ptr() as usize % 8 == 0;

        if is_aligned {
            // Aligned fast path - no per-block branch
            for _ in 0..full_blocks {
                absorb_aligned(&mut state, &input[offset..offset + RATE_IN_BYTES]);
                offset += RATE_IN_BYTES;
            }
        } else {
            // Unaligned path - no per-block branch
            for _ in 0..full_blocks {
                absorb_unaligned(&mut state, &input[offset..offset + RATE_IN_BYTES]);
                offset += RATE_IN_BYTES;
            }
        }

        // Final block with Keccak padding - use direct absorb
        let remaining = len - offset;
        absorb_final(&mut state, &input[offset..], remaining);
        to_bytes(state)
    }

    /// Absorbs a full block from the internal buffer into the state.
    #[inline(always)]
    fn absorb_buffer(&mut self) {
        #[cfg(target_endian = "little")]
        unsafe {
            // On little-endian, directly XOR the buffer as u64 words
            let buffer_words = self.buffer.as_ptr() as *const u64;
            for i in 0..RATE_IN_U64 {
                self.state[i] ^= *buffer_words.add(i);
            }
        }

        #[cfg(target_endian = "big")]
        {
            // For big-endian, convert each word from little-endian bytes
            for i in 0..RATE_IN_U64 {
                let word = u64::from_le_bytes(self.buffer[i * 8..(i + 1) * 8].try_into().unwrap());
                self.state[i] ^= word;
            }
        }

        unsafe {
            keccak_f(self.state.as_mut_ptr());
        }
        self.buffer_len = 0;
    }
}

impl Default for Keccak256 {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert state to output hash bytes.
#[inline(always)]
fn to_bytes(state: [u64; 25]) -> [u8; HASH_LEN] {
    let mut hash = [0u8; HASH_LEN];

    #[cfg(target_endian = "little")]
    {
        unsafe {
            core::ptr::copy_nonoverlapping(
                state.as_ptr() as *const u8,
                hash.as_mut_ptr(),
                HASH_LEN,
            );
        }
    }

    #[cfg(target_endian = "big")]
    {
        for i in 0..HASH_LEN / 8 {
            let bytes = state[i].to_le_bytes();
            hash[i * 8..(i + 1) * 8].copy_from_slice(&bytes);
        }
    }

    hash
}

/// Absorb a 136-byte aligned block into state.
/// Caller must ensure the block pointer is 8-byte aligned.
#[inline(always)]
fn absorb_aligned(state: &mut [u64; 25], block: &[u8]) {
    unsafe {
        let block_words = block.as_ptr() as *const u64;
        for (i, s) in state.iter_mut().enumerate().take(RATE_IN_U64) {
            *s ^= *block_words.add(i);
        }
        keccak_f(state.as_mut_ptr());
    }
}

/// Absorb a 136-byte unaligned block into state.
/// Safe for any alignment.
#[inline(always)]
fn absorb_unaligned(state: &mut [u64; 25], block: &[u8]) {
    let ptr = block.as_ptr();
    for (i, s) in state.iter_mut().enumerate().take(RATE_IN_U64) {
        let word = unsafe {
            let mut tmp = core::mem::MaybeUninit::<[u8; 8]>::uninit();
            core::ptr::copy_nonoverlapping(ptr.add(i * 8), tmp.as_mut_ptr() as *mut u8, 8);
            u64::from_le_bytes(tmp.assume_init())
        };
        *s ^= word;
    }
    unsafe {
        keccak_f(state.as_mut_ptr());
    }
}

/// Absorb final block with padding directly into state.
#[inline(always)]
fn absorb_final(state: &mut [u64; 25], input: &[u8], len: usize) {
    // Build padded block and XOR into state
    let mut block = [0u8; RATE_IN_BYTES];

    if len > 0 {
        unsafe {
            core::ptr::copy_nonoverlapping(input.as_ptr(), block.as_mut_ptr(), len);
        }
    }

    // Keccak padding: 0x01 at end of data, 0x80 at end of block
    block[len] = 0x01;
    block[RATE_IN_BYTES - 1] |= 0x80;

    // XOR padded block into state
    #[cfg(target_endian = "little")]
    unsafe {
        let block_words = block.as_ptr() as *const u64;
        #[allow(clippy::needless_range_loop)]
        for i in 0..RATE_IN_U64 {
            state[i] ^= *block_words.add(i);
        }
    }

    #[cfg(target_endian = "big")]
    {
        for i in 0..RATE_IN_U64 {
            let offset = i * 8;
            let word = u64::from_le_bytes([
                block[offset],
                block[offset + 1],
                block[offset + 2],
                block[offset + 3],
                block[offset + 4],
                block[offset + 5],
                block[offset + 6],
                block[offset + 7],
            ]);
            state[i] ^= word;
        }
    }

    unsafe {
        keccak_f(state.as_mut_ptr());
    }
}

/// Calls the Keccak-f[1600] permutation custom instruction.
///
/// # Arguments
/// * `state` - Pointer to the 25-word (200-byte) Keccak state, which will be permuted in-place.
///
/// # Safety
/// - `state` must be a valid pointer to 200 bytes of readable and writable memory.
/// - The pointer must be properly aligned for u64 access (8-byte alignment).
#[cfg(not(feature = "host"))]
pub(crate) unsafe fn keccak_f(state: *mut u64) {
    use crate::{INLINE_OPCODE, KECCAK256_FUNCT3, KECCAK256_FUNCT7};
    core::arch::asm!(
        ".insn r {opcode}, {funct3}, {funct7}, x0, {rs1}, x0",
        opcode = const INLINE_OPCODE,
        funct3 = const KECCAK256_FUNCT3,
        funct7 = const KECCAK256_FUNCT7,
        rs1 = in(reg) state,
        options(nostack)
    );
}

#[cfg(feature = "host")]
/// Calls the Keccak-f[1600] permutation reference implementation when running on
/// the host where the custom RISC-V instruction is not available.
///
/// # Safety
/// * `state` must point to 25 contiguous `u64` words (exactly 200 bytes) that are
///   writable for the duration of the call.
/// * The pointer must be non-null and 8-byte aligned.
/// * The memory referenced by `state` will be permuted **in-place**; callers must
///   ensure this side-effect is acceptable.
/// * Passing an invalid pointer, misaligned pointer, or insufficiently sized
///   memory region results in undefined behaviour.
pub(crate) unsafe fn keccak_f(state: *mut u64) {
    // On the host, we call our own reference implementation from the tracer crate.
    let state_slice = core::slice::from_raw_parts_mut(state, 25);
    crate::exec::execute_keccak_f(
        state_slice
            .try_into()
            .expect("State slice was not 25 words"),
    );
}

#[cfg(all(test, feature = "host"))]
mod tests {
    use super::*;
    use hex_literal::hex;

    #[test]
    fn test_keccak256_empty() {
        let hash = Keccak256::digest(b"");
        assert_eq!(
            hash,
            hex!("c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470")
        );
    }

    #[test]
    fn test_keccak256_aligned_vs_unaligned() {
        // Test various sizes including rate boundary (136 bytes)
        let test_sizes = [
            0, 1, 7, 8, 31, 32, 63, 64, 135, 136, 137, 200, 272, 512, 1024, 2048,
        ];

        for &size in &test_sizes {
            // Create aligned buffer
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
            let aligned_result = Keccak256::digest(&aligned);
            let unaligned_result = Keccak256::digest(unaligned);

            assert_eq!(
                aligned_result, unaligned_result,
                "Keccak256: aligned vs unaligned mismatch at size {size}"
            );

            // Also verify against reference implementation
            use sha3::{Digest, Keccak256 as RefKeccak};
            let expected: [u8; 32] = RefKeccak::digest(&aligned).into();
            assert_eq!(
                aligned_result, expected,
                "Keccak256: result doesn't match reference at size {size}"
            );
        }
    }
}
