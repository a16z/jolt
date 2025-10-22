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
                    self.buffer.as_mut_ptr().add(self.buffer_len),
                    remaining,
                );
            }
            self.buffer_len += remaining;
        }
    }

    /// Reads hash digest and consumes the hasher.
    #[inline(always)]
    pub fn finalize(mut self) -> [u8; HASH_LEN] {
        // Pad the message. Keccak uses `0x01` padding.
        // If buffer_len == RATE_IN_BYTES-1 both markers land in the same byte (0x01 | 0x80 = 0x81)
        self.buffer[self.buffer_len] = 0x01;

        // Zero the remaining bytes (except the last byte)
        if self.buffer_len + 1 < RATE_IN_BYTES - 1 {
            unsafe {
                core::ptr::write_bytes(
                    self.buffer.as_mut_ptr().add(self.buffer_len + 1),
                    0,
                    RATE_IN_BYTES - self.buffer_len - 2,
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

    /// Computes Keccak-256 hash of the input data in one call.
    #[inline(always)]
    pub fn digest(input: &[u8]) -> [u8; HASH_LEN] {
        let mut hasher = Self::new();
        hasher.update(input);
        hasher.finalize()
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

/// Calls the Keccak-f[1600] permutation custom instruction.
///
/// # Arguments
/// * `state` - Pointer to the 25-word (200-byte) Keccak state, which will be permuted in-place.
///
/// # Safety
/// - `state` must be a valid pointer to 200 bytes of readable and writable memory.
/// - The pointer must be properly aligned for u64 access (8-byte alignment).
#[cfg(not(feature = "host"))]
pub unsafe fn keccak_f(state: *mut u64) {
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
pub unsafe fn keccak_f(state: *mut u64) {
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
}
