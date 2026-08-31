//! Keccak-256 hash function implementation optimized for Jolt zkVM.
//!
//! This module provides an API similar to the `sha3` crate.
//! On the host

use crate::{RATE_IN_BYTES, RATE_IN_U64};

const HASH_LEN: usize = 32;

/// Keccak-256 hasher state.
pub struct Keccak256 {
    /// The 25-word (1600-bit) Keccak state.
    state: [u64; 25],
    /// Buffer for incomplete blocks.
    buffer: [u64; RATE_IN_U64],
    /// Number of bytes in the buffer.
    buffer_len: usize,
    /// Whether at least one block has been permuted into the state.
    initialized: bool,
}

impl Keccak256 {
    /// Creates a new Keccak-256 hasher.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            state: [0; 25],
            buffer: [0; RATE_IN_U64],
            buffer_len: 0,
            initialized: false,
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
                    self.buffer.as_mut_ptr().cast::<u8>().add(self.buffer_len),
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
                    self.buffer.as_mut_ptr().cast(),
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
                    self.buffer.as_mut_ptr().cast(),
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
        unsafe {
            let buffer = self.buffer.as_mut_ptr().cast::<u8>();
            *buffer.add(self.buffer_len) = 0x01;
            if self.buffer_len + 1 < RATE_IN_BYTES {
                core::ptr::write_bytes(
                    buffer.add(self.buffer_len + 1),
                    0,
                    RATE_IN_BYTES - self.buffer_len - 1,
                );
            }
            *buffer.add(RATE_IN_BYTES - 1) |= 0x80;
        }

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

        let is_aligned = (input.as_ptr() as usize).is_multiple_of(8);
        if full_blocks > 0 {
            if is_aligned {
                absorb_aligned_first(&mut state, &input[..RATE_IN_BYTES]);
            } else {
                absorb_unaligned_first(&mut state, &input[..RATE_IN_BYTES]);
            }
            offset = RATE_IN_BYTES;
        }

        if is_aligned {
            for _ in 1..full_blocks {
                absorb_aligned(&mut state, &input[offset..offset + RATE_IN_BYTES]);
                offset += RATE_IN_BYTES;
            }
        } else {
            for _ in 1..full_blocks {
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
        if self.initialized {
            unsafe {
                keccak256_absorb_permute(self.state.as_mut_ptr(), self.buffer.as_ptr().cast());
            }
        } else {
            absorb_words_first(&mut self.state, &self.buffer);
            self.initialized = true;
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
        keccak256_absorb_permute(state.as_mut_ptr(), block.as_ptr());
    }
}

#[inline(always)]
fn absorb_words_first(state: &mut [u64; 25], block: &[u64; RATE_IN_U64]) {
    for (lane, block_lane) in state.iter_mut().zip(block) {
        *lane ^= u64::from_le(*block_lane);
    }
    unsafe {
        keccak_f(state.as_mut_ptr());
    }
}

#[inline(always)]
fn absorb_aligned_first(state: &mut [u64; 25], block: &[u8]) {
    let block = unsafe { &*block.as_ptr().cast::<[u64; RATE_IN_U64]>() };
    absorb_words_first(state, block);
}

#[inline(always)]
fn absorb_unaligned_first(state: &mut [u64; 25], block: &[u8]) {
    let mut aligned = [0u64; RATE_IN_U64];
    unsafe {
        core::ptr::copy_nonoverlapping(block.as_ptr(), aligned.as_mut_ptr().cast(), RATE_IN_BYTES);
    }
    absorb_words_first(state, &aligned);
}

/// Absorb a 136-byte unaligned block into state.
/// Safe for any alignment.
#[inline(always)]
fn absorb_unaligned(state: &mut [u64; 25], block: &[u8]) {
    let mut aligned = [0u64; RATE_IN_U64];
    unsafe {
        core::ptr::copy_nonoverlapping(block.as_ptr(), aligned.as_mut_ptr().cast(), RATE_IN_BYTES);
        keccak256_absorb_permute(state.as_mut_ptr(), aligned.as_ptr().cast());
    }
}

/// Absorb final block with padding directly into state.
#[inline(always)]
fn absorb_final(state: &mut [u64; 25], input: &[u8], len: usize) {
    let mut block = [0u64; RATE_IN_U64];
    unsafe {
        core::ptr::copy_nonoverlapping(input.as_ptr(), block.as_mut_ptr().cast(), len);
        let block_bytes = block.as_mut_ptr().cast::<u8>();
        *block_bytes.add(len) = 0x01;
        *block_bytes.add(RATE_IN_BYTES - 1) |= 0x80;
    }
    absorb_words_first(state, &block);
}

/// Absorbs one Keccak-256 rate block and applies Keccak-f[1600].
///
/// # Safety
/// - `state` must point to 25 writable `u64` words and be 8-byte aligned.
/// - `block` must point to 136 readable bytes and be 8-byte aligned.
/// - The two memory regions must not overlap.
#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
pub unsafe fn keccak256_absorb_permute(state: *mut u64, block: *const u8) {
    use crate::{INLINE_OPCODE, KECCAK256_ABSORB_PERMUTE_FUNCT3, KECCAK256_FUNCT7};
    core::arch::asm!(
        ".insn r {opcode}, {funct3}, {funct7}, x0, {rs1}, {rs2}",
        opcode = const INLINE_OPCODE,
        funct3 = const KECCAK256_ABSORB_PERMUTE_FUNCT3,
        funct7 = const KECCAK256_FUNCT7,
        rs1 = in(reg) state,
        rs2 = in(reg) block,
        options(nostack)
    );
}

#[cfg(feature = "host")]
/// Host reference implementation of [`keccak256_absorb_permute`].
///
/// # Safety
/// - `state` must point to 25 writable `u64` words and be 8-byte aligned.
/// - `block` must point to 136 readable bytes and be 8-byte aligned.
/// - The two memory regions must not overlap.
pub unsafe fn keccak256_absorb_permute(state: *mut u64, block: *const u8) {
    let state = &mut *state.cast::<[u64; 25]>();
    let block = &*block.cast::<[u64; RATE_IN_U64]>();
    for (lane, block_lane) in state.iter_mut().zip(block) {
        *lane ^= u64::from_le(*block_lane);
    }
    crate::exec::execute_keccak_f(state);
}

#[cfg(all(
    not(feature = "host"),
    not(any(target_arch = "riscv32", target_arch = "riscv64"))
))]
pub unsafe fn keccak256_absorb_permute(_state: *mut u64, _block: *const u8) {
    panic!("keccak256_absorb_permute requires RISC-V target or host feature");
}

#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
unsafe fn keccak_f(state: *mut u64) {
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
unsafe fn keccak_f(state: *mut u64) {
    crate::exec::execute_keccak_f(&mut *state.cast());
}

#[cfg(all(
    not(feature = "host"),
    not(any(target_arch = "riscv32", target_arch = "riscv64"))
))]
unsafe fn keccak_f(_state: *mut u64) {
    panic!("keccak_f requires RISC-V target or host feature");
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
