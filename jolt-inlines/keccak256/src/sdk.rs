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
    /// Number of bytes in the buffer, always below `RATE_IN_BYTES`: `update`
    /// absorbs the buffer the moment it fills.
    buffer_len: usize,
}

impl Keccak256 {
    /// Creates a new Keccak-256 hasher.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            state: [0; 25],
            buffer: [0; RATE_IN_U64],
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

        if self.buffer_len > 0 {
            let buffer_len = self.buffer_len;
            let to_copy = (RATE_IN_BYTES - buffer_len).min(input.len());
            self.buffer_bytes()[buffer_len..buffer_len + to_copy]
                .copy_from_slice(&input[..to_copy]);
            self.buffer_len += to_copy;
            offset += to_copy;

            if self.buffer_len == RATE_IN_BYTES {
                self.absorb_buffer();
            }
        }

        // Complete blocks are absorbed straight from `input`; only the tail
        // is staged in the buffer.
        let remaining = absorb_full_blocks(&mut self.state, &input[offset..]);
        if !remaining.is_empty() {
            self.buffer_bytes()[..remaining.len()].copy_from_slice(remaining);
            self.buffer_len = remaining.len();
        }
    }

    /// Reads hash digest and consumes the hasher.
    #[inline(always)]
    pub fn finalize(mut self) -> [u8; HASH_LEN] {
        // Keccak padding is `0x01 .. 0x80`; both markers share a byte when
        // `buffer_len == RATE_IN_BYTES - 1` (0x81).
        let buffer_len = self.buffer_len;
        let bytes = self.buffer_bytes();
        bytes[buffer_len] = 0x01;
        bytes[buffer_len + 1..].fill(0);
        bytes[RATE_IN_BYTES - 1] |= 0x80;

        self.absorb_buffer();
        to_bytes(self.state)
    }

    /// Computes Keccak-256 hash in one call.
    /// Optimized for virtual cycles by avoiding intermediate buffer for final block.
    #[inline(always)]
    pub fn digest(input: &[u8]) -> [u8; HASH_LEN] {
        let mut state = [0u64; 25];
        let remaining = absorb_full_blocks(&mut state, input);
        absorb_final(&mut state, remaining);
        to_bytes(state)
    }

    #[inline(always)]
    fn buffer_bytes(&mut self) -> &mut [u8; RATE_IN_BYTES] {
        // SAFETY: `[u64; RATE_IN_U64]` is exactly `RATE_IN_BYTES` initialized
        // bytes and `u8` has no alignment requirement.
        unsafe { &mut *self.buffer.as_mut_ptr().cast() }
    }

    /// Absorbs the full block held in `buffer` into the state.
    #[inline(always)]
    fn absorb_buffer(&mut self) {
        // SAFETY: both arrays are 8-byte aligned fields of `self`, so they
        // are distinct and correctly sized for the inline's contract.
        unsafe {
            keccak256_absorb_permute(self.state.as_mut_ptr(), self.buffer.as_ptr().cast());
        }
        self.buffer_len = 0;
    }
}

impl Default for Keccak256 {
    fn default() -> Self {
        Self::new()
    }
}

/// The first `HASH_LEN` bytes of the state, lanes serialized little-endian.
#[inline(always)]
fn to_bytes(state: [u64; 25]) -> [u8; HASH_LEN] {
    let mut hash = [0u8; HASH_LEN];
    for (out, lane) in hash.chunks_exact_mut(size_of::<u64>()).zip(state) {
        out.copy_from_slice(&lane.to_le_bytes());
    }
    hash
}

/// Absorbs every complete rate block of `input` and returns the unabsorbed
/// tail (shorter than `RATE_IN_BYTES`).
///
/// 8-byte-aligned input feeds the fused inline directly from the caller's
/// memory; unaligned input is staged through an aligned stack copy per block.
/// The block stride is a multiple of 8, so alignment is decided once.
#[inline(always)]
fn absorb_full_blocks<'a>(state: &mut [u64; 25], input: &'a [u8]) -> &'a [u8] {
    let mut blocks = input.chunks_exact(RATE_IN_BYTES);
    if (input.as_ptr() as usize).is_multiple_of(8) {
        for block in &mut blocks {
            // SAFETY: `block` is `RATE_IN_BYTES` readable bytes at an 8-byte
            // aligned address; `state` is a distinct `[u64; 25]`.
            unsafe { keccak256_absorb_permute(state.as_mut_ptr(), block.as_ptr()) };
        }
    } else {
        for block in &mut blocks {
            let mut aligned = [0u64; RATE_IN_U64];
            // SAFETY: `block` and `aligned` are both `RATE_IN_BYTES` long, the
            // copy target is a fresh local, and `aligned` is 8-byte aligned.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    block.as_ptr(),
                    aligned.as_mut_ptr().cast(),
                    RATE_IN_BYTES,
                );
                keccak256_absorb_permute(state.as_mut_ptr(), aligned.as_ptr().cast());
            }
        }
    }
    blocks.remainder()
}

/// Pads the final partial block (`input.len() < RATE_IN_BYTES`) and absorbs it.
#[inline(always)]
fn absorb_final(state: &mut [u64; 25], input: &[u8]) {
    let mut block = [0u64; RATE_IN_U64];
    // SAFETY: `[u64; RATE_IN_U64]` is exactly `RATE_IN_BYTES` initialized
    // bytes and `u8` has no alignment requirement.
    let bytes: &mut [u8; RATE_IN_BYTES] = unsafe { &mut *block.as_mut_ptr().cast() };
    bytes[..input.len()].copy_from_slice(input);
    bytes[input.len()] = 0x01;
    bytes[RATE_IN_BYTES - 1] |= 0x80;
    // SAFETY: `block` is a fresh 8-byte aligned local distinct from `state`.
    unsafe { keccak256_absorb_permute(state.as_mut_ptr(), block.as_ptr().cast()) };
}

/// Absorbs one Keccak-256 rate block and applies Keccak-f[1600].
///
/// On RISC-V guests this is the fused inline; with the `host` feature it is
/// the reference implementation.
///
/// # Safety
/// - `state` must point to 25 writable `u64` words and be 8-byte aligned.
/// - `block` must point to 136 readable bytes and be 8-byte aligned.
/// - The two memory regions must not overlap.
pub unsafe fn keccak256_absorb_permute(state: *mut u64, block: *const u8) {
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    {
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
    {
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
    {
        let _ = (state, block);
        panic!("keccak256_absorb_permute requires RISC-V target or host feature");
    }
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
