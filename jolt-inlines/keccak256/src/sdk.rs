//! Keccak-256 hash function implementation optimized for Jolt zkVM.
//!
//! This module provides an API similar to the `sha3` crate.
//! On the host

use core::mem::MaybeUninit;

#[cfg(feature = "host")]
use crate::exec::{execute_absorb_permute, execute_init_absorb_permute};
use crate::{Keccak256State, RATE_IN_BYTES, RATE_IN_U64};
#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
use crate::{
    INLINE_OPCODE, KECCAK256_ABSORB_PERMUTE_FUNCT3, KECCAK256_ABSORB_PERMUTE_UNALIGNED_FUNCT3,
    KECCAK256_FUNCT7, KECCAK256_INIT_ABSORB_PERMUTE_FUNCT3,
    KECCAK256_INIT_ABSORB_PERMUTE_UNALIGNED_FUNCT3,
};

const HASH_LEN: usize = 32;

/// Keccak-256 hasher state.
pub struct Keccak256 {
    /// The 25-word (1600-bit) Keccak state.
    state: Keccak256State,
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
        // The buffer above `buffer_len` still holds bytes of earlier blocks:
        // clear the dead bytes of the partial word (the mask is empty when
        // the length is a whole number of words) and every word after it.
        let len = self.buffer_len;
        let word = len / 8;
        self.buffer[word] &= (1u64 << (8 * (len % 8))) - 1;
        self.buffer[word + 1..].fill(0);
        pad(&mut self.buffer, len);

        self.absorb_buffer();
        to_bytes(self.state)
    }

    /// Computes Keccak-256 hash in one call.
    ///
    /// The first block is absorbed into the zero state by the INIT inline, so
    /// no state is ever zeroed in memory; the final block is padded word-wise.
    #[inline(always)]
    pub fn digest(input: &[u8]) -> [u8; HASH_LEN] {
        let mut state = MaybeUninit::<Keccak256State>::uninit();
        let state_ptr = state.as_mut_ptr().cast::<u64>();
        if input.len() < RATE_IN_BYTES {
            let block = padded_block(input);
            // SAFETY: `block` is a fresh 8-byte aligned local and `state` 25
            // writable words the inline fills; the regions are distinct.
            unsafe { keccak256_init_absorb_permute(state_ptr, block.as_ptr().cast()) };
        } else {
            let (first, rest) = input.split_at(RATE_IN_BYTES);
            // SAFETY: `first` is `RATE_IN_BYTES` readable bytes, 8-byte
            // aligned on the aligned path; `state` is 25 writable words the
            // inline fills.
            unsafe {
                if is_aligned(first) {
                    keccak256_init_absorb_permute(state_ptr, first.as_ptr());
                } else {
                    keccak256_init_absorb_permute_unaligned(state_ptr, first.as_ptr());
                }
            }
            // SAFETY: the INIT inline wrote all 25 lanes.
            let state = unsafe { state.assume_init_mut() };
            let block = padded_block(absorb_full_blocks(state, rest));
            // SAFETY: `block` is a fresh 8-byte aligned local distinct from `state`.
            unsafe { keccak256_absorb_permute(state.as_mut_ptr(), block.as_ptr().cast()) };
        }
        // SAFETY: both paths start with an INIT inline, which wrote all 25 lanes.
        to_bytes(unsafe { state.assume_init() })
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
fn to_bytes(state: Keccak256State) -> [u8; HASH_LEN] {
    let mut hash = [0u8; HASH_LEN];
    for (out, lane) in hash.chunks_exact_mut(size_of::<u64>()).zip(state) {
        out.copy_from_slice(&lane.to_le_bytes());
    }
    hash
}

/// Whether `bytes` starts on an 8-byte boundary.
#[inline(always)]
fn is_aligned(bytes: &[u8]) -> bool {
    bytes.as_ptr().cast::<u64>().is_aligned()
}

/// Absorbs every complete rate block of `input` and returns the unabsorbed
/// tail (shorter than `RATE_IN_BYTES`).
///
/// Every block is fed to the inline straight from the caller's memory. The
/// block stride is a multiple of 8, so one alignment check selects the
/// aligned or the unaligned inline for all of them.
#[inline(always)]
fn absorb_full_blocks<'a>(state: &mut Keccak256State, input: &'a [u8]) -> &'a [u8] {
    let mut blocks = input.chunks_exact(RATE_IN_BYTES);
    if is_aligned(input) {
        for block in &mut blocks {
            // SAFETY: `block` is `RATE_IN_BYTES` readable bytes at an 8-byte
            // aligned address; `state` is a distinct `Keccak256State`.
            unsafe { keccak256_absorb_permute(state.as_mut_ptr(), block.as_ptr()) };
        }
    } else {
        for block in &mut blocks {
            // SAFETY: `block` is `RATE_IN_BYTES` readable bytes; `state` is a
            // distinct `Keccak256State`.
            unsafe { keccak256_absorb_permute_unaligned(state.as_mut_ptr(), block.as_ptr()) };
        }
    }
    blocks.remainder()
}

/// The final rate block: `tail` (shorter than `RATE_IN_BYTES`) followed by
/// the Keccak padding, built word-wise so the only sub-word memory traffic
/// is the platform `memcpy` of the tail.
#[inline(always)]
fn padded_block(tail: &[u8]) -> [u64; RATE_IN_U64] {
    debug_assert!(tail.len() < RATE_IN_BYTES);
    let mut block = [0u64; RATE_IN_U64];
    // SAFETY: `tail` is shorter than the `RATE_IN_BYTES`-byte fresh local
    // `block`, and `u8` has no alignment requirement.
    unsafe {
        core::ptr::copy_nonoverlapping(tail.as_ptr(), block.as_mut_ptr().cast::<u8>(), tail.len());
    }
    pad(&mut block, tail.len());
    block
}

/// Keccak padding `0x01 .. 0x80` after `len < RATE_IN_BYTES` message bytes,
/// word-wise; both markers share byte 135 when `len == RATE_IN_BYTES - 1`
/// (0x81).
#[inline(always)]
fn pad(block: &mut [u64; RATE_IN_U64], len: usize) {
    block[len / 8] |= 0x01 << (8 * (len % 8));
    block[RATE_IN_U64 - 1] |= 1 << 63;
}

/// The KECCAK256 inline selected by `FUNCT3`: `rs1 = state`, `rs2 = block`.
#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
#[inline(always)]
unsafe fn keccak256_insn<const FUNCT3: u32>(state: *mut u64, block: *const u8) {
    core::arch::asm!(
        ".insn r {opcode}, {funct3}, {funct7}, x0, {rs1}, {rs2}",
        opcode = const INLINE_OPCODE,
        funct3 = const FUNCT3,
        funct7 = const KECCAK256_FUNCT7,
        rs1 = in(reg) state,
        rs2 = in(reg) block,
        options(nostack)
    );
}

/// The 17 block words at an 8-byte aligned `block`.
#[cfg(feature = "host")]
unsafe fn read_block(block: *const u8) -> [u64; RATE_IN_U64] {
    (*block.cast::<[u64; RATE_IN_U64]>()).map(u64::from_le)
}

/// The 17 little-endian block words at any alignment. The inline reads the
/// doublewords containing the block instead; the model touches only its bytes.
#[cfg(feature = "host")]
unsafe fn read_block_unaligned(block: *const u8) -> [u64; RATE_IN_U64] {
    core::array::from_fn(|i| u64::from_le(block.cast::<u64>().add(i).read_unaligned()))
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
    keccak256_insn::<KECCAK256_ABSORB_PERMUTE_FUNCT3>(state, block);
    #[cfg(feature = "host")]
    execute_absorb_permute(&mut *state.cast::<Keccak256State>(), &read_block(block));
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    {
        let _ = (state, block);
        panic!("keccak256_absorb_permute requires RISC-V target or host feature");
    }
}

/// Absorbs one Keccak-256 rate block into the zero state and applies
/// Keccak-f[1600]; the prior contents of `state` are ignored.
///
/// On RISC-V guests this is the fused inline; with the `host` feature it is
/// the reference implementation.
///
/// # Safety
/// - `state` must point to 25 writable `u64` words and be 8-byte aligned.
/// - `block` must point to 136 readable bytes and be 8-byte aligned.
/// - The two memory regions must not overlap.
pub unsafe fn keccak256_init_absorb_permute(state: *mut u64, block: *const u8) {
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    keccak256_insn::<KECCAK256_INIT_ABSORB_PERMUTE_FUNCT3>(state, block);
    #[cfg(feature = "host")]
    state
        .cast::<Keccak256State>()
        .write(execute_init_absorb_permute(&read_block(block)));
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    {
        let _ = (state, block);
        panic!("keccak256_init_absorb_permute requires RISC-V target or host feature");
    }
}

/// [`keccak256_absorb_permute`] for a block at any alignment.
///
/// # Safety
/// - `state` must point to 25 writable `u64` words and be 8-byte aligned.
/// - `block` must point to 136 readable bytes. On the guest the inline reads
///   the 18 aligned doublewords containing them, which for a misaligned
///   `block` never extend past the block; an aligned `block` also reads the
///   doubleword after it.
/// - The two memory regions must not overlap.
pub unsafe fn keccak256_absorb_permute_unaligned(state: *mut u64, block: *const u8) {
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    keccak256_insn::<KECCAK256_ABSORB_PERMUTE_UNALIGNED_FUNCT3>(state, block);
    #[cfg(feature = "host")]
    execute_absorb_permute(
        &mut *state.cast::<Keccak256State>(),
        &read_block_unaligned(block),
    );
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    {
        let _ = (state, block);
        panic!("keccak256_absorb_permute_unaligned requires RISC-V target or host feature");
    }
}

/// [`keccak256_init_absorb_permute`] for a block at any alignment, with the
/// memory contract of [`keccak256_absorb_permute_unaligned`].
///
/// # Safety
/// - `state` must point to 25 writable `u64` words and be 8-byte aligned.
/// - `block` must point to 136 readable bytes; see
///   [`keccak256_absorb_permute_unaligned`] for the doublewords the guest reads.
/// - The two memory regions must not overlap.
pub unsafe fn keccak256_init_absorb_permute_unaligned(state: *mut u64, block: *const u8) {
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    keccak256_insn::<KECCAK256_INIT_ABSORB_PERMUTE_UNALIGNED_FUNCT3>(state, block);
    #[cfg(feature = "host")]
    state
        .cast::<Keccak256State>()
        .write(execute_init_absorb_permute(&read_block_unaligned(block)));
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    {
        let _ = (state, block);
        panic!("keccak256_init_absorb_permute_unaligned requires RISC-V target or host feature");
    }
}

#[cfg(all(test, feature = "host"))]
mod tests {
    use super::*;
    use hex_literal::hex;
    use sha3::{Digest, Keccak256 as RefKeccak};

    #[test]
    fn test_keccak256_empty() {
        let hash = Keccak256::digest(b"");
        assert_eq!(
            hash,
            hex!("c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470")
        );
    }

    const MAX_LEN: usize = 8192;

    #[repr(align(8))]
    struct Aligned([u8; MAX_LEN + 8]);

    /// Every pointer offset modulo 8 for every length up to several blocks,
    /// through `digest` and through `update` in several chunkings, so every
    /// alignment dispatch and both padding paths run against the reference.
    #[test]
    fn test_keccak256_every_offset_and_length() {
        let mut backing = Aligned([0; MAX_LEN + 8]);
        for (i, byte) in backing.0.iter_mut().enumerate() {
            *byte = (i * 37 + 11) as u8;
        }
        let lengths = (0..=600).chain([1024, MAX_LEN]);
        for len in lengths {
            for offset in 0..8 {
                let input = &backing.0[offset..offset + len];
                assert_eq!(input.as_ptr() as usize % 8, offset);
                let expected: [u8; HASH_LEN] = RefKeccak::digest(input).into();

                assert_eq!(
                    Keccak256::digest(input),
                    expected,
                    "digest: length {len}, offset {offset}"
                );
                for chunk in [1, 7, 64, 135, 136, 137] {
                    let mut hasher = Keccak256::new();
                    for piece in input.chunks(chunk) {
                        hasher.update(piece);
                    }
                    assert_eq!(
                        hasher.finalize(),
                        expected,
                        "streaming: length {len}, offset {offset}, chunk {chunk}"
                    );
                }
            }
        }
    }
}
