use crate::{RATE_IN_BYTES, RATE_IN_U64};
use sealed::Sealed;

const HASH_LEN: usize = 32;

/// A streaming hash over the Keccak-256 sponge (rate 136 bytes, 32-byte
/// output) whose first padding byte is `PAD`. Only [`Keccak256`] and
/// [`Sha3_256`] can be constructed.
pub struct Sponge256<const PAD: u8> {
    state: [u64; 25],
    buffer: [u64; RATE_IN_U64],
    /// Number of bytes in the buffer, always below `RATE_IN_BYTES`: `update`
    /// absorbs the buffer the moment it fills.
    buffer_len: usize,
}

/// Keccak-256: the original `0x01` pad byte.
pub type Keccak256 = Sponge256<0x01>;
/// SHA3-256 (FIPS 202): the `0x06` pad byte.
pub type Sha3_256 = Sponge256<0x06>;

mod sealed {
    use super::{Keccak256, Sha3_256};

    pub trait Sealed {}
    impl Sealed for Keccak256 {}
    impl Sealed for Sha3_256 {}
}

impl<const PAD: u8> Sponge256<PAD>
where
    Self: Sealed,
{
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            state: [0; 25],
            buffer: [0; RATE_IN_U64],
            buffer_len: 0,
        }
    }

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

        let remaining = absorb_full_blocks(&mut self.state, &input[offset..]);
        if !remaining.is_empty() {
            self.buffer_bytes()[..remaining.len()].copy_from_slice(remaining);
            self.buffer_len = remaining.len();
        }
    }

    /// Reads hash digest and consumes the hasher.
    #[inline(always)]
    pub fn finalize(mut self) -> [u8; HASH_LEN] {
        // Keccak padding is `PAD .. 0x80`; both markers share a byte when
        // `buffer_len == RATE_IN_BYTES - 1`.
        let buffer_len = self.buffer_len;
        let bytes = self.buffer_bytes();
        bytes[buffer_len] = PAD;
        bytes[buffer_len + 1..].fill(0);
        bytes[RATE_IN_BYTES - 1] |= 0x80;

        self.absorb_buffer();
        to_bytes(self.state)
    }

    /// Computes the hash in one call.
    /// Optimized for virtual cycles by avoiding intermediate buffer for final block.
    #[inline(always)]
    pub fn digest(input: &[u8]) -> [u8; HASH_LEN] {
        let mut state = [0u64; 25];
        let remaining = absorb_full_blocks(&mut state, input);
        absorb_final(&mut state, remaining, PAD);
        to_bytes(state)
    }

    #[inline(always)]
    fn buffer_bytes(&mut self) -> &mut [u8; RATE_IN_BYTES] {
        // SAFETY: `[u64; RATE_IN_U64]` is exactly `RATE_IN_BYTES` initialized
        // bytes and `u8` has no alignment requirement.
        unsafe { &mut *self.buffer.as_mut_ptr().cast() }
    }

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

impl<const PAD: u8> Default for Sponge256<PAD>
where
    Self: Sealed,
{
    fn default() -> Self {
        Self::new()
    }
}

/// The first `HASH_LEN` bytes of the state, lanes serialized little-endian.
#[inline(always)]
fn to_bytes(state: [u64; 25]) -> [u8; HASH_LEN] {
    let mut hash = [0u8; HASH_LEN];
    for (out, lane) in hash
        .as_chunks_mut::<{ size_of::<u64>() }>()
        .0
        .iter_mut()
        .zip(state)
    {
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
    let (blocks, remainder) = input.as_chunks::<RATE_IN_BYTES>();
    if (input.as_ptr() as usize).is_multiple_of(8) {
        for block in blocks {
            // SAFETY: `block` is `RATE_IN_BYTES` readable bytes at an 8-byte
            // aligned address; `state` is a distinct `[u64; 25]`.
            unsafe { keccak256_absorb_permute(state.as_mut_ptr(), block.as_ptr()) };
        }
    } else {
        for block in blocks {
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
    remainder
}

/// Pads the final partial block (`input.len() < RATE_IN_BYTES`) and absorbs it.
#[inline(always)]
fn absorb_final(state: &mut [u64; 25], input: &[u8], pad: u8) {
    let mut block = [0u64; RATE_IN_U64];
    // SAFETY: `[u64; RATE_IN_U64]` is exactly `RATE_IN_BYTES` initialized
    // bytes and `u8` has no alignment requirement.
    let bytes: &mut [u8; RATE_IN_BYTES] = unsafe { &mut *block.as_mut_ptr().cast() };
    bytes[..input.len()].copy_from_slice(input);
    bytes[input.len()] = pad;
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

    #[test]
    fn test_keccak256_aligned_vs_unaligned() {
        let test_sizes = [
            0, 1, 7, 8, 31, 32, 63, 64, 135, 136, 137, 200, 272, 512, 1024, 2048,
        ];

        for &size in &test_sizes {
            let aligned: Vec<u8> = (0..size).map(|i| (i * 37 + 11) as u8).collect();

            let mut unaligned_buf = vec![0u8; size + 1];
            unaligned_buf[1..].copy_from_slice(&aligned);
            let unaligned = &unaligned_buf[1..];

            if size > 0 {
                assert_ne!(
                    aligned.as_ptr() as usize % 8,
                    unaligned.as_ptr() as usize % 8,
                    "Test setup error: pointers should have different alignment"
                );
            }

            let aligned_result = Keccak256::digest(&aligned);
            let unaligned_result = Keccak256::digest(unaligned);

            assert_eq!(
                aligned_result, unaligned_result,
                "Keccak256: aligned vs unaligned mismatch at size {size}"
            );

            use sha3::{Digest, Keccak256 as RefKeccak, Sha3_256 as RefSha3};
            let expected: [u8; 32] = RefKeccak::digest(&aligned).into();
            assert_eq!(
                aligned_result, expected,
                "Keccak256: result doesn't match reference at size {size}"
            );

            let expected: [u8; 32] = RefSha3::digest(&aligned).into();
            assert_eq!(
                Sha3_256::digest(&aligned),
                expected,
                "Sha3_256: result doesn't match reference at size {size}"
            );
            assert_eq!(
                Sha3_256::digest(unaligned),
                expected,
                "Sha3_256: aligned vs unaligned mismatch at size {size}"
            );
        }
    }

    /// FIPS 202 known answers, and every two-part split of inputs around one
    /// and two rate blocks through the streaming interface.
    #[test]
    fn test_sha3_256_known_answers_and_split_updates() {
        use hex_literal::hex;
        use sha3::{Digest, Sha3_256 as RefSha3};

        assert_eq!(
            Sha3_256::digest(b""),
            hex!("a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a")
        );
        assert_eq!(
            Sha3_256::digest(b"abc"),
            hex!("3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532")
        );

        for size in [0, 1, 135, 136, 137, 271, 272, 273] {
            let input: Vec<u8> = (0..size).map(|i| (i * 37 + 11) as u8).collect();
            let expected: [u8; 32] = RefSha3::digest(&input).into();
            for split in 0..=size {
                let (first, second) = input.split_at(split);
                let mut hasher = Sha3_256::new();
                hasher.update(first);
                hasher.update(&[]);
                hasher.update(second);
                assert_eq!(hasher.finalize(), expected, "size {size}, split {split}");
            }
        }
    }
}
