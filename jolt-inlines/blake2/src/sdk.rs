//! High-level Blake2b hashing API for host and guest modes.
//!
//! Message bytes are packed into little-endian words as they arrive instead
//! of being staged through byte buffers. On a guest, variable-length byte
//! copies and fills lower to libc `memcpy`/`memset` calls that move data in
//! sub-word accesses, each a multi-row sequence; the word paths below load
//! and store whole aligned words.
use crate::{
    BLOCK_INPUT_SIZE_IN_BYTES, IV, MSG_BLOCK_LEN, PERSONA_SIZE_IN_BYTES, SALT_SIZE_IN_BYTES,
    STATE_VECTOR_LEN,
};
const OUTPUT_SIZE: usize = 64;
const WORD_BYTES: usize = 8;
const INITIAL_STATE: [u64; STATE_VECTOR_LEN] = {
    let mut h = IV;
    h[0] ^= 0x01010000 ^ (OUTPUT_SIZE as u64);
    h
};

/// One compression input: 16 message words, the byte counter, and the final
/// flag, laid out as the 18 consecutive words the inline reads.
#[repr(C)]
struct CompressionBlock {
    /// Little-endian message words. Every byte at or past the buffered
    /// length is zero, so absorbing ORs bytes into place and finalizing
    /// needs no padding pass.
    words: [u64; MSG_BLOCK_LEN],
    counter: u64,
    final_flag: u64,
}

impl CompressionBlock {
    #[inline(always)]
    fn new() -> Self {
        Self {
            words: [0; MSG_BLOCK_LEN],
            counter: 0,
            final_flag: 0,
        }
    }

    #[inline(always)]
    fn compress(&mut self, state: &mut [u64; STATE_VECTOR_LEN], is_final: bool) {
        self.final_flag = u64::from(is_final);
        // SAFETY: repr(C) lays out the 16 message words, the counter, and the
        // flag as exactly the 18 aligned words the inline reads.
        unsafe {
            blake2b_compress(state.as_mut_ptr(), core::ptr::from_ref(self).cast::<u64>());
        }
    }
}

const _: () = assert!(core::mem::size_of::<CompressionBlock>() == (MSG_BLOCK_LEN + 2) * 8);

/// Reads one little-endian message word from an 8-byte chunk.
///
/// The volatile reads keep LLVM from recognizing the callers' word loops as
/// copies and lowering them back into `memcpy` calls.
#[inline(always)]
fn load_word(chunk: &[u8], aligned: bool) -> u64 {
    debug_assert_eq!(chunk.len(), WORD_BYTES);
    let p = chunk.as_ptr();
    if aligned {
        // SAFETY: the caller checked that `chunk` starts on an 8-byte
        // boundary, and it holds 8 readable bytes.
        u64::from_le(unsafe { p.cast::<u64>().read_volatile() })
    } else {
        let mut word = 0;
        for i in 0..WORD_BYTES {
            // SAFETY: `i < 8 == chunk.len()`.
            word |= u64::from(unsafe { p.add(i).read_volatile() }) << (8 * i);
        }
        word
    }
}

/// ORs `src` into `words` starting at byte `pos`. Bytes `pos..pos + src.len()`
/// must be zero and lie within the block.
#[inline(always)]
fn or_into_words(words: &mut [u64; MSG_BLOCK_LEN], mut pos: usize, mut src: &[u8]) {
    while !pos.is_multiple_of(WORD_BYTES) {
        let Some((&byte, rest)) = src.split_first() else {
            return;
        };
        words[pos / WORD_BYTES] |= u64::from(byte) << (8 * (pos % WORD_BYTES));
        pos += 1;
        src = rest;
    }
    let aligned = (src.as_ptr() as usize).is_multiple_of(WORD_BYTES);
    let mut chunks = src.chunks_exact(WORD_BYTES);
    for chunk in &mut chunks {
        words[pos / WORD_BYTES] = load_word(chunk, aligned);
        pos += WORD_BYTES;
    }
    for (i, &byte) in chunks.remainder().iter().enumerate() {
        words[pos / WORD_BYTES] |= u64::from(byte) << (8 * i);
    }
}

/// Fills all 16 words from a full 128-byte block.
#[inline(always)]
fn load_block(words: &mut [u64; MSG_BLOCK_LEN], block: &[u8]) {
    debug_assert_eq!(block.len(), BLOCK_INPUT_SIZE_IN_BYTES);
    let aligned = (block.as_ptr() as usize).is_multiple_of(WORD_BYTES);
    for (word, chunk) in words.iter_mut().zip(block.chunks_exact(WORD_BYTES)) {
        *word = load_word(chunk, aligned);
    }
}

/// Zeroes `words`. Volatile stores keep LLVM from emitting a `memset` call.
#[inline(always)]
fn clear_words(words: &mut [u64]) {
    for word in words {
        // SAFETY: `word` is a valid, aligned `&mut u64`.
        unsafe { core::ptr::from_mut(word).write_volatile(0) };
    }
}

pub struct Blake2b {
    h: [u64; STATE_VECTOR_LEN],
    buffer: CompressionBlock,
    buffer_len: usize,
}

/// Copies only the buffered words; the rest of a block is zero by invariant.
/// A derived clone copies the whole 216-byte hasher, which a guest lowers to a
/// sub-word `memcpy` call, and sponges clone prepared hashers per phase.
impl Clone for Blake2b {
    #[inline(always)]
    fn clone(&self) -> Self {
        let mut buffer = CompressionBlock::new();
        buffer.counter = self.buffer.counter;
        let live = self.buffer_len.div_ceil(WORD_BYTES);
        for (word, source) in buffer.words.iter_mut().zip(&self.buffer.words[..live]) {
            // SAFETY: `source` is a valid, aligned `&u64`; the volatile read
            // keeps this loop from lowering to a `memcpy` call.
            *word = unsafe { core::ptr::from_ref(source).read_volatile() };
        }
        Self {
            h: self.h,
            buffer,
            buffer_len: self.buffer_len,
        }
    }
}

impl Blake2b {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            h: INITIAL_STATE,
            buffer: CompressionBlock::new(),
            buffer_len: 0,
        }
    }

    /// Blake2b with an `output_len`-byte digest (1..=64). The digest length is
    /// part of the parameter block folded into the IV, so a shorter digest is
    /// its own hash function, not a truncation of the 64-byte one.
    #[inline(always)]
    pub fn new_with_output_len(output_len: usize) -> Self {
        assert!(
            (1..=OUTPUT_SIZE).contains(&output_len),
            "Blake2b digest length must be 1..=64 bytes"
        );
        let mut h = IV;
        h[0] ^= 0x01010000 ^ (output_len as u64);
        Self {
            h,
            buffer: CompressionBlock::new(),
            buffer_len: 0,
        }
    }

    /// Finalize into `out`, which holds the digest length this hasher was
    /// created with (the leading bytes of the state).
    #[inline(always)]
    pub fn finalize_into(mut self, out: &mut [u8]) {
        self.compress_final();
        let len = out.len();
        if len.is_multiple_of(WORD_BYTES) && (out.as_mut_ptr() as usize).is_multiple_of(WORD_BYTES)
        {
            for (chunk, word) in out.chunks_exact_mut(WORD_BYTES).zip(&self.h) {
                // SAFETY: `out` starts on an 8-byte boundary and its length is
                // a multiple of 8, so every chunk is an aligned 8-byte word.
                unsafe { chunk.as_mut_ptr().cast::<u64>().write(word.to_le()) };
            }
        } else {
            out.copy_from_slice(&to_bytes(self.h)[..len]);
        }
    }

    /// Creates a new hasher with the given salt and personalization,
    /// matching the `blake2` crate's `new_with_params` semantics.
    ///
    /// Shorter values are zero-padded per the BLAKE2b specification.
    ///
    /// # Panics
    /// Panics if `salt` or `persona` is longer than 16 bytes.
    #[inline(always)]
    pub fn new_with_params(salt: &[u8], persona: &[u8]) -> Self {
        Self {
            h: initial_state_with_params(salt, persona),
            buffer: CompressionBlock::new(),
            buffer_len: 0,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, mut input: &[u8]) {
        if input.is_empty() {
            return;
        }

        // Fill a partial buffer first. The block is compressed only once more
        // input follows, so the final block always reaches `finalize`.
        if self.buffer_len != 0 {
            let take = (BLOCK_INPUT_SIZE_IN_BYTES - self.buffer_len).min(input.len());
            let (head, rest) = input.split_at(take);
            or_into_words(&mut self.buffer.words, self.buffer_len, head);
            self.buffer_len += take;
            input = rest;
            if input.is_empty() {
                return;
            }
            self.buffer.counter += BLOCK_INPUT_SIZE_IN_BYTES as u64;
            self.buffer.compress(&mut self.h, false);
            clear_words(&mut self.buffer.words);
            self.buffer_len = 0;
        }

        // Compress complete blocks, keeping at least one byte for the final
        // block.
        while input.len() > BLOCK_INPUT_SIZE_IN_BYTES {
            let (block, rest) = input.split_at(BLOCK_INPUT_SIZE_IN_BYTES);
            self.buffer.counter += BLOCK_INPUT_SIZE_IN_BYTES as u64;
            load_block(&mut self.buffer.words, block);
            self.buffer.compress(&mut self.h, false);
            clear_words(&mut self.buffer.words);
            input = rest;
        }

        or_into_words(&mut self.buffer.words, 0, input);
        self.buffer_len = input.len();
    }

    /// Compresses the buffered bytes as the final block. Bytes past the
    /// buffered length are already zero.
    #[inline(always)]
    fn compress_final(&mut self) {
        self.buffer.counter += self.buffer_len as u64;
        self.buffer.compress(&mut self.h, true);
    }

    #[inline(always)]
    pub fn finalize(mut self) -> [u8; OUTPUT_SIZE] {
        self.compress_final();
        to_bytes(self.h)
    }

    /// Computes BLAKE2b hash in one call.
    #[inline(always)]
    pub fn digest(input: &[u8]) -> [u8; OUTPUT_SIZE] {
        Self::digest_from_state(INITIAL_STATE, input)
    }

    /// Computes BLAKE2b hash with the given salt and personalization in one call.
    ///
    /// Shorter values are zero-padded per the BLAKE2b specification.
    ///
    /// # Panics
    /// Panics if `salt` or `persona` is longer than 16 bytes.
    #[inline(always)]
    pub fn digest_with_params(salt: &[u8], persona: &[u8], input: &[u8]) -> [u8; OUTPUT_SIZE] {
        Self::digest_from_state(initial_state_with_params(salt, persona), input)
    }

    /// One BLAKE2b compression on the inline: mixes the 16-word message block
    /// `m` into the state `h` in place, with byte counter `t` and final-block
    /// flag `last`. 12 rounds, 64-bit counter: the EIP-152 `BLAKE2F` primitive
    /// for `rounds == 12` with a zero high counter word.
    #[inline(always)]
    pub fn compress(h: &mut [u64; STATE_VECTOR_LEN], m: &[u64; MSG_BLOCK_LEN], t: u64, last: bool) {
        let mut block = [0u64; MSG_BLOCK_LEN + 2];
        block[..MSG_BLOCK_LEN].copy_from_slice(m);
        block[MSG_BLOCK_LEN] = t;
        block[MSG_BLOCK_LEN + 1] = last as u64;
        // SAFETY: `h` is 8 and `block` 18 aligned u64s, exactly what the
        // inline reads and writes.
        unsafe { blake2b_compress(h.as_mut_ptr(), block.as_ptr()) }
    }

    #[inline(always)]
    fn digest_from_state(h: [u64; STATE_VECTOR_LEN], input: &[u8]) -> [u8; OUTPUT_SIZE] {
        let mut hasher = Self {
            h,
            buffer: CompressionBlock::new(),
            buffer_len: 0,
        };
        hasher.update(input);
        hasher.finalize()
    }
}

/// Initial state with the salt and personalization words of the BLAKE2b parameter block XORed
/// into the IV, as defined by section 2.8 of the
/// [BLAKE2 specification](https://www.blake2.net/blake2.pdf).
#[inline(always)]
#[expect(clippy::unwrap_used)]
fn initial_state_with_params(salt: &[u8], persona: &[u8]) -> [u64; STATE_VECTOR_LEN] {
    assert!(
        salt.len() <= SALT_SIZE_IN_BYTES,
        "salt must be at most 16 bytes"
    );
    assert!(
        persona.len() <= PERSONA_SIZE_IN_BYTES,
        "persona must be at most 16 bytes"
    );

    let mut params = [0u8; SALT_SIZE_IN_BYTES + PERSONA_SIZE_IN_BYTES];
    params[..salt.len()].copy_from_slice(salt);
    params[SALT_SIZE_IN_BYTES..SALT_SIZE_IN_BYTES + persona.len()].copy_from_slice(persona);

    let mut h = INITIAL_STATE;
    for (word, chunk) in h[4..].iter_mut().zip(params.chunks_exact(8)) {
        *word ^= u64::from_le_bytes(chunk.try_into().unwrap());
    }
    h
}

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
#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
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

    let state_array: &mut [u64; 8] = state_slice
        .try_into()
        .expect("State pointer must reference exactly 8 u64 values");
    let message_array: [u64; 18] = message_slice
        .try_into()
        .expect("Message pointer must reference exactly 18 u64 values");

    crate::exec::execute_blake2b_compression(state_array, &message_array);
}

#[cfg(all(
    not(feature = "host"),
    not(any(target_arch = "riscv32", target_arch = "riscv64"))
))]
pub(crate) unsafe fn blake2b_compress(_state: *mut u64, _message: *const u64) {
    panic!("blake2b_compress requires RISC-V target or host feature");
}

#[cfg(all(test, feature = "host"))]
mod digest_tests {
    use super::*;

    /// EIP-152 vectors 5 (final) and 6 (not final): 12 rounds, h = BLAKE2b-512
    /// IV with the parameter block, m = "abc" zero-padded, t = 3.
    #[test]
    fn test_blake2b_compress_eip152_vectors() {
        let mut h = IV;
        h[0] ^= 0x0101_0000 ^ 64;
        let mut m = [0u64; MSG_BLOCK_LEN];
        m[0] = 0x0063_6261;
        let expected: [(bool, [u64; STATE_VECTOR_LEN]); 2] = [
            (
                true,
                [
                    0x0d4d_1c98_3fa5_80ba,
                    0xe9f6_129f_b697_276a,
                    0xb7c4_5a68_142f_214c,
                    0xd1a2_ffdb_6fbb_124b,
                    0x2d79_ab2a_39c5_877d,
                    0x95cc_3345_ded5_52c2,
                    0x5a92_f1db_a88a_d318,
                    0x2399_00d4_ed86_23b9,
                ],
            ),
            (
                false,
                [
                    0x2c56_0a19_d369_ab75,
                    0x7527_1c8f_d8f8_ae51,
                    0x2cc4_7072_4044_6987,
                    0x5287_d226_2c25_4498,
                    0xf2a2_5e6d_7f3e_7498,
                    0x1bd3_9c03_26d2_e8d3,
                    0x66d6_d3f2_c46a_424e,
                    0x3547_de6f_11c2_10a6,
                ],
            ),
        ];
        for (last, want) in expected {
            let mut got = h;
            Blake2b::compress(&mut got, &m, 3, last);
            assert_eq!(got, want, "last = {last}");
        }
    }

    /// Streaming updates match the `blake2` crate for every source alignment
    /// and split point, covering the aligned-word, unaligned-word, and
    /// partial-word absorb paths and the block-boundary compressions.
    #[test]
    fn streaming_matches_reference_at_every_alignment() {
        use blake2::{Blake2b512, Digest as RefDigest};
        let backing: Vec<u64> = (0..64u64)
            .map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15))
            .collect();
        // SAFETY: a u64 buffer viewed as its bytes; the base is 8-aligned.
        let bytes = unsafe { core::slice::from_raw_parts(backing.as_ptr().cast::<u8>(), 512) };
        for offset in 0..8 {
            for len in [0, 1, 7, 8, 9, 64, 127, 128, 129, 255, 256, 300] {
                let input = &bytes[offset..offset + len];
                let expected: [u8; 64] = Blake2b512::digest(input).into();
                for split in [0, 1, 3, 8, 13, len / 2, len.saturating_sub(1), len] {
                    let split = split.min(len);
                    let mut hasher = Blake2b::new();
                    hasher.update(&input[..split]);
                    hasher.clone().update(&[0xff]);
                    hasher.update(&input[split..]);
                    assert_eq!(
                        hasher.finalize(),
                        expected,
                        "offset {offset} len {len} split {split}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_blake2b_variable_input_lengths() {
        const MAX_LENGTH: usize = 1200;
        let input_buffer: [u8; MAX_LENGTH] = std::array::from_fn(|i| {
            let base = (i % 256) as u8;
            let modifier = ((i / 256) * 17 + (i % 7) * 31) as u8;
            base.wrapping_add(modifier)
        });

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
}

#[cfg(all(test, feature = "host"))]
#[expect(clippy::unwrap_used)]
mod params_tests {
    use super::*;

    fn reference_with_params(salt: &[u8], persona: &[u8], input: &[u8]) -> [u8; OUTPUT_SIZE] {
        use blake2::digest::Mac;
        let mut mac =
            blake2::Blake2bMac512::new_with_salt_and_personal(None, salt, persona).unwrap();
        mac.update(input);
        mac.finalize().into_bytes().into()
    }

    #[test]
    fn test_blake2b_params_against_reference() {
        let params: [(&[u8], &[u8]); 6] = [
            (b"", b"ZcashPrevoutHash"),
            (b"0123456789abcdef", b""),
            (b"0123456789abcdef", b"fedcba9876543210"),
            (b"salt", b"persona"),
            (b"s", b"p"),
            (b"", b""),
        ];
        let input_buffer: [u8; 1200] = std::array::from_fn(|i| ((i * 213 + 17) % 256) as u8);
        let lengths = [0, 1, 8, 64, 127, 128, 129, 255, 256, 257, 512, 1199, 1200];

        for (salt, persona) in params {
            for length in lengths {
                let input = &input_buffer[..length];
                let expected = reference_with_params(salt, persona, input);

                assert_eq!(
                    Blake2b::digest_with_params(salt, persona, input),
                    expected,
                    "digest_with_params mismatch: salt={salt:?}, persona={persona:?}, length={length}"
                );

                let mut hasher = Blake2b::new_with_params(salt, persona);
                hasher.update(input);
                assert_eq!(
                    hasher.finalize(),
                    expected,
                    "new_with_params streaming mismatch: salt={salt:?}, persona={persona:?}, length={length}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "salt must be at most 16 bytes")]
    fn test_blake2b_oversized_salt_panics() {
        let _ = Blake2b::new_with_params(b"01234567890123456", b"");
    }

    #[test]
    #[should_panic(expected = "persona must be at most 16 bytes")]
    fn test_blake2b_oversized_persona_panics() {
        let _ = Blake2b::digest_with_params(b"", b"01234567890123456", b"");
    }
}

#[cfg(all(test, feature = "host"))]
mod streaming_tests {
    use super::*;

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
    fn test_blake2b_streaming_incremental_updates() {
        use blake2::Digest as RefDigest;

        const MAX_LENGTH: usize = 512;
        let input_buffer: [u8; MAX_LENGTH] = std::array::from_fn(|i| ((i * 137 + 42) % 256) as u8);

        let chunk_sizes = [1, 3, 7, 16, 32, 63, 64, 65, 128];
        let test_lengths = [0, 1, 63, 64, 65, 127, 128, 129, 255, 256, 257, MAX_LENGTH];

        for &chunk_size in &chunk_sizes {
            for total_length in test_lengths {
                let input = &input_buffer[..total_length];
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
    fn test_blake2b_streaming_empty_updates() {
        use blake2::Digest as RefDigest;

        let test_data = b"Some test data for empty update testing";

        let mut hasher = Blake2b::new();
        hasher.update(b"");
        hasher.update(&test_data[..10]);
        hasher.update(b"");
        hasher.update(&test_data[10..]);
        hasher.update(b"");

        assert_eq!(
            hasher.finalize(),
            Into::<[u8; 64]>::into(blake2::Blake2b512::digest(test_data)),
            "Empty updates should not affect the result"
        );
    }

    #[test]
    fn test_blake2b_aligned_vs_unaligned() {
        let test_sizes = [
            0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128, 129, 256, 512, 1024, 2048,
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

            let aligned_result = Blake2b::digest(&aligned);
            let unaligned_result = Blake2b::digest(unaligned);

            assert_eq!(
                aligned_result, unaligned_result,
                "Blake2b: aligned vs unaligned mismatch at size {size}"
            );

            use blake2::Digest as RefDigest;
            let expected: [u8; 64] = blake2::Blake2b512::digest(&aligned).into();
            assert_eq!(
                aligned_result, expected,
                "Blake2b: result doesn't match reference at size {size}"
            );
        }
    }
}
