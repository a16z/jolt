//! Bit-exact model of the streaming keyed BLAKE3 chain behind
//! [`jolt_transcript::Blake3Transcript`], traced at half-G-step granularity.
//!
//! The `blake3` crate exposes no intermediate state, and the table needs every
//! word the compression function writes, so the compression function is
//! re-implemented here (from the BLAKE3 specification, §2.2–2.3) with a
//! per-half-step trace. [`Chain`] reproduces the transcript's segment rules:
//! one keyed chunk per segment, blocks compressed lazily (a full block is
//! compressed when the next byte arrives), a 1,024-byte chunk closed with
//! `CHUNK_END | ROOT`, and a squeeze finalizing the pending block the same way
//! and reading the 64-byte root output — words `0..8` re-key the next segment,
//! words `8..12` are the challenge.

pub const BLOCK_BYTES: usize = 64;
pub const SEGMENT_BYTES: usize = 1024;
pub const WORD_BITS: usize = 32;

pub const IV: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];
const MESSAGE_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

pub const CHUNK_START: u32 = 1;
pub const CHUNK_END: u32 = 2;
pub const ROOT: u32 = 8;
pub const KEYED_HASH: u32 = 16;

pub const ROUNDS: usize = 7;
pub const G_PER_ROUND: usize = 8;
/// Half-G steps per compression: 7 rounds × 8 G × 2 halves.
pub const HALF_STEPS: usize = ROUNDS * G_PER_ROUND * 2;

/// State indices `(a, b, c, d)` of the eight G steps of a round: the four
/// column steps, then the four diagonal steps.
pub const G_INDICES: [[usize; 4]; G_PER_ROUND] = [
    [0, 4, 8, 12],
    [1, 5, 9, 13],
    [2, 6, 10, 14],
    [3, 7, 11, 15],
    [0, 5, 10, 15],
    [1, 6, 11, 12],
    [2, 7, 8, 13],
    [3, 4, 9, 14],
];

/// Right-rotation amounts `(d, b)` of the two halves of a G step.
pub const ROTATIONS: [(u32, u32); 2] = [(16, 12), (8, 7)];

/// The state index each half-G step of a round writes last. Both halves of a
/// step write the same four indices; the diagonal steps come after the column
/// steps, so the last writer of every index is a diagonal step.
pub fn last_writer(index: usize) -> usize {
    (G_PER_ROUND / 2..G_PER_ROUND)
        .find(|&g| G_INDICES[g].contains(&index))
        .unwrap_or(0)
}

/// Original message index at schedule position `position` of `round`
/// (`schedule[0]` is the identity; each round permutes the previous one).
pub fn schedule(round: usize, position: usize) -> usize {
    let mut index = position;
    for _ in 0..round {
        index = MESSAGE_PERMUTATION[index];
    }
    index
}

/// One half of a G step: `a' = a + b + m; d' = (d ^ a') >>> r1; c' = c + d';
/// b' = (b ^ c') >>> r2`. The XOR outputs are kept un-rotated (the rotation is
/// a bit re-indexing the table's wiring performs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HalfStep {
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub d: u32,
    pub m: u32,
    pub a_out: u32,
    /// `(a + b + m) >> 32`, in `0..=3`.
    pub a_carry: u32,
    pub d_xor: u32,
    pub c_out: u32,
    /// `(c + d') >> 32`, in `0..=1`.
    pub c_carry: u32,
    pub b_xor: u32,
}

/// One compression: inputs, the half-step trace, the state after the seven
/// rounds and the 16-word output (`out[i] = v[i] ^ v[i + 8]`,
/// `out[8 + i] = v[8 + i] ^ cv[i]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Compression {
    pub cv: [u32; 8],
    pub block: [u32; 16],
    pub block_len: u32,
    pub flags: u32,
    /// Index `(round * 8 + g) * 2 + half`.
    pub steps: Vec<HalfStep>,
    pub v: [u32; 16],
    pub out: [u32; 16],
}

fn half_step(
    v: &mut [u32; 16],
    [a, b, c, d]: [usize; 4],
    m: u32,
    (r1, r2): (u32, u32),
) -> HalfStep {
    let (va, vb, vc, vd) = (v[a], v[b], v[c], v[d]);
    let sum = u64::from(va) + u64::from(vb) + u64::from(m);
    let a_out = sum as u32;
    let a_carry = (sum >> 32) as u32;
    let d_xor = vd ^ a_out;
    let d_out = d_xor.rotate_right(r1);
    let sum = u64::from(vc) + u64::from(d_out);
    let c_out = sum as u32;
    let c_carry = (sum >> 32) as u32;
    let b_xor = vb ^ c_out;
    v[a] = a_out;
    v[d] = d_out;
    v[c] = c_out;
    v[b] = b_xor.rotate_right(r2);
    HalfStep {
        a: va,
        b: vb,
        c: vc,
        d: vd,
        m,
        a_out,
        a_carry,
        d_xor,
        c_out,
        c_carry,
        b_xor,
    }
}

/// The compression function with chunk counter 0 — every compression of the
/// transcript chain is block ≤ 16 of chunk 0 of a fresh keyed hasher.
pub fn compress(cv: &[u32; 8], block: &[u32; 16], block_len: u32, flags: u32) -> Compression {
    let mut v = [0u32; 16];
    v[..8].copy_from_slice(cv);
    v[8..12].copy_from_slice(&IV[..4]);
    v[14] = block_len;
    v[15] = flags;
    let mut steps = Vec::with_capacity(HALF_STEPS);
    for round in 0..ROUNDS {
        for (g, indices) in G_INDICES.iter().enumerate() {
            for half in 0..2 {
                let m = block[schedule(round, 2 * g + half)];
                steps.push(half_step(&mut v, *indices, m, ROTATIONS[half]));
            }
        }
    }
    let mut out = [0u32; 16];
    for i in 0..8 {
        out[i] = v[i] ^ v[i + 8];
        out[i + 8] = v[i + 8] ^ cv[i];
    }
    Compression {
        cv: *cv,
        block: *block,
        block_len,
        flags,
        steps,
        v,
        out,
    }
}

pub fn words_from_bytes(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|chunk| {
            let mut word = [0u8; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            u32::from_le_bytes(word)
        })
        .collect()
}

pub fn bytes_from_words(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Which absorbed byte a block byte came from: the log item and the offset
/// within it. `None` is zero padding of a partial final block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ByteOrigin {
    pub item: u32,
    pub offset: u32,
}

/// One compression of the chain with the provenance of its 64 block bytes and
/// the squeeze it serves (the log item index of the challenge), if any.
#[derive(Clone, Debug)]
pub struct Block {
    pub compression: Compression,
    pub origins: Vec<Option<ByteOrigin>>,
    pub squeeze: Option<u32>,
}

/// The streaming keyed chain, byte-compatible with `Blake3Transcript`.
#[derive(Clone, Debug)]
pub struct Chain {
    cv: [u32; 8],
    buffer: Vec<u8>,
    origins: Vec<Option<ByteOrigin>>,
    blocks_in_segment: usize,
    segment_bytes: usize,
    pub blocks: Vec<Block>,
}

impl Chain {
    /// A chain keyed by `key` — for the transcript, `blake3::hash` of the
    /// zero-padded 32-byte domain label.
    pub fn new(key: &[u8; 32]) -> Self {
        let mut cv = [0u32; 8];
        cv.copy_from_slice(&words_from_bytes(key));
        Self {
            cv,
            buffer: Vec::with_capacity(BLOCK_BYTES),
            origins: Vec::with_capacity(BLOCK_BYTES),
            blocks_in_segment: 0,
            segment_bytes: 0,
            blocks: Vec::new(),
        }
    }

    fn block_words(&self) -> [u32; 16] {
        let mut padded = [0u8; BLOCK_BYTES];
        padded[..self.buffer.len()].copy_from_slice(&self.buffer);
        let mut words = [0u32; 16];
        words.copy_from_slice(&words_from_bytes(&padded));
        words
    }

    fn flags(&self, last: bool) -> u32 {
        let mut flags = KEYED_HASH;
        if self.blocks_in_segment == 0 {
            flags |= CHUNK_START;
        }
        if last {
            flags |= CHUNK_END | ROOT;
        }
        flags
    }

    /// The compression a finalize would perform now (the pending block with
    /// `CHUNK_END | ROOT`), without advancing the chain.
    fn finalize_compression(&self) -> Compression {
        compress(
            &self.cv,
            &self.block_words(),
            self.buffer.len() as u32,
            self.flags(true),
        )
    }

    fn push_block(&mut self, compression: Compression, squeeze: Option<u32>) {
        let mut origins = std::mem::take(&mut self.origins);
        origins.resize(BLOCK_BYTES, None);
        self.cv.copy_from_slice(&compression.out[..8]);
        self.blocks.push(Block {
            compression,
            origins,
            squeeze,
        });
        self.buffer.clear();
    }

    /// Compress the pending full block as a non-final block of the segment.
    fn compress_pending(&mut self) {
        let compression = compress(
            &self.cv,
            &self.block_words(),
            BLOCK_BYTES as u32,
            self.flags(false),
        );
        self.push_block(compression, None);
        self.blocks_in_segment += 1;
    }

    /// Finalize the segment: the pending (possibly empty or partial) block
    /// gets `CHUNK_END | ROOT`; `out[0..8]` keys the next segment.
    fn finalize(&mut self, squeeze: Option<u32>) -> [u32; 16] {
        let compression = self.finalize_compression();
        let out = compression.out;
        self.push_block(compression, squeeze);
        self.blocks_in_segment = 0;
        self.segment_bytes = 0;
        out
    }

    /// Absorb the bytes of log item `item`.
    pub fn absorb(&mut self, item: u32, bytes: &[u8]) {
        for (offset, &byte) in bytes.iter().enumerate() {
            if self.segment_bytes == SEGMENT_BYTES {
                let _ = self.finalize(None);
            }
            if self.buffer.len() == BLOCK_BYTES {
                self.compress_pending();
            }
            self.buffer.push(byte);
            self.origins.push(Some(ByteOrigin {
                item,
                offset: offset as u32,
            }));
            self.segment_bytes += 1;
        }
    }

    /// Squeeze for log item `item`: the 16 challenge bytes (root output
    /// words `8..12`).
    pub fn squeeze(&mut self, item: u32) -> [u8; 16] {
        let out = self.finalize(Some(item));
        let mut challenge = [0u8; 16];
        challenge.copy_from_slice(&bytes_from_words(&out[8..12]));
        challenge
    }

    /// Extend the chain with one padding compression: an empty block with
    /// `CHUNK_START | KEYED_HASH` chained from the current key, never
    /// finalized. Padding cells of the table continue the chain so the wiring
    /// stays uniform; their outputs are never linked.
    pub fn pad(&mut self) {
        // Chain from the last block kept (the schedule drops the blocks after
        // its final squeeze), not from the key of the dropped tail.
        let mut cv = self.cv;
        if let Some(last) = self.blocks.last() {
            cv.copy_from_slice(&last.compression.out[..8]);
        }
        let compression = compress(&cv, &[0; 16], 0, KEYED_HASH | CHUNK_START);
        self.cv.copy_from_slice(&compression.out[..8]);
        self.blocks.push(Block {
            compression,
            origins: vec![None; BLOCK_BYTES],
            squeeze: None,
        });
    }

    /// `Blake3Transcript::state()`: the keyed digest of the pending segment.
    pub fn state(&self) -> [u8; 32] {
        let mut state = [0u8; 32];
        state.copy_from_slice(&bytes_from_words(&self.finalize_compression().out[..8]));
        state
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{Rng, RngCore, SeedableRng};

    use super::*;

    fn random_bytes(rng: &mut StdRng, len: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; len];
        rng.fill_bytes(&mut bytes);
        bytes
    }

    #[test]
    fn keyed_single_chunk_matches_blake3() {
        let mut rng = StdRng::seed_from_u64(0xb1a3);
        for len in [0usize, 1, 63, 64, 65, 127, 128, 500, 1023, 1024] {
            let key: [u8; 32] = rng.gen();
            let input = random_bytes(&mut rng, len);
            let mut chain = Chain::new(&key);
            // Split the input into a few appends to exercise the lazy block boundary.
            let split = len / 3;
            chain.absorb(0, &input[..split]);
            chain.absorb(1, &input[split..]);
            let expected = blake3::Hasher::new_keyed(&key).update(&input).finalize();
            assert_eq!(chain.state(), *expected.as_bytes(), "state at len {len}");
            let mut xof = [0u8; 64];
            blake3::Hasher::new_keyed(&key)
                .update(&input)
                .finalize_xof()
                .fill(&mut xof);
            let challenge = chain.squeeze(2);
            assert_eq!(challenge, xof[32..48], "challenge at len {len}");
            let last = chain.blocks.last().unwrap();
            assert_eq!(bytes_from_words(&last.compression.out), xof.to_vec());
            assert_eq!(chain.blocks.len(), len.div_ceil(BLOCK_BYTES).max(1));
        }
    }

    #[test]
    fn origins_cover_every_absorbed_byte_once() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut chain = Chain::new(&[0u8; 32]);
        let lens = [32usize, 384, 5, 0, 1000, 64, 33];
        for (item, len) in lens.iter().enumerate() {
            chain.absorb(item as u32, &random_bytes(&mut rng, *len));
            if item % 3 == 2 {
                let _ = chain.squeeze(100 + item as u32);
            }
        }
        let _ = chain.squeeze(200);
        let mut seen = std::collections::HashSet::new();
        for block in &chain.blocks {
            assert_eq!(block.origins.len(), BLOCK_BYTES);
            for origin in block.origins.iter().flatten() {
                assert!(seen.insert(*origin), "duplicate origin {origin:?}");
            }
        }
        assert_eq!(seen.len(), lens.iter().sum::<usize>());
    }
}
