//! Streaming XOF cursor used by the signed-sparse fold-challenge sampler.
//!
//! Every fold coordinate gets a fresh indexed SHAKE256 stream. A sampler may
//! reset this cursor between coordinates and reuse its small squeeze buffer.
//!
//! The cursor's `next_*` helpers use bitmask rejection sampling, so every
//! returned value is uniform over the requested range with no modulo bias.

const SHAKE256_RATE: usize = 136;
const SHAKE_DOMAIN_SUFFIX: u8 = 0x1f;
const GROUP_ROOT_LEN: usize = 32;
const COORDINATE_INPUT_LEN: usize = GROUP_ROOT_LEN + size_of::<u64>();

/// SHAKE256 state after absorbing one dedicated group root. Cloning this state
/// gives each coordinate a fresh XOF without repeating the root absorption.
#[derive(Clone)]
pub(crate) struct IndexedXofPrefix {
    state: [u64; 25],
}

impl IndexedXofPrefix {
    pub(crate) fn new(seed: &[u8]) -> Result<Self, &'static str> {
        if seed.len() != GROUP_ROOT_LEN {
            return Err("indexed sparse challenge group root must be exactly 32 bytes");
        }
        let mut state = [0u64; 25];
        absorb_bytes(&mut state, 0, seed);
        Ok(Self { state })
    }

    fn reader(&self, coordinate_index: u64) -> IndexedShakeReader {
        let mut state = self.state;
        absorb_bytes(&mut state, GROUP_ROOT_LEN, &coordinate_index.to_le_bytes());
        xor_state_byte(&mut state, COORDINATE_INPUT_LEN, SHAKE_DOMAIN_SUFFIX);
        xor_state_byte(&mut state, SHAKE256_RATE - 1, 0x80);
        keccak::f1600(&mut state);
        IndexedShakeReader { state, pos: 0 }
    }
}

fn absorb_bytes(state: &mut [u64; 25], offset: usize, bytes: &[u8]) {
    for (index, &byte) in bytes.iter().enumerate() {
        xor_state_byte(state, offset + index, byte);
    }
}

fn xor_state_byte(state: &mut [u64; 25], index: usize, byte: u8) {
    state[index / 8] ^= u64::from(byte) << (8 * (index % 8));
}

struct IndexedShakeReader {
    state: [u64; 25],
    pos: usize,
}

impl IndexedShakeReader {
    fn read(&mut self, out: &mut [u8]) {
        let mut written = 0;
        while written < out.len() {
            if self.pos == SHAKE256_RATE {
                keccak::f1600(&mut self.state);
                self.pos = 0;
            }
            let available = SHAKE256_RATE - self.pos;
            let take = available.min(out.len() - written);
            let end = self.pos + take;
            while self.pos < end {
                let lane = self.state[self.pos / 8].to_le_bytes();
                let lane_offset = self.pos % 8;
                let lane_take = (8 - lane_offset).min(end - self.pos);
                out[written..written + lane_take]
                    .copy_from_slice(&lane[lane_offset..lane_offset + lane_take]);
                self.pos += lane_take;
                written += lane_take;
            }
        }
    }
}

/// One coordinate normally consumes well below one SHAKE256 rate block. Fill
/// a bounded buffer so resetting a short coordinate does not squeeze bytes
/// that will be discarded.
const XOF_BUFFER_SIZE: usize = 128;

/// Streaming cursor backed by a SHAKE256 XOF and a reusable sub-rate buffer.
pub(crate) struct XofCursor {
    reader: IndexedShakeReader,
    buf: [u8; XOF_BUFFER_SIZE],
    pos: usize,
    len: usize,
}

impl XofCursor {
    /// Allocate reusable cursor storage before its first indexed reset.
    pub(crate) fn new() -> Self {
        Self {
            reader: IndexedShakeReader {
                state: [0u64; 25],
                pos: SHAKE256_RATE,
            },
            buf: [0u8; XOF_BUFFER_SIZE],
            pos: 0,
            len: 0,
        }
    }

    /// Build the canonical stream for one claim-major fold coordinate.
    #[cfg(test)]
    pub(crate) fn from_indexed_prefix(prefix: &IndexedXofPrefix, coordinate_index: u64) -> Self {
        let mut xof = prefix.reader(coordinate_index);
        let mut buf = [0u8; XOF_BUFFER_SIZE];
        xof.read(&mut buf);
        Self {
            reader: xof,
            buf,
            pos: 0,
            len: XOF_BUFFER_SIZE,
        }
    }

    /// Reset to another coordinate stream without reallocating the buffer.
    pub(crate) fn reset_indexed_prefix(
        &mut self,
        prefix: &IndexedXofPrefix,
        coordinate_index: u64,
    ) {
        self.reader = prefix.reader(coordinate_index);
        self.pos = 0;
        self.len = 0;
        self.refill();
    }

    #[inline]
    fn refill(&mut self) {
        self.reader.read(&mut self.buf);
        self.pos = 0;
        self.len = XOF_BUFFER_SIZE;
    }

    #[inline]
    fn next_u8(&mut self) -> u8 {
        if self.pos >= self.len {
            self.refill();
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        b
    }

    /// Copy `out.len()` bytes from the buffered XOF stream in one pass.
    #[inline]
    pub(crate) fn fill_bytes(&mut self, out: &mut [u8]) {
        let mut off = 0;
        while off < out.len() {
            if self.pos >= self.len {
                self.refill();
            }
            let avail = self.len - self.pos;
            let take = avail.min(out.len() - off);
            out[off..off + take].copy_from_slice(&self.buf[self.pos..self.pos + take]);
            self.pos += take;
            off += take;
        }
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        if self.pos + 4 <= self.len {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&self.buf[self.pos..self.pos + 4]);
            self.pos += 4;
            u32::from_le_bytes(bytes)
        } else {
            let mut tmp = [0u8; 4];
            for b in &mut tmp {
                *b = self.next_u8();
            }
            u32::from_le_bytes(tmp)
        }
    }

    /// Uniformly sample from `0..modulus` using bitmask rejection sampling
    /// with minimal XOF consumption. Uses 1-byte reads when the modulus
    /// fits in 8 bits, 2-byte reads for 16 bits, else 4 bytes.
    #[inline]
    pub(crate) fn next_usize_mod(&mut self, modulus: usize) -> usize {
        debug_assert!(modulus > 0);
        if modulus == 1 {
            return 0;
        }
        let bits = usize::BITS - (modulus - 1).leading_zeros();
        if bits <= 8 {
            let mask = ((1u16 << bits) - 1) as u8;
            loop {
                let val = (self.next_u8() & mask) as usize;
                if val < modulus {
                    return val;
                }
            }
        } else if bits <= 16 {
            let mask = (1usize << bits) - 1;
            loop {
                let lo = self.next_u8() as usize;
                let hi = self.next_u8() as usize;
                let val = (lo | (hi << 8)) & mask;
                if val < modulus {
                    return val;
                }
            }
        } else {
            let mask: usize = (1 << bits) - 1;
            loop {
                let val = (self.next_u32() as usize) & mask;
                if val < modulus {
                    return val;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shake::digest::{ExtendableOutput, Update, XofReader};
    use shake::Shake256;

    #[test]
    fn indexed_cursor_uses_the_canonical_coordinate_input() {
        let seed = [0x5au8; 32];
        let index = 0x0102_0304_0506_0708u64;
        let mut expected_xof = Shake256::default();
        expected_xof.update(&seed);
        expected_xof.update(&index.to_le_bytes());
        let mut expected_reader = expected_xof.finalize_xof();
        let mut expected = [0u8; 384];
        expected_reader.read(&mut expected);

        let prefix = IndexedXofPrefix::new(&seed).unwrap();
        let mut cursor = XofCursor::from_indexed_prefix(&prefix, index);
        let mut actual = [0u8; 384];
        cursor.fill_bytes(&mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn indexed_prefix_requires_the_canonical_root_width() {
        assert_eq!(
            IndexedXofPrefix::new(&[0u8; GROUP_ROOT_LEN - 1]).err(),
            Some("indexed sparse challenge group root must be exactly 32 bytes")
        );
        assert_eq!(
            IndexedXofPrefix::new(&[0u8; GROUP_ROOT_LEN + 1]).err(),
            Some("indexed sparse challenge group root must be exactly 32 bytes")
        );
    }

    #[test]
    fn resetting_an_indexed_cursor_matches_a_fresh_cursor() {
        let seed = [0xabu8; 32];
        let prefix = IndexedXofPrefix::new(&seed).unwrap();
        let mut reused = XofCursor::from_indexed_prefix(&prefix, 0);
        reused.reset_indexed_prefix(&prefix, u64::MAX);
        let mut fresh = XofCursor::from_indexed_prefix(&prefix, u64::MAX);
        let mut reused_bytes = [0u8; 96];
        let mut fresh_bytes = [0u8; 96];
        reused.fill_bytes(&mut reused_bytes);
        fresh.fill_bytes(&mut fresh_bytes);
        assert_eq!(reused_bytes, fresh_bytes);
    }

    #[test]
    fn next_u32_preserves_stream_bytes_across_a_refill_boundary() {
        let prefix = IndexedXofPrefix::new(&[0x3cu8; GROUP_ROOT_LEN]).unwrap();
        let mut expected_cursor = XofCursor::from_indexed_prefix(&prefix, 17);
        let mut expected = [0u8; XOF_BUFFER_SIZE + 4];
        expected_cursor.fill_bytes(&mut expected);

        let mut cursor = XofCursor::from_indexed_prefix(&prefix, 17);
        let mut prefix_bytes = [0u8; XOF_BUFFER_SIZE - 2];
        cursor.fill_bytes(&mut prefix_bytes);
        assert_eq!(prefix_bytes, expected[..XOF_BUFFER_SIZE - 2]);
        assert_eq!(
            cursor.next_u32(),
            u32::from_le_bytes(
                expected[XOF_BUFFER_SIZE - 2..XOF_BUFFER_SIZE + 2]
                    .try_into()
                    .unwrap()
            )
        );
    }
}
