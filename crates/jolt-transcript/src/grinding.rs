//! Proof of work and its nonce codec.
//!
//! A prover grinds `bits` bits by finding a nonce whose absorption makes the
//! next 32 squeezed bytes start with `bits` zero bits, low bit first. Nonces
//! are searched over `bits + GRINDING_NONCE_SLACK_BITS` bits, so an honest
//! search exhausts its range with probability about `exp(-2^7)`.

use std::num::NonZeroU8;

/// Extra nonce bits searched beyond the difficulty.
pub const GRINDING_NONCE_SLACK_BITS: u8 = 7;
/// Largest supported difficulty; the nonce range then fills a `u32`.
pub const MAX_GRINDING_BITS: u8 = u32::BITS as u8 - GRINDING_NONCE_SLACK_BITS;
/// Byte length of the squeezed proof-of-work predicate.
pub const GRINDING_PREDICATE_LEN: usize = 32;

/// Largest encoded nonce: a `u32` in 7-bit groups.
const NONCE_MAX_BYTES: usize = 5;

/// Returns whether the low `bits` bits of `predicate`, low byte first and low
/// bit first within each byte, are all zero.
#[must_use]
pub fn grinding_predicate_accepts(
    predicate: &[u8; GRINDING_PREDICATE_LEN],
    bits: NonZeroU8,
) -> bool {
    let bits = usize::from(bits.get());
    let (whole, partial) = predicate.split_at(bits / 8);
    let remaining = bits % 8;
    whole.iter().all(|&byte| byte == 0)
        && (remaining == 0
            || partial
                .first()
                .is_some_and(|&byte| byte & ((1u8 << remaining) - 1) == 0))
}

/// Nonce search width for difficulty `bits`.
pub(crate) fn nonce_bits(bits: NonZeroU8) -> Option<u8> {
    (bits.get() <= MAX_GRINDING_BITS).then(|| bits.get() + GRINDING_NONCE_SLACK_BITS)
}

/// The canonical encoding of a nonce: little-endian base-128 groups with a
/// continuation bit and no redundant trailing zero group.
pub(crate) struct NonceBytes {
    bytes: [u8; NONCE_MAX_BYTES],
    len: usize,
}

impl NonceBytes {
    pub(crate) fn as_slice(&self) -> &[u8] {
        self.bytes.split_at(self.len).0
    }
}

pub(crate) fn encode_nonce(mut value: u32) -> NonceBytes {
    let mut encoded = NonceBytes {
        bytes: [0; NONCE_MAX_BYTES],
        len: 0,
    };
    for slot in &mut encoded.bytes {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        *slot = byte;
        encoded.len += 1;
        if value == 0 {
            break;
        }
    }
    encoded
}

/// Decodes a canonical nonce below `2^nonce_bits` from the front of `bytes`,
/// returning it with its encoded length. Rejects non-minimal encodings.
pub(crate) fn decode_nonce(bytes: &[u8], nonce_bits: u8) -> Option<(u32, usize)> {
    let mut value = 0u64;
    for (index, &byte) in bytes.iter().take(NONCE_MAX_BYTES).enumerate() {
        let payload = byte & 0x7f;
        value |= u64::from(payload) << (7 * index);
        if byte & 0x80 == 0 {
            if index != 0 && payload == 0 {
                return None;
            }
            let value = u32::try_from(value).ok()?;
            return (u64::from(value) >> nonce_bits == 0).then_some((value, index + 1));
        }
    }
    None
}

/// Searches `0..2^nonce_bits` for the first nonce whose predicate accepts.
pub(crate) fn search_nonce(
    bits: NonZeroU8,
    nonce_bits: u8,
    mut predicate_for: impl FnMut(u32) -> [u8; GRINDING_PREDICATE_LEN],
) -> Option<u32> {
    let attempts = 1u64 << nonce_bits;
    (0..attempts)
        .filter_map(|candidate| u32::try_from(candidate).ok())
        .find(|&nonce| grinding_predicate_accepts(&predicate_for(nonce), bits))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "test code")]
mod tests {
    use super::*;

    #[test]
    fn predicate_checks_low_bits_at_byte_boundaries() {
        for bits in [1, 7, 8, 9, 16, 17, MAX_GRINDING_BITS] {
            let nonzero = NonZeroU8::new(bits).unwrap();
            let mut predicate = [0u8; GRINDING_PREDICATE_LEN];
            assert!(grinding_predicate_accepts(&predicate, nonzero));
            let last = usize::from(bits - 1);
            predicate[last / 8] = 1 << (last % 8);
            assert!(!grinding_predicate_accepts(&predicate, nonzero));
            predicate[last / 8] = 0;
            let first_free = usize::from(bits);
            predicate[first_free / 8] |= 1 << (first_free % 8);
            assert!(grinding_predicate_accepts(&predicate, nonzero));
        }
    }

    #[test]
    fn nonce_codec_is_canonical() {
        for value in [0, 1, 0x7f, 0x80, 0x3fff, 0x4000, u32::MAX] {
            let encoded = encode_nonce(value);
            assert_eq!(
                decode_nonce(encoded.as_slice(), 32),
                Some((value, encoded.as_slice().len()))
            );
        }
        assert_eq!(
            decode_nonce(&[0x80, 0x00], 32),
            None,
            "redundant zero group"
        );
        assert_eq!(decode_nonce(&[0x80], 32), None, "truncated");
        assert_eq!(
            decode_nonce(&[0xff, 0xff, 0xff, 0xff, 0x1f], 32),
            None,
            "exceeds u32"
        );
        assert_eq!(decode_nonce(&[0x80, 0x01], 7), None, "exceeds nonce_bits");
        assert_eq!(decode_nonce(&[0x7f], 7), Some((0x7f, 1)));
    }
}
