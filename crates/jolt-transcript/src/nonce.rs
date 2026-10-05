//! The nonce message: a `u32` in canonical unsigned LEB128.

use spongefish::{Encoding, NargDeserialize, VerificationError, VerificationResult};

/// Largest encoded nonce: a `u32` in 7-bit groups.
const NONCE_MAX_BYTES: usize = 5;

/// A prover-chosen search counter (proof of work, fold response) as a prover
/// message: canonical unsigned LEB128, so small counters cost few bytes. The
/// encoding is little-endian base-128 groups with a continuation bit and no
/// redundant trailing zero group; any other byte string is rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nonce(pub u32);

impl Nonce {
    /// The encoded byte length.
    #[must_use]
    pub const fn encoded_len(self) -> usize {
        let bits = u32::BITS - self.0.leading_zeros();
        if bits == 0 {
            1
        } else {
            bits.div_ceil(7) as usize
        }
    }
}

impl Encoding<[u8]> for Nonce {
    fn encode(&self) -> impl AsRef<[u8]> {
        let mut value = self.0;
        let mut out = Vec::with_capacity(NONCE_MAX_BYTES);
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                out.push(byte);
                return out;
            }
            out.push(byte | 0x80);
        }
    }
}

impl NargDeserialize for Nonce {
    fn deserialize_from_narg(buf: &mut &[u8]) -> VerificationResult<Self> {
        let mut value = 0u64;
        for (index, &byte) in buf.iter().take(NONCE_MAX_BYTES).enumerate() {
            let payload = byte & 0x7f;
            value |= u64::from(payload) << (7 * index);
            if byte & 0x80 == 0 {
                if index != 0 && payload == 0 {
                    return Err(VerificationError);
                }
                let value = u32::try_from(value).map_err(|_| VerificationError)?;
                *buf = buf.split_at(index + 1).1;
                return Ok(Self(value));
            }
        }
        Err(VerificationError)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test code")]
mod tests {
    use super::*;

    fn decode(bytes: &[u8]) -> Option<(u32, usize)> {
        let mut rest = bytes;
        let nonce = Nonce::deserialize_from_narg(&mut rest).ok()?;
        Some((nonce.0, bytes.len() - rest.len()))
    }

    #[test]
    fn nonce_codec_is_canonical() {
        for value in [0, 1, 0x7f, 0x80, 0x3fff, 0x4000, u32::MAX] {
            let encoded = Nonce(value).encode().as_ref().to_vec();
            assert_eq!(encoded.len(), Nonce(value).encoded_len());
            assert_eq!(decode(&encoded), Some((value, encoded.len())));
        }
        assert_eq!(decode(&[0x80, 0x00]), None, "redundant zero group");
        assert_eq!(decode(&[0x80]), None, "truncated");
        assert_eq!(decode(&[0xff, 0xff, 0xff, 0xff, 0x1f]), None, "exceeds u32");
        assert_eq!(decode(&[0x7f, 0x01]).unwrap(), (0x7f, 1));
    }
}
