//! Local codecs for absorbing / decoding Jolt-native messages over a
//! byte-oriented spongefish sponge.
//!
//! Spongefish ships optional arkworks codec features; we don't enable them
//! because Jolt patches `ark-ff` / `ark-serialize` to a fork. These local
//! codecs are injective and prefix-free.

use jolt_field::{CanonicalBytes, FixedByteSize, FromPrimitiveInt, ReducingBytes};
use spongefish::{Decoding, Encoding, NargDeserialize, VerificationError, VerificationResult};

const FR_TRUNCATED_BYTES: usize = 16;
/// Bytes drawn per full-field challenge. 64 bytes mod a ≤254-bit field
/// modulus is within `2^{-130}` statistical distance of uniform. Tuned for
/// BN254; safe for any field up to that width.
const FR_UNIFORM_BYTES: usize = 64;

/// Wraps a field element for absorption / decoding as little-endian bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldEl<F>(pub F);

impl<F> From<F> for FieldEl<F> {
    fn from(f: F) -> Self {
        Self(f)
    }
}

impl<F: CanonicalBytes> Encoding<[u8]> for FieldEl<F> {
    fn encode(&self) -> impl AsRef<[u8]> {
        self.0.to_bytes_le_vec()
    }
}

/// 64-byte squeeze buffer used as the [`Decoding::Repr`] for full-field
/// challenges. See `FR_UNIFORM_BYTES`.
#[derive(Clone, Copy)]
pub struct UniformFrBytes(pub [u8; FR_UNIFORM_BYTES]);

impl Default for UniformFrBytes {
    fn default() -> Self {
        Self([0u8; FR_UNIFORM_BYTES])
    }
}

impl AsMut<[u8]> for UniformFrBytes {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl<F: ReducingBytes> Decoding<[u8]> for FieldEl<F> {
    type Repr = UniformFrBytes;
    fn decode(buf: Self::Repr) -> Self {
        FieldEl(F::from_le_bytes_mod_order(&buf.0))
    }
}

impl<F: FixedByteSize + ReducingBytes> NargDeserialize for FieldEl<F> {
    fn deserialize_from_narg(buf: &mut &[u8]) -> VerificationResult<Self> {
        let n = F::NUM_BYTES;
        if buf.len() < n {
            return Err(VerificationError);
        }
        let (head, tail) = buf.split_at(n);
        *buf = tail;
        Ok(FieldEl(F::from_le_bytes_mod_order(head)))
    }
}

/// 128-bit-truncating challenge wrapper. Decodes 16 squeezed bytes via
/// `F::from_u128`. Used only as a verifier message; the `Encoding` impl
/// is the same little-endian form as [`FieldEl`] so that absorbing one of
/// these symmetrically with the other type stays a code error rather than
/// a wire-format hazard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldElOptimized<F>(pub F);

impl<F: CanonicalBytes> Encoding<[u8]> for FieldElOptimized<F> {
    fn encode(&self) -> impl AsRef<[u8]> {
        self.0.to_bytes_le_vec()
    }
}

impl<F: FromPrimitiveInt> Decoding<[u8]> for FieldElOptimized<F> {
    type Repr = [u8; FR_TRUNCATED_BYTES];
    fn decode(buf: Self::Repr) -> Self {
        FieldElOptimized(F::from_u128(u128::from_le_bytes(buf)))
    }
}

/// Length-prefixed byte string. 8-byte LE length keeps `BytesMsg(a) ; BytesMsg(b)`
/// distinguishable from `BytesMsg(a||b)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytesMsg(pub Vec<u8>);

impl BytesMsg {
    /// Returns the inner bytes.
    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }
}

impl From<Vec<u8>> for BytesMsg {
    fn from(v: Vec<u8>) -> Self {
        Self(v)
    }
}

impl Encoding<[u8]> for BytesMsg {
    fn encode(&self) -> impl AsRef<[u8]> {
        let mut out = Vec::with_capacity(8 + self.0.len());
        out.extend_from_slice(&(self.0.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.0);
        out
    }
}

impl NargDeserialize for BytesMsg {
    fn deserialize_from_narg(buf: &mut &[u8]) -> VerificationResult<Self> {
        if buf.len() < 8 {
            return Err(VerificationError);
        }
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&buf[..8]);
        let len = u64::from_le_bytes(len_bytes) as usize;
        let total = 8usize.checked_add(len).ok_or(VerificationError)?;
        if buf.len() < total {
            return Err(VerificationError);
        }
        let body = buf[8..total].to_vec();
        *buf = &buf[total..];
        Ok(BytesMsg(body))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;

    #[test]
    fn fr_le_bytes_round_trip() {
        for i in 0u64..32 {
            let f = Fr::from(i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let bytes = f.to_bytes_le_vec();
            assert_eq!(Fr::from_le_bytes_mod_order(&bytes), f);
        }
    }

    #[test]
    fn bytes_msg_is_length_prefixed() {
        let m = BytesMsg(vec![1, 2, 3, 4]);
        let enc = m.encode();
        let bytes = enc.as_ref();
        assert_eq!(bytes.len(), 8 + 4);
        assert_eq!(&bytes[..8], &4u64.to_le_bytes());
        assert_eq!(&bytes[8..], &[1, 2, 3, 4]);
    }

    #[test]
    fn bytes_msg_narg_rejects_truncation() {
        let m = BytesMsg(vec![9, 8, 7]);
        let mut narg: Vec<u8> = Vec::new();
        narg.extend_from_slice(m.encode().as_ref());
        let _ = narg.pop();
        let mut cursor: &[u8] = &narg;
        let before = cursor.len();
        let result = BytesMsg::deserialize_from_narg(&mut cursor);
        assert!(result.is_err());
        assert_eq!(cursor.len(), before, "cursor must not advance on error");
    }

    #[test]
    fn bytes_msg_narg_rejects_oversized_length() {
        let mut narg = Vec::new();
        narg.extend_from_slice(&u64::MAX.to_le_bytes());
        let mut cursor: &[u8] = &narg;
        let before = cursor.len();
        let result = BytesMsg::deserialize_from_narg(&mut cursor);
        assert!(result.is_err());
        assert_eq!(cursor.len(), before, "cursor must not advance on error");
    }

    #[test]
    fn field_el_optimized_decodes_u128() {
        let buf = 12345u128.to_le_bytes();
        let FieldElOptimized(f) = FieldElOptimized::<Fr>::decode(buf);
        assert_eq!(f, Fr::from(12345u128));
    }
}
