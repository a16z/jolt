//! Local codecs for absorbing / decoding Jolt-native messages over a
//! byte-oriented spongefish sponge.
//!
//! Spongefish ships optional arkworks codec features; we don't enable them
//! because Jolt patches `ark-ff` / `ark-serialize` to a fork. These local
//! codecs are injective and prefix-free.

use ark_bn254::Fr;
use ark_ff::{BigInteger, PrimeField};
use spongefish::{Decoding, Encoding, NargDeserialize, VerificationError, VerificationResult};

const FR_LE_BYTES: usize = 32;
const FR_TRUNCATED_BYTES: usize = 16;
/// Bytes drawn per full-field challenge. 64 bytes mod the BN254 modulus
/// is within `2^{-130}` statistical distance of uniform.
const FR_UNIFORM_BYTES: usize = 64;

fn fr_to_le_bytes(f: &Fr) -> [u8; FR_LE_BYTES] {
    let bytes = f.into_bigint().to_bytes_le();
    debug_assert_eq!(
        bytes.len(),
        FR_LE_BYTES,
        "BN254 Fr LE serialization is fixed-width 32 bytes"
    );
    let mut out = [0u8; FR_LE_BYTES];
    out.copy_from_slice(&bytes);
    out
}

/// Wraps a BN254 `Fr` for absorption / decoding as 32 little-endian bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldEl(pub Fr);

impl From<Fr> for FieldEl {
    fn from(f: Fr) -> Self {
        Self(f)
    }
}

impl Encoding<[u8]> for FieldEl {
    fn encode(&self) -> impl AsRef<[u8]> {
        fr_to_le_bytes(&self.0)
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

impl Decoding<[u8]> for FieldEl {
    type Repr = UniformFrBytes;
    fn decode(buf: Self::Repr) -> Self {
        FieldEl(Fr::from_le_bytes_mod_order(&buf.0))
    }
}

impl NargDeserialize for FieldEl {
    fn deserialize_from_narg(buf: &mut &[u8]) -> VerificationResult<Self> {
        if buf.len() < FR_LE_BYTES {
            return Err(VerificationError);
        }
        let (head, tail) = buf.split_at(FR_LE_BYTES);
        *buf = tail;
        Ok(FieldEl(Fr::from_le_bytes_mod_order(head)))
    }
}

/// 128-bit-truncating challenge wrapper. Decodes 16 squeezed bytes via
/// `Fr::from(u128)`. Used only as a verifier message; the `Encoding` impl
/// is the same 32-byte LE form as [`FieldEl`] so that absorbing one of
/// these symmetrically with the other type stays a code error rather than
/// a wire-format hazard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldElOptimized(pub Fr);

impl Encoding<[u8]> for FieldElOptimized {
    fn encode(&self) -> impl AsRef<[u8]> {
        fr_to_le_bytes(&self.0)
    }
}

impl Decoding<[u8]> for FieldElOptimized {
    type Repr = [u8; FR_TRUNCATED_BYTES];
    fn decode(buf: Self::Repr) -> Self {
        FieldElOptimized(Fr::from(u128::from_le_bytes(buf)))
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
        if buf.len() < 8 + len {
            return Err(VerificationError);
        }
        let body = buf[8..8 + len].to_vec();
        *buf = &buf[8 + len..];
        Ok(BytesMsg(body))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fr_le_bytes_round_trip() {
        for i in 0u64..32 {
            let f = Fr::from(i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let bytes = fr_to_le_bytes(&f);
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
    fn field_el_optimized_decodes_u128() {
        let buf = 12345u128.to_le_bytes();
        let FieldElOptimized(f) = FieldElOptimized::decode(buf);
        assert_eq!(f, Fr::from(12345u128));
    }
}
