//! Local codecs for absorbing / decoding Jolt-native messages over a
//! byte-oriented spongefish sponge.
//!
//! Field-element codecs come from spongefish's `ark-ff` feature
//! (`spongefish::Encoding<[u8]>` is implemented for every `ark_ff::Fp<C, N>`
//! using big-endian canonical encoding per RFC8017). 128-bit-truncated
//! challenges decode through spongefish's built-in `u128` codec directly;
//! see [`crate::prover::OptimizedChallenge`].
//!
//! This module keeps only `BytesMsg`, the length-prefixed byte string
//! framing that spongefish 0.6 does not provide.

use spongefish::{Encoding, NargDeserialize, VerificationError, VerificationResult};

pub(crate) fn encode_bytes_frame(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + bytes.len());
    out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(bytes);
    out
}

/// Length-prefixed byte string. 8-byte LE length keeps `BytesMsg(a) ; BytesMsg(b)`
/// distinguishable from `BytesMsg(a||b)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytesMsg(pub Vec<u8>);

impl Encoding<[u8]> for BytesMsg {
    fn encode(&self) -> impl AsRef<[u8]> {
        encode_bytes_frame(&self.0)
    }
}

impl NargDeserialize for BytesMsg {
    fn deserialize_from_narg(buf: &mut &[u8]) -> VerificationResult<Self> {
        let Some((len_bytes, rest)) = buf.split_first_chunk::<8>() else {
            return Err(VerificationError);
        };
        let len = usize::try_from(u64::from_le_bytes(*len_bytes)).map_err(|_| VerificationError)?;
        let body = rest.get(..len).ok_or(VerificationError)?.to_vec();
        *buf = &rest[len..];
        Ok(BytesMsg(body))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    // On 32-bit this exercises the usize::try_from rejection path.
    // On 64-bit it exercises the checked_add overflow path.
    fn rejects_non_representable_lengths() {
        #[cfg(target_pointer_width = "32")]
        let len: u64 = (u32::MAX as u64) + 1;

        #[cfg(target_pointer_width = "64")]
        let len: u64 = u64::MAX;

        let mut narg = Vec::new();
        narg.extend_from_slice(&len.to_le_bytes());

        let mut cursor: &[u8] = &narg;
        let before = cursor.len();
        assert!(BytesMsg::deserialize_from_narg(&mut cursor).is_err());
        assert_eq!(cursor.len(), before, "cursor must not advance on error");
    }
}
