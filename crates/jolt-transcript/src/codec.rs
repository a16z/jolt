//! Length-prefixed byte messages for byte-oriented spongefish transcripts.
//!
//! `BytesMsg` uses an eight-byte little-endian length, followed by the body.
//! Challenges use spongefish's built-in `u128` codec.

use spongefish::{Encoding, FromNarg, NargReader, VerificationError};

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

impl Encoding for BytesMsg {
    fn encode(&self) -> impl AsRef<[u8]> {
        let mut out = Vec::with_capacity(8 + self.0.len());
        out.extend_from_slice(&(self.0.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.0);
        out
    }
}

impl FromNarg for BytesMsg {
    fn from_narg(reader: &mut NargReader<'_>) -> Result<Self, VerificationError> {
        reader.read_with(|reader| {
            let len_bytes = reader.take_array::<8>()?;
            let len =
                usize::try_from(u64::from_le_bytes(len_bytes)).map_err(|_| VerificationError)?;
            let body = reader.take(len)?;
            Ok(BytesMsg(body.to_vec()))
        })
    }
}

#[cfg(test)]
#[expect(clippy::indexing_slicing, reason = "tests index fixture data")]
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
        let mut reader = NargReader::new(&narg);
        let result = BytesMsg::from_narg(&mut reader);
        assert!(result.is_err());
        assert!(reader.is_poisoned());
        assert!(reader.read::<u8>().is_err());
    }

    #[test]
    fn bytes_msg_narg_rejects_oversized_length() {
        let mut narg = Vec::new();
        narg.extend_from_slice(&u64::MAX.to_le_bytes());
        let mut reader = NargReader::new(&narg);
        let result = BytesMsg::from_narg(&mut reader);
        assert!(result.is_err());
        assert!(reader.is_poisoned());
        assert!(reader.read::<u8>().is_err());
    }

    #[test]
    // A representable length that exceeds the remaining buffer must be
    // rejected by NargReader::take BEFORE the payload
    // copy — the allocation is bounded by the caller-supplied NARG size. If the
    // copy ever moves ahead of the bounds check, this test attempts a 1 TiB
    // allocation and aborts.
    fn rejects_huge_representable_length_without_allocating() {
        let mut narg = Vec::new();
        narg.extend_from_slice(&(1u64 << 40).to_le_bytes());
        narg.extend_from_slice(&[0u8; 16]);
        let mut reader = NargReader::new(&narg);
        assert!(BytesMsg::from_narg(&mut reader).is_err());
        assert!(reader.is_poisoned());
        assert!(reader.read::<u8>().is_err());
    }

    #[test]
    // On 32-bit this exercises the usize::try_from rejection path.
    // On 64-bit it exercises the length-vs-remaining rejection with a
    // length near usize::MAX.
    fn rejects_non_representable_lengths() {
        #[cfg(target_pointer_width = "32")]
        let len: u64 = (u32::MAX as u64) + 1;

        #[cfg(target_pointer_width = "64")]
        let len: u64 = u64::MAX;

        let mut narg = Vec::new();
        narg.extend_from_slice(&len.to_le_bytes());

        let mut reader = NargReader::new(&narg);
        assert!(BytesMsg::from_narg(&mut reader).is_err());
        assert!(reader.is_poisoned());
        assert!(reader.read::<u8>().is_err());
    }
}
