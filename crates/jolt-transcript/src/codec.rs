//! Message encodings for byte-oriented spongefish transcripts.
//!
//! [`BytesMsg`] is a byte string with an eight-byte little-endian length,
//! followed by the body. [`Label`], [`LabelWithCount`] and [`U64Word`] are the
//! 32-byte words that name what the next messages are.

use spongefish::{Encoding, FromNarg, NargReader, VerificationError};

/// Maximum label length in bytes accepted by [`Label`].
pub const MAX_LABEL_LEN: usize = 32;

/// Length-prefixed byte string, owned (`Vec<u8>`, the default) or borrowed.
/// The 8-byte LE length keeps `BytesMsg(a) ; BytesMsg(b)` distinguishable
/// from `BytesMsg(a||b)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytesMsg<B = Vec<u8>>(pub B);

impl<B: AsRef<[u8]>> BytesMsg<B> {
    /// Returns the inner bytes.
    pub fn as_slice(&self) -> &[u8] {
        self.0.as_ref()
    }
}

impl From<Vec<u8>> for BytesMsg {
    fn from(v: Vec<u8>) -> Self {
        Self(v)
    }
}

impl<B: AsRef<[u8]>> Encoding for BytesMsg<B> {
    fn encode(&self) -> impl AsRef<[u8]> {
        let body = self.0.as_ref();
        let mut out = Vec::with_capacity(8 + body.len());
        out.extend_from_slice(&(body.len() as u64).to_le_bytes());
        out.extend_from_slice(body);
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

/// 32-byte zero-padded label word.
///
/// Labels are protocol constants; encoding one longer than
/// [`MAX_LABEL_LEN`] bytes panics.
#[derive(Clone, Copy, Debug)]
pub struct Label(pub &'static [u8]);

impl Encoding for Label {
    fn encode(&self) -> impl AsRef<[u8]> {
        assert!(
            self.0.len() <= MAX_LABEL_LEN,
            "label {:?} exceeds 32 bytes",
            core::str::from_utf8(self.0)
        );
        let mut padded = [0u8; 32];
        let (head, _) = padded.split_at_mut(self.0.len());
        head.copy_from_slice(self.0);
        padded
    }
}

/// Packed label (24 bytes) + count (8-byte big-endian) in one 32-byte word.
///
/// Labels are protocol constants; encoding one longer than 24 bytes panics.
#[derive(Clone, Copy, Debug)]
pub struct LabelWithCount(pub &'static [u8], pub u64);

impl Encoding for LabelWithCount {
    fn encode(&self) -> impl AsRef<[u8]> {
        assert!(
            self.0.len() <= 24,
            "label {:?} exceeds 24 bytes",
            core::str::from_utf8(self.0)
        );
        let mut packed = [0u8; 32];
        let (head, tail) = packed.split_at_mut(24);
        let (label, _) = head.split_at_mut(self.0.len());
        label.copy_from_slice(self.0);
        tail.copy_from_slice(&self.1.to_be_bytes());
        packed
    }
}

/// EVM-compatible left-padded u64: 24 zero bytes + 8-byte BE value.
#[derive(Clone, Copy, Debug)]
pub struct U64Word(pub u64);

impl Encoding for U64Word {
    fn encode(&self) -> impl AsRef<[u8]> {
        let mut packed = [0u8; 32];
        let (_, value) = packed.split_at_mut(24);
        value.copy_from_slice(&self.0.to_be_bytes());
        packed
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
    fn borrowed_and_owned_bytes_encode_alike() {
        let owned = BytesMsg(vec![5u8, 6, 7]);
        let borrowed = BytesMsg(&[5u8, 6, 7][..]);
        assert_eq!(owned.encode().as_ref(), borrowed.encode().as_ref());
    }

    #[test]
    fn label_words_pad_and_pack() {
        let label = Label(b"abc");
        let encoded = label.encode();
        assert_eq!(&encoded.as_ref()[..3], b"abc");
        assert!(encoded.as_ref()[3..].iter().all(|&byte| byte == 0));

        let counted = LabelWithCount(b"xy", 0x0102);
        let encoded = counted.encode();
        assert_eq!(&encoded.as_ref()[..2], b"xy");
        assert_eq!(&encoded.as_ref()[24..], &0x0102u64.to_be_bytes());

        let word = U64Word(7);
        let encoded = word.encode();
        assert!(encoded.as_ref()[..31].iter().all(|&byte| byte == 0));
        assert_eq!(encoded.as_ref()[31], 7);
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
