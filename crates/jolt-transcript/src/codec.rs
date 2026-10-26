//! Message encodings for byte-oriented spongefish transcripts.
//!
//! Every absorbed message starts and ends on an eight-byte word boundary.
//! [`BytesMsg`] is a byte string absorbed as an eight-byte little-endian
//! length, the body, then zero padding to the next word. Its NARG form is
//! the length and the body with no padding: [`NargMessage`] sends it so.
//! Counts and other integers are bare `u64` words, which spongefish encodes
//! little-endian. [`protocol_tag`] pads a protocol name to the 32-byte tag
//! that opens a sub-protocol.

use spongefish::{Encoding, FromNarg, NargReader, VerificationError};

/// Bytes in one absorbed word.
pub const WORD_BYTES: usize = 8;

/// Bytes in a protocol tag: four words.
pub const TAG_BYTES: usize = 32;

/// Zero bytes that pad `len` bytes to the next word.
#[must_use]
pub const fn word_padding(len: usize) -> usize {
    (WORD_BYTES - len % WORD_BYTES) % WORD_BYTES
}

/// `name` zero-padded to a 32-byte protocol tag.
///
/// A tag names a protocol and its version and opens that protocol's part of
/// the transcript. It has a fixed width, so it carries no length. Tags are
/// constants: a name longer than 32 bytes fails const evaluation.
///
/// The zero padding makes names that differ only in trailing NUL bytes, such
/// as `b"x"` and `b"x\0"`, the same tag. Pass a fixed constant name with no
/// NUL bytes; never build a tag from runtime or prover-supplied data.
#[must_use]
pub const fn protocol_tag(name: &[u8]) -> [u8; TAG_BYTES] {
    assert!(name.len() <= TAG_BYTES, "protocol tag exceeds 32 bytes");
    let mut tag = [0u8; TAG_BYTES];
    let mut i = 0;
    while i < name.len() {
        #[expect(
            clippy::indexing_slicing,
            reason = "i < name.len() <= 32 == tag.len(); const-evaluated, so an out-of-bounds index fails compilation"
        )]
        {
            tag[i] = name[i];
        }
        i += 1;
    }
    tag
}

/// Length-prefixed byte string, owned (`Vec<u8>`, the default) or borrowed.
///
/// Absorbed as an eight-byte LE length, the body, then zero padding to the
/// next word. The length keeps `BytesMsg(a) ; BytesMsg(b)` distinguishable
/// from `BytesMsg(a||b)` and fixes the number of padding bytes. In the NARG
/// string it is the length and the body, with no padding.
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
        let pad = word_padding(body.len());
        let mut out = Vec::with_capacity(WORD_BYTES + body.len() + pad);
        out.extend_from_slice(&(body.len() as u64).to_le_bytes());
        out.extend_from_slice(body);
        out.resize(out.len() + pad, 0);
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

/// A prover message: absorbed as its [`Encoding`], and written to the NARG
/// string by [`write_narg`](Self::write_narg).
///
/// The two forms differ only where the absorbed form pads to a word. The NARG
/// form must be the one the message's [`FromNarg`] reads, so the verifier
/// parses it and absorbs the same [`Encoding`] as the prover.
pub trait NargMessage: Encoding {
    /// Appends the message's NARG form to `narg`.
    fn write_narg(&self, narg: &mut Vec<u8>);
}

impl<B: AsRef<[u8]>> NargMessage for BytesMsg<B> {
    fn write_narg(&self, narg: &mut Vec<u8>) {
        let body = self.0.as_ref();
        narg.extend_from_slice(&(body.len() as u64).to_le_bytes());
        narg.extend_from_slice(body);
    }
}

impl NargMessage for u64 {
    fn write_narg(&self, narg: &mut Vec<u8>) {
        narg.extend_from_slice(&self.to_le_bytes());
    }
}

impl<const N: usize> NargMessage for [u8; N] {
    fn write_narg(&self, narg: &mut Vec<u8>) {
        narg.extend_from_slice(self);
    }
}

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    reason = "tests index fixture data"
)]
mod tests {
    use super::*;

    #[test]
    fn bytes_msg_is_length_prefixed_and_word_padded() {
        let m = BytesMsg(vec![1, 2, 3, 4]);
        let enc = m.encode();
        let bytes = enc.as_ref();
        assert_eq!(bytes.len(), 8 + 8);
        assert_eq!(&bytes[..8], &4u64.to_le_bytes());
        assert_eq!(&bytes[8..12], &[1, 2, 3, 4]);
        assert_eq!(&bytes[12..], &[0; 4]);
        for len in 0..=17 {
            let encoded = BytesMsg(vec![0xA5u8; len]).encode().as_ref().to_vec();
            assert_eq!(encoded.len(), 8 + len.div_ceil(8) * 8, "length {len}");
            assert_eq!(&encoded[..8], &(len as u64).to_le_bytes());
            assert!(encoded[8 + len..].iter().all(|&byte| byte == 0));
        }
    }

    #[test]
    fn bytes_msg_narg_form_has_no_padding_and_round_trips() {
        let m = BytesMsg(vec![9u8, 8, 7]);
        let mut narg = Vec::new();
        m.write_narg(&mut narg);
        assert_eq!(narg.len(), 8 + 3);
        assert_eq!(&narg[..8], &3u64.to_le_bytes());
        assert_eq!(&narg[8..], &[9, 8, 7]);
        let mut reader = NargReader::new(&narg);
        assert_eq!(BytesMsg::from_narg(&mut reader).unwrap(), m);
        assert!(reader.read::<u8>().is_err());
    }

    #[test]
    fn borrowed_and_owned_bytes_encode_alike() {
        let owned = BytesMsg(vec![5u8, 6, 7]);
        let borrowed = BytesMsg(&[5u8, 6, 7][..]);
        assert_eq!(owned.encode().as_ref(), borrowed.encode().as_ref());
    }

    #[test]
    fn protocol_tags_pad_to_four_words() {
        let tag = protocol_tag(b"abc");
        assert_eq!(tag.len(), 32);
        assert_eq!(&tag[..3], b"abc");
        assert!(tag[3..].iter().all(|&byte| byte == 0));
        let full = [0x41u8; 32];
        assert_eq!(
            protocol_tag(&full),
            full,
            "a 32-byte name is passed through"
        );
        assert_eq!(word_padding(0), 0);
        assert_eq!(word_padding(1), 7);
        assert_eq!(word_padding(8), 0);
        assert_eq!(word_padding(13), 3);
    }

    #[test]
    fn bytes_msg_narg_rejects_truncation() {
        let m = BytesMsg(vec![9, 8, 7]);
        let mut narg: Vec<u8> = Vec::new();
        m.write_narg(&mut narg);
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
