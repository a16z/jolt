//! Domain-separation helpers matching jolt-core's labeled transcript encoding.
//!
//! These types implement [`AppendToTranscript`] and reproduce the exact byte
//! patterns that jolt-core absorbs before payload data:
//!
//! - [`Label`] — 32-byte zero-padded label word (matches `raw_append_label`)
//! - [`LabelWithCount`] — 24-byte label + 8-byte BE count (matches `raw_append_label_with_len`)
//! - [`U64Word`] — 24 zero bytes + 8-byte BE u64 (matches `raw_append_u64`)

use crate::transcript::{AppendToTranscript, Transcript};

/// 32-byte zero-padded label word.
///
/// Matches jolt-core's `raw_append_label(label)`: the label bytes are placed
/// at the start of a 32-byte buffer, with the remainder zero-filled.
///
/// # Panics
///
/// Panics if the label exceeds 32 bytes. Silent truncation would allow two
/// distinct labels sharing a 32-byte prefix to collide in Fiat-Shamir.
pub struct Label(pub &'static [u8]);

impl AppendToTranscript for Label {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        assert!(
            self.0.len() <= 32,
            "label {:?} exceeds 32 bytes",
            core::str::from_utf8(self.0)
        );
        let mut padded = [0u8; 32];
        padded[..self.0.len()].copy_from_slice(self.0);
        transcript.append_bytes(&padded);
    }
}

/// Packed label (24 bytes) + count (8 bytes BE) in one 32-byte word.
///
/// Matches jolt-core's `raw_append_label_with_len(label, count)`: the label
/// occupies bytes `[0..24)` and the count is big-endian in `[24..32)`.
///
/// # Panics
///
/// Panics if the label exceeds 24 bytes. Silent truncation would allow two
/// distinct labels sharing a 24-byte prefix to collide in Fiat-Shamir.
pub struct LabelWithCount(pub &'static [u8], pub u64);

impl AppendToTranscript for LabelWithCount {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        assert!(
            self.0.len() <= 24,
            "label {:?} exceeds 24 bytes",
            core::str::from_utf8(self.0)
        );
        let mut packed = [0u8; 32];
        packed[..self.0.len()].copy_from_slice(self.0);
        packed[24..32].copy_from_slice(&self.1.to_be_bytes());
        transcript.append_bytes(&packed);
    }
}

/// EVM-compatible left-padded u64: 24 zero bytes + 8-byte BE value.
///
/// Matches jolt-core's `raw_append_u64(x)`.
pub struct U64Word(pub u64);

impl AppendToTranscript for U64Word {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        let mut packed = [0u8; 32];
        packed[24..].copy_from_slice(&self.0.to_be_bytes());
        transcript.append_bytes(&packed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Blake2bTranscript;
    use jolt_field::Fr;

    #[test]
    fn label_pads_to_32_bytes() {
        let mut t1 = Blake2bTranscript::<Fr>::new(b"test");
        t1.append(&Label(b"hello"));

        let mut t2 = Blake2bTranscript::<Fr>::new(b"test");
        let mut buf = [0u8; 32];
        buf[..5].copy_from_slice(b"hello");
        t2.append_bytes(&buf);

        assert_eq!(t1.state(), t2.state());
    }

    #[test]
    fn label_with_count_packs_correctly() {
        let mut t1 = Blake2bTranscript::<Fr>::new(b"test");
        t1.append(&LabelWithCount(b"sumcheck_poly", 5));

        let mut t2 = Blake2bTranscript::<Fr>::new(b"test");
        let mut buf = [0u8; 32];
        buf[..13].copy_from_slice(b"sumcheck_poly");
        buf[24..32].copy_from_slice(&5u64.to_be_bytes());
        t2.append_bytes(&buf);

        assert_eq!(t1.state(), t2.state());
    }

    #[test]
    fn u64_word_left_pads() {
        let mut t1 = Blake2bTranscript::<Fr>::new(b"test");
        t1.append(&U64Word(42));

        let mut t2 = Blake2bTranscript::<Fr>::new(b"test");
        let mut buf = [0u8; 32];
        buf[24..].copy_from_slice(&42u64.to_be_bytes());
        t2.append_bytes(&buf);

        assert_eq!(t1.state(), t2.state());
    }
}
