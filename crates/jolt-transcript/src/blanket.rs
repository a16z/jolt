//! Blanket implementations of [`AppendToTranscript`] for common types.

use crate::transcript::{AppendToTranscript, Transcript};

impl AppendToTranscript for [u8] {
    #[inline]
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append_bytes(self);
    }
}

impl<const N: usize> AppendToTranscript for [u8; N] {
    #[inline]
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append_bytes(self);
    }
}

/// Absorbs as 8 big-endian bytes.
impl AppendToTranscript for u64 {
    #[inline]
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append_bytes(&self.to_be_bytes());
    }
}

/// Absorbs as 16 big-endian bytes.
impl AppendToTranscript for u128 {
    #[inline]
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append_bytes(&self.to_be_bytes());
    }
}

/// Absorbs as 8 big-endian bytes (cast to `u64`).
impl AppendToTranscript for usize {
    #[inline]
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append_bytes(&(*self as u64).to_be_bytes());
    }
}
