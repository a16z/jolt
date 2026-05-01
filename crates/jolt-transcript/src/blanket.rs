//! Blanket implementation of [`AppendToTranscript`] for field elements.

use jolt_field::{CanonicalBytes, FixedByteSize};

use crate::transcript::{AppendToTranscript, Transcript};

/// Absorbs any field element as big-endian bytes (reversed from the canonical
/// LE layout) for EVM compatibility.
impl<F: CanonicalBytes + FixedByteSize> AppendToTranscript for F {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        let mut buf = vec![0u8; F::NUM_BYTES];
        self.to_bytes_le(&mut buf);
        buf.reverse();
        transcript.append_bytes(&buf);
    }
}
