//! Blanket implementation of [`AppendToTranscript`] for field elements.

use jolt_field::Field;

use crate::transcript::{AppendToTranscript, Transcript};

/// Absorbs any field element as big-endian bytes (reversed from the canonical
/// LE layout) for EVM compatibility.
impl<F: Field> AppendToTranscript for F {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        let mut buf = self.to_bytes();
        buf.reverse();
        transcript.append_bytes(&buf);
    }
}
