//! Transcript adapter bridging jolt-transcript to dory-pcs.
//!
//! The bridging must produce the exact same transcript state as jolt-core's
//! `JoltToDoryTranscript` (in `jolt-core/src/poly/commitment/dory/wrappers.rs`).
//! jolt-core's adapter uses labeled methods (e.g. `append_bytes(b"dory_bytes", data)`),
//! which internally absorb a 32-byte label/length header before the payload.
//! We reproduce those headers here using jolt-transcript's `Label` and `LabelWithCount`.

use dory::backends::arkworks::BN254;
use dory::primitives::arithmetic::Group as DoryGroup;
use dory::primitives::transcript::Transcript as DoryTranscript;
use dory::primitives::DorySerialize;
use jolt_field::Fr;
use jolt_transcript::domain::{Label, LabelWithCount};
use jolt_transcript::{AppendToTranscript, Transcript};

use crate::scheme::{ark_to_jolt_fr, jolt_fr_to_ark, ArkFr};

pub struct JoltToDoryTranscript<'a, T: Transcript<Challenge = Fr>> {
    transcript: &'a mut T,
}

impl<'a, T: Transcript<Challenge = Fr>> JoltToDoryTranscript<'a, T> {
    pub fn new(transcript: &'a mut T) -> Self {
        Self { transcript }
    }
}

impl<T: Transcript<Challenge = Fr>> Default for JoltToDoryTranscript<'_, T> {
    fn default() -> Self {
        panic!("JoltToDoryTranscript must be constructed via JoltToDoryTranscript::new")
    }
}

impl<T: Transcript<Challenge = Fr>> DoryTranscript for JoltToDoryTranscript<'_, T> {
    type Curve = BN254;

    fn append_bytes(&mut self, _label: &[u8], bytes: &[u8]) {
        // Matches jolt-core: transcript.append_bytes(b"dory_bytes", bytes)
        // = raw_append_label_with_len(b"dory_bytes", len) + raw_append_bytes(bytes)
        self.transcript
            .append(&LabelWithCount(b"dory_bytes", bytes.len() as u64));
        self.transcript.append_bytes(bytes);
    }

    fn append_field(&mut self, _label: &[u8], x: &ArkFr) {
        // Matches jolt-core: transcript.append_scalar(b"dory_field", &jolt_scalar)
        // = raw_append_label(b"dory_field") + raw_append_scalar(&scalar)
        // raw_append_scalar serializes uncompressed, reverses for BE, then hashes.
        // Field::append_to_transcript does the same: to_bytes() + reverse + append_bytes.
        let jolt_scalar = ark_to_jolt_fr(x);
        self.transcript.append(&Label(b"dory_field"));
        jolt_scalar.append_to_transcript(self.transcript);
    }

    fn append_group<G: DoryGroup>(&mut self, _label: &[u8], g: &G) {
        // Matches jolt-core: transcript.append_bytes(b"dory_group", &buffer)
        let mut buffer = Vec::new();
        g.serialize_compressed(&mut buffer)
            .expect("group serialization should not fail");
        self.transcript
            .append(&LabelWithCount(b"dory_group", buffer.len() as u64));
        self.transcript.append_bytes(&buffer);
    }

    fn append_serde<S: DorySerialize>(&mut self, _label: &[u8], s: &S) {
        // Matches jolt-core: transcript.append_bytes(b"dory_serde", &buffer)
        let mut buffer = Vec::new();
        s.serialize_compressed(&mut buffer)
            .expect("DorySerialize serialization should not fail");
        self.transcript
            .append(&LabelWithCount(b"dory_serde", buffer.len() as u64));
        self.transcript.append_bytes(&buffer);
    }

    fn challenge_scalar(&mut self, _label: &[u8]) -> ArkFr {
        // Matches jolt-core: transcript.challenge_scalar::<Fr>()
        // Both squeeze 16 bytes, reverse, interpret as field element.
        let challenge: Fr = self.transcript.challenge();
        jolt_fr_to_ark(&challenge)
    }

    fn reset(&mut self, _domain_label: &[u8]) {
        panic!("reset is not supported on JoltToDoryTranscript")
    }
}
