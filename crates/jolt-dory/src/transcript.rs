//! Transcript adapter bridging `jolt-transcript::Transcript` to `dory-pcs`'s
//! transcript interface.

use dory::backends::arkworks::BN254;
use dory::primitives::arithmetic::Group as DoryGroup;
use dory::primitives::transcript::Transcript as DoryTranscript;
use dory::primitives::DorySerialize;
use jolt_field::Field;
use jolt_transcript::Transcript;

use crate::scheme::ark_to_jolt_fr;

type InnerFr = dory::backends::arkworks::ArkFr;

/// Bridges a Jolt transcript to dory-pcs's `Transcript` trait.
pub struct JoltToDoryTranscript<'a, T: Transcript> {
    transcript: &'a mut T,
}

impl<'a, T: Transcript> JoltToDoryTranscript<'a, T> {
    pub fn new(transcript: &'a mut T) -> Self {
        Self { transcript }
    }
}

impl<T: Transcript> Default for JoltToDoryTranscript<'_, T> {
    fn default() -> Self {
        panic!("JoltToDoryTranscript must be constructed via JoltToDoryTranscript::new")
    }
}

impl<T: Transcript> DoryTranscript for JoltToDoryTranscript<'_, T> {
    type Curve = BN254;

    fn append_bytes(&mut self, _label: &[u8], bytes: &[u8]) {
        self.transcript.append_bytes(bytes);
    }

    fn append_field(&mut self, _label: &[u8], x: &InnerFr) {
        let jolt_scalar = ark_to_jolt_fr(x);
        self.transcript.append_bytes(&jolt_scalar.to_bytes());
    }

    fn append_group<G: DoryGroup>(&mut self, _label: &[u8], g: &G) {
        let mut buffer = Vec::new();
        g.serialize_compressed(&mut buffer)
            .expect("group serialization should not fail");
        self.transcript.append_bytes(&buffer);
    }

    fn append_serde<S: DorySerialize>(&mut self, _label: &[u8], s: &S) {
        let mut buffer = Vec::new();
        s.serialize_compressed(&mut buffer)
            .expect("DorySerialize serialization should not fail");
        self.transcript.append_bytes(&buffer);
    }

    fn challenge_scalar(&mut self, _label: &[u8]) -> InnerFr {
        let state = *self.transcript.state();
        let fr = jolt_field::Field::from_bytes(&state);
        let _: <T as Transcript>::Challenge = self.transcript.challenge();
        crate::scheme::jolt_fr_to_ark(&fr)
    }

    fn reset(&mut self, _domain_label: &[u8]) {
        panic!("reset is not supported on JoltToDoryTranscript")
    }
}
