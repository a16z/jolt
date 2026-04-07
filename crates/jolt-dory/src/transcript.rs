//! Transcript adapter bridging jolt-transcript to dory-pcs.

use dory::backends::arkworks::BN254;
use dory::primitives::arithmetic::Group as DoryGroup;
use dory::primitives::transcript::Transcript as DoryTranscript;
use dory::primitives::DorySerialize;
use jolt_field::{Field, Fr};
use jolt_transcript::Transcript;

use crate::scheme::{jolt_fr_to_ark, ArkFr};

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
        self.transcript.append_bytes(bytes);
    }

    fn append_field(&mut self, _label: &[u8], x: &ArkFr) {
        let jolt_scalar = crate::scheme::ark_to_jolt_fr(x);
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

    fn challenge_scalar(&mut self, _label: &[u8]) -> ArkFr {
        let challenge: Fr = self.transcript.challenge();
        jolt_fr_to_ark(&challenge)
    }

    fn reset(&mut self, _domain_label: &[u8]) {
        panic!("reset is not supported on JoltToDoryTranscript")
    }
}
