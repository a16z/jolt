//! Bridges the `jolt-transcript` framework into dory-pcs's `DoryTranscript` trait.
//!
//! Prover/verifier parity within `jolt-dory` is by construction: both sides
//! traverse this adapter. Cross-framework parity with `jolt-core`'s adapter
//! or with dory-pcs's reference `Blake2bTranscript` is NOT guaranteed —
//! `jolt-transcript` prepends a per-absorb domain-tag byte and squeezes
//! 16-byte challenges, neither of which the other frameworks do.

#![expect(
    clippy::expect_used,
    reason = "transcript serialization failures are fatal"
)]

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

impl<T: Transcript<Challenge = Fr>> DoryTranscript for JoltToDoryTranscript<'_, T> {
    type Curve = BN254;

    fn append_bytes(&mut self, _label: &[u8], bytes: &[u8]) {
        self.transcript
            .append(&LabelWithCount(b"dory_bytes", bytes.len() as u64));
        self.transcript.append_bytes(bytes);
    }

    fn append_field(&mut self, _label: &[u8], x: &ArkFr) {
        let jolt_scalar = ark_to_jolt_fr(x);
        self.transcript.append(&Label(b"dory_field"));
        jolt_scalar.append_to_transcript(self.transcript);
    }

    fn append_group<G: DoryGroup>(&mut self, _label: &[u8], g: &G) {
        let mut buffer = Vec::new();
        g.serialize_compressed(&mut buffer)
            .expect("group serialization should not fail");
        self.transcript
            .append(&LabelWithCount(b"dory_group", buffer.len() as u64));
        self.transcript.append_bytes(&buffer);
    }

    fn append_serde<S: DorySerialize>(&mut self, _label: &[u8], s: &S) {
        let mut buffer = Vec::new();
        s.serialize_compressed(&mut buffer)
            .expect("DorySerialize serialization should not fail");
        self.transcript
            .append(&LabelWithCount(b"dory_serde", buffer.len() as u64));
        self.transcript.append_bytes(&buffer);
    }

    fn challenge_scalar(&mut self, _label: &[u8]) -> ArkFr {
        let challenge: Fr = self.transcript.challenge();
        jolt_fr_to_ark(&challenge)
    }

    fn reset(&mut self, _domain_label: &[u8]) {
        unreachable!("reset is not invoked by dory-pcs and is intentionally unsupported")
    }
}
