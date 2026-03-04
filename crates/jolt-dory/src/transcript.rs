//! Transcript adapter bridging `jolt-transcript::Transcript` to `dory-pcs`'s
//! transcript interface.
//!
//! The `dory-pcs` crate defines its own `Transcript` trait for Fiat-Shamir
//! challenges. This module provides [`JoltToDoryTranscript`], a wrapper that
//! delegates all dory-pcs transcript operations to an underlying Jolt transcript.

use ark_serialize::CanonicalSerialize;
use dory::backends::arkworks::BN254;
use dory::primitives::arithmetic::Group as DoryGroup;
use dory::primitives::transcript::Transcript as DoryTranscript;
use dory::primitives::DorySerialize;
use jolt_transcript::Transcript;

use crate::types::ark_to_jolt_fr;

type InnerFr = dory::backends::arkworks::ArkFr;

/// Bridges a Jolt transcript to dory-pcs's `Transcript` trait.
///
/// Holds a mutable reference to the caller's Jolt transcript and forwards
/// all absorb/challenge operations, ensuring the Fiat-Shamir state stays
/// synchronized between Jolt and dory-pcs.
pub struct JoltToDoryTranscript<'a, T: Transcript> {
    transcript: &'a mut T,
}

impl<'a, T: Transcript> JoltToDoryTranscript<'a, T> {
    /// Wraps an existing Jolt transcript for use with dory-pcs.
    pub fn new(transcript: &'a mut T) -> Self {
        Self { transcript }
    }
}

impl<T: Transcript> Default for JoltToDoryTranscript<'_, T> {
    fn default() -> Self {
        // dory-pcs requires Default but we never use a default-constructed transcript.
        panic!("JoltToDoryTranscript must be constructed via JoltToDoryTranscript::new")
    }
}

impl<T: Transcript> DoryTranscript for JoltToDoryTranscript<'_, T> {
    type Curve = BN254;

    /// Forwards raw bytes to the Jolt transcript, ignoring dory-pcs labels
    /// since Jolt transcripts use implicit domain separation via round counters.
    fn append_bytes(&mut self, _label: &[u8], bytes: &[u8]) {
        self.transcript.append_bytes(bytes);
    }

    fn append_field(&mut self, _label: &[u8], x: &InnerFr) {
        let jolt_scalar = ark_to_jolt_fr(x);
        let mut buf = Vec::new();
        jolt_scalar
            .serialize_compressed(&mut buf)
            .expect("field serialization should not fail");
        self.transcript.append_bytes(&buf);
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

    /// Squeezes a challenge from the Jolt transcript's current state and
    /// converts it to the dory-pcs field type (`ark_bn254::Fr`), then
    /// advances the transcript so subsequent calls produce fresh challenges.
    fn challenge_scalar(&mut self, _label: &[u8]) -> InnerFr {
        let state = *self.transcript.state();
        let fr = jolt_field::Field::from_bytes(&state);
        let _: <T as Transcript>::Challenge = self.transcript.challenge();
        crate::types::jolt_fr_to_ark(&fr)
    }

    fn reset(&mut self, _domain_label: &[u8]) {
        panic!("reset is not supported on JoltToDoryTranscript")
    }
}
