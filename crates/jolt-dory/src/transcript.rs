//! Bridges the `jolt-transcript` framework into dory-pcs's `DoryTranscript` trait.
//!
//! Prover/verifier parity within `jolt-dory` is by construction: both sides
//! traverse this adapter. The surrounding Jolt transcript is responsible for
//! matching the core Fiat-Shamir byte layout before this adapter is entered.

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
        let challenge: Fr = self.transcript.challenge_scalar();
        jolt_fr_to_ark(&challenge)
    }

    fn reset(&mut self, _domain_label: &[u8]) {
        unreachable!("reset is not invoked by dory-pcs and is intentionally unsupported")
    }
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "tests unwrap infallible serialization of well-formed group elements"
    )]

    use super::*;
    use crate::scheme::jolt_fr_to_ark;
    use ark_ec::PrimeGroup;
    use dory::backends::arkworks::ArkG1;
    use jolt_field::FromPrimitiveInt;
    use jolt_transcript::Blake2bTranscript;

    // The framing words below are reconstructed from the documented layout
    // (legacy.rs `Label` / `LabelWithCount`), NOT by calling those helpers:
    // a `Label` is one 32-byte zero-padded word; a `LabelWithCount` packs the
    // label into bytes 0..24 and the count big-endian into bytes 24..32. Each
    // word and each payload is absorbed as its own `append_bytes` call.

    fn label_word(label: &[u8]) -> [u8; 32] {
        assert!(label.len() <= 32);
        let mut word = [0u8; 32];
        word[..label.len()].copy_from_slice(label);
        word
    }

    fn label_with_count_word(label: &[u8], count: u64) -> [u8; 32] {
        assert!(label.len() <= 24);
        let mut word = [0u8; 32];
        word[..label.len()].copy_from_slice(label);
        word[24..].copy_from_slice(&count.to_be_bytes());
        word
    }

    fn transcript() -> Blake2bTranscript {
        Blake2bTranscript::new(b"dory-adapter-framing")
    }

    /// `ArkFr` in scope is a type alias, which cannot be used in constructor
    /// position; this builds the dory wrapper explicitly.
    fn dory_ark_fr(inner: ark_bn254::Fr) -> ArkFr {
        dory::backends::arkworks::ArkFr(inner)
    }

    #[test]
    fn append_bytes_frames_as_dory_bytes_count_word_then_payload() {
        let payload = [0xaa, 0xbb, 0xcc, 0xdd, 0xee];

        let mut actual = transcript();
        JoltToDoryTranscript::new(&mut actual).append_bytes(b"caller-label", &payload);

        let mut expected = transcript();
        expected.append_bytes(&label_with_count_word(b"dory_bytes", 5));
        expected.append_bytes(&payload);
        assert_eq!(actual.state(), expected.state());

        // The count is load-bearing: the same payload under a wrong count
        // must diverge, otherwise the golden comparison proves nothing.
        let mut wrong_count = transcript();
        wrong_count.append_bytes(&label_with_count_word(b"dory_bytes", 4));
        wrong_count.append_bytes(&payload);
        assert_ne!(actual.state(), wrong_count.state());

        // Word and payload are separate absorptions, not one concatenated
        // buffer (the sponge length-prefixes each `append_bytes` call).
        let mut merged = transcript();
        let mut buffer = label_with_count_word(b"dory_bytes", 5).to_vec();
        buffer.extend_from_slice(&payload);
        merged.append_bytes(&buffer);
        assert_ne!(actual.state(), merged.state());
    }

    #[test]
    fn append_field_frames_as_dory_field_label_then_big_endian_scalar() {
        let mut actual = transcript();
        JoltToDoryTranscript::new(&mut actual).append_field(
            b"caller-label",
            &dory_ark_fr(ark_bn254::Fr::from(0xdead_beefu64)),
        );

        // Fr absorbs as its 32-byte big-endian canonical form: 24 zero bytes
        // then the value, reconstructed here without CanonicalBytes.
        let mut scalar_be = [0u8; 32];
        scalar_be[24..].copy_from_slice(&0xdead_beefu64.to_be_bytes());

        let mut expected = transcript();
        expected.append_bytes(&label_word(b"dory_field"));
        expected.append_bytes(&scalar_be);
        assert_eq!(actual.state(), expected.state());

        // Little-endian absorption would be an invisible-to-roundtrip bug.
        let mut little_endian = transcript();
        let mut scalar_le = [0u8; 32];
        scalar_le[..8].copy_from_slice(&0xdead_beefu64.to_le_bytes());
        little_endian.append_bytes(&label_word(b"dory_field"));
        little_endian.append_bytes(&scalar_le);
        assert_ne!(actual.state(), little_endian.state());
    }

    #[test]
    fn append_group_frames_as_dory_group_count_word_then_compressed_point() {
        let generator = ArkG1(ark_bn254::G1Projective::generator());

        let mut actual = transcript();
        JoltToDoryTranscript::new(&mut actual).append_group(b"caller-label", &generator);

        // Payload reconstructed via arkworks compressed serialization of the
        // inner point; the adapter's framing is the labeled count word.
        let mut payload = Vec::new();
        ark_serialize::CanonicalSerialize::serialize_compressed(&generator.0, &mut payload)
            .expect("compressed G1 serialization should not fail");
        assert_eq!(payload.len(), 32);

        let mut expected = transcript();
        expected.append_bytes(&label_with_count_word(b"dory_group", payload.len() as u64));
        expected.append_bytes(&payload);
        assert_eq!(actual.state(), expected.state());
    }

    #[test]
    fn append_serde_frames_as_dory_serde_count_word_then_compressed_value() {
        let value = dory_ark_fr(ark_bn254::Fr::from(42u64));

        let mut actual = transcript();
        JoltToDoryTranscript::new(&mut actual).append_serde(b"caller-label", &value);

        let mut payload = Vec::new();
        ark_serialize::CanonicalSerialize::serialize_compressed(&value.0, &mut payload)
            .expect("compressed Fr serialization should not fail");
        assert_eq!(payload.len(), 32);

        let mut expected = transcript();
        expected.append_bytes(&label_with_count_word(b"dory_serde", payload.len() as u64));
        expected.append_bytes(&payload);
        assert_eq!(actual.state(), expected.state());
    }

    /// Domain separation comes from the fixed `dory_*` labels; the caller's
    /// dory-side label is deliberately dropped by the adapter.
    #[test]
    fn caller_labels_do_not_reach_the_transcript() {
        let payload = [1u8, 2, 3];

        let mut first = transcript();
        JoltToDoryTranscript::new(&mut first).append_bytes(b"label-one", &payload);

        let mut second = transcript();
        JoltToDoryTranscript::new(&mut second).append_bytes(b"label-two", &payload);

        assert_eq!(first.state(), second.state());
        assert_ne!(first.state(), transcript().state());
    }

    /// The adapter's challenge is the Jolt transcript's scalar challenge,
    /// converted — so after identical absorptions both sides must squeeze the
    /// same scalar, and the adapter's state advances exactly like the direct
    /// transcript's.
    #[test]
    fn challenge_scalar_matches_underlying_jolt_transcript() {
        let mut adapted = transcript();
        let mut adapter = JoltToDoryTranscript::new(&mut adapted);
        adapter.append_bytes(b"caller-label", b"shared-absorption");
        let adapter_challenge = adapter.challenge_scalar(b"caller-label");

        let mut direct = transcript();
        direct.append_bytes(&label_with_count_word(b"dory_bytes", 17));
        direct.append_bytes(b"shared-absorption");
        let direct_challenge: Fr = direct.challenge_scalar();

        assert_eq!(adapter_challenge, jolt_fr_to_ark(&direct_challenge));
        assert_ne!(adapter_challenge, dory_ark_fr(ark_bn254::Fr::from(0u64)));
        assert_eq!(adapted.state(), direct.state());

        // Fr conversion sanity: the transmute-based bridge is the identity on
        // canonical values.
        assert_eq!(
            jolt_fr_to_ark(&Fr::from_u64(7)),
            dory_ark_fr(ark_bn254::Fr::from(7u64)),
        );
    }
}
