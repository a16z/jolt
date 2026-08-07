//! The legacy labeled-append vocabulary is consensus-critical: provers and
//! verifiers absorb commitments through these exact 32-byte framings. Each
//! helper is pinned against a manually packed `append_bytes` equivalent on
//! the same sponge, so any framing change shows up as a state divergence.

#![cfg(feature = "transcript-blake2b")]

use jolt_field::{Fr, FromPrimitiveInt};
use jolt_transcript::{
    append_length_prefixed, AppendToTranscript, Blake2bTranscript, Label, LabelWithCount,
    Transcript, U64Word,
};

type T = Blake2bTranscript<Fr>;

fn state_after(f: impl FnOnce(&mut T)) -> [u8; 32] {
    let mut transcript = T::new(b"legacy-facade");
    f(&mut transcript);
    transcript.state()
}

#[test]
fn label_is_a_zero_padded_32_byte_word() {
    let framed = state_after(|t| t.append(&Label(b"tag")));
    let mut manual = [0u8; 32];
    manual[..3].copy_from_slice(b"tag");
    assert_eq!(framed, state_after(|t| t.append_bytes(&manual)));
}

#[test]
fn label_with_count_packs_label_and_big_endian_count() {
    let framed = state_after(|t| t.append(&LabelWithCount(b"claims", 300)));
    let mut manual = [0u8; 32];
    manual[..6].copy_from_slice(b"claims");
    manual[24..].copy_from_slice(&300u64.to_be_bytes());
    assert_eq!(framed, state_after(|t| t.append_bytes(&manual)));
}

#[test]
fn u64_word_is_a_left_padded_big_endian_word() {
    let framed = state_after(|t| t.append(&U64Word(0x0102_0304)));
    let mut manual = [0u8; 32];
    manual[24..].copy_from_slice(&0x0102_0304u64.to_be_bytes());
    assert_eq!(framed, state_after(|t| t.append_bytes(&manual)));
}

#[test]
#[should_panic(expected = "exceeds 32 bytes")]
fn label_over_32_bytes_panics() {
    let _ = state_after(|t| t.append(&Label(&[b'x'; 33])));
}

#[test]
#[should_panic(expected = "exceeds 24 bytes")]
fn counted_label_over_24_bytes_panics() {
    let _ = state_after(|t| t.append(&LabelWithCount(&[b'x'; 25], 1)));
}

#[test]
fn append_labeled_is_label_then_value() {
    let value = Fr::from_u64(77);
    let combined = state_after(|t| t.append_labeled(b"opening", &value));
    let manual = state_after(|t| {
        t.append(&Label(b"opening"));
        t.append(&value);
    });
    assert_eq!(combined, manual);
}

struct CountedPayload(Vec<u8>);

impl AppendToTranscript for CountedPayload {
    fn append_to_transcript<Tr: Transcript>(&self, transcript: &mut Tr) {
        transcript.append_bytes(&self.0);
    }

    fn transcript_payload_len(&self) -> Option<u64> {
        Some(self.0.len() as u64)
    }
}

/// `append_length_prefixed` must pick `LabelWithCount` framing exactly when
/// the payload exposes a transcript length, and plain `Label` framing
/// otherwise — mixing the two up would silently change the Fiat-Shamir
/// stream while both sides stay consistent.
#[test]
fn length_prefixed_append_framing_follows_the_payload_length() {
    let field_value = Fr::from_u64(9);
    assert_eq!(
        AppendToTranscript::transcript_payload_len(&field_value),
        None,
        "field elements are fixed-width and expose no payload length"
    );
    let unlabeled = state_after(|t| append_length_prefixed(t, b"value", &field_value));
    let manual_unlabeled = state_after(|t| {
        t.append(&Label(b"value"));
        t.append(&field_value);
    });
    assert_eq!(unlabeled, manual_unlabeled);

    let payload = CountedPayload(vec![1, 2, 3]);
    assert_eq!(payload.transcript_payload_len(), Some(3));
    let counted = state_after(|t| append_length_prefixed(t, b"blob", &payload));
    let manual_counted = state_after(|t| {
        t.append(&LabelWithCount(b"blob", 3));
        t.append_bytes(&[1, 2, 3]);
    });
    assert_eq!(counted, manual_counted);
    assert_ne!(
        counted, unlabeled,
        "the two framings must be distinguishable"
    );
}
