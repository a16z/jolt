//! Tests for the legacy digest-chained transcript (`DigestTranscript`).
//!
//! `LegacyBlake2bTranscript` exists solely for byte compatibility with
//! `jolt-prover-legacy`'s chained-digest transcript, so beyond the shared
//! `transcript_tests!` behavioral suite, every state transition is checked
//! against an independently recomputed Blake2b-256 hash chain:
//! `state0 = H(label || 0^pad)`, append is `H(state || round || payload)`,
//! squeeze is `H(state || round)` with the 4-byte big-endian round counter
//! right-aligned in a 32-byte word.

#![cfg(feature = "transcript-blake2b")]

mod common;

use blake2::{digest::consts::U32, Blake2b, Digest};
use jolt_field::{CanonicalEncoding, Fr};
use jolt_transcript::LegacyBlake2bTranscript;

type LegacyB2b = LegacyBlake2bTranscript<Fr>;
type Blake2b256 = Blake2b<U32>;

transcript_tests!(LegacyB2b);

fn manual_step(state: &[u8; 32], round: u32, payload: &[u8]) -> [u8; 32] {
    let mut round_word = [0u8; 32];
    round_word[28..].copy_from_slice(&round.to_be_bytes());
    let mut hasher = Blake2b256::new();
    hasher.update(state);
    hasher.update(round_word);
    hasher.update(payload);
    hasher.finalize().into()
}

fn manual_label_state(label: &[u8]) -> [u8; 32] {
    let mut padded = [0u8; 32];
    padded[..label.len()].copy_from_slice(label);
    Blake2b256::digest(padded).into()
}

#[test]
fn appends_and_challenges_follow_the_legacy_hash_chain() {
    let mut transcript = LegacyB2b::new(b"legacy-chain");
    let mut expected = manual_label_state(b"legacy-chain");

    transcript.append_bytes(b"first message");
    expected = manual_step(&expected, 0, b"first message");
    assert_eq!(
        transcript.state(),
        expected,
        "append must chain the payload"
    );

    transcript.append_bytes(&[]);
    expected = manual_step(&expected, 1, &[]);
    assert_eq!(
        transcript.state(),
        expected,
        "an empty append must still advance the round counter"
    );

    // The first two challenges after an absorb squeeze
    // `Fr::challenge_width(1) = Fr::challenge_width(2) = 52` bytes: two chunks.
    for (draw, round) in [
        (LegacyB2b::challenge as fn(&mut LegacyB2b) -> Fr, 2),
        (LegacyB2b::challenge_scalar, 4),
    ] {
        let challenge = draw(&mut transcript);
        let first = manual_step(&expected, round, &[]);
        expected = manual_step(&first, round + 1, &[]);
        assert_eq!(transcript.state(), expected, "challenge must chain state");
        let squeezed = [first.as_slice(), &expected[..20]].concat();
        assert_eq!(
            challenge,
            Fr::from_bytes_le_reduced(&squeezed),
            "challenge must reduce the 52 squeezed bytes"
        );
    }
}

#[test]
fn clone_replays_identically_until_inputs_diverge() {
    let mut original = LegacyB2b::new(b"legacy-clone");
    original.append_bytes(b"shared prefix");
    let mut cloned = original.clone();
    assert_eq!(original.state(), cloned.state());
    assert_eq!(
        original.challenge(),
        cloned.challenge(),
        "clones must squeeze identical challenges"
    );

    original.append_bytes(b"left");
    cloned.append_bytes(b"right");
    assert_ne!(
        original.state(),
        cloned.state(),
        "divergent appends must yield divergent states"
    );
}
