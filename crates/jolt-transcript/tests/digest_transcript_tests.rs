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

    let challenge = transcript.challenge();
    expected = manual_step(&expected, 2, &[]);
    assert_eq!(transcript.state(), expected, "challenge must chain state");
    assert_eq!(
        challenge,
        Fr::from_challenge_bytes(&expected[..16]),
        "challenge must decode the first 16 squeezed bytes"
    );

    let scalar = transcript.challenge_scalar();
    expected = manual_step(&expected, 3, &[]);
    assert_eq!(
        scalar,
        Fr::from_scalar_challenge_bytes(&expected[..16]),
        "scalar challenge must use the non-optimized decoding path"
    );
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
