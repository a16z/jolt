//! Native transcript domain separation and challenge bounds.

#![cfg(all(feature = "bn254", feature = "transcript-blake2b"))]

use jolt_field::{CanonicalBytes, Fr};
use jolt_transcript::{prover_transcript, verifier_transcript, OptimizedChallenge};
use spongefish::instantiations::Blake2b512;

const SESSION: &[u8] = b"native-traits";
const INSTANCE: [u8; 32] = [0x77; 32];

#[test]
fn optimized_challenge_is_128_bit_truncated() {
    let mut prover = prover_transcript(SESSION, INSTANCE, Blake2b512::default());
    let mut verifier = verifier_transcript(SESSION, INSTANCE, Blake2b512::default(), &[]);

    for challenge in [prover.challenge_128(), verifier.challenge_128()] {
        let mut bytes = [0u8; 32];
        challenge.to_bytes_le(&mut bytes);
        assert!(bytes.iter().skip(16).all(|&byte| byte == 0));
    }
}

#[test]
fn distinct_sessions_diverge() {
    let mut prover_a = prover_transcript(b"a", INSTANCE, Blake2b512::default());
    let mut prover_b = prover_transcript(b"b", INSTANCE, Blake2b512::default());
    let a: Fr = prover_a.challenge_128();
    let b: Fr = prover_b.challenge_128();
    assert_ne!(a, b);

    let mut verifier_a = verifier_transcript(b"a", INSTANCE, Blake2b512::default(), &[]);
    let mut verifier_b = verifier_transcript(b"b", INSTANCE, Blake2b512::default(), &[]);
    assert_ne!(verifier_a.challenge_128(), verifier_b.challenge_128());
}

#[test]
fn distinct_instances_diverge() {
    let mut prover_a = prover_transcript(SESSION, [0x11; 32], Blake2b512::default());
    let mut prover_b = prover_transcript(SESSION, [0x22; 32], Blake2b512::default());
    assert_ne!(prover_a.challenge_128(), prover_b.challenge_128());

    let mut verifier_a = verifier_transcript(SESSION, [0x11; 32], Blake2b512::default(), &[]);
    let mut verifier_b = verifier_transcript(SESSION, [0x22; 32], Blake2b512::default(), &[]);
    assert_ne!(verifier_a.challenge_128(), verifier_b.challenge_128());
}
