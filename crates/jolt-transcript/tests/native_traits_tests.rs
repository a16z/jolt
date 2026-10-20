//! Native transcript domain separation and challenge bounds.

#![cfg(all(feature = "bn254", feature = "transcript-blake2b"))]

use blake2::Blake2b512;
use jolt_field::{CanonicalBytes, Fr};
use jolt_transcript::{prover_transcript, verifier_transcript, OptimizedChallenge};
use spongefish::instantiations::Hash;

const SESSION: &[u8] = b"native-traits";
const INSTANCE: [u8; 32] = [0x77; 32];

#[test]
fn optimized_challenge_is_128_bit_truncated() {
    let mut prover = prover_transcript::<Hash<Blake2b512>>(SESSION, INSTANCE);
    let mut verifier = verifier_transcript::<Hash<Blake2b512>>(SESSION, INSTANCE, &[]);

    for challenge in [prover.challenge_128(), verifier.challenge_128()] {
        let mut bytes = [0u8; 32];
        challenge.to_bytes_le(&mut bytes);
        assert!(bytes.iter().skip(16).all(|&byte| byte == 0));
    }
}

#[test]
fn distinct_sessions_diverge() {
    let mut prover_a = prover_transcript::<Hash<Blake2b512>>(b"a", INSTANCE);
    let mut prover_b = prover_transcript::<Hash<Blake2b512>>(b"b", INSTANCE);
    let a: Fr = prover_a.challenge_128();
    let b: Fr = prover_b.challenge_128();
    assert_ne!(a, b);

    let mut verifier_a = verifier_transcript::<Hash<Blake2b512>>(b"a", INSTANCE, &[]);
    let mut verifier_b = verifier_transcript::<Hash<Blake2b512>>(b"b", INSTANCE, &[]);
    assert_ne!(verifier_a.challenge_128(), verifier_b.challenge_128());
}

#[test]
fn distinct_instances_diverge() {
    let mut prover_a = prover_transcript::<Hash<Blake2b512>>(SESSION, [0x11; 32]);
    let mut prover_b = prover_transcript::<Hash<Blake2b512>>(SESSION, [0x22; 32]);
    assert_ne!(prover_a.challenge_128(), prover_b.challenge_128());

    let mut verifier_a = verifier_transcript::<Hash<Blake2b512>>(SESSION, [0x11; 32], &[]);
    let mut verifier_b = verifier_transcript::<Hash<Blake2b512>>(SESSION, [0x22; 32], &[]);
    assert_ne!(verifier_a.challenge_128(), verifier_b.challenge_128());
}
