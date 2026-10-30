//! Native transcript domain separation.

#![cfg(all(feature = "bn254", feature = "transcript-blake2b"))]
#![expect(clippy::unwrap_used, reason = "tests")]

use blake2::Blake2b512;
use jolt_field::Fr;
use jolt_transcript::FieldTranscript;
use jolt_transcript::{protocol_tag, prover_transcript, verifier_transcript, TAG_BYTES};
use spongefish::instantiations::Hash;

const SESSION: [u8; TAG_BYTES] = protocol_tag(b"native-traits");
const INSTANCE: [u8; 32] = [0x77; 32];

fn prover_challenge(session: [u8; TAG_BYTES], instance: [u8; 32]) -> Vec<Fr> {
    let mut prover = prover_transcript::<Hash<Blake2b512>>(&session, &instance);
    FieldTranscript::<Fr>::challenge_vector(&mut prover, 1).unwrap()
}

fn verifier_challenge(session: [u8; TAG_BYTES], instance: [u8; 32]) -> Vec<Fr> {
    let mut verifier = verifier_transcript::<Hash<Blake2b512>>(&session, &instance, &[]);
    FieldTranscript::<Fr>::challenge_vector(&mut verifier, 1).unwrap()
}

#[test]
fn distinct_sessions_diverge() {
    assert_ne!(
        prover_challenge(protocol_tag(b"a"), INSTANCE),
        prover_challenge(protocol_tag(b"b"), INSTANCE)
    );
    assert_ne!(
        verifier_challenge(protocol_tag(b"a"), INSTANCE),
        verifier_challenge(protocol_tag(b"b"), INSTANCE)
    );
}

#[test]
fn distinct_instances_diverge() {
    assert_ne!(
        prover_challenge(SESSION, [0x11; 32]),
        prover_challenge(SESSION, [0x22; 32])
    );
    assert_ne!(
        verifier_challenge(SESSION, [0x11; 32]),
        verifier_challenge(SESSION, [0x22; 32])
    );
}
