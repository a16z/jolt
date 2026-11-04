//! Tests for the Poseidon sponge.
//!
//! Poseidon's squeezed bytes are not uniform, so it draws no field
//! challenges; these tests exercise it as a byte sponge.

use jolt_transcript::{
    protocol_tag, prover_transcript, verifier_transcript, PoseidonSponge, ProverState, Transcript,
};

const SESSION: [u8; 32] = protocol_tag(b"jolt-transcript/tests");

fn squeeze_after<F: FnOnce(&mut ProverState<PoseidonSponge>)>(f: F) -> [u8; 32] {
    let mut transcript = prover_transcript::<PoseidonSponge>(&SESSION, b"Jolt");
    f(&mut transcript);
    transcript.verifier_message::<[u8; 32]>()
}

#[test]
fn test_poseidon_determinism_and_order() {
    let a = squeeze_after(|t| {
        t.append(&1u64);
        t.append(&2u64);
    });
    let b = squeeze_after(|t| {
        t.append(&1u64);
        t.append(&2u64);
    });
    let c = squeeze_after(|t| {
        t.append(&2u64);
        t.append(&1u64);
    });
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn test_poseidon_prover_and_verifier_agree() {
    let mut prover = prover_transcript::<PoseidonSponge>(&SESSION, b"Jolt");
    let mut verifier = verifier_transcript::<PoseidonSponge>(&SESSION, b"Jolt", &[]);
    prover.append_bytes(b"msg");
    verifier.append_bytes(b"msg");
    assert_eq!(
        prover.verifier_message::<[u8; 32]>(),
        verifier.verifier_message::<[u8; 32]>()
    );
}

#[cfg_attr(
    feature = "keccak256-digests",
    ignore = "pinned for SHA3-256 digests; keccak256-digests changes the session identifier"
)]
#[test]
fn test_poseidon_known_vector() {
    let got = squeeze_after(|t| t.append(&12345u64));
    // Pins the session derived from PROTOCOL_ID, the instance absorb, one
    // `u64` public message and a 32-byte squeeze.
    assert_eq!(
        hex(&got),
        "f156ab8ed8a565f069d4798debca31d19d3deed946d89ce9642a9b81c764112f",
        "Poseidon known-vector regression"
    );
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut hex, byte| {
        let _ = write!(hex, "{byte:02x}");
        hex
    })
}
