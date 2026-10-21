//! Tests for Blake2bTranscript implementation.

mod common;

use jolt_field::Fr;
use jolt_transcript::Blake2bTranscript;

type B2b = Blake2bTranscript<Fr>;

transcript_tests!(B2b);

#[test]
fn test_blake2b_known_vector() {
    use ark_ff::PrimeField;
    use jolt_transcript::Transcript;

    let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
    transcript.append_bytes(&12345u64.to_be_bytes());
    let challenge: Fr = transcript.challenge();

    // Pins the v0.8 session derived from PROTOCOL_ID and the single
    // length-prefixed label instance absorb, then one 54-byte challenge
    // squeeze reduced mod r.
    let expected: ark_bn254::Fr = ark_bn254::Fr::from_le_bytes_mod_order(&[
        0xB5, 0x56, 0x22, 0x6C, 0xED, 0x90, 0x17, 0xF8, 0xE3, 0x9B, 0x87, 0x81, 0xA5, 0x6F, 0x70,
        0x2E, 0x2B, 0x9E, 0x14, 0x15, 0x89, 0x94, 0xD8, 0x11, 0x11, 0xDF, 0x68, 0x72, 0xAE, 0x19,
        0xDF, 0x25,
    ]);
    let got: ark_bn254::Fr = challenge.into();
    assert_eq!(got, expected, "Blake2b known-vector regression");
}
