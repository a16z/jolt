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
    // length-prefixed label instance absorb, then one 52-byte challenge
    // squeeze reduced mod r.
    let expected: ark_bn254::Fr = ark_bn254::Fr::from_le_bytes_mod_order(&[
        0xF9, 0xD1, 0xC9, 0xE3, 0xFF, 0xFA, 0xC9, 0xFA, 0x19, 0x41, 0xCE, 0x75, 0xDB, 0xD7, 0x77,
        0x87, 0x2C, 0x12, 0x37, 0x00, 0x83, 0x9C, 0x90, 0xED, 0x0D, 0xD1, 0x70, 0x90, 0xCD, 0xE7,
        0x46, 0x22,
    ]);
    let got: ark_bn254::Fr = challenge.into();
    assert_eq!(got, expected, "Blake2b known-vector regression");
}
