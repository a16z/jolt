//! Tests for PoseidonTranscript implementation.

mod common;

use jolt_field::Fr;
use jolt_transcript::PoseidonTranscript;

type Pos = PoseidonTranscript<Fr>;

transcript_tests!(Pos);

#[test]
fn test_poseidon_known_vector() {
    use ark_ff::PrimeField;
    use jolt_transcript::Transcript;

    let mut transcript = PoseidonTranscript::<Fr>::new(b"Jolt");
    transcript.append_bytes(&12345u64.to_be_bytes());
    let challenge: Fr = transcript.challenge();

    // Pins the v0.8 session derived from PROTOCOL_ID and the single
    // length-prefixed label instance absorb, then one 54-byte challenge
    // squeeze reduced mod r.
    let expected: ark_bn254::Fr = ark_bn254::Fr::from_le_bytes_mod_order(&[
        0x16, 0x73, 0xD0, 0x6A, 0xA5, 0xDE, 0xD3, 0xEA, 0x70, 0x29, 0x76, 0x3F, 0x47, 0x51, 0x60,
        0x0C, 0xD9, 0xAF, 0x04, 0xDA, 0x89, 0x5E, 0xB6, 0xA0, 0xF0, 0xE0, 0x50, 0x78, 0x34, 0x8F,
        0xAD, 0x15,
    ]);
    let got: ark_bn254::Fr = challenge.into();
    assert_eq!(got, expected, "Poseidon known-vector regression");
}
