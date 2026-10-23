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
    // length-prefixed label instance absorb, then one 52-byte challenge
    // squeeze reduced mod r.
    let expected: ark_bn254::Fr = ark_bn254::Fr::from_le_bytes_mod_order(&[
        0x52, 0x31, 0x0C, 0x87, 0xD8, 0x93, 0x5C, 0x65, 0x2E, 0x83, 0x71, 0x19, 0xF2, 0x81, 0x1E,
        0x79, 0xC0, 0xAE, 0x1A, 0xDB, 0xE6, 0xC5, 0xED, 0xD2, 0x12, 0xDC, 0x69, 0x7D, 0xBD, 0xF4,
        0x8E, 0x27,
    ]);
    let got: ark_bn254::Fr = challenge.into();
    assert_eq!(got, expected, "Poseidon known-vector regression");
}
