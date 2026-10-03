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

    // Pinned wire-format check; see Blake2b counterpart for rationale.
    let expected: ark_bn254::Fr = ark_bn254::Fr::from_le_bytes_mod_order(&[
        0xF7, 0x54, 0x3B, 0x32, 0x71, 0x47, 0x68, 0xEE, 0x04, 0x09, 0xEC, 0xAB, 0x9B, 0x91, 0x2E,
        0x8A, 0xD0, 0x51, 0x7E, 0x7C, 0x6E, 0xB2, 0xB8, 0x77, 0x52, 0x59, 0x2B, 0x10, 0x38, 0x67,
        0x78, 0x06,
    ]);
    let got: ark_bn254::Fr = challenge.into();
    assert_eq!(got, expected, "Poseidon known-vector regression");
}
