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
        0x4A, 0x77, 0xE9, 0xA1, 0x87, 0x58, 0xEA, 0x45, 0x42, 0x2A, 0x91, 0xD8, 0xCF, 0xCD, 0x8B,
        0xBA, 0x02, 0x9F, 0x63, 0x4F, 0xF0, 0x05, 0x6C, 0x75, 0xE8, 0x43, 0xE8, 0x77, 0x33, 0x0B,
        0x02, 0x14,
    ]);
    let got: ark_bn254::Fr = challenge.into();
    assert_eq!(got, expected, "Poseidon known-vector regression");
}
