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
    // length-prefixed label instance absorb.
    let expected: ark_bn254::Fr = ark_bn254::Fr::from_le_bytes_mod_order(&[
        0x99, 0x49, 0x1C, 0x25, 0x03, 0x5F, 0x22, 0x07, 0x64, 0x0B, 0x56, 0x74, 0x7B, 0x92, 0xD7,
        0x0F, 0x33, 0x8C, 0x91, 0x96, 0xD5, 0xD5, 0xBA, 0x15, 0xC4, 0xAC, 0xA9, 0xBA, 0x11, 0xE0,
        0xC9, 0x00,
    ]);
    let got: ark_bn254::Fr = challenge.into();
    assert_eq!(got, expected, "Poseidon known-vector regression");
}
