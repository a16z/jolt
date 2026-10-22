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
    // length-prefixed label instance absorb.
    let expected: ark_bn254::Fr = ark_bn254::Fr::from_le_bytes_mod_order(&[
        0x6C, 0xB9, 0xC2, 0x88, 0xC8, 0xD7, 0x24, 0x12, 0xCD, 0xD9, 0x55, 0xF8, 0x57, 0xB7, 0xEC,
        0x59, 0xEE, 0xAF, 0xA2, 0x01, 0x57, 0x0D, 0x39, 0x72, 0x19, 0x0F, 0x43, 0x58, 0xEB, 0xC1,
        0x72, 0x26,
    ]);
    let got: ark_bn254::Fr = challenge.into();
    assert_eq!(got, expected, "Blake2b known-vector regression");
}
