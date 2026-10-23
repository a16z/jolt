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
        0x72, 0x35, 0x16, 0x7B, 0x7C, 0x6A, 0xDE, 0x91, 0x4D, 0xE7, 0xBD, 0x04, 0x4B, 0x91, 0x30,
        0xF4, 0xAE, 0x68, 0xB7, 0x74, 0x86, 0xC4, 0xEE, 0xBB, 0x67, 0xDD, 0xDF, 0x66, 0x0C, 0x96,
        0x82, 0x1B,
    ]);
    let got: ark_bn254::Fr = challenge.into();
    assert_eq!(got, expected, "Blake2b known-vector regression");
}
