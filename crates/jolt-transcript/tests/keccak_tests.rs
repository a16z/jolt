//! Tests for KeccakTranscript implementation.

mod common;

use jolt_field::Fr;
use jolt_transcript::KeccakTranscript;

type Kec = KeccakTranscript<Fr>;

transcript_tests!(Kec);

#[test]
fn test_keccak_known_vector() {
    use ark_ff::PrimeField;
    use jolt_transcript::Transcript;

    let mut transcript = KeccakTranscript::<Fr>::new(b"Jolt");
    transcript.append_bytes(&12345u64.to_be_bytes());
    let challenge: Fr = transcript.challenge();

    // Pinned wire-format check; see Blake2b counterpart for rationale.
    let expected: ark_bn254::Fr = ark_bn254::Fr::from_le_bytes_mod_order(&[
        0x2E, 0xDF, 0x34, 0x68, 0x85, 0xEE, 0x1C, 0x8B, 0xEC, 0xBD, 0x68, 0xA6, 0x3E, 0x23, 0x00,
        0x9F, 0x10, 0x00, 0xBC, 0xA3, 0xC4, 0xBA, 0x1C, 0xF4, 0x63, 0xDC, 0x84, 0x8D, 0x45, 0xD9,
        0xDD, 0x1E,
    ]);
    let got: ark_bn254::Fr = challenge.into();
    assert_eq!(got, expected, "Keccak known-vector regression");
}
