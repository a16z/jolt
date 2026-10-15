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

    // Pinned wire-format check: any change to PROTOCOL_ID, the session
    // encoding, the append_bytes layout, or the challenge decoder will
    // flip these bytes. Update only with an audit trail.
    //
    // Audit trail: `from_challenge_bytes` now builds the field element from the
    // squeezed bytes via the 125-bit Montgomery-friendly decode rather than a
    // plain little-endian reduction, so the canonical challenge value changed.
    let expected: ark_bn254::Fr = ark_bn254::Fr::from_le_bytes_mod_order(&[
        0xAE, 0x28, 0x1C, 0xAE, 0x3E, 0x93, 0x36, 0xA9, 0xE2, 0x57, 0xE0, 0x30, 0x2F, 0xC0, 0x48,
        0x2E, 0xCF, 0xC9, 0x89, 0x1B, 0x7D, 0x65, 0x4E, 0x6A, 0xB5, 0xEF, 0x55, 0x46, 0x36, 0x71,
        0x8A, 0x29,
    ]);
    let got: ark_bn254::Fr = challenge.into();
    assert_eq!(got, expected, "Blake2b known-vector regression");
}
