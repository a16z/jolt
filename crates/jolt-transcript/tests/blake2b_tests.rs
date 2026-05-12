//! Tests for Blake2bTranscript implementation.

mod common;

use jolt_field::{Fr, FromPrimitiveInt};
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
    let expected: ark_bn254::Fr = ark_bn254::Fr::from_le_bytes_mod_order(&[
        0x6B, 0xAE, 0x98, 0xBB, 0x70, 0x31, 0xDE, 0xEA, 0x8B, 0x57, 0x22, 0xB0, 0x0F, 0xC5, 0x83,
        0x62, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00,
    ]);
    let got: ark_bn254::Fr = challenge.into();
    assert_eq!(got, expected, "Blake2b known-vector regression");
}

#[test]
fn test_field_zero_one_distinct_states() {
    use jolt_field::Fr;
    use jolt_transcript::{AppendToTranscript, Transcript};
    use num_traits::{One, Zero};

    let mut t_zero = Blake2bTranscript::<Fr>::new(b"field_test");
    Fr::zero().append_to_transcript(&mut t_zero);
    let c_zero: Fr = t_zero.challenge();

    let mut t_one = Blake2bTranscript::<Fr>::new(b"field_test");
    Fr::one().append_to_transcript(&mut t_one);
    let c_one: Fr = t_one.challenge();

    assert_ne!(
        c_zero, c_one,
        "Fr::zero() and Fr::one() must produce distinct transcript states"
    );
}

#[test]
fn test_field_element_ordering_sensitivity() {
    use jolt_transcript::{AppendToTranscript, Transcript};

    let a = Fr::from_u64(42);
    let b = Fr::from_u64(99);

    let mut t1 = Blake2bTranscript::<Fr>::new(b"order_test");
    a.append_to_transcript(&mut t1);
    b.append_to_transcript(&mut t1);
    let c1: Fr = t1.challenge();

    let mut t2 = Blake2bTranscript::<Fr>::new(b"order_test");
    b.append_to_transcript(&mut t2);
    a.append_to_transcript(&mut t2);
    let c2: Fr = t2.challenge();

    assert_ne!(
        c1, c2,
        "append(a, b) and append(b, a) must produce different challenges"
    );
}
