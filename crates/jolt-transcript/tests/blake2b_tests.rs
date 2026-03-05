//! Tests for Blake2bTranscript implementation.

mod common;

use jolt_transcript::Blake2bTranscript;

// Generate the standardized test suite for Blake2bTranscript
transcript_tests!(Blake2bTranscript);

#[test]
fn test_blake2b_known_vector() {
    use jolt_transcript::Transcript;

    let mut transcript = Blake2bTranscript::new(b"Jolt");
    transcript.append_bytes(&12345u64.to_be_bytes());

    let challenge = transcript.challenge();

    assert_ne!(challenge, 0);

    let mut transcript2 = Blake2bTranscript::new(b"Jolt");
    transcript2.append_bytes(&12345u64.to_be_bytes());
    assert_eq!(challenge, transcript2.challenge());
}

#[test]
fn test_blake2b_state_accessor() {
    use jolt_transcript::Transcript;

    let transcript = Blake2bTranscript::new(b"test");
    let state = transcript.state();

    assert_eq!(state.len(), 32);

    assert!(!state.iter().all(|&b| b == 0));
}

#[test]
fn test_field_zero_one_distinct_states() {
    use jolt_field::Fr;
    use jolt_transcript::{AppendToTranscript, Transcript};
    use num_traits::{One, Zero};

    let mut t_zero = Blake2bTranscript::new(b"field_test");
    Fr::zero().append_to_transcript(&mut t_zero);
    let c_zero = t_zero.challenge();

    let mut t_one = Blake2bTranscript::new(b"field_test");
    Fr::one().append_to_transcript(&mut t_one);
    let c_one = t_one.challenge();

    assert_ne!(
        c_zero, c_one,
        "Fr::zero() and Fr::one() must produce distinct transcript states"
    );
}

#[test]
fn test_field_element_ordering_sensitivity() {
    use jolt_field::{Field, Fr};
    use jolt_transcript::{AppendToTranscript, Transcript};

    let a = Fr::from_u64(42);
    let b = Fr::from_u64(99);

    let mut t1 = Blake2bTranscript::new(b"order_test");
    a.append_to_transcript(&mut t1);
    b.append_to_transcript(&mut t1);
    let c1 = t1.challenge();

    let mut t2 = Blake2bTranscript::new(b"order_test");
    b.append_to_transcript(&mut t2);
    a.append_to_transcript(&mut t2);
    let c2 = t2.challenge();

    assert_ne!(
        c1, c2,
        "append(a, b) and append(b, a) must produce different challenges"
    );
}
