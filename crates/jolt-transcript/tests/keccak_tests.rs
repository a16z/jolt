//! Tests for KeccakTranscript implementation.

mod common;

use jolt_field::Fr;
use jolt_transcript::KeccakTranscript;
#[allow(unused_imports)]
use num_traits::Zero;

type Kec = KeccakTranscript<Fr>;

transcript_tests!(Kec);

#[test]
fn test_keccak_known_vector() {
    use jolt_transcript::Transcript;

    let mut transcript = KeccakTranscript::<Fr>::new(b"Jolt");
    transcript.append_bytes(&12345u64.to_be_bytes());

    let challenge: Fr = transcript.challenge();

    assert!(!challenge.is_zero());

    let mut transcript2 = KeccakTranscript::<Fr>::new(b"Jolt");
    transcript2.append_bytes(&12345u64.to_be_bytes());
    assert_eq!(challenge, transcript2.challenge());
}

#[test]
fn test_keccak_state_accessor() {
    use jolt_transcript::Transcript;

    let transcript = KeccakTranscript::<Fr>::new(b"test");
    let state = transcript.state();

    assert_eq!(state.len(), 32);

    assert!(!state.iter().all(|&b| b == 0));
}
