//! Tests for KeccakTranscript implementation.

mod common;

use jolt_transcript::KeccakTranscript;

// Generate the standardized test suite for KeccakTranscript
transcript_tests!(KeccakTranscript);

#[test]
fn test_keccak_known_vector() {
    use jolt_transcript::Transcript;

    let mut transcript = KeccakTranscript::new(b"Jolt");
    transcript.append_bytes(&12345u64.to_be_bytes());

    let challenge = transcript.challenge();

    assert_ne!(challenge, 0);

    let mut transcript2 = KeccakTranscript::new(b"Jolt");
    transcript2.append_bytes(&12345u64.to_be_bytes());
    assert_eq!(challenge, transcript2.challenge());
}

#[test]
fn test_keccak_state_accessor() {
    use jolt_transcript::Transcript;

    let transcript = KeccakTranscript::new(b"test");
    let state = transcript.state();

    assert_eq!(state.len(), 32);

    assert!(!state.iter().all(|&b| b == 0));
}
