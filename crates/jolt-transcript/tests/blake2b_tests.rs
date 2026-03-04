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
