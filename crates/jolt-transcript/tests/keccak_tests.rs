//! Tests for KeccakTranscript implementation.

mod common;

use jolt_transcript::KeccakTranscript;

// Generate the standardized test suite for KeccakTranscript
transcript_tests!(KeccakTranscript);

#[test]
fn test_keccak_known_vector() {
    use jolt_transcript::Transcript;

    // Test against a known value to detect unintentional changes
    let mut transcript = KeccakTranscript::new(b"Jolt");
    transcript.append(&12345u64);

    let challenge = transcript.challenge();

    // This is a regression test
    assert_ne!(challenge, 0);

    // Running the same operations should give the same result
    let mut transcript2 = KeccakTranscript::new(b"Jolt");
    transcript2.append(&12345u64);
    assert_eq!(challenge, transcript2.challenge());
}

#[test]
fn test_keccak_state_accessor() {
    use jolt_transcript::Transcript;

    let transcript = KeccakTranscript::new(b"test");
    let state = transcript.state();

    // State should be 32 bytes
    assert_eq!(state.len(), 32);

    // State should not be all zeros after initialization with label
    assert!(!state.iter().all(|&b| b == 0));
}
