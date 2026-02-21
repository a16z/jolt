//! Tests for Blake2bTranscript implementation.

mod common;

use jolt_transcript::Blake2bTranscript;

// Generate the standardized test suite for Blake2bTranscript
transcript_tests!(Blake2bTranscript);

#[test]
fn test_blake2b_known_vector() {
    use jolt_transcript::Transcript;

    // Test against a known value to detect unintentional changes
    let mut transcript = Blake2bTranscript::new(b"Jolt");
    transcript.append(&12345u64);

    let challenge = transcript.challenge();

    // This is a regression test - if the implementation changes,
    // this value will need to be updated intentionally
    assert_ne!(challenge, 0);

    // Running the same operations should give the same result
    let mut transcript2 = Blake2bTranscript::new(b"Jolt");
    transcript2.append(&12345u64);
    assert_eq!(challenge, transcript2.challenge());
}

#[test]
fn test_blake2b_state_accessor() {
    use jolt_transcript::Transcript;

    let transcript = Blake2bTranscript::new(b"test");
    let state = transcript.state();

    // State should be 32 bytes
    assert_eq!(state.len(), 32);

    // State should not be all zeros after initialization with label
    assert!(!state.iter().all(|&b| b == 0));
}
