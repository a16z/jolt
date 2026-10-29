//! Tests for the Blake2b-512 transcript.

#![expect(clippy::unwrap_used, reason = "tests")]

mod common;

use jolt_transcript::Blake2b512;

transcript_tests!(Blake2b512);

#[cfg_attr(
    feature = "keccak256-digests",
    ignore = "pinned for SHA3-256 digests; keccak256-digests changes the session identifier"
)]
#[test]
fn test_blake2b_known_vector() {
    let mut transcript = transcript(b"Jolt");
    transcript.append(&12345u64);
    let challenge: Fr = transcript.challenge().unwrap();
    // Pins the session derived from PROTOCOL_ID, the instance absorb, one
    // `u64` public message and one 56-byte challenge squeeze reduced mod r.
    assert_eq!(
        format!("{challenge:?}"),
        "19511679008748546059954614320539583065670345019378152892361345655216605672333"
    );
}
