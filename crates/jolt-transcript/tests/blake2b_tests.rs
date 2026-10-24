//! Tests for the Blake2b-512 transcript.

#![expect(clippy::unwrap_used, reason = "tests")]

mod common;

use jolt_transcript::Blake2b512;

transcript_tests!(Blake2b512);

#[test]
fn test_blake2b_known_vector() {
    let mut transcript = transcript(b"Jolt");
    transcript.append(&12345u64);
    let challenge: Fr = transcript.challenge().unwrap();
    // Pins the session derived from PROTOCOL_ID, the instance absorb, one
    // `u64` public message and one 56-byte challenge squeeze reduced mod r.
    assert_eq!(
        format!("{challenge:?}"),
        "18793778572527438931167002525786312417669516031467570196084319916574544498697"
    );
}
