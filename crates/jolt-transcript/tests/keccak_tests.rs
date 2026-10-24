//! Tests for the Keccak transcript.

#![expect(clippy::unwrap_used, reason = "tests")]

mod common;

use jolt_transcript::Keccak;

transcript_tests!(Keccak);

/// A field without room for the short width is refused before anything is
/// squeezed, and the verifier is poisoned: loss 1 needs 128 bits, the whole
/// bit length of this field.
#[cfg(feature = "solinas")]
#[test]
fn short_scalar_refuses_a_field_without_room() {
    use jolt_field::Prime128Offset275;

    let mut refused = transcript(b"narw");
    let mut untouched = transcript(b"narw");
    assert!(FieldTranscript::<Prime128Offset275>::challenge_short_scalar(&mut refused, 1).is_err());
    let next: Fr = refused.challenge().unwrap();
    assert_eq!(next, untouched.challenge().unwrap());

    let mut verifier = verifier_transcript::<Keccak>(&SESSION, b"narw", &[]);
    assert!(
        FieldTranscript::<Prime128Offset275>::challenge_short_scalar(&mut verifier, 1).is_err()
    );
    assert!(verifier.check_eof().is_err());
}

#[test]
fn test_keccak_known_vector() {
    let mut transcript = transcript(b"Jolt");
    transcript.append(&12345u64);
    let challenge: Fr = transcript.challenge().unwrap();
    // Pins the session derived from PROTOCOL_ID, the instance absorb, one
    // `u64` public message and one 56-byte challenge squeeze reduced mod r.
    assert_eq!(
        format!("{challenge:?}"),
        "12245913755514758687269313047689373645644388575530658576000096080939340770669"
    );
}

#[test]
fn public_message_after_challenge_agrees_including_empty() {
    use jolt_transcript::{BytesMsg, ProverTranscript};

    const PUBLIC: [u8; 32] = protocol_tag(b"keccak-public");
    for public in [b"public after squeeze".as_slice(), &[]] {
        let mut prover = prover_transcript::<Keccak>(&PUBLIC, &[0u8; 32]);
        let first: [u8; 17] = prover.verifier_message();
        prover.public_message_as(public, |bytes| bytes);
        let second: [u8; 32] = prover.verifier_message();
        prover.send(&BytesMsg(b"response".to_vec()));
        let third: [u8; 32] = prover.verifier_message();

        let mut verifier = verifier_transcript::<Keccak>(&PUBLIC, &[0u8; 32], prover.narg_string());
        assert_eq!(verifier.verifier_message::<[u8; 17]>(), first);
        verifier.public_message_as(public, |bytes| bytes);
        assert_eq!(verifier.verifier_message::<[u8; 32]>(), second);
        assert_eq!(
            verifier.prover_message::<BytesMsg>().unwrap(),
            BytesMsg(b"response".to_vec()),
        );
        assert_eq!(verifier.verifier_message::<[u8; 32]>(), third);
        verifier.check_eof().unwrap();

        let mut uninterrupted = prover_transcript::<Keccak>(&PUBLIC, &[0u8; 32]);
        let stream: [u8; 49] = uninterrupted.verifier_message();
        assert_eq!(&stream[..17], &first);
        if public.is_empty() {
            assert_eq!(
                &stream[17..],
                &second,
                "empty absorb preserves squeeze stream"
            );
        } else {
            assert_ne!(
                &stream[17..],
                &second,
                "public bytes bind the next challenge"
            );
        }
    }
}
