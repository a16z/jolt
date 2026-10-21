//! Tests for the Keccak transcript.

#![expect(clippy::unwrap_used, reason = "tests")]

mod common;

transcript_tests!(jolt_transcript::Keccak);

/// A field without room for the short width is refused before anything is
/// squeezed: loss 1 needs 128 bits, the whole bit length of this field.
#[cfg(feature = "solinas")]
#[test]
fn short_scalar_refuses_a_field_without_room() {
    use jolt_field::Prime128Offset275;
    use jolt_transcript::{GrindBitsError, Transcript};

    let mut short = KeccakTranscript::<Prime128Offset275>::new(b"narrow");
    let untouched = short.clone();
    assert_eq!(
        short.challenge_short_scalar(1),
        Err(GrindBitsError::FieldTooSmall)
    );
    assert_eq!(short.state(), untouched.state());
}

#[test]
fn test_keccak_known_vector() {
    let mut transcript = transcript(b"Jolt");
    transcript.append(&12345u64);
    let challenge: Fr = transcript.challenge().unwrap();
    // Pins the session derived from PROTOCOL_ID, the instance absorb, one
    // `u64` public message and one 52-byte challenge squeeze reduced mod r.
    assert_eq!(
        format!("{challenge:?}"),
        "1766718655065384849885355460798243624214294877158123096419747417635580052889"
    );
}

#[test]
fn public_message_after_challenge_agrees_including_empty() {
    use jolt_transcript::{BytesMsg, Keccak};

    for public in [b"public after squeeze".as_slice(), &[]] {
        let mut prover = prover_transcript::<Keccak>(b"keccak-public", &[0u8; 32]);
        let first: [u8; 17] = prover.verifier_message();
        prover.public_message_as(public, |bytes| bytes);
        let second: [u8; 32] = prover.verifier_message();
        prover.prover_message(&BytesMsg(b"response".to_vec()));
        let third: [u8; 32] = prover.verifier_message();

        let mut verifier =
            verifier_transcript::<Keccak>(b"keccak-public", &[0u8; 32], prover.narg_string());
        assert_eq!(verifier.verifier_message::<[u8; 17]>(), first);
        verifier.public_message_as(public, |bytes| bytes);
        assert_eq!(verifier.verifier_message::<[u8; 32]>(), second);
        assert_eq!(
            verifier.prover_message::<BytesMsg>().unwrap(),
            BytesMsg(b"response".to_vec()),
        );
        assert_eq!(verifier.verifier_message::<[u8; 32]>(), third);
        verifier.check_eof().unwrap();

        let mut uninterrupted = prover_transcript::<Keccak>(b"keccak-public", &[0u8; 32]);
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
