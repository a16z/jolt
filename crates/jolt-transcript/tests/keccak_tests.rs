//! Tests for KeccakTranscript implementation.

#![expect(clippy::unwrap_used, reason = "tests")]

mod common;

use jolt_field::Fr;
use jolt_transcript::KeccakTranscript;

type Kec = KeccakTranscript<Fr>;

transcript_tests!(Kec);

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
    use ark_ff::PrimeField;
    use jolt_transcript::Transcript;

    let mut transcript = KeccakTranscript::<Fr>::new(b"Jolt");
    transcript.append_bytes(&12345u64.to_be_bytes());
    let challenge: Fr = transcript.challenge();

    // Pins the v0.8 session derived from PROTOCOL_ID and the single
    // length-prefixed label instance absorb, then one 52-byte challenge
    // squeeze reduced mod r.
    let expected: ark_bn254::Fr = ark_bn254::Fr::from_le_bytes_mod_order(&[
        0x76, 0x68, 0xDD, 0x4A, 0x05, 0x14, 0xFF, 0x3E, 0x89, 0x0B, 0xDA, 0x07, 0xF5, 0x0D, 0x78,
        0x8B, 0xFC, 0xAB, 0x80, 0x07, 0xEA, 0x25, 0x75, 0xFC, 0x96, 0xE4, 0x5A, 0x7F, 0x00, 0x55,
        0x23, 0x00,
    ]);
    let got: ark_bn254::Fr = challenge.into();
    assert_eq!(got, expected, "Keccak known-vector regression");
}

#[test]
fn public_message_after_challenge_agrees_including_empty() {
    use jolt_transcript::{prover_transcript, verifier_transcript, BytesMsg};
    use spongefish::instantiations::Keccak;

    for public in [b"public after squeeze".as_slice(), &[]] {
        let mut prover = prover_transcript::<Keccak>(b"keccak-public", [0; 32]);
        let first: [u8; 17] = prover.verifier_message();
        prover.public_message_as(public, |bytes| bytes);
        let second: [u8; 32] = prover.verifier_message();
        prover.prover_message(&BytesMsg(b"response".to_vec()));
        let third: [u8; 32] = prover.verifier_message();

        let mut verifier =
            verifier_transcript::<Keccak>(b"keccak-public", [0; 32], prover.narg_string());
        assert_eq!(verifier.verifier_message::<[u8; 17]>(), first);
        verifier.public_message_as(public, |bytes| bytes);
        assert_eq!(verifier.verifier_message::<[u8; 32]>(), second);
        assert_eq!(
            verifier.prover_message::<BytesMsg>().unwrap(),
            BytesMsg(b"response".to_vec()),
        );
        assert_eq!(verifier.verifier_message::<[u8; 32]>(), third);
        verifier.check_eof().unwrap();

        let mut uninterrupted = prover_transcript::<Keccak>(b"keccak-public", [0; 32]);
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
