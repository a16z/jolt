//! Tests for KeccakTranscript implementation.

#![expect(clippy::unwrap_used, reason = "tests")]

mod common;

use jolt_field::Fr;
use jolt_transcript::KeccakTranscript;

type Kec = KeccakTranscript<Fr>;

transcript_tests!(Kec);

#[test]
fn test_keccak_known_vector() {
    use ark_ff::PrimeField;
    use jolt_transcript::Transcript;

    let mut transcript = KeccakTranscript::<Fr>::new(b"Jolt");
    transcript.append_bytes(&12345u64.to_be_bytes());
    let challenge: Fr = transcript.challenge();

    // Pins the v0.8 session derived from PROTOCOL_ID and the single
    // length-prefixed label instance absorb, then one 54-byte challenge
    // squeeze reduced mod r.
    let expected: ark_bn254::Fr = ark_bn254::Fr::from_le_bytes_mod_order(&[
        0x9E, 0xFA, 0xD4, 0xB5, 0x97, 0x2A, 0x98, 0xD6, 0xB2, 0x60, 0x36, 0x5B, 0x1A, 0xD3, 0x91,
        0xA0, 0x73, 0x48, 0x30, 0xDB, 0xCA, 0x91, 0x8A, 0x5F, 0x31, 0x0E, 0xDE, 0x41, 0x9F, 0x9D,
        0x58, 0x05,
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
