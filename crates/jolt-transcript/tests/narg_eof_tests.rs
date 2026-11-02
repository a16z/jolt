//! Soundness regression: NARG strings appended with trailing garbage must
//! be rejected by `check_eof`. Without this check, `valid_proof || garbage`
//! would round-trip through verification, making top-level proof bytes
//! malleable.
//!
//! See PR #1455 review for the original report.

#![cfg(feature = "transcript-blake2b")]
#![expect(clippy::expect_used, reason = "tests")]

use blake2::Blake2b512;
use jolt_transcript::{
    protocol_tag, prover_transcript, verifier_transcript, BytesMsg, ProverTranscript,
};
use spongefish::instantiations::Hash;
use spongefish::VerifierState;

const SESSION: [u8; 32] = protocol_tag(b"narg-eof-test");
const INSTANCE: [u8; 32] = [0x42; 32];

fn build_valid_narg(messages: &[&[u8]]) -> Vec<u8> {
    let mut prover = prover_transcript::<Hash<Blake2b512>>(&SESSION, &INSTANCE);
    for m in messages {
        prover.send(&BytesMsg(m.to_vec()));
    }
    prover.narg_string().to_vec()
}

fn build_verifier(narg: &[u8]) -> VerifierState<'_, Hash<Blake2b512>> {
    verifier_transcript::<Hash<Blake2b512>>(&SESSION, &INSTANCE, narg)
}

#[test]
fn check_eof_accepts_exact_narg() {
    let msgs: &[&[u8]] = &[b"alpha", b"beta", b"gamma"];
    let narg = build_valid_narg(msgs);

    let mut verifier = build_verifier(&narg);
    for expected in msgs {
        let got: BytesMsg = verifier
            .prover_message()
            .expect("valid prover message must deserialize");
        assert_eq!(got.as_slice(), *expected);
    }
    verifier
        .check_eof()
        .expect("exact narg must pass check_eof");
}

#[test]
fn check_eof_rejects_trailing_garbage() {
    let msgs: &[&[u8]] = &[b"alpha", b"beta", b"gamma"];
    let mut narg = build_valid_narg(msgs);
    let original_len = narg.len();
    narg.extend_from_slice(&[0xFFu8; 7]);

    let mut verifier = build_verifier(&narg);
    for expected in msgs {
        let got: BytesMsg = verifier
            .prover_message()
            .expect("valid prefix must deserialize");
        assert_eq!(got.as_slice(), *expected);
    }
    let result = verifier.check_eof();
    assert!(
        result.is_err(),
        "narg with {} trailing bytes (original_len={}) must fail check_eof",
        7,
        original_len,
    );
}

#[test]
fn check_eof_rejects_single_trailing_byte() {
    let narg = {
        let mut n = build_valid_narg(&[b"only"]);
        n.push(0x00);
        n
    };
    let mut verifier = build_verifier(&narg);
    let _: BytesMsg = verifier
        .prover_message()
        .expect("valid prefix must deserialize");
    assert!(
        verifier.check_eof().is_err(),
        "even a single trailing byte must fail check_eof",
    );
}

#[test]
fn check_eof_rejects_unread_messages() {
    let msgs: &[&[u8]] = &[b"alpha", b"beta"];
    let narg = build_valid_narg(msgs);

    let mut verifier = build_verifier(&narg);
    let _: BytesMsg = verifier
        .prover_message()
        .expect("first message must deserialize");
    assert!(
        verifier.check_eof().is_err(),
        "leaving prover messages unread must fail check_eof",
    );
}
