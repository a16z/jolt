//! Known-answer tests for `LegacyBlake2bTranscript` (`DigestTranscript`).
//!
//! This transcript exists solely for byte-compatibility with the
//! `jolt-prover-legacy` Fiat-Shamir transcript, so prover/verifier round-trips
//! cannot catch a chaining regression — both sides would drift together. The
//! hex vectors below were generated with an independent BLAKE2b-256
//! implementation (Python `hashlib.blake2b(digest_size=32)`) from the
//! documented formula:
//!
//! ```text
//! state_0        = H(label zero-padded to 32 bytes)
//! round_word(n)  = 28 zero bytes || n as u32 big-endian
//! append(bytes)  : state' = H(state || round_word(n) || bytes), n += 1
//! squeeze        : out    = H(state || round_word(n)),  state' = out, n += 1
//! ```

use jolt_field::{Fr, TranscriptChallenge};
use jolt_transcript::{Label, LabelWithCount, LegacyBlake2bTranscript, Transcript, U64Word};

type T = LegacyBlake2bTranscript<Fr>;

const KAT_LABEL: &[u8] = b"jolt transcript kat";

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut out, byte| {
        // writing to a String is infallible
        let _ = write!(out, "{byte:02x}");
        out
    })
}

#[test]
fn initial_state_is_hash_of_zero_padded_label() {
    let transcript = T::new(KAT_LABEL);
    assert_eq!(
        hex(&transcript.state()),
        "b57f1b2110a7b1720489cf99300d3a5641498671160090254ef89dc6f926b19e",
    );
}

#[test]
fn empty_label_hashes_thirty_two_zero_bytes() {
    let transcript = T::new(b"");
    assert_eq!(
        hex(&transcript.state()),
        "89eb0d6a8a691dae2cd15ed0369931ce0a949ecafa5c3f93f8121833646e15c3",
    );
}

#[test]
fn append_bytes_chains_state_with_round_counter() {
    let mut transcript = T::new(KAT_LABEL);
    transcript.append_bytes(b"jolt says hello");
    assert_eq!(
        hex(&transcript.state()),
        "e0581c34f7f60080a2b1cac4282e5420a88034e11a876c47f94374122fa1f905",
    );
}

#[test]
fn challenge_draw_rotates_state_and_truncates_to_sixteen_bytes() {
    const EXPECTED_CHUNK: &str = "25a496912b5b75d6504cbc204a25b13d6121e6266ac056589689c43dc06f9142";

    let mut transcript = T::new(KAT_LABEL);
    transcript.append_bytes(b"jolt says hello");
    let mut buf = [0u8; 16];
    transcript.raw_challenge_bytes(&mut buf);

    assert_eq!(
        hex(&buf),
        &EXPECTED_CHUNK[..32],
        "low 16 bytes of the chunk"
    );
    assert_eq!(
        hex(&transcript.state()),
        EXPECTED_CHUNK,
        "state becomes the squeezed chunk"
    );
}

#[test]
fn multi_chunk_draw_concatenates_successive_squeezes() {
    let mut transcript = T::new(KAT_LABEL);
    let mut out = [0u8; 48];
    transcript.raw_challenge_bytes(&mut out);
    assert_eq!(
        hex(&out),
        "0db99faadc348c2ecdd3faf44d5f94cf27b2a8b93e757d60eae8a2bc1c057d95\
         b6d6e19661c63658f73f61b9f25d4e70",
    );
}

#[test]
fn challenge_decodes_the_pinned_bytes_via_from_challenge_bytes() {
    let mut transcript = T::new(KAT_LABEL);
    transcript.append_bytes(b"jolt says hello");
    let challenge: Fr = transcript.challenge();

    let pinned: [u8; 16] = [
        0x25, 0xa4, 0x96, 0x91, 0x2b, 0x5b, 0x75, 0xd6, 0x50, 0x4c, 0xbc, 0x20, 0x4a, 0x25, 0xb1,
        0x3d,
    ];
    assert_eq!(challenge, Fr::from_challenge_bytes(&pinned));
}

#[test]
fn challenge_scalar_uses_the_scalar_decode_path() {
    let mut transcript = T::new(KAT_LABEL);
    transcript.append_bytes(b"jolt says hello");
    let challenge: Fr = transcript.challenge_scalar();

    let pinned: [u8; 16] = [
        0x25, 0xa4, 0x96, 0x91, 0x2b, 0x5b, 0x75, 0xd6, 0x50, 0x4c, 0xbc, 0x20, 0x4a, 0x25, 0xb1,
        0x3d,
    ];
    assert_eq!(challenge, Fr::from_scalar_challenge_bytes(&pinned));
}

#[test]
fn challenge_scalar_powers_are_consecutive_powers_of_one_draw() {
    let mut probe = T::new(KAT_LABEL);
    let gamma: Fr = probe.challenge_scalar();

    let mut transcript = T::new(KAT_LABEL);
    let powers = transcript.challenge_scalar_powers(4);

    assert_eq!(powers[0], Fr::from(1u64));
    assert_eq!(powers[1], gamma);
    assert_eq!(powers[2], gamma * gamma);
    assert_eq!(powers[3], gamma * gamma * gamma);
    assert_eq!(
        transcript.state(),
        probe.state(),
        "powers consume exactly one draw"
    );
}

#[test]
fn label_helper_packs_zero_padded_thirty_two_byte_word() {
    let mut via_helper = T::new(KAT_LABEL);
    via_helper.append(&Label(b"opening claim"));

    let mut explicit = T::new(KAT_LABEL);
    let mut word = [0u8; 32];
    word[..13].copy_from_slice(b"opening claim");
    explicit.append_bytes(&word);

    assert_eq!(via_helper.state(), explicit.state());
}

#[test]
fn label_with_count_packs_label_and_big_endian_count() {
    let mut via_helper = T::new(KAT_LABEL);
    via_helper.append(&LabelWithCount(b"round polys", 7));

    let mut explicit = T::new(KAT_LABEL);
    let mut word = [0u8; 32];
    word[..11].copy_from_slice(b"round polys");
    word[24..].copy_from_slice(&7u64.to_be_bytes());
    explicit.append_bytes(&word);

    assert_eq!(via_helper.state(), explicit.state());
}

#[test]
fn u64_word_packs_left_padded_big_endian_value() {
    let mut via_helper = T::new(KAT_LABEL);
    via_helper.append(&U64Word(0xDEAD_BEEF));

    let mut explicit = T::new(KAT_LABEL);
    let mut word = [0u8; 32];
    word[24..].copy_from_slice(&0xDEAD_BEEFu64.to_be_bytes());
    explicit.append_bytes(&word);

    assert_eq!(via_helper.state(), explicit.state());
}

#[test]
#[should_panic(expected = "label must be at most")]
fn overlong_transcript_label_is_rejected() {
    let _ = T::new(b"this label is thirty-three bytes!");
}
