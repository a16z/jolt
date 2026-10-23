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

use jolt_field::{CanonicalBytes, CanonicalEncoding, Fr};
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
    transcript.challenge_bytes(&mut buf);

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
    transcript.challenge_bytes(&mut out);
    assert_eq!(
        hex(&out),
        "0db99faadc348c2ecdd3faf44d5f94cf27b2a8b93e757d60eae8a2bc1c057d95\
         b6d6e19661c63658f73f61b9f25d4e70",
    );
}

/// The first challenge after an absorb squeezes `Fr::challenge_width(1) = 52`
/// bytes, two chunks, and reduces them modulo `r` as a little-endian integer.
/// The value was computed independently (Python
/// `int.from_bytes(.., "little") % r`).
#[test]
fn challenge_reduces_fifty_two_squeezed_bytes() {
    const SQUEEZED: &str = "25a496912b5b75d6504cbc204a25b13d6121e6266ac056589689c43dc06f9142\
                            66bf5655d23fb7f4e92e96ea1f0df9727d7f0e9a";
    const REDUCED: &str = "71854c4909f4c3e785ab8e719f895fdb6a0ea3df7952cabbe4595baa5d355d1f";
    assert_eq!(Fr::challenge_width(1), 52);

    let mut probe = T::new(KAT_LABEL);
    probe.append_bytes(b"jolt says hello");
    let mut squeezed = [0u8; 52];
    probe.challenge_bytes(&mut squeezed);
    assert_eq!(hex(&squeezed), SQUEEZED.replace(' ', ""));

    for draw in [T::challenge, T::challenge_scalar] {
        let mut transcript = T::new(KAT_LABEL);
        transcript.append_bytes(b"jolt says hello");
        let challenge: Fr = draw(&mut transcript);
        assert_eq!(hex(&challenge.to_bytes_le_vec()), REDUCED);
        assert_eq!(transcript.state(), probe.state());
    }
}

/// Each draw since the last absorb is wider than the one before it may be
/// (52 bytes for draws 1–2, 53 for draws 3–32), and an absorb starts over.
#[test]
fn challenge_width_grows_with_draws_and_resets_on_absorb() {
    let mut transcript = T::new(KAT_LABEL);
    let mut probe = T::new(KAT_LABEL);
    let check = |transcript: &mut T, probe: &mut T, widths: &[usize]| {
        for (draw, &width) in widths.iter().enumerate() {
            assert_eq!(Fr::challenge_width(draw as u64 + 1), width);
            let mut bytes = vec![0u8; width];
            probe.challenge_bytes(&mut bytes);
            let challenge: Fr = transcript.challenge();
            assert_eq!(challenge, Fr::from_bytes_le_reduced(&bytes), "draw {draw}");
        }
    };
    check(&mut transcript, &mut probe, &[52, 52, 53, 53, 53]);
    transcript.append_bytes(b"next message");
    probe.append_bytes(b"next message");
    check(&mut transcript, &mut probe, &[52, 52, 53]);
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
