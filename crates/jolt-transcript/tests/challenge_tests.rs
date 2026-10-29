#![cfg(all(
    feature = "transcript-keccak",
    any(feature = "solinas", feature = "bn254", feature = "binary")
))]
#![expect(clippy::unwrap_used, reason = "tests")]

use jolt_field::CanonicalBytes;
use jolt_transcript::{
    protocol_tag, prover_transcript, verifier_transcript, BytesMsg, ChallengeCodec,
    FieldTranscript, ProverTranscript, MAX_CHALLENGE_BYTES,
};
use spongefish::instantiations::Keccak;
use std::fmt::{Debug, Write};

#[cfg(feature = "solinas")]
use jolt_field::{
    Ext2, ExtField, FpExt4, FpExt8, Prime128Offset275, Prime128OffsetA7F7, Prime24Offset3,
    Prime30Offset35, Prime31Offset19, Prime32Offset99, Prime40Offset195, Prime48Offset59,
    Prime56Offset27, Prime64Offset59,
};
#[cfg(feature = "bn254")]
use jolt_field::{Fq, Fr};
#[cfg(feature = "binary")]
use jolt_field::{F128, F192, F64, F8};
const SESSION: [u8; 32] = protocol_tag(b"jolt/challenge-codecs/v1");
const INSTANCE: [u8; 32] = [0x42; 32];

fn round_trip<F: ChallengeCodec + PartialEq + Debug>() {
    for len in [1, 17, 28] {
        let mut prover = prover_transcript::<Keccak>(&SESSION, &INSTANCE);
        prover.send(&BytesMsg(b"before".to_vec()));
        let expected = FieldTranscript::<F>::challenge_vector(&mut prover, len).unwrap();
        assert_eq!(expected.len(), len);
        prover.send(&BytesMsg(b"after".to_vec()));
        let tail: [u8; 32] = prover.verifier_message();

        let mut verifier = verifier_transcript::<Keccak>(&SESSION, &INSTANCE, prover.narg_string());
        assert_eq!(verifier.prover_message::<BytesMsg>().unwrap().0, b"before");
        assert_eq!(
            FieldTranscript::<F>::challenge_vector(&mut verifier, len).unwrap(),
            expected
        );
        assert_eq!(verifier.prover_message::<BytesMsg>().unwrap().0, b"after");
        assert_eq!(verifier.verifier_message::<[u8; 32]>(), tail);
        verifier.check_eof().unwrap();

        assert_eq!(F::challenge_bytes(len).unwrap() % 8, 0, "whole words");
        assert_eq!(F::challenge_bytes(0), Some(0));
        assert_eq!(F::decode_challenges(&[], 0), Some(vec![]));
        assert!(F::decode_challenges(&[], len).is_none());
        assert!(F::decode_challenges(&[0], 0).is_none());
        assert!(F::decode_challenges(&[], usize::MAX).is_none());
    }
}

#[cfg(feature = "solinas")]
#[test]
fn registered_primes_and_valid_towers_round_trip() {
    macro_rules! base_and_towers {
        ($($base:ty),+ $(,)?) => {$(
            round_trip::<$base>();
            round_trip::<Ext2<$base>>();
            round_trip::<FpExt4<$base>>();
            round_trip::<FpExt8<$base>>();
        )+};
    }
    base_and_towers!(
        Prime24Offset3,
        Prime30Offset35,
        Prime31Offset19,
        Prime32Offset99,
        Prime40Offset195,
        Prime48Offset59,
        Prime56Offset27,
        Prime64Offset59,
        Prime128Offset275,
    );
    round_trip::<Prime128OffsetA7F7>();
}

#[cfg(feature = "bn254")]
#[test]
fn bn254_round_trip() {
    round_trip::<Fr>();
    round_trip::<Fq>();
    // 56 bytes, seven words, per element at every length up to the byte cap.
    let longest = MAX_CHALLENGE_BYTES / 56;
    assert_eq!(longest, 299_593);
    for len in (1..=2048).chain([longest - 1, longest]) {
        assert_eq!(Fr::challenge_bytes(len), Some(len * 56));
        assert_eq!(Fq::challenge_bytes(len), Some(len * 56));
    }
    // The TV bound needs 8 * 56 >= 254 + 159 + ceil(log2(len)): at the cap
    // len < 2^19, and 448 >= 254 + 159 + 19 = 432.
    const { assert!(MAX_CHALLENGE_BYTES / 56 < 1 << 19 && 8 * 56 >= 254 + 159 + 19) };
    assert!(Fr::challenge_bytes(longest + 1).unwrap() > MAX_CHALLENGE_BYTES);
    assert_eq!(Fr::challenge_bytes(usize::MAX), None);
}

#[cfg(feature = "binary")]
#[test]
fn binary_round_trip_and_all_ones_are_canonical() {
    use jolt_field::CanonicalEncoding;

    fn binary<F: ChallengeCodec + CanonicalEncoding>() {
        round_trip::<F>();
        // One element squeezes a whole word and decodes its first bytes.
        let tape = vec![0xff; F::NUM_BYTES.next_multiple_of(8)];
        assert_eq!(F::challenge_bytes(1), Some(tape.len()));
        let element = F::from_bytes_le_checked(tape.get(..F::NUM_BYTES).unwrap()).unwrap();
        assert_eq!(F::decode_challenges(&tape, 1), Some(vec![element]));
    }
    binary::<F8>();
    binary::<F64>();
    binary::<F128>();
    binary::<F192>();
    assert_eq!(F192::challenge_bytes(usize::MAX), None);
    // F8 messages pad to a word; the padding bytes are not decoded.
    for len in 0..=33 {
        assert_eq!(F8::challenge_bytes(len), Some(len.div_ceil(8) * 8));
    }
    let mut tape = vec![0u8; 8];
    tape[..3].copy_from_slice(&[1, 2, 3]);
    let decoded = F8::decode_challenges(&tape, 3).unwrap();
    tape[3..].fill(0xff);
    assert_eq!(F8::decode_challenges(&tape, 3).unwrap(), decoded);
    assert!(F8::decode_challenges(&tape[..3], 3).is_none());
}

#[cfg(feature = "solinas")]
#[test]
fn pooled_message_sizes_meet_owner_bounds() {
    fn check<F: ChallengeCodec>(minimum: [usize; 3], actual: [usize; 3]) {
        for ((len, minimum), actual) in [1, 17, 28].into_iter().zip(minimum).zip(actual) {
            let bytes = F::challenge_bytes(len).unwrap();
            assert_eq!(bytes, actual);
            assert!((minimum..=minimum + 16).contains(&bytes));
        }
        assert_eq!(F::challenge_bytes(usize::MAX), None);
    }
    check::<Prime128OffsetA7F7>([32, 288, 464], [32, 288, 464]);
    check::<Prime64Offset59>([24, 152, 240], [24, 160, 248]);
    // Four-byte candidates come in pairs: 25 candidates round up to 26, and
    // 11 to 12.
    check::<Prime32Offset99>([28, 96, 140], [32, 104, 144]);
    check::<Ext2<Prime64Offset59>>([32, 288, 472], [32, 296, 472]);
    check::<FpExt4<Prime32Offset99>>([40, 300, 476], [48, 304, 480]);
}

#[cfg(feature = "solinas")]
#[test]
fn pooled_messages_are_whole_words() {
    fn check<F: ChallengeCodec>() {
        for len in 1..=300 {
            assert_eq!(F::challenge_bytes(len).unwrap() % 8, 0, "length {len}");
        }
    }
    check::<Prime24Offset3>();
    check::<Prime30Offset35>();
    check::<Prime31Offset19>();
    check::<Prime32Offset99>();
    check::<Prime40Offset195>();
    check::<Prime64Offset59>();
    check::<Prime128Offset275>();
    check::<Ext2<Prime31Offset19>>();
    check::<FpExt4<Prime24Offset3>>();
    check::<FpExt8<Prime32Offset99>>();
}

#[cfg(feature = "solinas")]
#[test]
fn unsupported_lengths_return_transcript_errors() {
    for len in [1 << 22, usize::MAX] {
        assert_eq!(Prime24Offset3::challenge_bytes(len), None);
        let mut prover = prover_transcript::<Keccak>(&SESSION, &INSTANCE);
        assert!(FieldTranscript::<Prime24Offset3>::challenge_vector(&mut prover, len).is_err());
        let mut verifier = verifier_transcript::<Keccak>(&SESSION, &INSTANCE, &[]);
        assert!(FieldTranscript::<Prime24Offset3>::challenge_vector(&mut verifier, len).is_err());
        assert!(verifier.check_eof().is_err());
    }
}

#[cfg(feature = "binary")]
#[test]
fn messages_above_the_byte_cap_are_rejected_before_touching_the_sponge() {
    let mut prover = prover_transcript::<Keccak>(&SESSION, &INSTANCE);
    let full = FieldTranscript::<F8>::challenge_vector(&mut prover, MAX_CHALLENGE_BYTES).unwrap();
    assert_eq!(full.len(), MAX_CHALLENGE_BYTES);
    let mut verifier = verifier_transcript::<Keccak>(&SESSION, &INSTANCE, &[]);
    assert_eq!(
        FieldTranscript::<F8>::challenge_vector(&mut verifier, MAX_CHALLENGE_BYTES).unwrap(),
        full
    );

    for len in [MAX_CHALLENGE_BYTES + 1, usize::MAX] {
        let mut prover = prover_transcript::<Keccak>(&SESSION, &INSTANCE);
        assert!(FieldTranscript::<F8>::challenge_vector(&mut prover, len).is_err());
        let mut fresh = prover_transcript::<Keccak>(&SESSION, &INSTANCE);
        assert_eq!(
            prover.verifier_message::<[u8; 16]>(),
            fresh.verifier_message::<[u8; 16]>()
        );
        let mut verifier = verifier_transcript::<Keccak>(&SESSION, &INSTANCE, &[]);
        assert!(FieldTranscript::<F8>::challenge_vector(&mut verifier, len).is_err());
        assert!(verifier.check_eof().is_err());
    }
}

/// An empty message skips the squeeze, so it leaves every sponge untouched,
/// including the Blake2b bridge, where an empty squeeze is not a no-op.
#[cfg(all(feature = "bn254", feature = "transcript-blake2b"))]
#[test]
fn empty_messages_leave_the_sponge_untouched() {
    use jolt_transcript::Blake2b512;
    use jolt_transcript::UniformBytes;
    use spongefish::DuplexSpongeInit;

    fn check<H: DuplexSpongeInit<U = u8> + UniformBytes>() {
        let mut prover = prover_transcript::<H>(&SESSION, &INSTANCE);
        assert!(FieldTranscript::<Fr>::challenge_vector(&mut prover, 0)
            .unwrap()
            .is_empty());
        let mut fresh = prover_transcript::<H>(&SESSION, &INSTANCE);
        assert_eq!(
            prover.verifier_message::<[u8; 16]>(),
            fresh.verifier_message::<[u8; 16]>()
        );
        let mut verifier = verifier_transcript::<H>(&SESSION, &INSTANCE, &[]);
        assert!(FieldTranscript::<Fr>::challenge_vector(&mut verifier, 0)
            .unwrap()
            .is_empty());
        let mut fresh = verifier_transcript::<H>(&SESSION, &INSTANCE, &[]);
        assert_eq!(
            verifier.verifier_message::<[u8; 16]>(),
            fresh.verifier_message::<[u8; 16]>()
        );
    }
    check::<Keccak>();
    check::<Blake2b512>();
}

#[cfg(feature = "solinas")]
#[test]
fn rejection_exhaustion_and_accepted_order() {
    use jolt_field::Ring;

    let mut tape = vec![0xff; Prime64Offset59::challenge_bytes(2).unwrap()];
    assert_eq!(Prime64Offset59::decode_challenges(&tape, 2), None);
    tape.get_mut(8..16)
        .unwrap()
        .copy_from_slice(&7u64.to_le_bytes());
    tape.get_mut(16..24)
        .unwrap()
        .copy_from_slice(&11u64.to_le_bytes());
    assert_eq!(
        Prime64Offset59::decode_challenges(&tape, 2),
        Some(vec![
            Prime64Offset59::from_u64(7),
            Prime64Offset59::from_u64(11)
        ])
    );
    assert!(Prime64Offset59::decode_challenges(tape.get(..tape.len() - 1).unwrap(), 2).is_none());
}

#[cfg(feature = "solinas")]
#[test]
fn subword_candidates_mask_unused_high_bits() {
    use jolt_field::Ring;

    fn check<F: ChallengeCodec + PartialEq + Debug + Ring>(candidate: [u8; 4]) {
        let mut tape = vec![0xff; F::challenge_bytes(1).unwrap()];
        tape.get_mut(..4).unwrap().copy_from_slice(&candidate);
        assert_eq!(F::decode_challenges(&tape, 1), Some(vec![F::from_u64(7)]));
    }
    check::<Prime24Offset3>([7, 0, 0, 0xff]);
    check::<Prime30Offset35>([7, 0, 0, 0xc0]);
    check::<Prime31Offset19>([7, 0, 0, 0x80]);
}

#[cfg(feature = "solinas")]
#[test]
fn candidates_at_the_modulus_boundary() {
    use jolt_field::PseudoMersenne;

    fn check<F: ChallengeCodec + PseudoMersenne + PartialEq + Debug>() {
        let p = (u128::MAX >> (128 - F::MODULUS_BITS)) - (F::OFFSET - 1);
        let width = <F as CanonicalBytes>::NUM_BYTES;
        let mut tape = vec![0xff; F::challenge_bytes(1).unwrap()];
        assert!(
            tape.len() >= 2 * width,
            "the pool has at least two candidates"
        );
        tape.get_mut(..width)
            .unwrap()
            .copy_from_slice(p.to_le_bytes().get(..width).unwrap());
        let largest = (p - 1).to_le_bytes();
        tape.get_mut(width..2 * width)
            .unwrap()
            .copy_from_slice(largest.get(..width).unwrap());
        let expected = F::from_bytes_le_checked(largest.get(..width).unwrap()).unwrap();
        assert_eq!(F::decode_challenges(&tape, 1), Some(vec![expected]));
    }
    check::<Prime24Offset3>();
    check::<Prime30Offset35>();
    check::<Prime31Offset19>();
    check::<Prime32Offset99>();
    check::<Prime40Offset195>();
    check::<Prime48Offset59>();
    check::<Prime56Offset27>();
    check::<Prime64Offset59>();
    check::<Prime128Offset275>();
}

#[cfg(feature = "solinas")]
#[test]
fn extension_coordinates_follow_the_single_base_pool() {
    type Base = Prime32Offset99;
    fn check<E: ChallengeCodec + ExtField<Base>>() {
        let count = 2 * E::DEGREE;
        let mut tape = vec![0xff; E::challenge_bytes(2).unwrap()];
        for (value, candidate) in (1u32..).zip(tape.chunks_exact_mut(Base::NUM_BYTES).skip(1)) {
            candidate.copy_from_slice(&value.to_le_bytes());
        }
        let bases = Base::decode_challenges(&tape, count).unwrap();
        let extensions = E::decode_challenges(&tape, 2).unwrap();
        for (element, coordinates) in extensions.iter().zip(bases.chunks_exact(E::DEGREE)) {
            assert_eq!(element.to_base_vec(), coordinates);
        }
    }
    check::<Ext2<Base>>();
    check::<FpExt4<Base>>();
    check::<FpExt8<Base>>();
}

fn hex_bytes(bytes: &[u8]) -> String {
    let mut hex = String::new();
    for byte in bytes {
        write!(&mut hex, "{byte:02x}").unwrap();
    }
    hex
}

fn canonical_hex<F: CanonicalBytes>(values: &[F]) -> String {
    let mut encoded = vec![0; F::NUM_BYTES * values.len()];
    for (value, bytes) in values.iter().zip(encoded.chunks_exact_mut(F::NUM_BYTES)) {
        value.to_bytes_le(bytes);
    }
    hex_bytes(&encoded)
}

fn known_answer<F: ChallengeCodec + CanonicalBytes + PartialEq + Debug>(
    len: usize,
    expected: &str,
    continuation: &str,
) {
    let mut prover = prover_transcript::<Keccak>(&SESSION, &INSTANCE);
    prover.send(&BytesMsg(b"before".to_vec()));
    let values = FieldTranscript::<F>::challenge_vector(&mut prover, len).unwrap();
    let tail: [u8; 16] = prover.verifier_message();
    prover.send(&BytesMsg(b"after".to_vec()));
    let mut verifier = verifier_transcript::<Keccak>(&SESSION, &INSTANCE, prover.narg_string());
    assert_eq!(verifier.prover_message::<BytesMsg>().unwrap().0, b"before");
    assert_eq!(
        FieldTranscript::<F>::challenge_vector(&mut verifier, len).unwrap(),
        values
    );
    assert_eq!(verifier.verifier_message::<[u8; 16]>(), tail);
    assert_eq!(verifier.prover_message::<BytesMsg>().unwrap().0, b"after");
    verifier.check_eof().unwrap();
    assert_eq!(canonical_hex(&values), expected);
    assert_eq!(hex_bytes(&tail), continuation);
}

#[cfg_attr(
    feature = "keccak256-digests",
    ignore = "pinned for SHA3-256 digests; keccak256-digests changes the session identifier"
)]
#[cfg(feature = "solinas")]
#[test]
fn solinas_known_answers() {
    // Pools of 16- and 8-byte candidates are already whole words; the Fp32
    // extension pool rounds its candidate count up to even (R5), so its next
    // squeeze starts at byte 48 instead of 44.
    known_answer::<Prime128OffsetA7F7>(
        1,
        "e1fd4d691fdfd5431ee0b45796a1b3ad",
        "663164ebd1ed79c41375a2c02aae4edf",
    );
    known_answer::<Prime64Offset59>(1, "e1fd4d691fdfd543", "1469a531ac276dd1663164ebd1ed79c4");
    known_answer::<FpExt4<Prime32Offset99>>(
        1,
        "e1fd4d691fdfd5431ee0b45796a1b3ad",
        "b5121e2e2bfabaae6f0832afec310e0b",
    );
}

#[cfg_attr(
    feature = "keccak256-digests",
    ignore = "pinned for SHA3-256 digests; keccak256-digests changes the session identifier"
)]
#[cfg(feature = "bn254")]
#[test]
fn bn254_known_answer() {
    // One element squeezes a fixed 56 bytes (R5), so the next squeeze starts
    // at byte 56.
    known_answer::<Fr>(
        1,
        "19726053acccc3deea86bae864e97dfddb01905d53d6899046819f89f9527906",
        "6f0832afec310e0b705df269fea7021a",
    );
}

#[cfg_attr(
    feature = "keccak256-digests",
    ignore = "pinned for SHA3-256 digests; keccak256-digests changes the session identifier"
)]
#[cfg(feature = "binary")]
#[test]
fn binary_known_answer() {
    known_answer::<F128>(
        1,
        "e1fd4d691fdfd5431ee0b45796a1b3ad",
        "7c6c0cc83ae5c82f1469a531ac276dd1",
    );
}

#[cfg_attr(
    feature = "keccak256-digests",
    ignore = "pinned for SHA3-256 digests; keccak256-digests changes the session identifier"
)]
#[cfg(feature = "binary")]
#[test]
fn keccak_rate_boundary_known_answers() {
    assert_eq!(F8::challenge_bytes(136), Some(136));
    assert_eq!(F8::challenge_bytes(137), Some(144));
    // 136 bytes fill one rate block exactly. 137 elements squeeze 144 bytes
    // and decode the first 137 (R5), so their continuation is the 136-element
    // stream from byte 144: it starts with the second half of that tail.
    known_answer::<F8>(
        136,
        concat!(
            "e1fd4d691fdfd5431ee0b45796a1b3ad7c6c0cc83ae5c82f1469a531ac276dd1",
            "663164ebd1ed79c41375a2c02aae4edfb5121e2e2bfabaae6f0832afec310e0b",
            "705df269fea7021a301b510ce1f141c953d3aed27fb138fea5eae4780cf8ee65",
            "dd5f1e33442d2393b7b1fd435f8945c5f1358f60155dd4e348924d872ea1e160",
            "a3f8ed296bcd9bce",
        ),
        "3dd469cc1281b9292e8e4b027efe5163",
    );
    known_answer::<F8>(
        137,
        concat!(
            "e1fd4d691fdfd5431ee0b45796a1b3ad7c6c0cc83ae5c82f1469a531ac276dd1",
            "663164ebd1ed79c41375a2c02aae4edfb5121e2e2bfabaae6f0832afec310e0b",
            "705df269fea7021a301b510ce1f141c953d3aed27fb138fea5eae4780cf8ee65",
            "dd5f1e33442d2393b7b1fd435f8945c5f1358f60155dd4e348924d872ea1e160",
            "a3f8ed296bcd9bce3d",
        ),
        "2e8e4b027efe5163565eeaaa98babf82",
    );
}

/// `append_values` against one absorb per value: the challenge after each is
/// the same exactly when every value's encoding is a whole number of words.
fn append_values_matches_per_value_absorbs<M: spongefish::Encoding>(values: &[M]) {
    use jolt_transcript::Transcript;
    let width = values[0].encode().as_ref().len();
    assert_eq!(width % 8, 0, "{width}-byte encoding");
    let mut batched = prover_transcript::<Keccak>(&SESSION, &INSTANCE);
    batched.append_values(values);
    let mut separate = prover_transcript::<Keccak>(&SESSION, &INSTANCE);
    for value in values {
        separate.public_message(value);
    }
    assert_eq!(
        batched.verifier_message::<[u8; 32]>(),
        separate.verifier_message::<[u8; 32]>(),
        "{width}-byte values"
    );
}

/// For every production width the one padded absorb is byte-identical to the
/// per-value absorbs it replaced: BN254 `Fr` and `Fq` (32 bytes), `Fp64` (8),
/// `Fp128` (16), their extensions, and Dory's compressed G1, G2 and GT bytes
/// (32, 64 and 384 one-byte values).
#[cfg(all(feature = "solinas", feature = "bn254"))]
#[test]
fn append_values_is_unchanged_for_whole_word_encodings() {
    use jolt_field::Ring;
    fn elements<F: Ring>(len: u64) -> Vec<F> {
        (1..=len)
            .map(|i| F::from_u64(i * 0x1234_5678_9abc))
            .collect()
    }
    for len in [1, 3, 8] {
        append_values_matches_per_value_absorbs(&elements::<Fr>(len));
        append_values_matches_per_value_absorbs(&elements::<Fq>(len));
        append_values_matches_per_value_absorbs(&elements::<Prime64Offset59>(len));
        append_values_matches_per_value_absorbs(&elements::<Prime128OffsetA7F7>(len));
        append_values_matches_per_value_absorbs(&elements::<Prime128Offset275>(len));
        append_values_matches_per_value_absorbs(&elements::<FpExt4<Prime32Offset99>>(len));
        append_values_matches_per_value_absorbs(&elements::<Ext2<Prime64Offset59>>(len));
    }
    for len in [32usize, 64, 384] {
        let group: Vec<u8> = (0..len).map(|i| i as u8 ^ 0x5a).collect();
        let mut batched = prover_transcript::<Keccak>(&SESSION, &INSTANCE);
        jolt_transcript::Transcript::append_values(&mut batched, &group);
        let mut separate = prover_transcript::<Keccak>(&SESSION, &INSTANCE);
        for byte in &group {
            separate.public_message(byte);
        }
        assert_eq!(
            batched.verifier_message::<[u8; 32]>(),
            separate.verifier_message::<[u8; 32]>(),
            "{len} group bytes"
        );
    }
}

/// An odd count of 4-byte values is padded once, at the end, to a whole word
/// (R6): three `Prime32Offset99` values absorb as their 12 bytes and 4 zeros,
/// so the next message starts on a word boundary. Keccak absorbs by
/// overwriting state bytes. In this fixture the 4 padding bytes land on state
/// bytes that are still zero and stay inside one rate block, so padding alone
/// would leave the next squeeze unchanged here. The check therefore absorbs
/// one more word after the padding.
#[cfg(feature = "solinas")]
#[test]
fn append_values_pads_an_odd_count_of_four_byte_values() {
    use jolt_field::Ring;
    use jolt_transcript::Transcript;
    use spongefish::Encoding;
    let values: Vec<Prime32Offset99> = (1..=3).map(|i| Prime32Offset99::from_u64(i * 77)).collect();
    let mut padded = [0u8; 16];
    for (chunk, value) in padded.chunks_mut(4).zip(&values) {
        let bytes = value.encode();
        assert_eq!(bytes.as_ref().len(), 4);
        chunk.copy_from_slice(bytes.as_ref());
    }
    let next = 0x0123_4567_89ab_cdef_u64;
    let mut batched = prover_transcript::<Keccak>(&SESSION, &INSTANCE);
    batched.append_values(&values);
    batched.append(&next);
    let mut expected = prover_transcript::<Keccak>(&SESSION, &INSTANCE);
    expected.public_message(&padded);
    expected.public_message(&next);
    let mut unpadded = prover_transcript::<Keccak>(&SESSION, &INSTANCE);
    for value in &values {
        unpadded.public_message(value);
    }
    unpadded.public_message(&next);
    let challenge = batched.verifier_message::<[u8; 32]>();
    assert_eq!(challenge, expected.verifier_message::<[u8; 32]>());
    assert_ne!(challenge, unpadded.verifier_message::<[u8; 32]>());
}

/// `append` absorbs an encoding as it is, so it accepts only whole words: a
/// bare 4-byte `Prime32Offset99` would leave the next message off a word
/// boundary. Debug builds reject it; such a value goes through
/// `append_values`, which pads it.
#[cfg(all(feature = "solinas", debug_assertions))]
#[test]
#[should_panic(expected = "whole number of words")]
fn append_rejects_a_sub_word_encoding() {
    use jolt_field::Ring;
    use jolt_transcript::Transcript;
    let mut transcript = prover_transcript::<Keccak>(&SESSION, &INSTANCE);
    transcript.append(&Prime32Offset99::from_u64(77));
}
