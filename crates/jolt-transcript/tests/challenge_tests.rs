#![cfg(all(
    feature = "transcript-keccak",
    any(feature = "solinas", feature = "bn254", feature = "binary")
))]
#![expect(clippy::unwrap_used, reason = "tests")]

use jolt_field::CanonicalBytes;
use jolt_transcript::{
    prover_transcript, verifier_transcript, BytesMsg, ChallengeCodec, ProverTranscript,
    VerifierTranscript,
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
#[cfg(feature = "binary")]
use jolt_transcript::MAX_CHALLENGE_BYTES;

const SESSION: &[u8] = b"jolt/challenge-codecs/v1";
const INSTANCE: [u8; 32] = [0x42; 32];

fn round_trip<F: ChallengeCodec + PartialEq + Debug>() {
    for len in [1, 17, 28] {
        let mut prover = prover_transcript::<Keccak>(SESSION, INSTANCE);
        prover.prover_message(&BytesMsg(b"before".to_vec()));
        let expected = ProverTranscript::field_challenges::<F>(&mut prover, len).unwrap();
        assert_eq!(expected.len(), len);
        prover.prover_message(&BytesMsg(b"after".to_vec()));
        let tail: [u8; 32] = prover.verifier_message();

        let mut verifier = verifier_transcript::<Keccak>(SESSION, INSTANCE, prover.narg_string());
        assert_eq!(verifier.prover_message::<BytesMsg>().unwrap().0, b"before");
        assert_eq!(
            VerifierTranscript::field_challenges::<F>(&mut verifier, len).unwrap(),
            expected
        );
        assert_eq!(verifier.prover_message::<BytesMsg>().unwrap().0, b"after");
        assert_eq!(verifier.verifier_message::<[u8; 32]>(), tail);
        verifier.check_eof().unwrap();

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
    for len in 1..=8 {
        assert_eq!(Fr::challenge_bytes(len), Some(len * 52));
        assert_eq!(Fq::challenge_bytes(len), Some(len * 52));
    }
    for len in 9..=128 {
        assert_eq!(Fr::challenge_bytes(len), Some(len * 53));
        assert_eq!(Fq::challenge_bytes(len), Some(len * 53));
    }
    for (len, minimum) in [(1, 51), (17, 901), (28, 1484)] {
        assert!(Fr::challenge_bytes(len).unwrap() >= minimum);
        assert!(Fq::challenge_bytes(len).unwrap() >= minimum);
    }
    assert_eq!(Fr::challenge_bytes(usize::MAX), None);
}

#[cfg(feature = "binary")]
#[test]
fn binary_round_trip_and_all_ones_are_canonical() {
    use jolt_field::CanonicalEncoding;

    fn binary<F: ChallengeCodec + CanonicalEncoding>() {
        round_trip::<F>();
        let tape = vec![0xff; F::NUM_BYTES];
        assert_eq!(F::challenge_bytes(1), Some(tape.len()));
        assert_eq!(
            F::decode_challenges(&tape, 1),
            Some(vec![F::from_bytes_le_checked(&tape).unwrap()])
        );
    }
    binary::<F8>();
    binary::<F64>();
    binary::<F128>();
    binary::<F192>();
    assert_eq!(F192::challenge_bytes(usize::MAX), None);
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
    check::<Prime32Offset99>([28, 96, 140], [32, 100, 144]);
    check::<Ext2<Prime64Offset59>>([32, 288, 472], [32, 296, 472]);
    check::<FpExt4<Prime32Offset99>>([40, 300, 476], [44, 304, 480]);
}

#[cfg(feature = "solinas")]
#[test]
fn unsupported_lengths_return_transcript_errors() {
    for len in [1 << 22, usize::MAX] {
        assert_eq!(Prime24Offset3::challenge_bytes(len), None);
        let mut prover = prover_transcript::<Keccak>(SESSION, INSTANCE);
        assert!(ProverTranscript::field_challenges::<Prime24Offset3>(&mut prover, len).is_err());
        let mut verifier = verifier_transcript::<Keccak>(SESSION, INSTANCE, &[]);
        assert!(
            VerifierTranscript::field_challenges::<Prime24Offset3>(&mut verifier, len).is_err()
        );
        assert!(verifier.check_eof().is_err());
    }
}

#[cfg(feature = "binary")]
#[test]
fn messages_above_the_byte_cap_are_rejected_before_touching_the_sponge() {
    let mut prover = prover_transcript::<Keccak>(SESSION, INSTANCE);
    let full = ProverTranscript::field_challenges::<F8>(&mut prover, MAX_CHALLENGE_BYTES).unwrap();
    assert_eq!(full.len(), MAX_CHALLENGE_BYTES);
    let mut verifier = verifier_transcript::<Keccak>(SESSION, INSTANCE, &[]);
    assert_eq!(
        VerifierTranscript::field_challenges::<F8>(&mut verifier, MAX_CHALLENGE_BYTES).unwrap(),
        full
    );

    for len in [MAX_CHALLENGE_BYTES + 1, usize::MAX] {
        let mut prover = prover_transcript::<Keccak>(SESSION, INSTANCE);
        assert!(ProverTranscript::field_challenges::<F8>(&mut prover, len).is_err());
        let mut fresh = prover_transcript::<Keccak>(SESSION, INSTANCE);
        assert_eq!(
            prover.verifier_message::<[u8; 16]>(),
            fresh.verifier_message::<[u8; 16]>()
        );
        let mut verifier = verifier_transcript::<Keccak>(SESSION, INSTANCE, &[]);
        assert!(VerifierTranscript::field_challenges::<F8>(&mut verifier, len).is_err());
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
        let mut prover = prover_transcript::<H>(SESSION, INSTANCE);
        assert!(ProverTranscript::field_challenges::<Fr>(&mut prover, 0)
            .unwrap()
            .is_empty());
        let mut fresh = prover_transcript::<H>(SESSION, INSTANCE);
        assert_eq!(
            prover.verifier_message::<[u8; 16]>(),
            fresh.verifier_message::<[u8; 16]>()
        );
        let mut verifier = verifier_transcript::<H>(SESSION, INSTANCE, &[]);
        assert!(VerifierTranscript::field_challenges::<Fr>(&mut verifier, 0)
            .unwrap()
            .is_empty());
        let mut fresh = verifier_transcript::<H>(SESSION, INSTANCE, &[]);
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
    let mut prover = prover_transcript::<Keccak>(SESSION, INSTANCE);
    prover.prover_message(&BytesMsg(b"before".to_vec()));
    let values = ProverTranscript::field_challenges::<F>(&mut prover, len).unwrap();
    let tail: [u8; 16] = prover.verifier_message();
    prover.prover_message(&BytesMsg(b"after".to_vec()));
    let mut verifier = verifier_transcript::<Keccak>(SESSION, INSTANCE, prover.narg_string());
    assert_eq!(verifier.prover_message::<BytesMsg>().unwrap().0, b"before");
    assert_eq!(
        VerifierTranscript::field_challenges::<F>(&mut verifier, len).unwrap(),
        values
    );
    assert_eq!(verifier.verifier_message::<[u8; 16]>(), tail);
    assert_eq!(verifier.prover_message::<BytesMsg>().unwrap().0, b"after");
    verifier.check_eof().unwrap();
    assert_eq!(canonical_hex(&values), expected);
    assert_eq!(hex_bytes(&tail), continuation);
}

#[cfg(feature = "solinas")]
#[test]
fn solinas_known_answers() {
    known_answer::<Prime128OffsetA7F7>(
        1,
        "a8ae1872291b8d7798b5ab4ab3b352ea",
        "763298e14f37d14cf41bb84cc8d9626c",
    );
    known_answer::<Prime64Offset59>(1, "a8ae1872291b8d77", "d1398de82241f4bf763298e14f37d14c");
    known_answer::<FpExt4<Prime32Offset99>>(
        1,
        "a8ae1872291b8d7798b5ab4ab3b352ea",
        "c8d9626c9b7d64e129482e411eb04927",
    );
}

#[cfg(feature = "bn254")]
#[test]
fn bn254_known_answer() {
    known_answer::<Fr>(
        1,
        "0303b9b655d450a18ec44f2bd452e5dfcddb2a1843904d06df1e9c6044adfc27",
        "29482e411eb0492749644af23906381a",
    );
}

#[cfg(feature = "binary")]
#[test]
fn binary_known_answer() {
    known_answer::<F128>(
        1,
        "a8ae1872291b8d7798b5ab4ab3b352ea",
        "6a30a074554faa50d1398de82241f4bf",
    );
}

#[cfg(feature = "binary")]
#[test]
fn keccak_rate_boundary_known_answers() {
    assert_eq!(F8::challenge_bytes(136), Some(136));
    assert_eq!(F8::challenge_bytes(137), Some(137));
    known_answer::<F8>(
        136,
        concat!(
            "a8ae1872291b8d7798b5ab4ab3b352ea6a30a074554faa50d1398de82241f4bf",
            "763298e14f37d14cf41bb84cc8d9626c9b7d64e129482e411eb0492749644af2",
            "3906381aac448b91a19086d90f1a056563ff682bf8217185805153d605e02d2e",
            "f0900a979cb17c13ae4fe682c11a32547071a1f1f56a4553c145fa2195cde255",
            "79946a922be123d5",
        ),
        "ecb3e78abcfe26e9c7da670f99e1eb2a",
    );
    known_answer::<F8>(
        137,
        concat!(
            "a8ae1872291b8d7798b5ab4ab3b352ea6a30a074554faa50d1398de82241f4bf",
            "763298e14f37d14cf41bb84cc8d9626c9b7d64e129482e411eb0492749644af2",
            "3906381aac448b91a19086d90f1a056563ff682bf8217185805153d605e02d2e",
            "f0900a979cb17c13ae4fe682c11a32547071a1f1f56a4553c145fa2195cde255",
            "79946a922be123d5ec",
        ),
        "b3e78abcfe26e9c7da670f99e1eb2abc",
    );
}
