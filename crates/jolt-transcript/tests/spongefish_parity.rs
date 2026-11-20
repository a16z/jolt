#![cfg(any(
    feature = "transcript-keccak",
    feature = "transcript-blake2b",
    feature = "transcript-poseidon"
))]
#![expect(clippy::unwrap_used, reason = "tests")]

#[cfg(all(feature = "bn254", feature = "transcript-blake2b"))]
use jolt_field::Fq;
#[cfg(all(feature = "bn254", feature = "transcript-keccak"))]
use jolt_field::Fr;
#[cfg(all(
    feature = "binary",
    any(feature = "transcript-keccak", feature = "transcript-blake2b")
))]
use jolt_field::F8;
#[cfg(all(feature = "binary", feature = "transcript-keccak"))]
use jolt_field::{F192, F64};
use jolt_transcript::{protocol_tag, prover_transcript, verifier_transcript, Encoding, Transcript};
#[cfg(all(
    any(feature = "bn254", feature = "binary"),
    any(feature = "transcript-keccak", feature = "transcript-blake2b")
))]
use jolt_transcript::{ChallengeCodec, FieldTranscript, UniformBytes};
use spongefish::DuplexSpongeInit;
#[cfg(all(
    any(feature = "bn254", feature = "binary"),
    any(feature = "transcript-keccak", feature = "transcript-blake2b")
))]
use std::fmt::Debug;

const SESSION: [u8; 32] = protocol_tag(b"jolt/spongefish-parity");
const STATEMENT: &[u8; 4] = b"stmt";

struct Chunk<'a>(&'a [u8]);

impl Encoding for Chunk<'_> {
    fn encode(&self) -> impl AsRef<[u8]> {
        self.0
    }
}

fn batch_parity<H: DuplexSpongeInit<U = u8>>() {
    let mut prover = prover_transcript::<H>(&SESSION, STATEMENT);
    let mut verifier = verifier_transcript::<H>(&SESSION, STATEMENT, &[]);
    let mut stock = verifier_transcript::<H>(&SESSION, STATEMENT, &[]);
    let payload: Vec<u8> = (0..=u8::MAX).collect();
    let unaligned = &payload[1..];

    prover.append_values::<Chunk<'_>>(&[]);
    verifier.append_values::<Chunk<'_>>(&[]);
    for len in [0usize, 7, 8, 9, 63, 64, 65, 127, 128, 129, 135, 136, 137] {
        let message = &unaligned[..len];
        let (first, rest) = message.split_at(len.min(3));
        let (middle, last) = rest.split_at(rest.len().min(62));
        let chunks = [Chunk(first), Chunk(middle), Chunk(last)];
        let mut padded = message.to_vec();
        padded.resize(len.next_multiple_of(8), 0);
        stock.public_message(&Chunk(&padded));
        prover.append_values(&chunks);
        verifier.append_values(&chunks);

        let expected = stock.verifier_message_as(8, |tape| <[u8; 8]>::try_from(tape).unwrap());
        assert_eq!(
            prover.verifier_message::<[u8; 8]>(),
            expected,
            "length {len}"
        );
        assert_eq!(
            verifier.verifier_message::<[u8; 8]>(),
            expected,
            "length {len}"
        );
    }
    verifier.check_eof().unwrap();
}

#[cfg(feature = "transcript-keccak")]
#[test]
fn spilled_batch_releases_each_encoding_before_the_next() {
    use jolt_transcript::Keccak;
    use std::cell::{RefCell, RefMut};

    struct ScratchValue<'a> {
        byte: u8,
        scratch: &'a RefCell<[u8; 47]>,
    }

    struct ScratchEncoding<'a>(RefMut<'a, [u8; 47]>);

    impl AsRef<[u8]> for ScratchEncoding<'_> {
        fn as_ref(&self) -> &[u8] {
            &*self.0
        }
    }

    impl Encoding for ScratchValue<'_> {
        fn encode(&self) -> impl AsRef<[u8]> {
            let mut scratch = self.scratch.borrow_mut();
            scratch.fill(self.byte);
            ScratchEncoding(scratch)
        }
    }

    let scratch = RefCell::new([0; 47]);
    let values = [1, 2, 3].map(|byte| ScratchValue {
        byte,
        scratch: &scratch,
    });
    let mut verifier = verifier_transcript::<Keccak>(&SESSION, STATEMENT, &[]);
    verifier.append_values(&values);

    let mut stock = verifier_transcript::<Keccak>(&SESSION, STATEMENT, &[]);
    let mut expected = [[1; 47], [2; 47], [3; 47]].concat();
    expected.extend_from_slice(&[0; 3]);
    stock.public_message(&Chunk(&expected));
    assert_eq!(
        verifier.verifier_message::<[u8; 64]>(),
        stock.verifier_message::<[u8; 64]>()
    );
    verifier.check_eof().unwrap();
}

#[cfg(all(
    any(feature = "bn254", feature = "binary"),
    any(feature = "transcript-keccak", feature = "transcript-blake2b")
))]
fn field_parity<H, F>(lengths: &[usize])
where
    H: UniformBytes + DuplexSpongeInit<U = u8>,
    F: ChallengeCodec + Eq + Debug,
{
    let mut prover = prover_transcript::<H>(&SESSION, STATEMENT);
    let mut verifier = verifier_transcript::<H>(&SESSION, STATEMENT, &[]);
    let mut stock = verifier_transcript::<H>(&SESSION, STATEMENT, &[]);
    for &len in lengths {
        prover.append(&0x0123_4567_89ab_cdefu64);
        verifier.append(&0x0123_4567_89ab_cdefu64);
        stock.public_message(&0x0123_4567_89ab_cdefu64);
        let expected = if len == 0 {
            Vec::new()
        } else {
            stock.verifier_message_as(F::challenge_bytes(len).unwrap(), |tape| {
                F::decode_challenges(tape, len).unwrap()
            })
        };
        if len == 1 {
            let expected = expected.into_iter().next().unwrap();
            assert_eq!(
                FieldTranscript::<F>::challenge(&mut prover).unwrap(),
                expected
            );
            assert_eq!(
                FieldTranscript::<F>::challenge(&mut verifier).unwrap(),
                expected
            );
        } else {
            assert_eq!(
                FieldTranscript::<F>::challenge_vector(&mut prover, len).unwrap(),
                expected
            );
            assert_eq!(
                FieldTranscript::<F>::challenge_vector(&mut verifier, len).unwrap(),
                expected
            );
        }

        let tail = stock.verifier_message_as(16, |tape| <[u8; 16]>::try_from(tape).unwrap());
        assert_eq!(prover.verifier_message::<[u8; 16]>(), tail);
        assert_eq!(verifier.verifier_message::<[u8; 16]>(), tail);
    }
    verifier.check_eof().unwrap();
}

#[cfg(feature = "transcript-keccak")]
#[test]
fn keccak_matches_stock_spongefish() {
    use jolt_transcript::Keccak;

    batch_parity::<Keccak>();
    #[cfg(feature = "bn254")]
    field_parity::<Keccak, Fr>(&[0, 1, 2, 3]);
    #[cfg(feature = "binary")]
    {
        field_parity::<Keccak, F8>(&[0, 1, 8, 9, 17, 56, 128, 136, 137]);
        field_parity::<Keccak, F64>(&[1, 2, 3, 7, 16, 17, 18]);
        field_parity::<Keccak, F192>(&[1, 2, 6]);
    }
}

#[cfg(feature = "transcript-blake2b")]
#[test]
fn blake2b_matches_stock_spongefish() {
    use jolt_transcript::Blake2b512;

    batch_parity::<Blake2b512>();
    #[cfg(feature = "bn254")]
    field_parity::<Blake2b512, Fq>(&[0, 1, 2, 3]);
    #[cfg(feature = "binary")]
    field_parity::<Blake2b512, F8>(&[0, 1, 56, 64, 65]);
}

#[cfg(feature = "transcript-poseidon")]
#[test]
fn poseidon_matches_stock_spongefish() {
    use jolt_transcript::PoseidonSponge;

    batch_parity::<PoseidonSponge>();
}
