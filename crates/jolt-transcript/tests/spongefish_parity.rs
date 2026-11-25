#![cfg(any(feature = "transcript-keccak", feature = "transcript-poseidon"))]
#![expect(clippy::unwrap_used, reason = "tests")]

#[cfg(feature = "transcript-poseidon")]
use jolt_transcript::prover_transcript;
use jolt_transcript::{protocol_tag, verifier_transcript, Encoding, Transcript};

const SESSION: [u8; 32] = protocol_tag(b"jolt/spongefish-parity");
const STATEMENT: &[u8; 4] = b"stmt";

struct Chunk<'a>(&'a [u8]);

impl Encoding for Chunk<'_> {
    fn encode(&self) -> impl AsRef<[u8]> {
        self.0
    }
}

// Poseidon frames each absorb with its length, so a batch split across
// absorbs fails here as well as wrong bytes.
#[cfg(feature = "transcript-poseidon")]
#[test]
fn poseidon_absorbs_each_batch_once() {
    use jolt_transcript::PoseidonSponge;

    let mut prover = prover_transcript::<PoseidonSponge>(&SESSION, STATEMENT);
    let mut verifier = verifier_transcript::<PoseidonSponge>(&SESSION, STATEMENT, &[]);
    let mut stock = verifier_transcript::<PoseidonSponge>(&SESSION, STATEMENT, &[]);
    let payload: Vec<u8> = (0..=u8::MAX).collect();
    let unaligned = &payload[1..];

    prover.append_values::<Chunk<'_>>(&[]);
    verifier.append_values::<Chunk<'_>>(&[]);
    for len in [0usize, 7, 8, 9, 63, 64, 65, 129] {
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
