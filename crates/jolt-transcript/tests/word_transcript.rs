#![expect(clippy::unwrap_used, reason = "tests construct canonical fixtures")]
use jolt_field::solinas::Prime128OffsetA7F7 as Field;
use jolt_field::CanonicalEncoding;
use jolt_transcript::{AppendToTranscript, Transcript};

#[derive(Default)]
struct Capture {
    messages: Vec<Vec<u8>>,
}

impl Transcript for Capture {
    type Challenge = ();

    fn new(_: &'static [u8]) -> Self {
        Self::default()
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        self.messages.push(bytes.to_vec());
    }

    fn challenge(&mut self) {}

    fn state(&self) -> [u8; 32] {
        [0; 32]
    }
}

#[test]
fn word_scalar_message_matches_wire_contract() {
    let value = Field::from_u128_checked(0x0102_0304_0506_0708_1112_1314_1516_1718).unwrap();
    let mut transcript = Capture::default();
    transcript.append_scalars(b"x", &[value]);
    let mut expected = vec![0; 48];
    expected[0] = b'x';
    expected[31] = 1;
    expected[32..].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 19, 20, 21, 22, 23, 24]);
    assert_eq!(transcript.messages, vec![expected]);

    let mut scalar = Capture::default();
    value.append_to_transcript(&mut scalar);
    assert_eq!(
        scalar.messages,
        vec![vec![1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 19, 20, 21, 22, 23, 24,]]
    );
}

#[test]
fn word_empty_and_multiple_scalars_keep_one_message() {
    let mut empty = Capture::default();
    empty.append_scalars::<Field>(b"", &[]);
    assert_eq!(empty.messages, vec![vec![0; 32]]);

    let mut transcript = Capture::default();
    let values = [
        Field::from_u128_checked(0).unwrap(),
        Field::from_u128_checked(1).unwrap(),
    ];
    transcript.append_scalars(b"", &values);
    let mut expected = vec![0; 64];
    expected[31] = 2;
    expected[63] = 1;
    assert_eq!(transcript.messages, vec![expected]);
}

#[cfg(feature = "transcript-blake2b")]
#[test]
fn word_scalar_draw_matches_existing_independent_digest_vector() {
    use jolt_transcript::LegacyBlake2bTranscript;

    let mut transcript = LegacyBlake2bTranscript::<Field>::new(b"jolt transcript kat");
    transcript.append_bytes(b"jolt says hello");
    // Same independent hashlib vector as legacy_digest_tests.rs.
    let state = [
        0x25, 0xa4, 0x96, 0x91, 0x2b, 0x5b, 0x75, 0xd6, 0x50, 0x4c, 0xbc, 0x20, 0x4a, 0x25, 0xb1,
        0x3d, 0x61, 0x21, 0xe6, 0x26, 0x6a, 0xc0, 0x56, 0x58, 0x96, 0x89, 0xc4, 0x3d, 0xc0, 0x6f,
        0x91, 0x42,
    ];
    let challenge = transcript.challenge_scalar();
    assert_eq!(transcript.state(), state);
    assert_eq!(
        challenge.to_u128_checked(),
        Some(0x3db1_254a_20bc_4c50_d675_5b2b_9196_a425)
    );
}
