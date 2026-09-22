//! Constrained unkeyed BLAKE2b, legacy Jolt framing, and spongefish duplex state.
//!
//! Implements RFC 7693 sections 2.5–3.3. Digest size is part of initialization:
//! BLAKE2b-256 is not truncated BLAKE2b-512. Message length and round counter
//! are public circuit-shape parameters, not unconstrained private witnesses.
use jolt_field::Fr;
use jolt_r1cs::bn254_bits::{BitsError, ByteVar, Word64Var};
use jolt_r1cs::R1csBuilder;
use thiserror::Error;

use crate::MAX_LABEL_LEN;

const IV: [u64; 8] = [
    0x6a09_e667_f3bc_c908,
    0xbb67_ae85_84ca_a73b,
    0x3c6e_f372_fe94_f82b,
    0xa54f_f53a_5f1d_36f1,
    0x510e_527f_ade6_82d1,
    0x9b05_688c_2b3e_6c1f,
    0x1f83_d9ab_fb41_bd6b,
    0x5be0_cd19_137e_2179,
];
const SIGMA: [[usize; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

/// Invalid public framing, variable indices, or internal fixed-size layout.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum Blake2bR1csError {
    /// A byte expression references an unavailable variable.
    #[error(transparent)]
    Bits(#[from] BitsError),
    /// The padded message length cannot be represented.
    #[error("message length exceeds the addressable circuit layout")]
    MessageTooLong,
    /// An internal conversion violated the fixed block or digest shape.
    #[error("internal BLAKE2b digest or block length mismatch")]
    InvalidLength,
    /// Public initialization labels have at most 32 bytes.
    #[error("legacy transcript label exceeds 32 bytes")]
    LabelTooLong,
    /// The next transition would overflow the public u32 schedule.
    #[error("legacy transcript round counter exhausted")]
    RoundOverflow,
    /// The requested read exceeds the complete u64 block-counter domain.
    #[error("Blake2b counter stream exhausted")]
    StreamExhausted,
}

/// Full unkeyed BLAKE2b-256, with message bytes constrained by their handles.
pub fn blake2b256(
    builder: &mut R1csBuilder<Fr>,
    message: &[ByteVar],
) -> Result<[ByteVar; 32], Blake2bR1csError> {
    hash::<32>(builder, message)
}

/// Full unkeyed BLAKE2b-512; this does not implement spongefish's duplex framing.
pub fn blake2b512(
    builder: &mut R1csBuilder<Fr>,
    message: &[ByteVar],
) -> Result<[ByteVar; 64], Blake2bR1csError> {
    hash::<64>(builder, message)
}

#[expect(
    clippy::indexing_slicing,
    reason = "fixed IV and eight-word output use indices generated within their bounds"
)]
fn hash<const N: usize>(
    builder: &mut R1csBuilder<Fr>,
    message: &[ByteVar],
) -> Result<[ByteVar; N], Blake2bR1csError> {
    for byte in message {
        byte.validate_indices(builder)?;
    }
    let blocks = message.len().div_ceil(128).max(1);
    let padded_len = blocks
        .checked_mul(128)
        .ok_or(Blake2bR1csError::MessageTooLong)?;
    let mut padded = message.to_vec();
    padded.resize_with(padded_len, || ByteVar::constant(0));
    let mut h: [Word64Var; 8] = std::array::from_fn(|i| {
        Word64Var::constant(IV[i] ^ if i == 0 { 0x0101_0000 ^ N as u64 } else { 0 })
    });
    for (index, chunk) in padded.chunks_exact(128).enumerate() {
        let words: Vec<_> = chunk
            .chunks_exact(8)
            .map(|bytes| {
                let bytes: &[ByteVar; 8] = bytes
                    .try_into()
                    .map_err(|_| Blake2bR1csError::InvalidLength)?;
                Ok(Word64Var::from_le_bytes(bytes))
            })
            .collect::<Result<_, Blake2bR1csError>>()?;
        let m: [Word64Var; 16] = words
            .try_into()
            .map_err(|_| Blake2bR1csError::InvalidLength)?;
        let last = index + 1 == blocks;
        let counter = if last {
            message.len() as u128
        } else {
            (index as u128 + 1) * 128
        };
        compress(builder, &mut h, &m, counter, last)?;
    }
    let bytes: Vec<_> = h.iter().flat_map(Word64Var::to_le_bytes).take(N).collect();
    bytes
        .try_into()
        .map_err(|_| Blake2bR1csError::InvalidLength)
}

#[expect(
    clippy::indexing_slicing,
    reason = "all state and message indices are the fixed RFC 7693 schedule"
)]
fn compress(
    builder: &mut R1csBuilder<Fr>,
    h: &mut [Word64Var; 8],
    m: &[Word64Var; 16],
    counter: u128,
    last: bool,
) -> Result<(), Blake2bR1csError> {
    let mut v: [Word64Var; 16] = std::array::from_fn(|i| {
        if i < 8 {
            h[i].clone()
        } else {
            Word64Var::constant(IV[i - 8])
        }
    });
    v[12] = v[12].xor(builder, &Word64Var::constant(counter as u64))?;
    v[13] = v[13].xor(builder, &Word64Var::constant((counter >> 64) as u64))?;
    v[14] = v[14].xor(
        builder,
        &Word64Var::constant(if last { u64::MAX } else { 0 }),
    )?;
    for round in 0..12 {
        let s = SIGMA[round % 10];
        for (a, b, c, d, x, y) in [
            (0, 4, 8, 12, 0, 1),
            (1, 5, 9, 13, 2, 3),
            (2, 6, 10, 14, 4, 5),
            (3, 7, 11, 15, 6, 7),
            (0, 5, 10, 15, 8, 9),
            (1, 6, 11, 12, 10, 11),
            (2, 7, 8, 13, 12, 13),
            (3, 4, 9, 14, 14, 15),
        ] {
            mix(builder, &mut v, [a, b, c, d], &m[s[x]], &m[s[y]])?;
        }
    }
    for i in 0..8 {
        h[i] = h[i].xor(builder, &v[i])?.xor(builder, &v[i + 8])?;
    }
    Ok(())
}

#[expect(
    clippy::indexing_slicing,
    reason = "private caller supplies only the eight fixed valid G index tuples"
)]
fn mix(
    builder: &mut R1csBuilder<Fr>,
    v: &mut [Word64Var; 16],
    [a, b, c, d]: [usize; 4],
    x: &Word64Var,
    y: &Word64Var,
) -> Result<(), BitsError> {
    v[a] = v[a].add_three(builder, &v[b], x)?;
    v[d] = v[d].xor(builder, &v[a])?.rotate_right(32);
    v[c] = v[c].add(builder, &v[d])?;
    v[b] = v[b].xor(builder, &v[c])?.rotate_right(24);
    v[a] = v[a].add_three(builder, &v[b], y)?;
    v[d] = v[d].xor(builder, &v[a])?.rotate_right(16);
    v[c] = v[c].add(builder, &v[d])?;
    v[b] = v[b].xor(builder, &v[c])?.rotate_right(63);
    Ok(())
}

/// Legacy Jolt's chained Blake2b-256 state with a fixed, public round schedule.
///
/// This is not the spongefish `Blake2bTranscript`. Values must remain in their
/// allocating builder. Output-state bytes remain constrained private handles
/// until the caller binds them to its statement or subsequent transcript work.
#[derive(Clone, Debug)]
pub struct LegacyBlake2bVar {
    state: [ByteVar; 32],
    round: u32,
}

impl LegacyBlake2bVar {
    /// Hash the public label padded with zeros to exactly 32 bytes.
    pub fn new(builder: &mut R1csBuilder<Fr>, label: &[u8]) -> Result<Self, Blake2bR1csError> {
        if label.len() > MAX_LABEL_LEN {
            return Err(Blake2bR1csError::LabelTooLong);
        }
        let mut padded: Vec<_> = label.iter().copied().map(ByteVar::constant).collect();
        padded.resize_with(MAX_LABEL_LEN, || ByteVar::constant(0));
        Ok(Self {
            state: blake2b256(builder, &padded)?,
            round: 0,
        })
    }

    /// Hash state || 28 zero bytes || round_be_u32 || payload, then advance round.
    pub fn append_bytes(
        &mut self,
        builder: &mut R1csBuilder<Fr>,
        payload: &[ByteVar],
    ) -> Result<(), Blake2bR1csError> {
        let next = self
            .round
            .checked_add(1)
            .ok_or(Blake2bR1csError::RoundOverflow)?;
        let capacity = payload
            .len()
            .checked_add(64)
            .ok_or(Blake2bR1csError::MessageTooLong)?;
        let mut message = Vec::with_capacity(capacity);
        message.extend(self.state.iter().cloned());
        message.extend((0..28).map(|_| ByteVar::constant(0)));
        message.extend(self.round.to_be_bytes().into_iter().map(ByteVar::constant));
        message.extend(payload.iter().cloned());
        self.state = blake2b256(builder, &message)?;
        self.round = next;
        Ok(())
    }

    /// Return constrained state bytes for subsequent transitions or statement binding.
    pub fn state(&self) -> &[ByteVar; 32] {
        &self.state
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests bind vectors and mutate complete witnesses"
)]
mod tests {
    use super::*;
    use blake2::{
        digest::{consts::U32, Digest},
        Blake2b, Blake2b512,
    };
    use jolt_field::Ring;
    use jolt_r1cs::{LinearCombination, Variable};

    fn bind(builder: &mut R1csBuilder<Fr>, output: &[ByteVar], expected: &[u8]) {
        assert_eq!(output.len(), expected.len());
        for (byte, &value) in output.iter().zip(expected) {
            builder.assert_equal(
                byte.expression(),
                LinearCombination::constant(Fr::from_u64(u64::from(value))),
            );
        }
    }

    #[test]
    fn hash_matches_native_across_block_boundaries() {
        for length in [0, 3, 127, 128, 129] {
            let input: Vec<_> = (0..length).map(|i| (i % 251) as u8).collect();
            for wide in [false, true] {
                let mut builder = R1csBuilder::new();
                let bytes: Vec<_> = input
                    .iter()
                    .map(|&x| ByteVar::allocate(&mut builder, Some(x)))
                    .collect();
                if wide {
                    let hash = blake2b512(&mut builder, &bytes).unwrap();
                    bind(&mut builder, &hash, &Blake2b512::digest(&input));
                } else {
                    let hash = blake2b256(&mut builder, &bytes).unwrap();
                    bind(&mut builder, &hash, &Blake2b::<U32>::digest(&input));
                }
                let witness = builder.witness().unwrap();
                assert!(
                    builder.into_matrices().check_witness(&witness).is_ok(),
                    "length={length}, wide={wide}"
                );
            }
        }
    }

    #[test]
    fn rfc7693_abc_vector_and_completed_witness_tampering() {
        let expected_hex = "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d17d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923";
        let expected: Vec<_> = (0..64)
            .map(|i| {
                let pair = expected_hex.as_bytes().get(2 * i..2 * i + 2).unwrap();
                u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap()
            })
            .collect();
        let mut builder = R1csBuilder::new();
        let input: Vec<_> = b"abc"
            .iter()
            .map(|&x| ByteVar::allocate(&mut builder, Some(x)))
            .collect();
        let output = blake2b512(&mut builder, &input).unwrap();
        let claimed: Vec<_> = expected
            .iter()
            .map(|&x| ByteVar::allocate(&mut builder, Some(x)))
            .collect();
        for (actual, claimed) in output.iter().zip(&claimed) {
            builder.assert_equal(actual.expression(), claimed.expression());
        }
        let witness = builder.witness().unwrap();
        let matrices = builder.into_matrices();
        assert!(matrices.check_witness(&witness).is_ok());
        for byte in [&input[0], &claimed[0]] {
            let variable = byte
                .expression()
                .terms
                .iter()
                .find(|(v, _)| *v != Variable::ONE)
                .unwrap()
                .0;
            let mut changed = witness.clone();
            changed[variable.index()] = Fr::from_u64(1) - changed[variable.index()];
            assert!(matrices.check_witness(&changed).is_err());
            changed[variable.index()] = Fr::from_u64(2);
            assert!(matrices.check_witness(&changed).is_err());
        }
    }

    #[test]
    fn hash_shape_does_not_depend_on_witness() {
        let emit = |value| {
            let mut builder = R1csBuilder::new();
            let byte = ByteVar::allocate(&mut builder, value);
            let _ = blake2b256(&mut builder, &[byte]).unwrap();
            builder.into_matrices()
        };
        let unknown = emit(None);
        let known = emit(Some(171));
        assert_eq!(unknown.num_vars, known.num_vars);
        assert_eq!(unknown.a, known.a);
        assert_eq!(unknown.b, known.b);
        assert_eq!(unknown.c, known.c);
    }

    #[test]
    #[cfg(feature = "transcript-blake2b")]
    fn chained_transition_matches_actual_legacy_transcript() {
        use crate::{LegacyBlake2bTranscript, Transcript};
        let mut native = LegacyBlake2bTranscript::<Fr>::new(b"wrapper-fixture");
        let mut builder = R1csBuilder::new();
        let mut circuit = LegacyBlake2bVar::new(&mut builder, b"wrapper-fixture").unwrap();
        bind(&mut builder, circuit.state(), &native.state());
        for payload in [&[42u8; 80][..], &[17u8; 2][..]] {
            native.append_bytes(payload);
            let bytes: Vec<_> = payload
                .iter()
                .map(|&x| ByteVar::allocate(&mut builder, Some(x)))
                .collect();
            circuit.append_bytes(&mut builder, &bytes).unwrap();
            bind(&mut builder, circuit.state(), &native.state());
        }
        let witness = builder.witness().unwrap();
        assert!(builder.into_matrices().check_witness(&witness).is_ok());
    }

    #[test]
    fn public_shape_errors_precede_constraint_emission() {
        let mut builder = R1csBuilder::new();
        assert!(matches!(
            LegacyBlake2bVar::new(&mut builder, &[0; 33]),
            Err(Blake2bR1csError::LabelTooLong)
        ));
        assert_eq!(builder.num_vars(), 1);
        let mut exhausted = LegacyBlake2bVar {
            state: std::array::from_fn(|_| ByteVar::constant(0)),
            round: u32::MAX,
        };
        assert!(matches!(
            exhausted.append_bytes(&mut builder, &[]),
            Err(Blake2bR1csError::RoundOverflow)
        ));
        assert_eq!(builder.num_vars(), 1);
        let byte = ByteVar::allocate(&mut builder, Some(0));
        let mut other = R1csBuilder::new();
        assert!(matches!(
            blake2b256(&mut other, &[byte]),
            Err(Blake2bR1csError::Bits(BitsError::UnknownVariable { .. }))
        ));
        assert_eq!(other.num_vars(), 1);
    }
}

mod duplex;
pub use duplex::Blake2bDuplexVar;

mod stream;
pub use stream::{Blake2bStreamVar, BLAKE2B_STREAM_BYTES};
