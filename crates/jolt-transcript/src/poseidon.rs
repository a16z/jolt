//! `PoseidonSponge` — a duplex sponge over BN254 `Fr`, on the
//! Circom-compatible Poseidon permutation.
//!
//! # Sponge
//!
//! The sponge is spongefish's overwrite-mode [`DuplexSponge`] over the unit
//! [`FrUnit`], with the width-4 Poseidon permutation of circomlib (x^5 S-box,
//! 8 full and 56 partial rounds, the constants light-poseidon ships for four
//! state elements). Rate is 3 units (state slots 0–2) and capacity is 1 unit
//! (slot 3), so the generic bound is about `2^127` permutation calls.
//! Absorb and squeeze are associative: absorbing in several calls equals one
//! absorb of the concatenation.
//!
//! Field elements are absorbed and squeezed natively, one unit each. A
//! squeezed unit is a uniform `Fr` up to the sponge's security, so `Fr`
//! challenges need no decoding and have no reduction bias. Challenges in any
//! other field are not offered on this sponge.
//!
//! # Byte packing
//!
//! Bytes enter the sponge only through these encodings:
//!
//! - A byte string of length `n` (a [`BytesMsg`], the session identifier, and
//!   both parts of the transcript instance) becomes the unit `n`, then
//!   `ceil(n / 31)` units, each the little-endian integer of the next 31 bytes
//!   (the last chunk may be shorter). Every chunk is below `2^248 < p`, so it
//!   is a canonical element, and the leading length fixes the number of
//!   chunks and the size of the last one: the map is injective and
//!   prefix-free.
//! - A `u8` is one unit, the byte's value. The proof-of-work nonce is sent
//!   this way, one byte per prover message. Arrays `[u8; N]` therefore cost
//!   one unit per byte; use [`BytesMsg`] for byte strings.
//! - A [`FieldMsg<Fr>`] is one unit, the element itself.
//!
//! [`BytesMsg`]: crate::BytesMsg
//! [`FieldMsg<Fr>`]: crate::FieldMsg

use std::sync::OnceLock;

use ark_bn254::Fr as InnerFr;
use ark_ff::{AdditiveGroup, Field, PrimeField};
use jolt_field::Fr;
use light_poseidon::parameters::bn254_x5::get_poseidon_parameters;
use spongefish::{DuplexSponge, EncodedSessionId, Encoding, Permutation, Unit};

use crate::challenge::sealed::Sealed;
use crate::challenge::ChallengeSponge;
use crate::codec::{BytesMsg, FieldMsg};
use crate::pow::PowUnit;
use crate::MAX_CHALLENGE_BYTES;

/// State width of the permutation, in field elements.
const WIDTH: usize = 4;
/// Units absorbed or squeezed per permutation call.
const RATE: usize = 3;
/// Bytes per packed chunk: `2^248 < p`, so every chunk is canonical.
const CHUNK_BYTES: usize = 31;

/// Duplex sponge over BN254 `Fr` with the Circom-compatible width-4 Poseidon
/// permutation. See the [module documentation](self) for the layout and the
/// byte packing.
pub type PoseidonSponge = DuplexSponge<PoseidonPermutation, WIDTH, RATE>;

/// One BN254 `Fr` element: the alphabet of [`PoseidonSponge`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrUnit(InnerFr);

impl Unit for FrUnit {
    const ZERO: Self = Self(InnerFr::ZERO);
}

impl From<Fr> for FrUnit {
    fn from(value: Fr) -> Self {
        Self(value.into())
    }
}

impl From<FrUnit> for Fr {
    fn from(unit: FrUnit) -> Self {
        unit.0.into()
    }
}

/// Packs a byte string as its length, then 31-byte little-endian chunks.
fn pack_bytes(bytes: &[u8]) -> Vec<FrUnit> {
    let mut units = Vec::with_capacity(1 + bytes.len().div_ceil(CHUNK_BYTES));
    units.push(FrUnit(InnerFr::from(bytes.len() as u64)));
    units.extend(
        bytes
            .chunks(CHUNK_BYTES)
            .map(|chunk| FrUnit(InnerFr::from_le_bytes_mod_order(chunk))),
    );
    units
}

impl EncodedSessionId for FrUnit {
    fn encode_bytes(bytes: &[u8]) -> impl AsRef<[Self]> {
        pack_bytes(bytes)
    }
}

impl Encoding<FrUnit> for BytesMsg {
    fn encode(&self) -> impl AsRef<[FrUnit]> {
        pack_bytes(self.as_slice())
    }
}

/// The transcript instance: the packed session, then the packed digest.
impl Encoding<FrUnit> for (BytesMsg, [u8; 32]) {
    fn encode(&self) -> impl AsRef<[FrUnit]> {
        let mut units = pack_bytes(self.0.as_slice());
        units.extend(pack_bytes(&self.1));
        units
    }
}

impl Encoding<FrUnit> for u8 {
    fn encode(&self) -> impl AsRef<[FrUnit]> {
        [FrUnit(InnerFr::from(*self))]
    }
}

impl Encoding<FrUnit> for FieldMsg<Fr> {
    fn encode(&self) -> impl AsRef<[FrUnit]> {
        [FrUnit::from(self.0)]
    }
}

/// Round constants and MDS matrix of the width-4 Circom permutation.
struct Constants {
    /// One row of `WIDTH` constants per round.
    ark: Vec<[InnerFr; WIDTH]>,
    mds: [[InnerFr; WIDTH]; WIDTH],
    full_rounds: usize,
    partial_rounds: usize,
}

#[expect(
    clippy::expect_used,
    reason = "light-poseidon ships width-4 Circom parameters of this shape; a mismatch is a build defect caught by the first test"
)]
fn constants() -> &'static Constants {
    static CONSTANTS: OnceLock<Constants> = OnceLock::new();
    CONSTANTS.get_or_init(|| {
        let params = get_poseidon_parameters::<InnerFr>(WIDTH as u8)
            .expect("light-poseidon: width-4 Circom parameters");
        assert_eq!(params.width, WIDTH, "Poseidon width");
        assert_eq!(params.alpha, 5, "Poseidon S-box exponent");
        let rounds = params.full_rounds + params.partial_rounds;
        assert_eq!(params.ark.len(), rounds * WIDTH, "Poseidon round constants");
        let ark = params
            .ark
            .chunks_exact(WIDTH)
            .map(|row| <[InnerFr; WIDTH]>::try_from(row).expect("row of WIDTH constants"))
            .collect();
        let rows: Vec<[InnerFr; WIDTH]> = params
            .mds
            .iter()
            .map(|row| <[InnerFr; WIDTH]>::try_from(row.as_slice()).expect("MDS row"))
            .collect();
        let mds = <[[InnerFr; WIDTH]; WIDTH]>::try_from(rows).expect("WIDTH MDS rows");
        Constants {
            ark,
            mds,
            full_rounds: params.full_rounds,
            partial_rounds: params.partial_rounds,
        }
    })
}

fn sbox(x: InnerFr) -> InnerFr {
    x.square().square() * x
}

/// The width-4 Poseidon permutation with circomlib's constants: the same
/// rounds as light-poseidon's `Poseidon::new_circom(3)`, which applies it to
/// `[0, a, b, c]` and returns slot 0.
#[derive(Clone, Copy, Debug, Default)]
pub struct PoseidonPermutation;

impl Permutation<WIDTH> for PoseidonPermutation {
    type U = FrUnit;

    fn permute_mut(&self, state: &mut [FrUnit; WIDTH]) {
        let constants = constants();
        let half_full = constants.full_rounds / 2;
        let partial = half_full..half_full + constants.partial_rounds;
        let mut s = state.map(|unit| unit.0);
        for (round, ark) in constants.ark.iter().enumerate() {
            for (x, c) in s.iter_mut().zip(ark) {
                *x += c;
            }
            if partial.contains(&round) {
                if let Some(first) = s.first_mut() {
                    *first = sbox(*first);
                }
            } else {
                for x in &mut s {
                    *x = sbox(*x);
                }
            }
            let mut mixed = [InnerFr::ZERO; WIDTH];
            for (out, row) in mixed.iter_mut().zip(&constants.mds) {
                *out = row
                    .iter()
                    .zip(&s)
                    .fold(InnerFr::ZERO, |acc, (m, x)| acc + *m * x);
            }
            s = mixed;
        }
        *state = s.map(FrUnit);
    }
}

impl Sealed for PoseidonSponge {}

/// `Fr` challenges are squeezed units, one per element. Messages are capped
/// at [`MAX_CHALLENGE_BYTES`] of 32-byte elements.
impl ChallengeSponge<Fr> for PoseidonSponge {
    fn challenge_units(len: usize) -> Option<usize> {
        (len <= MAX_CHALLENGE_BYTES / 32).then_some(len)
    }

    fn decode_challenges(units: &[FrUnit], len: usize) -> Option<Vec<Fr>> {
        (units.len() == len).then(|| units.iter().copied().map(Fr::from).collect())
    }
}

impl Sealed for FrUnit {}

/// The check is one squeezed element, tested on the low bits of its canonical
/// integer. For a uniform element and `bits <= MAX_POW_BITS`, those bits are
/// within `2^bits / p < 2^-228` of uniform.
///
/// [`MAX_POW_BITS`]: crate::MAX_POW_BITS
impl PowUnit for FrUnit {
    fn check_len(_bits: u32) -> usize {
        1
    }

    fn low_word(check: &[Self]) -> Option<u32> {
        let [unit] = check else {
            return None;
        };
        let low_limb = unit.0.into_bigint().0.first().copied()?;
        Some(low_limb as u32)
    }
}

#[cfg(test)]
#[expect(
    unused_results,
    reason = "DuplexSpongeInterface methods return &mut Self for chaining"
)]
#[expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;
    use ark_ff::BigInteger;
    use light_poseidon::{Poseidon, PoseidonHasher};
    use spongefish::{DuplexSpongeInit, DuplexSpongeInterface};

    fn unit(value: u64) -> FrUnit {
        FrUnit(InnerFr::from(value))
    }

    /// The permutation agrees with light-poseidon's Circom hash, which puts a
    /// zero domain tag in slot 0 and returns slot 0.
    #[test]
    fn permutation_matches_the_circom_hash() {
        let mut hasher = Poseidon::<InnerFr>::new_circom(3).unwrap();
        for inputs in [[0u64, 0, 0], [1, 2, 3], [u64::MAX, 7, 1 << 40]] {
            let [a, b, c] = inputs.map(InnerFr::from);
            let mut state = [FrUnit::ZERO, FrUnit(a), FrUnit(b), FrUnit(c)];
            PoseidonPermutation.permute_mut(&mut state);
            assert_eq!(state[0].0, hasher.hash(&[a, b, c]).unwrap());
        }
    }

    #[test]
    fn byte_packing_is_length_then_31_byte_chunks() {
        assert_eq!(pack_bytes(&[]), [unit(0)]);
        assert_eq!(pack_bytes(&[0]), [unit(1), unit(0)]);
        let bytes: Vec<u8> = (1..=32).collect();
        let packed = pack_bytes(&bytes);
        assert_eq!(packed.len(), 3);
        assert_eq!(packed[0], unit(32));
        assert_eq!(packed[1].0, InnerFr::from_le_bytes_mod_order(&bytes[..31]));
        assert_eq!(packed[2], unit(32));
        // A full chunk of 0xff is 2^248 - 1, below the modulus.
        let top = pack_bytes(&[0xff; 31]);
        assert_eq!(top[1].0.into_bigint().to_bytes_le()[..31], [0xff; 31]);
    }

    /// Byte strings that alias at the chunk level stay distinct through the
    /// length, and the packing never reads one string as a prefix of another.
    #[test]
    fn byte_packing_separates_lengths() {
        let strings: [&[u8]; 6] = [&[], &[0], &[0, 0], &[0; 31], &[0; 32], &[0; 62]];
        for (i, a) in strings.iter().enumerate() {
            for b in &strings[i + 1..] {
                let (a, b) = (pack_bytes(a), pack_bytes(b));
                assert_ne!(a, b);
                assert!(!b.starts_with(&a) && !a.starts_with(&b));
            }
        }
    }

    fn squeeze_after(input: &[FrUnit]) -> [FrUnit; 4] {
        let mut sponge = PoseidonSponge::init(&[7u8; 32]);
        sponge.absorb(input);
        sponge.squeeze_array::<4>()
    }

    #[test]
    fn absorb_and_squeeze_are_associative() {
        let input: Vec<FrUnit> = (0..8).map(unit).collect();
        let mut split = PoseidonSponge::init(&[7u8; 32]);
        split
            .absorb(&input[..1])
            .absorb(&input[1..5])
            .absorb(&input[5..]);
        let mut first = [FrUnit::ZERO; 1];
        let mut rest = [FrUnit::ZERO; 3];
        split.squeeze(&mut first).squeeze(&mut rest);
        assert_eq!([first.as_slice(), &rest].concat(), squeeze_after(&input));
    }

    #[test]
    fn distinct_inputs_and_sessions_diverge() {
        assert_ne!(squeeze_after(&[unit(0)]), squeeze_after(&[unit(1)]));
        assert_ne!(squeeze_after(&[unit(0)]), squeeze_after(&[unit(0); 2]));
        let mut other = PoseidonSponge::init(&[8u8; 32]);
        other.absorb(&[unit(0)]);
        assert_ne!(other.squeeze_array::<4>(), squeeze_after(&[unit(0)]));
    }

    #[test]
    fn clones_resume_a_partially_consumed_rate_block() {
        let mut original = PoseidonSponge::init(&[1u8; 32]);
        original.absorb(&[unit(5)]);
        let mut prefix = [FrUnit::ZERO; 2];
        original.squeeze(&mut prefix);
        let mut cloned = original.clone();
        assert_eq!(original.squeeze_array::<5>(), cloned.squeeze_array::<5>());
    }

    #[test]
    fn pow_check_reads_the_low_word_of_one_element() {
        assert_eq!(
            FrUnit::low_word(&[unit(0x1234_5678_9abc)]),
            Some(0x5678_9abc)
        );
        assert_eq!(FrUnit::low_word(&[]), None);
        assert_eq!(FrUnit::low_word(&[unit(0); 2]), None);
    }
}
