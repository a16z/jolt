//! `PoseidonSponge` — a duplex sponge over BN254 `Fr`, on the
//! Circom-compatible Poseidon permutation.
//!
//! # Sponge
//!
//! The sponge is spongefish's overwrite-mode [`DuplexSponge`] over the unit
//! [`FrUnit`], with the width-5 Poseidon permutation of circomlib (x^5 S-box,
//! 8 full and 60 partial rounds, the constants light-poseidon ships for five
//! state elements). Rate is 3 units (state slots 0–2) and capacity is 2 units
//! (slots 3–4). The generic bound is a birthday collision on the capacity,
//! about `2^253` permutation calls, so security rests on the permutation's
//! own 128-bit design target; one unit of capacity would give only about
//! `2^127`.
//! Absorb and squeeze are associative: absorbing in several calls equals one
//! absorb of the concatenation.
//!
//! Challenges are squeezed natively, one unit each. A squeezed unit is a
//! uniform `Fr` up to the sponge's security, so `Fr` challenges need no
//! decoding and have no reduction bias. Challenges in any other field are not
//! offered on this sponge.
//!
//! # Byte packing
//!
//! Every message, the session identifier and the transcript instance enter
//! the sponge as byte strings, each packed on its own: a byte string of
//! length `n` becomes the unit `n`, then `ceil(n / 31)` units, each the
//! little-endian integer of the next 31 bytes (the last chunk may be
//! shorter). Every chunk is below `2^248 < p`, so it is a canonical element,
//! and the leading length fixes the number of chunks and the size of the last
//! one: the map is injective and prefix-free, so the messages absorbed
//! between two squeezes decode uniquely from their concatenation.
//!
//! A message is packed from its byte [`Encoding`]. A proof-of-work nonce,
//! one eight-byte word, is two units; a 32-byte field element is three. The
//! word padding of the byte layout is packed like any other byte.

use std::sync::OnceLock;

use ark_bn254::Fr as InnerFr;
use ark_ff::{AdditiveGroup, Field, PrimeField};
use jolt_field::Fr;
use light_poseidon::parameters::bn254_x5::get_poseidon_parameters;
use spongefish::{DuplexSponge, EncodedSessionId, Encoding, Permutation, Unit};

use crate::challenge::sealed::Sealed as SpongeSealed;
use crate::codec::sealed::Sealed as UnitSealed;
use crate::{ChallengeCodec, ChallengeSetSize, ChallengeSponge, SpongeUnit, MAX_CHALLENGE_BYTES};

/// State width of the permutation, in field elements.
const WIDTH: usize = 5;
/// Units absorbed or squeezed per permutation call.
const RATE: usize = 3;
/// Bytes per packed chunk: `2^248 < p`, so every chunk is canonical.
const CHUNK_BYTES: usize = 31;

/// Duplex sponge over BN254 `Fr` with the Circom-compatible width-5 Poseidon
/// permutation: rate 3 elements, capacity 2. Each message is packed on its
/// own as its byte length, then its bytes in 31-byte little-endian chunks.
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

impl UnitSealed for FrUnit {}

/// A message is its packed byte encoding. The proof-of-work check is one
/// squeezed element, tested on the low bits of its canonical integer: for a
/// uniform element and `bits <= MAX_POW_BITS`, those bits are within
/// `2^bits / p < 2^-196` of uniform.
///
/// [`MAX_POW_BITS`]: crate::MAX_POW_BITS
impl SpongeUnit for FrUnit {
    const CHECK_LEN: usize = 1;

    fn encode_message<M: Encoding + ?Sized>(message: &M) -> impl AsRef<[Self]> {
        pack_bytes(message.encode().as_ref())
    }

    fn encode_word(word: u64) -> impl AsRef<[Self]> {
        pack_bytes(&word.to_le_bytes())
    }

    fn low_word(check: &[Self]) -> Option<u64> {
        let [unit] = check else {
            return None;
        };
        unit.0.into_bigint().0.first().copied()
    }
}

/// Round constants and MDS matrix of the width-5 Circom permutation.
struct Constants {
    /// One row of `WIDTH` constants per round.
    ark: Vec<[InnerFr; WIDTH]>,
    mds: [[InnerFr; WIDTH]; WIDTH],
    full_rounds: usize,
    partial_rounds: usize,
}

#[expect(
    clippy::expect_used,
    reason = "light-poseidon ships width-5 Circom parameters of this shape; a mismatch is a build defect caught by the first test"
)]
fn constants() -> &'static Constants {
    static CONSTANTS: OnceLock<Constants> = OnceLock::new();
    CONSTANTS.get_or_init(|| {
        let params = get_poseidon_parameters::<InnerFr>(WIDTH as u8)
            .expect("light-poseidon: width-5 Circom parameters");
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

/// The width-5 Poseidon permutation with circomlib's constants: the same
/// rounds as light-poseidon's `Poseidon::new_circom(4)`, which applies it to
/// `[0, a, b, c, d]` and returns slot 0, and as circomlib's `PoseidonEx` with
/// four inputs, an initial state and all five outputs.
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

impl SpongeSealed for PoseidonSponge {}

/// `Fr` challenges are squeezed units, one per element, from the whole field.
/// Messages are capped at [`MAX_CHALLENGE_BYTES`] of 32-byte elements.
impl ChallengeSponge<Fr> for PoseidonSponge {
    const CHALLENGE_SET: ChallengeSetSize = <Fr as ChallengeCodec>::CHALLENGE_SET_SIZE;

    fn challenge_units(len: usize) -> Option<usize> {
        (len <= MAX_CHALLENGE_BYTES / 32).then_some(len)
    }

    fn decode_challenges(units: &[FrUnit], len: usize) -> Option<Vec<Fr>> {
        (units.len() == len).then(|| units.iter().copied().map(Fr::from).collect())
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
    use std::str::FromStr;

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
        let mut hasher = Poseidon::<InnerFr>::new_circom(4).unwrap();
        for inputs in [[0u64, 0, 0, 0], [1, 2, 3, 4], [u64::MAX, 7, 1 << 40, 9]] {
            let [a, b, c, d] = inputs.map(InnerFr::from);
            let mut state = [FrUnit::ZERO, FrUnit(a), FrUnit(b), FrUnit(c), FrUnit(d)];
            PoseidonPermutation.permute_mut(&mut state);
            assert_eq!(state[0].0, hasher.hash(&[a, b, c, d]).unwrap());
        }
    }

    /// Full input and output states from circomlibjs 0.1.7: `poseidon(inputs,
    /// init, 5)` permutes `[init, inputs..]` and returns all five slots. This
    /// pins the slots the sponge writes and reads, not only slot 0.
    #[test]
    fn permutation_matches_circomlibjs_on_the_full_state() {
        let vectors: [([&str; WIDTH], [&str; WIDTH]); 3] = [
            (
                ["0", "1", "2", "3", "4"],
                [
                    "18821383157269793795438455681495246036402687001665670618754263018637548127333",
                    "7817711165059374331357136443537800893307845083525445872661165200086166013245",
                    "16733335996448830230979566039396561240864200624113062088822991822580465420551",
                    "6644334865470350789317807668685953492649391266180911382577082600917830417726",
                    "3372108894677221197912083238087960099443657816445944159266857514496320565191",
                ],
            ),
            (
                ["5", "6", "7", "8", "9"],
                [
                    "18579283463335299535026317327209621916637121867533110620302664491185669653790",
                    "4839293927659934187583387626954013301637664044190093833281890908670902790478",
                    "3533800487131377481301584166615267513872620666740963507842462725216737178402",
                    "5940255035692576900005793592585174844567995327116752015751005272509148573319",
                    "20404954242403156198400108044020687119358875604606616587752779986720487318498",
                ],
            ),
            (
                [
                    "1606938044258990275541962092341162602522202993782792835301387",
                    "515377520732011331036461129765621272702107522001",
                    "0",
                    "1",
                    "1809251394333065553493296640760748560207343510400633813116524750123642650624",
                ],
                [
                    "10283392003478364169555601049796176772009100738157568382521096879459620708057",
                    "20080835164471701059339308672369966090281333042336046927504392591494914030429",
                    "16354268304952869921319451770559219642228841318443983767906418135619628518778",
                    "10967621804834771612841019768306877278929285037226631454369680010856400098637",
                    "9874277098345336612192428837053126420843492328588816763190316741189242710163",
                ],
            ),
        ];
        let parse = |digits: [&str; WIDTH]| digits.map(|d| FrUnit(InnerFr::from_str(d).unwrap()));
        for (input, output) in vectors {
            let mut state = parse(input);
            PoseidonPermutation.permute_mut(&mut state);
            assert_eq!(state, parse(output));
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
        assert_eq!(FrUnit::CHECK_LEN, 1);
        let wide = FrUnit(InnerFr::from(u128::MAX));
        assert_eq!(FrUnit::low_word(&[wide]), Some(u64::MAX));
        assert_eq!(
            FrUnit::low_word(&[unit(0x1234_5678_9abc)]),
            Some(0x1234_5678_9abc)
        );
        assert_eq!(FrUnit::low_word(&[]), None);
        assert_eq!(FrUnit::low_word(&[unit(0); 2]), None);
    }

    /// The owned nonce encoding is the packing of the nonce's word.
    #[test]
    fn a_nonce_word_packs_as_its_eight_bytes() {
        let word = 0x0102_0304_0506_0708u64;
        let packed = FrUnit::encode_word(word);
        assert_eq!(
            packed.as_ref(),
            FrUnit::encode_message(&word.to_le_bytes()).as_ref()
        );
        assert_eq!(packed.as_ref(), [unit(8), unit(word)]);
    }
}
