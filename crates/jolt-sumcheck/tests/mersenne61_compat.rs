#![expect(clippy::unwrap_used, reason = "tests may panic on assertion failures")]

//! Compatibility test for non-BN254 sumcheck verifier plumbing.
//!
//! `Mersenne61` is intentionally small and exists here only to prove that the
//! transcript and verifier APIs no longer depend on BN254-specific helper
//! surface. It is not a production proving field. Real sumcheck soundness with
//! this base field would require an adequately large extension field.

use std::{
    fmt::{Debug, Display},
    hash::Hash,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use jolt_field::{
    AdditiveGroup, CanonicalBitLength, CanonicalBytes, CanonicalU64, FieldCore, FixedByteSize,
    FixedBytes, FromPrimitiveInt, Invertible, MulPow2, MulPrimitiveInt, NaiveAccumulator,
    RandomSampling, ReducingBytes, RingCore, TranscriptChallenge, WithAccumulator,
};
use jolt_sumcheck::{
    BooleanHypercube, ClearRound, EvaluationClaim, RoundMessage, SumcheckClaim, SumcheckVerifier,
};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, KeccakTranscript, Transcript};
use num_traits::{One, Zero};

const MODULUS: u64 = (1u64 << 61) - 1;

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
struct Mersenne61(u64);

impl Mersenne61 {
    fn reduce_u128(x: u128) -> Self {
        let p = MODULUS as u128;
        let mut y = (x & p) + (x >> 61);
        y = (y & p) + (y >> 61);
        if y >= p {
            y -= p;
        }
        Self(y as u64)
    }

    fn pow(self, mut exp: u64) -> Self {
        let mut base = self;
        let mut acc = Self::one();
        while exp > 0 {
            if exp & 1 == 1 {
                acc *= base;
            }
            base *= base;
            exp >>= 1;
        }
        acc
    }
}

impl Debug for Mersenne61 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl Display for Mersenne61 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl Zero for Mersenne61 {
    fn zero() -> Self {
        Self(0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl One for Mersenne61 {
    fn one() -> Self {
        Self(1)
    }

    fn is_one(&self) -> bool {
        self.0 == 1
    }
}

impl Add for Mersenne61 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut sum = self.0 + rhs.0;
        if sum >= MODULUS {
            sum -= MODULUS;
        }
        Self(sum)
    }
}

impl Add<&Self> for Mersenne61 {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        self + *rhs
    }
}

impl AddAssign for Mersenne61 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Mersenne61 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.0 >= rhs.0 {
            Self(self.0 - rhs.0)
        } else {
            Self(MODULUS - (rhs.0 - self.0))
        }
    }
}

impl Sub<&Self> for Mersenne61 {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        self - *rhs
    }
}

impl SubAssign for Mersenne61 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for Mersenne61 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        if self.is_zero() {
            self
        } else {
            Self(MODULUS - self.0)
        }
    }
}

impl Mul for Mersenne61 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::reduce_u128(self.0 as u128 * rhs.0 as u128)
    }
}

impl Mul<&Self> for Mersenne61 {
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self::Output {
        self * *rhs
    }
}

impl MulAssign for Mersenne61 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Sum for Mersenne61 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a> Sum<&'a Mersenne61> for Mersenne61 {
    fn sum<I: Iterator<Item = &'a Mersenne61>>(iter: I) -> Self {
        iter.copied().sum()
    }
}

impl Product for Mersenne61 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<'a> Product<&'a Mersenne61> for Mersenne61 {
    fn product<I: Iterator<Item = &'a Mersenne61>>(iter: I) -> Self {
        iter.copied().product()
    }
}

impl AdditiveGroup for Mersenne61 {}
impl RingCore for Mersenne61 {}

impl Invertible for Mersenne61 {
    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            Some(self.pow(MODULUS - 2))
        }
    }
}

impl FieldCore for Mersenne61 {}

impl FromPrimitiveInt for Mersenne61 {
    fn from_u64(v: u64) -> Self {
        Self::reduce_u128(v as u128)
    }

    fn from_i64(v: i64) -> Self {
        if v >= 0 {
            Self::from_u64(v as u64)
        } else {
            -Self::from_u64(v.unsigned_abs())
        }
    }

    fn from_u128(v: u128) -> Self {
        Self::reduce_u128(v)
    }

    fn from_i128(v: i128) -> Self {
        if v >= 0 {
            Self::from_u128(v as u128)
        } else {
            -Self::from_u128(v.unsigned_abs())
        }
    }
}

impl RandomSampling for Mersenne61 {
    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        Self::from_u64(rng.next_u64())
    }
}

impl CanonicalBytes for Mersenne61 {
    fn to_bytes_le(&self, out: &mut [u8]) {
        assert_eq!(out.len(), 8);
        out.copy_from_slice(&self.0.to_le_bytes());
    }
}

impl ReducingBytes for Mersenne61 {
    fn from_le_bytes_mod_order(bytes: &[u8]) -> Self {
        let mut buf = [0u8; 16];
        let len = bytes.len().min(16);
        buf[..len].copy_from_slice(&bytes[..len]);
        Self::from_u128(u128::from_le_bytes(buf))
    }
}

impl TranscriptChallenge for Mersenne61 {
    fn from_challenge_bytes(bytes: &[u8]) -> Self {
        Self::from_le_bytes_mod_order(bytes)
    }
}

impl FixedByteSize for Mersenne61 {
    const NUM_BYTES: usize = 8;
}

impl FixedBytes<8> for Mersenne61 {}

impl CanonicalBitLength for Mersenne61 {
    fn num_bits(&self) -> u32 {
        u64::BITS - self.0.leading_zeros()
    }
}

impl CanonicalU64 for Mersenne61 {
    fn to_canonical_u64_checked(&self) -> Option<u64> {
        Some(self.0)
    }
}

impl WithAccumulator for Mersenne61 {
    type Accumulator = NaiveAccumulator<Mersenne61>;
}

impl MulPow2 for Mersenne61 {}
impl MulPrimitiveInt for Mersenne61 {}

#[derive(Clone, Debug)]
struct LinearRound {
    coeffs: [Mersenne61; 2],
}

impl RoundMessage for LinearRound {
    fn degree(&self) -> usize {
        1
    }

    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        self.coeffs[0].append_to_transcript(transcript);
        self.coeffs[1].append_to_transcript(transcript);
    }
}

impl ClearRound<Mersenne61> for LinearRound {
    fn evaluate(&self, challenge: Mersenne61) -> Mersenne61 {
        self.coeffs[0] + self.coeffs[1] * challenge
    }

    fn coefficient_linear_combination(&self, coefficients: &[Mersenne61]) -> Mersenne61 {
        self.coeffs
            .iter()
            .zip(coefficients)
            .map(|(&coefficient, &scale)| coefficient * scale)
            .sum()
    }
}

fn build_rounds() -> (
    SumcheckClaim<Mersenne61>,
    Vec<LinearRound>,
    EvaluationClaim<Mersenne61>,
) {
    let mut transcript = Blake2bTranscript::<Mersenne61>::new(b"mersenne61");
    let mut running_sum = Mersenne61::from_u64(10);
    let mut point = Vec::new();
    let mut rounds = Vec::new();

    for _ in 0..4 {
        let c0 = running_sum * Mersenne61::from_u64(3);
        let c1 = running_sum - c0 - c0;
        let round = LinearRound { coeffs: [c0, c1] };
        round.append_to_transcript(&mut transcript);
        let r = transcript.challenge();
        running_sum = round.evaluate(r);
        point.push(r);
        rounds.push(round);
    }

    (
        SumcheckClaim::new(4, 1, Mersenne61::from_u64(10)),
        rounds,
        EvaluationClaim::new(point, running_sum),
    )
}

#[test]
fn hash_transcripts_accept_mersenne61_without_bn254_field_surface() {
    let mut blake = Blake2bTranscript::<Mersenne61>::new(b"compat");
    let mut keccak = KeccakTranscript::<Mersenne61>::new(b"compat");
    Mersenne61::from_u64(42).append_to_transcript(&mut blake);
    Mersenne61::from_u64(42).append_to_transcript(&mut keccak);

    let _: Mersenne61 = blake.challenge();
    let _: Mersenne61 = keccak.challenge();
}

#[test]
fn sumcheck_verifier_accepts_mersenne61_round_proof() {
    let (claim, rounds, expected) = build_rounds();
    let mut verifier_transcript = Blake2bTranscript::<Mersenne61>::new(b"mersenne61");
    let actual =
        SumcheckVerifier::verify(&claim, &rounds, BooleanHypercube, &mut verifier_transcript)
            .unwrap();
    assert_eq!(actual, expected);
}
