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
    AdditiveGroup, CanonicalBytes, CanonicalDecode, CanonicalEncoding, Field, NaiveAccumulator,
    Ring, WithAccumulator,
};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{
    send_full_round, BooleanHypercube, EvaluationClaim, SumcheckClaim, SumcheckVerifier,
};
use jolt_transcript::{
    Blake2b512, Channel, Keccak, ProtocolId, ProverTranscript, Sponge, VerifierTranscript,
};
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

impl Field for Mersenne61 {
    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            Some(self.pow(MODULUS - 2))
        }
    }

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        Self::from_u64(rng.next_u64())
    }
}

impl Ring for Mersenne61 {
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

impl CanonicalBytes for Mersenne61 {
    const NUM_BYTES: usize = 8;

    fn to_bytes_le(&self, out: &mut [u8]) {
        assert_eq!(out.len(), 8);
        out.copy_from_slice(&self.0.to_le_bytes());
    }
}

impl spongefish::Encoding<[u8]> for Mersenne61 {
    fn encode(&self) -> impl AsRef<[u8]> {
        jolt_field::narg::encode(self)
    }
}

impl CanonicalDecode for Mersenne61 {
    fn from_bytes_le_checked(bytes: &[u8]) -> Option<Self> {
        let arr: [u8; 8] = bytes.try_into().ok()?;
        Self::from_u128_checked(u64::from_le_bytes(arr) as u128)
    }
}

impl spongefish::NargDeserialize for Mersenne61 {
    fn deserialize_from_narg(buf: &mut &[u8]) -> spongefish::VerificationResult<Self> {
        jolt_field::narg::deserialize(buf)
    }
}

impl CanonicalEncoding for Mersenne61 {
    const MODULUS_BITS: u32 = 61;

    fn from_bytes_le_reduced(bytes: &[u8]) -> Self {
        let mut buf = [0u8; 16];
        let len = bytes.len().min(16);
        buf[..len].copy_from_slice(&bytes[..len]);
        Self::from_u128(u128::from_le_bytes(buf))
    }

    fn to_u128_checked(&self) -> Option<u128> {
        Some(self.0 as u128)
    }

    fn to_u64_checked(&self) -> Option<u64> {
        Some(self.0)
    }

    fn from_u128_checked(v: u128) -> Option<Self> {
        (v < MODULUS as u128).then_some(Self(v as u64))
    }

    fn from_u128_reduced(v: u128) -> Self {
        Self::reduce_u128(v)
    }

    fn num_bits(&self) -> u32 {
        u64::BITS - self.0.leading_zeros()
    }

    fn from_scalar_challenge_bytes(bytes: &[u8]) -> Self {
        Self::from_bytes_le_reduced(bytes)
    }
}

impl WithAccumulator for Mersenne61 {
    type Accumulator = NaiveAccumulator<Mersenne61>;
    type SmallScalarAccumulator = NaiveAccumulator<Mersenne61>;
    type SignedProductAccumulator = NaiveAccumulator<Mersenne61>;
}

/// Proves four degree-1 rounds of the claim 10 under sponge `H` (each round
/// splits its running sum as `3s + (s - 6s) X`), then verifies them.
fn roundtrip<H: Sponge>() {
    let protocol = ProtocolId::new::<H>("jolt-sumcheck/tests/mersenne61");
    let claim = SumcheckClaim::new(4, 1, Mersenne61::from_u64(10));

    let mut prover = ProverTranscript::<H>::new(&protocol, b"mersenne61");
    let mut running_sum = claim.claimed_sum;
    let mut point = Vec::new();
    for _ in 0..claim.num_vars {
        let c0 = running_sum * Mersenne61::from_u64(3);
        let round = UnivariatePoly::new(vec![c0, running_sum - c0 - c0]);
        send_full_round(&round, claim.degree, &mut prover).unwrap();
        let r: Mersenne61 = prover.challenge_small();
        running_sum = round.evaluate(r);
        point.push(r);
    }
    let narg = prover.finish();
    assert_eq!(narg.len(), claim.num_vars * 2 * Mersenne61::NUM_BYTES);

    let mut verifier = VerifierTranscript::<H>::new(&protocol, b"mersenne61", &narg);
    let actual = SumcheckVerifier::verify(&claim, BooleanHypercube, &mut verifier).unwrap();
    verifier.finish().unwrap();
    assert_eq!(actual, EvaluationClaim::new(point, running_sum));
}

#[test]
fn sumcheck_roundtrips_mersenne61_under_blake2b_and_keccak() {
    roundtrip::<Blake2b512>();
    roundtrip::<Keccak>();
}
