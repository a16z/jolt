use super::{FieldOps, JoltField};
use crate::utils::counters::{INVERSE_COUNT, MULT_COUNT};
use allocative::Allocative;
use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_ff::{One, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::Rng;
use std::default::Default;
use std::fmt;
use std::iter::{Product, Sum};
use std::mem::transmute;
use std::ops::{Add, Deref, DerefMut, Div, Mul, Sub};
use std::ops::{AddAssign, MulAssign, Neg, SubAssign};
use std::sync::atomic::Ordering;

#[derive(
    Clone, Default, Copy, PartialEq, Eq, Hash, Debug, CanonicalSerialize, CanonicalDeserialize,
)]
pub struct TrackedFr(pub ark_bn254::Fr);

impl Allocative for TrackedFr {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

impl Deref for TrackedFr {
    type Target = Fr;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl UniformRand for TrackedFr {
    fn rand<R: Rng + ?Sized>(rng: &mut R) -> Self {
        TrackedFr(ark_bn254::Fr::rand(rng))
    }
}
impl DerefMut for TrackedFr {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// Owned + Owned
impl Add for TrackedFr {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        // No MULT_COUNT increment for add, but add if you want
        Self(self.0 + rhs.0)
    }
}

// Owned + Borrowed
impl<'a> Add<&'a TrackedFr> for TrackedFr {
    type Output = Self;
    fn add(self, rhs: &'a TrackedFr) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

// Owned - Owned
impl Sub for TrackedFr {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

// Owned - Borrowed
impl<'a> Sub<&'a TrackedFr> for TrackedFr {
    type Output = Self;
    fn sub(self, rhs: &'a TrackedFr) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

// Owned * owned
impl Mul<TrackedFr> for TrackedFr {
    type Output = TrackedFr;
    fn mul(self, rhs: TrackedFr) -> Self::Output {
        MULT_COUNT.fetch_add(1, Ordering::Relaxed);
        TrackedFr(self.0 * rhs.0)
    }
}

// Owned * borrowed
impl<'a> Mul<&'a TrackedFr> for TrackedFr {
    type Output = TrackedFr;
    fn mul(self, rhs: &'a TrackedFr) -> Self::Output {
        MULT_COUNT.fetch_add(1, Ordering::Relaxed);
        TrackedFr(self.0 * rhs.0)
    }
}

// Borrowed * owned
impl Mul<TrackedFr> for &TrackedFr {
    type Output = TrackedFr;

    fn mul(self, rhs: TrackedFr) -> Self::Output {
        MULT_COUNT.fetch_add(1, Ordering::Relaxed);
        TrackedFr(self.0 * rhs.0)
    }
}

// Owned / Owned
#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<TrackedFr> for TrackedFr {
    type Output = TrackedFr;
    fn div(self, rhs: TrackedFr) -> Self::Output {
        MULT_COUNT.fetch_add(1, Ordering::Relaxed);
        let inv = rhs.0.inverse().expect("division by zero");
        TrackedFr(self.0 * inv)
    }
}
// Owned / Borrowed
#[allow(clippy::suspicious_arithmetic_impl)]
impl<'a> Div<&'a TrackedFr> for TrackedFr {
    type Output = TrackedFr;
    fn div(self, rhs: &'a TrackedFr) -> Self::Output {
        MULT_COUNT.fetch_add(1, Ordering::Relaxed);
        TrackedFr(self.0 * ark_ff::Field::inverse(&rhs.0).unwrap())
    }
}

// Borrowed / Owned
#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<TrackedFr> for &TrackedFr {
    type Output = TrackedFr;
    fn div(self, rhs: TrackedFr) -> Self::Output {
        MULT_COUNT.fetch_add(1, Ordering::Relaxed);
        TrackedFr(self.0 * ark_ff::Field::inverse(&rhs.0).unwrap())
    }
}

impl PartialEq<Fr> for TrackedFr {
    fn eq(&self, other: &Fr) -> bool {
        self.0 == *other
    }
}
impl PartialEq<TrackedFr> for Fr {
    fn eq(&self, other: &TrackedFr) -> bool {
        *self == other.0
    }
}

// Display delegates to Debug or inner Display
impl fmt::Display for TrackedFr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// Negation
impl Neg for TrackedFr {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

// AddAssign, SubAssign, MulAssign
impl AddAssign for TrackedFr {
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl SubAssign for TrackedFr {
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
    }
}

impl MulAssign for TrackedFr {
    fn mul_assign(&mut self, other: Self) {
        MULT_COUNT.fetch_add(1, Ordering::Relaxed);
        self.0 *= other.0;
    }
}

// Zero and One
impl Zero for TrackedFr {
    fn zero() -> Self {
        Self(Fr::zero())
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl One for TrackedFr {
    fn one() -> Self {
        Self(Fr::one())
    }
    fn is_one(&self) -> bool {
        self.0.is_one()
    }
}

// Sum and Product for iterators
impl Sum for TrackedFr {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| Self(a.0 + b.0))
    }
}

impl<'a> Sum<&'a Self> for TrackedFr {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| Self(a.0 + b.0))
    }
}

impl Product for TrackedFr {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |a, b| {
            MULT_COUNT.fetch_add(1, Ordering::Relaxed);
            Self(a.0 * b.0)
        })
    }
}

impl<'a> Product<&'a Self> for TrackedFr {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |a, b| {
            MULT_COUNT.fetch_add(1, Ordering::Relaxed);
            Self(a.0 * b.0)
        })
    }
}

// &TrackedFr + &TrackedFr
#[allow(clippy::needless_lifetimes)]
impl<'a, 'b> Add<&'b TrackedFr> for &'a TrackedFr {
    type Output = TrackedFr;
    fn add(self, rhs: &'b TrackedFr) -> Self::Output {
        TrackedFr(self.0 + rhs.0)
    }
}

// &TrackedFr - &TrackedFr
#[allow(clippy::needless_lifetimes)]
impl<'a, 'b> Sub<&'b TrackedFr> for &'a TrackedFr {
    type Output = TrackedFr;
    fn sub(self, rhs: &'b TrackedFr) -> Self::Output {
        TrackedFr(self.0 - rhs.0)
    }
}

// &TrackedFr * &TrackedFr
#[allow(clippy::needless_lifetimes)]
impl<'a, 'b> Mul<&'b TrackedFr> for &'a TrackedFr {
    type Output = TrackedFr;
    fn mul(self, rhs: &'b TrackedFr) -> Self::Output {
        MULT_COUNT.fetch_add(1, Ordering::Relaxed);
        TrackedFr(self.0 * rhs.0)
    }
}

// &TrackedFr / &Tracked
#[allow(clippy::needless_lifetimes)]
#[allow(clippy::suspicious_arithmetic_impl)]
impl<'a, 'b> Div<&'b TrackedFr> for &'a TrackedFr {
    type Output = TrackedFr;
    fn div(self, rhs: &'b TrackedFr) -> Self::Output {
        MULT_COUNT.fetch_add(1, Ordering::Relaxed);
        TrackedFr(self.0 * ark_ff::Field::inverse(&rhs.0).unwrap())
    }
}

impl FieldOps for TrackedFr {}
impl FieldOps<&TrackedFr, TrackedFr> for TrackedFr {}

impl JoltField for TrackedFr {
    const NUM_BYTES: usize = <ark_bn254::Fr as JoltField>::NUM_BYTES;
    type SmallValueLookupTables = <ark_bn254::Fr as JoltField>::SmallValueLookupTables;

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        TrackedFr(<ark_bn254::Fr as JoltField>::random(rng))
    }

    fn compute_lookup_tables() -> Self::SmallValueLookupTables {
        <ark_bn254::Fr as JoltField>::compute_lookup_tables()
    }

    fn from_bool(val: bool) -> Self {
        TrackedFr(<ark_bn254::Fr as JoltField>::from_bool(val))
    }

    fn from_u8(n: u8) -> Self {
        TrackedFr(<ark_bn254::Fr as JoltField>::from_u8(n))
    }

    fn from_u16(n: u16) -> Self {
        TrackedFr(<ark_bn254::Fr as JoltField>::from_u16(n))
    }

    fn from_u32(n: u32) -> Self {
        TrackedFr(<ark_bn254::Fr as JoltField>::from_u32(n))
    }

    fn from_u64(n: u64) -> Self {
        TrackedFr(<ark_bn254::Fr as JoltField>::from_u64(n))
    }

    fn from_i64(n: i64) -> Self {
        TrackedFr(<ark_bn254::Fr as JoltField>::from_i64(n))
    }

    fn from_i128(n: i128) -> Self {
        TrackedFr(<ark_bn254::Fr as JoltField>::from_i128(n))
    }

    fn from_u128(n: u128) -> Self {
        TrackedFr(<ark_bn254::Fr as JoltField>::from_u128(n))
    }

    fn to_u64(&self) -> Option<u64> {
        self.0.to_u64()
    }

    fn square(&self) -> Self {
        MULT_COUNT.fetch_add(1, Ordering::Relaxed);
        TrackedFr(self.0.square())
    }

    fn inverse(&self) -> Option<Self> {
        INVERSE_COUNT.fetch_add(1, Ordering::Relaxed);
        self.0.inverse().map(TrackedFr)
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        TrackedFr(<ark_bn254::Fr as JoltField>::from_bytes(bytes))
    }

    fn num_bits(&self) -> u32 {
        self.0.num_bits()
    }

    fn mul_u64(&self, n: u64) -> Self {
        MULT_COUNT.fetch_add(1, Ordering::Relaxed);
        Self(<Fr as JoltField>::mul_u64(&self.0, n))
    }

    fn mul_i64(&self, n: i64) -> Self {
        MULT_COUNT.fetch_add(1, Ordering::Relaxed);
        Self(<Fr as JoltField>::mul_i64(&self.0, n))
    }

    fn mul_u128(&self, n: u128) -> Self {
        MULT_COUNT.fetch_add(1, Ordering::Relaxed);
        Self(<Fr as JoltField>::mul_u128(&self.0, n))
    }

    fn mul_i128(&self, n: i128) -> Self {
        MULT_COUNT.fetch_add(1, Ordering::Relaxed);
        Self(<Fr as JoltField>::mul_i128(&self.0, n))
    }

    fn linear_combination_u64(pairs: &[(Self, u64)], add_terms: &[Self]) -> Self {
        MULT_COUNT.fetch_add(pairs.len(), Ordering::Relaxed);
        Self(<Fr as JoltField>::linear_combination_u64(
            unsafe { transmute::<&[(Self, u64)], &[(Fr, u64)]>(pairs) },
            unsafe { transmute::<&[Self], &[Fr]>(add_terms) },
        ))
    }

    fn linear_combination_i64(
        pos: &[(Self, u64)],
        neg: &[(Self, u64)],
        pos_add: &[Self],
        neg_add: &[Self],
    ) -> Self {
        MULT_COUNT.fetch_add(pos.len() + neg.len(), Ordering::Relaxed);
        Self(<Fr as JoltField>::linear_combination_i64(
            unsafe { transmute::<&[(Self, u64)], &[(Fr, u64)]>(pos) },
            unsafe { transmute::<&[(Self, u64)], &[(Fr, u64)]>(neg) },
            unsafe { transmute::<&[Self], &[Fr]>(pos_add) },
            unsafe { transmute::<&[Self], &[Fr]>(neg_add) },
        ))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::op_ref)]
    use crate::field::tracked_ark::TrackedFr as Fr;
    use crate::field::{JoltField, OptimizedMul};
    use crate::utils::counters::{
        get_inverse_count, get_mult_count, reset_inverse_count, reset_mult_count,
    };
    use std::ops::MulAssign;

    #[test]
    fn test_if_trackers_are_working() {
        reset_mult_count();
        let a = Fr::from_u8(12);
        let b = Fr::from_u8(12);
        let _ = a.mul_0_optimized(b);
        let num_mults = get_mult_count();
        assert_eq!(num_mults, 1);

        reset_mult_count();
        let a = Fr::from_u8(12);
        let b = Fr::from_u8(12);
        let _ = a * &b;
        let num_mults = get_mult_count();
        assert_eq!(num_mults, 1);

        reset_mult_count();
        let a = Fr::from_u8(12);
        let b = Fr::from_u8(12);
        let _ = &a * b;
        let num_mults = get_mult_count();
        assert_eq!(num_mults, 1);

        reset_mult_count();
        let a = Fr::from_u8(12);
        let b = Fr::from_u8(12);
        let _ = &a * &b;
        let num_mults = get_mult_count();
        assert_eq!(num_mults, 1);

        reset_mult_count();
        let a = Fr::from_u8(12);
        let b = Fr::from_u8(12);
        let _ = a * b;
        let num_mults = get_mult_count();
        assert_eq!(num_mults, 1);

        reset_inverse_count();
        let mut a = Fr::from_u8(12);
        let _b = Fr::from_u8(12);
        let c = a.inverse().unwrap();
        let num_mults = get_inverse_count();
        assert_eq!(num_mults, 1);

        reset_mult_count();
        a.mul_assign(b);
        let num_mults = get_mult_count();
        assert_eq!(num_mults, 1);

        reset_mult_count();
        let vals = [&a, &b, &c];
        let _: Fr = vals.into_iter().product();
        let num_mults = get_mult_count();
        assert_eq!(num_mults, 3);
    }
}
