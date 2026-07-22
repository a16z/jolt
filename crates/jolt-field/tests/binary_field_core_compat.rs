//! Compatibility test for the slim algebraic trait layer.
//!
//! `Gf2` is intentionally a toy characteristic-2 field. It proves that the
//! core field traits do not bake in prime-field or large-field assumptions, but
//! it is not a production proving field. Real sumcheck soundness over such a
//! small base field would require an appropriately large extension field.

use std::{
    fmt::{Debug, Display},
    hash::Hash,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use jolt_field::{AdditiveGroup, CanonicalRepr, FieldCore, RingCore};
use num_traits::{One, Zero};

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
struct Gf2(bool);

impl Debug for Gf2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&(self.0 as u8), f)
    }
}

impl Display for Gf2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&(self.0 as u8), f)
    }
}

impl Zero for Gf2 {
    fn zero() -> Self {
        Self(false)
    }

    fn is_zero(&self) -> bool {
        !self.0
    }
}

impl One for Gf2 {
    fn one() -> Self {
        Self(true)
    }

    fn is_one(&self) -> bool {
        self.0
    }
}

#[expect(clippy::suspicious_arithmetic_impl)]
impl Add for Gf2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl Add<&Self> for Gf2 {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        self + *rhs
    }
}

impl AddAssign for Gf2 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

#[expect(clippy::suspicious_arithmetic_impl)]
impl Sub for Gf2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + rhs
    }
}

impl Sub<&Self> for Gf2 {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        self - *rhs
    }
}

impl SubAssign for Gf2 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for Gf2 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self
    }
}

#[expect(clippy::suspicious_arithmetic_impl)]
impl Mul for Gf2 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl Mul<&Self> for Gf2 {
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self::Output {
        self * *rhs
    }
}

impl MulAssign for Gf2 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Sum for Gf2 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a> Sum<&'a Gf2> for Gf2 {
    fn sum<I: Iterator<Item = &'a Gf2>>(iter: I) -> Self {
        iter.copied().sum()
    }
}

impl Product for Gf2 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<'a> Product<&'a Gf2> for Gf2 {
    fn product<I: Iterator<Item = &'a Gf2>>(iter: I) -> Self {
        iter.copied().product()
    }
}

impl AdditiveGroup for Gf2 {}
impl RingCore for Gf2 {}

impl FieldCore for Gf2 {
    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            Some(Self::one())
        }
    }

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        Self(rng.next_u32() & 1 == 1)
    }
}

impl CanonicalRepr for Gf2 {
    const NUM_BYTES: usize = 1;

    fn to_bytes_le(&self, out: &mut [u8]) {
        assert_eq!(out.len(), 1);
        out[0] = self.0 as u8;
    }

    fn from_le_bytes_mod_order(bytes: &[u8]) -> Self {
        Self(bytes.iter().fold(0u8, |acc, b| acc ^ (b & 1)) == 1)
    }

    fn to_canonical_u64_checked(&self) -> Option<u64> {
        Some(self.0 as u64)
    }

    fn num_bits(&self) -> u32 {
        self.0 as u32
    }
}

fn accepts_field_core<F: FieldCore + CanonicalRepr>(x: F) -> F {
    x.square()
}

#[test]
fn characteristic_two_field_fits_algebraic_layer() {
    assert_eq!(accepts_field_core(Gf2::one()), Gf2::one());
    assert_eq!(Gf2::one() + Gf2::one(), Gf2::zero());
}
