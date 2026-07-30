#![allow(missing_docs)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

use crate::primitives::arithmetic::Field;
use ark_bn254::Fr;
use ark_ff::{Field as ArkField, UniformRand, Zero as ArkZero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::ops::{Add, Mul, Neg, Sub};

#[derive(Clone, Copy, PartialEq, Eq, Debug, CanonicalSerialize, CanonicalDeserialize)]
#[repr(transparent)]
pub struct ArkFr(pub Fr);

impl Field for ArkFr {
    fn zero() -> Self {
        ArkFr(Fr::from(0u64))
    }

    fn one() -> Self {
        ArkFr(Fr::from(1u64))
    }

    fn is_zero(&self) -> bool {
        ArkZero::is_zero(&self.0)
    }

    fn add(&self, rhs: &Self) -> Self {
        ArkFr(self.0 + rhs.0)
    }

    fn sub(&self, rhs: &Self) -> Self {
        ArkFr(self.0 - rhs.0)
    }

    fn mul(&self, rhs: &Self) -> Self {
        ArkFr(self.0 * rhs.0)
    }

    fn inv(self) -> Option<Self> {
        ArkField::inverse(&self.0).map(ArkFr)
    }

    fn random() -> Self {
        ArkFr(Fr::rand(&mut rand_core::OsRng))
    }

    fn from_u64(val: u64) -> Self {
        ArkFr(Fr::from(val))
    }

    fn from_i64(val: i64) -> Self {
        if val >= 0 {
            ArkFr(Fr::from(val as u64))
        } else {
            ArkFr(-Fr::from((-val) as u64))
        }
    }
}

impl Add for ArkFr {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        ArkFr(self.0 + rhs.0)
    }
}

impl Sub for ArkFr {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        ArkFr(self.0 - rhs.0)
    }
}

impl Mul for ArkFr {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        ArkFr(self.0 * rhs.0)
    }
}

impl Neg for ArkFr {
    type Output = Self;
    fn neg(self) -> Self {
        ArkFr(-self.0)
    }
}

impl<'a> Add<&'a ArkFr> for ArkFr {
    type Output = ArkFr;
    fn add(self, rhs: &'a ArkFr) -> ArkFr {
        ArkFr(self.0 + rhs.0)
    }
}

impl<'a> Sub<&'a ArkFr> for ArkFr {
    type Output = ArkFr;
    fn sub(self, rhs: &'a ArkFr) -> ArkFr {
        ArkFr(self.0 - rhs.0)
    }
}

impl<'a> Mul<&'a ArkFr> for ArkFr {
    type Output = ArkFr;
    fn mul(self, rhs: &'a ArkFr) -> ArkFr {
        ArkFr(self.0 * rhs.0)
    }
}
