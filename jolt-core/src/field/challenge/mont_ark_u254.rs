//! This module implements a base Challenge type whose range is exactly that the same as the
//! concrete type being used as JoltField.
//!

use crate::field::OptimizedMul;
use crate::field::{tracked_ark::TrackedFr, JoltField};
//use crate::impl_field_ops_inline;
use allocative::Allocative;
use ark_ff::UniformRand;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use num::{One, Zero};
use rand::{Rng, RngCore};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::*;

#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    PartialEq,
    Eq,
    Hash,
    CanonicalSerialize,
    CanonicalDeserialize,
    Allocative,
)]
pub struct Mont254BitChallenge<F: JoltField> {
    value: F,
}

impl<F: JoltField> Mont254BitChallenge<F> {
    pub fn new(value: F) -> Self {
        Self { value }
    }

    pub fn value(&self) -> F {
        self.value
    }

    pub fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self::from(rng.gen::<u128>())
    }
}

impl<F: JoltField> UniformRand for Mont254BitChallenge<F> {
    fn rand<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        Self::from(rng.gen::<u128>())
    }
}

impl<F: JoltField> Display for Mont254BitChallenge<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mont254BitChallenge({})", self.value)
    }
}

impl<F: JoltField> Neg for Mont254BitChallenge<F> {
    type Output = F;

    fn neg(self) -> F {
        -self.value
    }
}

impl<F: JoltField> From<u128> for Mont254BitChallenge<F> {
    fn from(val: u128) -> Self {
        Self::new(F::from_u128(val))
    }
}

impl<F: JoltField> From<F> for Mont254BitChallenge<F> {
    fn from(value: F) -> Self {
        Self::new(value)
    }
}
impl Into<ark_bn254::Fr> for Mont254BitChallenge<ark_bn254::Fr> {
    #[inline(always)]
    fn into(self) -> ark_bn254::Fr {
        self.value()
    }
}

impl Into<ark_bn254::Fr> for &Mont254BitChallenge<ark_bn254::Fr> {
    #[inline(always)]
    fn into(self) -> ark_bn254::Fr {
        self.value()
    }
}

impl_field_ops_inline!(Mont254BitChallenge<ark_bn254::Fr>, ark_bn254::Fr, standard);

impl Into<TrackedFr> for Mont254BitChallenge<TrackedFr> {
    #[inline(always)]
    fn into(self) -> TrackedFr {
        TrackedFr(self.value().0)
    }
}

impl Into<TrackedFr> for &Mont254BitChallenge<TrackedFr> {
    #[inline(always)]
    fn into(self) -> TrackedFr {
        TrackedFr(self.value().0)
    }
}

impl_field_ops_inline!(Mont254BitChallenge<TrackedFr>, TrackedFr, standard);

impl OptimizedMul<ark_bn254::Fr, ark_bn254::Fr> for Mont254BitChallenge<ark_bn254::Fr> {
    fn mul_0_optimized(self, other: ark_bn254::Fr) -> Self::Output {
        if other.is_zero() {
            ark_bn254::Fr::zero()
        } else {
            self * other
        }
    }

    fn mul_1_optimized(self, other: ark_bn254::Fr) -> Self::Output {
        if other.is_one() {
            self.into()
        } else {
            self * other
        }
    }

    fn mul_01_optimized(self, other: ark_bn254::Fr) -> Self::Output {
        if other.is_zero() {
            ark_bn254::Fr::zero()
        } else if other.is_one() {
            self.into()
        } else {
            self * other
        }
    }
}

impl OptimizedMul<TrackedFr, TrackedFr> for Mont254BitChallenge<TrackedFr> {
    fn mul_0_optimized(self, other: TrackedFr) -> Self::Output {
        if other.is_zero() {
            TrackedFr::zero()
        } else {
            self * other
        }
    }

    fn mul_1_optimized(self, other: TrackedFr) -> Self::Output {
        if other.is_one() {
            self.into()
        } else {
            self * other
        }
    }

    fn mul_01_optimized(self, other: TrackedFr) -> Self::Output {
        if other.is_zero() {
            TrackedFr::zero()
        } else if other.is_one() {
            self.into()
        } else {
            self * other
        }
    }
}
