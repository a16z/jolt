//! Optimized Challenge field for faster polynomial operations
//!
//! This module implements a specialized Challenge type that is a 125-bit subset of JoltField
//! with the two least significant bits zeroed out. This constraint enables ~1.6x faster
//! multiplication with ark_bn254::Fr elements, resulting in ~1.3x speedup for polynomial
//! binding operations.
//!
//! For implementation details and benchmarks, see: *TODO: LINK*

use crate::field::OptimizedMul;
use crate::field::{tracked_ark::TrackedFr, JoltField};
//use crate::impl_field_ops_inline;
use allocative::Allocative;
use ark_ff::{BigInt, PrimeField, UniformRand};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use num::{One, Zero};
use rand::{Rng, RngCore};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Sub};
#[derive(
    Copy, Clone, Default, PartialEq, Eq, Hash, CanonicalSerialize, CanonicalDeserialize, Allocative,
)]
pub struct MontU128Challenge<F: JoltField> {
    value: [u64; 4],
    _marker: PhantomData<F>,
}

impl<F: JoltField> Display for MontU128Challenge<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bigint = BigInt::new(self.value);
        write!(f, "{bigint}")
    }
}

impl<F: JoltField> Debug for MontU128Challenge<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bigint = BigInt::new(self.value);
        write!(f, "{bigint}")
    }
}

impl<F: JoltField> From<u128> for MontU128Challenge<F> {
    fn from(value: u128) -> Self {
        Self::new(value)
    }
}

impl<F: JoltField> MontU128Challenge<F> {
    pub fn new(value: u128) -> Self {
        // MontU128 can always be represented by 125 bits.
        // This guarantees that the big integer is never greater than the
        // bn254 modulus
        let val_masked = value & (u128::MAX >> 3);
        let low = val_masked as u64;
        let high = (val_masked >> 64) as u64;
        Self {
            value: [0, 0, low, high],
            _marker: PhantomData,
        }
    }

    pub fn value(&self) -> [u64; 4] {
        self.value
    }

    pub fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self::from(rng.gen::<u128>())
    }
}

impl<F: JoltField> UniformRand for MontU128Challenge<F> {
    fn rand<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        Self::from(rng.gen::<u128>())
    }
}

impl Into<ark_bn254::Fr> for MontU128Challenge<ark_bn254::Fr> {
    #[inline(always)]
    fn into(self) -> ark_bn254::Fr {
        ark_bn254::Fr::from_bigint_unchecked(BigInt::new(self.value())).unwrap()
    }
}
impl Into<ark_bn254::Fr> for &MontU128Challenge<ark_bn254::Fr> {
    #[inline(always)]
    fn into(self) -> ark_bn254::Fr {
        ark_bn254::Fr::from_bigint_unchecked(BigInt::new(self.value())).unwrap()
    }
}

impl_field_ops_inline!(MontU128Challenge<ark_bn254::Fr>, ark_bn254::Fr, optimized);

impl Into<TrackedFr> for MontU128Challenge<TrackedFr> {
    #[inline(always)]
    fn into(self) -> TrackedFr {
        TrackedFr(ark_bn254::Fr::from_bigint_unchecked(BigInt::new(self.value())).unwrap())
    }
}

impl Into<TrackedFr> for &MontU128Challenge<TrackedFr> {
    #[inline(always)]
    fn into(self) -> TrackedFr {
        TrackedFr(ark_bn254::Fr::from_bigint_unchecked(BigInt::new(self.value())).unwrap())
    }
}

impl_field_ops_inline!(MontU128Challenge<TrackedFr>, TrackedFr, optimized);

impl OptimizedMul<ark_bn254::Fr, ark_bn254::Fr> for MontU128Challenge<ark_bn254::Fr> {
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

impl OptimizedMul<TrackedFr, TrackedFr> for MontU128Challenge<TrackedFr> {
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
