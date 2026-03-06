//! Optimized Challenge field for faster polynomial operations
//!
//! This module implements a specialized Challenge type that is a 125-bit subset of Field
//! with the top 3 bits zeroed out. This constraint enables ~1.6x faster
//! multiplication with Fr elements, resulting in ~1.3x speedup for polynomial
//! binding operations.

use crate::arkworks::bn254::Fr;
use crate::{Challenge, Field, Limbs, OptimizedMul};
#[cfg(feature = "allocative")]
use allocative::Allocative;
use ark_ff::UniformRand;
use num_traits::{One, Zero};
use rand::{Rng, RngCore};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Sub};

/// Challenge type storing two u64 limbs (low, high).
/// Only uses 125 bits (top 3 bits of high limb are always zero).
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct MontU128Challenge<F: Field> {
    pub low: u64,
    pub high: u64,
    _marker: PhantomData<F>,
}

impl<F: Field> MontU128Challenge<F> {
    /// Creates a new challenge from a u128 value.
    /// Only the low 125 bits are used (top 3 bits are masked off).
    #[inline(always)]
    pub fn new(value: u128) -> Self {
        let low = value as u64;
        let high = ((value >> 64) as u64) & (u64::MAX >> 3);
        Self {
            low,
            high,
            _marker: PhantomData,
        }
    }

    /// Returns the value as a [u64; 4] BigInt array for field conversion.
    /// Format: [0, 0, low, high] - zeros in lower limbs, value in upper limbs.
    /// This layout is required for the optimized `mul_by_hi_2limbs` multiplication path.
    #[inline(always)]
    pub fn to_bigint_array(&self) -> [u64; 4] {
        [0, 0, self.low, self.high]
    }

    #[inline(always)]
    pub fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self::from(rng.gen::<u128>())
    }
}

impl<F: Field> From<u128> for MontU128Challenge<F> {
    #[inline(always)]
    fn from(value: u128) -> Self {
        Self::new(value)
    }
}

impl<F: Field> UniformRand for MontU128Challenge<F> {
    #[inline(always)]
    fn rand<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        Self::from(rng.gen::<u128>())
    }
}

impl<F> Challenge<F> for MontU128Challenge<F>
where
    F: Field,
    Self: From<u128> + Into<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<F, Output = F>,
{
    fn rand<R: RngCore>(rng: &mut R) -> Self {
        <Self as UniformRand>::rand(rng)
    }
}

impl<F: Field> Display for MontU128Challenge<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let limbs = Limbs::new(self.to_bigint_array());
        write!(f, "{limbs}")
    }
}

impl<F: Field> Debug for MontU128Challenge<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let limbs = Limbs::new(self.to_bigint_array());
        write!(f, "{limbs}")
    }
}

impl From<MontU128Challenge<Fr>> for Fr {
    #[inline(always)]
    fn from(challenge: MontU128Challenge<Fr>) -> Fr {
        Fr::from_bigint_unchecked(Limbs::new(challenge.to_bigint_array())).unwrap()
    }
}

impl From<&MontU128Challenge<Fr>> for Fr {
    #[inline(always)]
    fn from(challenge: &MontU128Challenge<Fr>) -> Fr {
        Fr::from_bigint_unchecked(Limbs::new(challenge.to_bigint_array())).unwrap()
    }
}

impl_field_ops_inline!(MontU128Challenge<Fr>, Fr, optimized);

impl OptimizedMul<Fr, Fr> for MontU128Challenge<Fr> {
    #[inline(always)]
    fn mul_0_optimized(self, other: Fr) -> Self::Output {
        if other.is_zero() {
            Fr::zero()
        } else {
            self * other
        }
    }

    #[inline(always)]
    fn mul_1_optimized(self, other: Fr) -> Self::Output {
        if other.is_one() {
            self.into()
        } else {
            self * other
        }
    }

    #[inline(always)]
    fn mul_01_optimized(self, other: Fr) -> Self::Output {
        if other.is_zero() {
            Fr::zero()
        } else if other.is_one() {
            self.into()
        } else {
            self * other
        }
    }
}
