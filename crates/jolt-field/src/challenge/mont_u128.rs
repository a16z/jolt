//! Optimized Challenge field for faster polynomial operations
//!
//! This module implements a specialized Challenge type that is a 125-bit subset of Field
//! with the top 3 bits zeroed out. This constraint enables ~1.6x faster
//! multiplication with Fr elements, resulting in ~1.3x speedup for polynomial
//! binding operations.

use crate::{Field, OptimizedMul, Challenge};
#[cfg(feature = "allocative")]
use allocative::Allocative;
use ark_bn254::Fr;
use ark_ff::{BigInt, PrimeField, UniformRand};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid,
    Validate, Write,
};
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
    /// Low 64 bits of the 125-bit value
    pub low: u64,
    /// High 61 bits of the 125-bit value (top 3 bits always zero)
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
    #[inline(always)]
    pub fn to_bigint_array(&self) -> [u64; 4] {
        [self.low, self.high, 0, 0]
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

impl<F: Field> Valid for MontU128Challenge<F> {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<F: Field> CanonicalSerialize for MontU128Challenge<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.to_bigint_array()
            .serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        [0u64; 4].serialized_size(compress)
    }
}

impl<F: Field> CanonicalDeserialize for MontU128Challenge<F> {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let arr = <[u64; 4]>::deserialize_with_mode(reader, compress, validate)?;
        Ok(Self {
            low: arr[0],
            high: arr[1],
            _marker: PhantomData,
        })
    }
}

impl<F: Field> Display for MontU128Challenge<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bigint = BigInt::new(self.to_bigint_array());
        write!(f, "{bigint}")
    }
}

impl<F: Field> Debug for MontU128Challenge<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bigint = BigInt::new(self.to_bigint_array());
        write!(f, "{bigint}")
    }
}

impl From<MontU128Challenge<Fr>> for Fr {
    #[inline(always)]
    fn from(challenge: MontU128Challenge<Fr>) -> Fr {
        Fr::from_bigint(BigInt::new(challenge.to_bigint_array())).unwrap()
    }
}

impl From<&MontU128Challenge<Fr>> for Fr {
    #[inline(always)]
    fn from(challenge: &MontU128Challenge<Fr>) -> Fr {
        Fr::from_bigint(BigInt::new(challenge.to_bigint_array())).unwrap()
    }
}

impl_field_ops_inline!(MontU128Challenge<Fr>, Fr, standard);

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