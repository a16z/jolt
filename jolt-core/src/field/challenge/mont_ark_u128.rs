//! Optimized Challenge field for faster polynomial operations
//!
//! This module implements a specialized Challenge type that is a 125-bit subset of JoltField
//! with the top 3 bits zeroed out. This constraint enables ~1.6x faster
//! multiplication with ark_bn254::Fr elements, resulting in ~1.3x speedup for polynomial
//! binding operations.
//!
//! For implementation details and benchmarks, see: *TODO: LINK*

use crate::field::OptimizedMul;
use crate::field::{tracked_ark::TrackedFr, JoltField};
use allocative::Allocative;
use ark_ff::{BigInt, PrimeField, UniformRand};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use num::{One, Zero};
use rand::{Rng, RngCore};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::ops::{Add, Mul, Sub};

/// Challenge type storing two u64 limbs (low, high) matching arkworks' internal representation.
/// Only uses 125 bits (top 3 bits of high limb are always zero).
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash, Allocative)]
pub struct MontU128Challenge<F: JoltField> {
    /// Low 64 bits of the 125-bit value
    pub low: u64,
    /// High 61 bits of the 125-bit value (top 3 bits always zero)
    pub high: u64,
    _marker: PhantomData<F>,
}

// Custom serialization: serialize as [u64; 4] for compatibility with field element format
impl<F: JoltField> Valid for MontU128Challenge<F> {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<F: JoltField> CanonicalSerialize for MontU128Challenge<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        // Serialize as [u64; 4] for field element compatibility
        self.to_bigint_array()
            .serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        [0u64; 4].serialized_size(compress)
    }
}

impl<F: JoltField> CanonicalDeserialize for MontU128Challenge<F> {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let arr = <[u64; 4]>::deserialize_with_mode(reader, compress, validate)?;
        // arr[0] and arr[1] should be 0, arr[2] is low, arr[3] is high
        Ok(Self {
            low: arr[2],
            high: arr[3],
            _marker: PhantomData,
        })
    }
}

impl<F: JoltField> Display for MontU128Challenge<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bigint = BigInt::new(self.to_bigint_array());
        write!(f, "{bigint}")
    }
}

impl<F: JoltField> Debug for MontU128Challenge<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bigint = BigInt::new(self.to_bigint_array());
        write!(f, "{bigint}")
    }
}

impl<F: JoltField> From<u128> for MontU128Challenge<F> {
    #[inline(always)]
    fn from(value: u128) -> Self {
        Self::new(value)
    }
}

impl<F: JoltField> MontU128Challenge<F> {
    /// Creates a new challenge from a u128 value.
    /// Only the low 125 bits are used (top 3 bits are masked off).
    #[inline(always)]
    pub fn new(value: u128) -> Self {
        // Only mask the high limb - low limb passes through unchanged.
        // Top 3 bits of high limb are zeroed to ensure value < bn254 modulus.
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
    #[inline(always)]
    pub fn to_bigint_array(&self) -> [u64; 4] {
        [0, 0, self.low, self.high]
    }

    #[inline(always)]
    pub fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self::from(rng.gen::<u128>())
    }
}

impl<F: JoltField> UniformRand for MontU128Challenge<F> {
    #[inline(always)]
    fn rand<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        Self::from(rng.gen::<u128>())
    }
}

impl From<MontU128Challenge<ark_bn254::Fr>> for ark_bn254::Fr {
    #[inline(always)]
    fn from(challenge: MontU128Challenge<ark_bn254::Fr>) -> ark_bn254::Fr {
        ark_bn254::Fr::from_bigint_unchecked(BigInt::new(challenge.to_bigint_array())).unwrap()
    }
}

impl From<&MontU128Challenge<ark_bn254::Fr>> for ark_bn254::Fr {
    #[inline(always)]
    fn from(challenge: &MontU128Challenge<ark_bn254::Fr>) -> ark_bn254::Fr {
        ark_bn254::Fr::from_bigint_unchecked(BigInt::new(challenge.to_bigint_array())).unwrap()
    }
}

impl_field_ops_inline!(MontU128Challenge<ark_bn254::Fr>, ark_bn254::Fr, optimized);

impl From<MontU128Challenge<TrackedFr>> for TrackedFr {
    #[inline(always)]
    fn from(challenge: MontU128Challenge<TrackedFr>) -> TrackedFr {
        TrackedFr(
            ark_bn254::Fr::from_bigint_unchecked(BigInt::new(challenge.to_bigint_array())).unwrap(),
        )
    }
}

impl From<&MontU128Challenge<TrackedFr>> for TrackedFr {
    #[inline(always)]
    fn from(challenge: &MontU128Challenge<TrackedFr>) -> TrackedFr {
        TrackedFr(
            ark_bn254::Fr::from_bigint_unchecked(BigInt::new(challenge.to_bigint_array())).unwrap(),
        )
    }
}

impl_field_ops_inline!(MontU128Challenge<TrackedFr>, TrackedFr, optimized);

impl OptimizedMul<ark_bn254::Fr, ark_bn254::Fr> for MontU128Challenge<ark_bn254::Fr> {
    #[inline(always)]
    fn mul_0_optimized(self, other: ark_bn254::Fr) -> Self::Output {
        if other.is_zero() {
            ark_bn254::Fr::zero()
        } else {
            self * other
        }
    }

    #[inline(always)]
    fn mul_1_optimized(self, other: ark_bn254::Fr) -> Self::Output {
        if other.is_one() {
            self.into()
        } else {
            self * other
        }
    }

    #[inline(always)]
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
    #[inline(always)]
    fn mul_0_optimized(self, other: TrackedFr) -> Self::Output {
        if other.is_zero() {
            TrackedFr::zero()
        } else {
            self * other
        }
    }

    #[inline(always)]
    fn mul_1_optimized(self, other: TrackedFr) -> Self::Output {
        if other.is_one() {
            self.into()
        } else {
            self * other
        }
    }

    #[inline(always)]
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
