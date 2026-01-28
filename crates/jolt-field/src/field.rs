#[cfg(feature = "allocative")]
use allocative::Allocative;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use num_traits::{One, Zero};
use rand_core::RngCore;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::UnreducedField;

/// Core field trait with minimal bounds
pub trait Field:
    'static
    + Sized
    + Zero
    + One
    + Neg<Output = Self>
    + Add<Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + Sub<Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + Mul<Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + Div<Output = Self>
    + for<'a> Div<&'a Self, Output = Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + core::iter::Sum<Self>
    + for<'a> core::iter::Sum<&'a Self>
    + core::iter::Product<Self>
    + for<'a> core::iter::Product<&'a Self>
    + Eq
    + Copy
    + Sync
    + Send
    + Display
    + Debug
    + Default
    + CanonicalSerialize
    + CanonicalDeserialize
    + Hash
    + MaybeAllocative
{
    const NUM_BYTES: usize;

    fn random<R: RngCore>(rng: &mut R) -> Self;
    fn from_bytes(bytes: &[u8]) -> Self;
    fn to_u64(&self) -> Option<u64>;
    fn num_bits(&self) -> u32;
    fn square(&self) -> Self;
    fn inverse(&self) -> Option<Self>;

    fn from_bool(val: bool) -> Self;
    fn from_u8(n: u8) -> Self;
    fn from_u16(n: u16) -> Self;
    fn from_u32(n: u32) -> Self;
    fn from_u64(n: u64) -> Self;
    fn from_i64(val: i64) -> Self;
    fn from_i128(val: i128) -> Self;
    fn from_u128(val: u128) -> Self;
}

/// Trait for operations with unreduced representations
pub trait UnreducedOps: Field {
    type UnreducedType: UnreducedField<Self>;

    fn as_unreduced_ref(&self) -> &Self::UnreducedType;
    fn mul_unreduced(self, other: Self) -> Self::UnreducedType;
    fn mul_u64_unreduced(self, other: u64) -> Self::UnreducedType;
    fn mul_u128_unreduced(self, other: u128) -> Self::UnreducedType;
}

/// Trait for field reduction operations
pub trait ReductionOps: UnreducedOps {
    const MONTGOMERY_R: Self;
    const MONTGOMERY_R_SQUARE: Self;

    fn from_montgomery_reduce(unreduced: Self::UnreducedType) -> Self;
    fn from_barrett_reduce(unreduced: Self::UnreducedType) -> Self;
}

/// Challenge trait with minimal bounds
pub trait Challenge<F: Field>:
    Copy + Send + Sync + From<u128> + Into<F> +
    Add<F, Output = F> + Sub<F, Output = F> + Mul<F, Output = F>
{
    fn rand<R: RngCore>(rng: &mut R) -> Self;
}

/// Trait for fields that support challenge types
pub trait WithChallenge: Field {
    type Challenge: Challenge<Self>;
}

#[cfg(feature = "allocative")]
pub trait MaybeAllocative: Allocative {}
#[cfg(feature = "allocative")]
impl<T: Allocative> MaybeAllocative for T {}
#[cfg(not(feature = "allocative"))]
pub trait MaybeAllocative {}
#[cfg(not(feature = "allocative"))]
impl<T> MaybeAllocative for T {}

pub trait OptimizedMul<Rhs, Output>: Sized + Mul<Rhs, Output = Output> {
    fn mul_0_optimized(self, other: Rhs) -> Self::Output;
    fn mul_1_optimized(self, other: Rhs) -> Self::Output;
    fn mul_01_optimized(self, other: Rhs) -> Self::Output;
}

impl<F> OptimizedMul<F, F> for F
where
    F: Field,
{
    #[inline(always)]
    fn mul_0_optimized(self, other: F) -> F {
        if self.is_zero() || other.is_zero() {
            Self::zero()
        } else {
            self * other
        }
    }

    #[inline(always)]
    fn mul_1_optimized(self, other: F) -> F {
        if self.is_one() {
            other
        } else if other.is_one() {
            self
        } else {
            self * other
        }
    }

    #[inline(always)]
    fn mul_01_optimized(self, other: F) -> F {
        if self.is_zero() || other.is_zero() {
            Self::zero()
        } else {
            self.mul_1_optimized(other)
        }
    }
}