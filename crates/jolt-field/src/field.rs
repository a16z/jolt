#[cfg(all(feature = "allocative", not(feature = "akita")))]
use allocative::Allocative;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::Mul;

use crate::{
    CanonicalBitLength, CanonicalBytes, CanonicalU64, FieldCore, FixedByteSize, FromPrimitiveInt,
    MulPow2, MulPrimitiveInt, RandomSampling, ReducingBytes, RingCore, TranscriptChallenge,
    WithAccumulator,
};

/// Prime field element abstraction used throughout Jolt.
///
/// This trait provides a backend-agnostic interface over a prime-order scalar
/// field.
///
/// All arithmetic is modular over the field's prime order. Elements are `Copy`,
/// thread-safe, and cheaply serializable. Negative integers are mapped via
/// their canonical representative modulo `p`.
pub trait Field:
    'static
    + Sized
    + Copy
    + Sync
    + Send
    + Default
    + Eq
    + Hash
    + Display
    + Debug
    + FieldCore
    + FromPrimitiveInt
    + CanonicalBytes
    + ReducingBytes
    + TranscriptChallenge
    + FixedByteSize
    + CanonicalBitLength
    + CanonicalU64
    + RandomSampling
    + WithAccumulator
    + MulPow2
    + MulPrimitiveInt
    + MaybeAllocative
{
}

// Akita's field is a foreign type, so the feature-gated bridge cannot add an
// `Allocative` impl for it. Keep the allocation-profiling bound for normal Jolt
// fields, but relax the umbrella while the Akita bridge is enabled.
#[cfg(all(feature = "allocative", not(feature = "akita")))]
pub trait MaybeAllocative: Allocative {}
#[cfg(all(feature = "allocative", not(feature = "akita")))]
impl<T: Allocative> MaybeAllocative for T {}
#[cfg(any(not(feature = "allocative"), feature = "akita"))]
pub trait MaybeAllocative {}
#[cfg(any(not(feature = "allocative"), feature = "akita"))]
impl<T> MaybeAllocative for T {}

/// Multiplication with fast-path short-circuits for zero and one.
///
/// In sumcheck hot loops many evaluations multiply by 0 or 1.
/// These methods avoid the full Montgomery multiplication in those cases.
pub trait OptimizedMul<Rhs, Output>: Sized + Mul<Rhs, Output = Output> {
    /// Returns `zero()` immediately if either operand is zero.
    fn mul_0_optimized(self, other: Rhs) -> Self::Output;
    /// Returns the other operand immediately if either is one.
    fn mul_1_optimized(self, other: Rhs) -> Self::Output;
    /// Combined: short-circuits on both zero and one.
    fn mul_01_optimized(self, other: Rhs) -> Self::Output;
}

impl<F> OptimizedMul<F, F> for F
where
    F: RingCore,
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
