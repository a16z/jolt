#[cfg(feature = "allocative")]
use allocative::Allocative;
use num_traits::{One, Zero};
use rand_core::RngCore;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

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
    + Hash
    + Serialize
    + for<'de> Deserialize<'de>
    + MaybeAllocative
{
    /// Accumulator for deferred-reduction fused multiply-add.
    ///
    /// For BN254 Fr, this is a wide 9-limb integer that defers Montgomery
    /// reduction. For other fields, use [`NaiveAccumulator`](crate::NaiveAccumulator).
    type Accumulator: crate::FieldAccumulator<Field = Self>;

    /// Byte length of a canonical (compressed) serialized element.
    const NUM_BYTES: usize;

    /// Serializes to compressed canonical form (little-endian, 32 bytes).
    fn to_bytes(&self) -> [u8; 32];

    /// Samples a uniformly random field element.
    fn random<R: RngCore>(rng: &mut R) -> Self;
    /// Deserializes from little-endian bytes, reducing modulo the field prime.
    fn from_bytes(bytes: &[u8]) -> Self;
    /// Returns the value as `u64` if it fits, or `None` if >= 2^64.
    fn to_u64(&self) -> Option<u64>;
    /// Number of significant bits in the canonical representation.
    fn num_bits(&self) -> u32;
    /// Returns `self * self`.
    fn square(&self) -> Self;
    /// Multiplicative inverse, or `None` for the zero element.
    fn inverse(&self) -> Option<Self>;

    fn from_bool(val: bool) -> Self {
        if val {
            Self::one()
        } else {
            Self::zero()
        }
    }
    fn from_u8(n: u8) -> Self {
        Self::from_u64(n as u64)
    }
    fn from_u16(n: u16) -> Self {
        Self::from_u64(n as u64)
    }
    fn from_u32(n: u32) -> Self {
        Self::from_u64(n as u64)
    }
    fn from_u64(n: u64) -> Self;
    /// Maps a signed integer to its canonical field representative: negative
    /// values become `p - |val|`.
    fn from_i64(val: i64) -> Self;
    /// Maps a signed integer to its canonical field representative: negative
    /// values become `p - |val|`.
    fn from_i128(val: i128) -> Self;
    fn from_u128(val: u128) -> Self;

    fn mul_u64(&self, n: u64) -> Self {
        *self * Self::from_u64(n)
    }

    fn mul_i64(&self, n: i64) -> Self {
        *self * Self::from_i64(n)
    }

    fn mul_u128(&self, n: u128) -> Self {
        *self * Self::from_u128(n)
    }

    fn mul_i128(&self, n: i128) -> Self {
        *self * Self::from_i128(n)
    }

    /// Multiplication of a field element and a power of 2.
    /// Split into chunks of 63 bits, then multiply and accumulate.
    fn mul_pow_2(&self, mut pow: usize) -> Self {
        assert!(pow <= 255, "pow > 255");
        let mut res = *self;
        while pow >= 64 {
            res = res.mul_u64(1 << 63);
            pow -= 63;
        }
        res.mul_u64(1 << pow)
    }
}

#[cfg(feature = "allocative")]
pub trait MaybeAllocative: Allocative {}
#[cfg(feature = "allocative")]
impl<T: Allocative> MaybeAllocative for T {}
#[cfg(not(feature = "allocative"))]
pub trait MaybeAllocative {}
#[cfg(not(feature = "allocative"))]
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
