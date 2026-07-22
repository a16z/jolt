//! Core algebraic ladder: additive groups, rings, and fields, plus
//! primitive-integer embedding.

use num_traits::{One, Zero};
use rand_core::RngCore;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Minimal additive group operations shared by fields, rings, and accumulators.
pub trait AdditiveGroup:
    Sized
    + Clone
    + Copy
    + Send
    + Sync
    + Zero
    + Add<Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + AddAssign<Self>
    + Sub<Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + SubAssign<Self>
    + Neg<Output = Self>
{
}

/// Core ring arithmetic: additive group plus multiplication and one.
pub trait RingCore:
    AdditiveGroup
    + One
    + PartialEq
    + Eq
    + Default
    + Debug
    + Display
    + Hash
    + Mul<Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + MulAssign<Self>
    + Sum<Self>
    + for<'a> Sum<&'a Self>
    + Product<Self>
    + for<'a> Product<&'a Self>
{
    /// Returns `self * self`.
    #[inline]
    fn square(&self) -> Self {
        *self * *self
    }

    #[inline]
    fn pow2(exponent: usize) -> Self {
        let mut result = Self::one();
        let mut base = Self::one() + Self::one();
        let mut remaining = exponent;

        while remaining > 0 {
            if remaining % 2 == 1 {
                result *= base;
            }
            remaining /= 2;
            if remaining > 0 {
                base = base.square();
            }
        }

        result
    }
}

/// Algebraic field: ring arithmetic plus explicit inversion and sampling.
pub trait FieldCore: RingCore {
    /// Multiplicative inverse, or `None` for the zero element.
    fn inverse(&self) -> Option<Self>;

    /// Multiplicative inverse with zero mapped to zero.
    #[inline]
    fn inv_or_zero(self) -> Self {
        self.inverse().unwrap_or_else(Self::zero)
    }

    /// Samples a random element (RNG-backed, for tests and witnesses).
    fn random<R: RngCore>(rng: &mut R) -> Self;
}

/// Embed primitive integer values and multiply by primitive integer scalars.
pub trait FromPrimitiveInt: RingCore {
    #[inline]
    fn from_bool(v: bool) -> Self {
        if v {
            Self::from_u64(1)
        } else {
            Self::from_u64(0)
        }
    }

    #[inline]
    fn from_u8(v: u8) -> Self {
        Self::from_u64(v as u64)
    }

    #[inline]
    fn from_i8(v: i8) -> Self {
        Self::from_i64(v as i64)
    }

    #[inline]
    fn from_u16(v: u16) -> Self {
        Self::from_u64(v as u64)
    }

    #[inline]
    fn from_i16(v: i16) -> Self {
        Self::from_i64(v as i64)
    }

    #[inline]
    fn from_u32(v: u32) -> Self {
        Self::from_u64(v as u64)
    }

    #[inline]
    fn from_i32(v: i32) -> Self {
        Self::from_i64(v as i64)
    }

    fn from_u64(v: u64) -> Self;
    fn from_i64(v: i64) -> Self;
    fn from_u128(v: u128) -> Self;
    fn from_i128(v: i128) -> Self;

    /// Multiplies by a `u64`.
    #[inline(always)]
    fn mul_u64(&self, n: u64) -> Self {
        *self * Self::from_u64(n)
    }

    /// Multiplies by an `i64`.
    #[inline(always)]
    fn mul_i64(&self, n: i64) -> Self {
        *self * Self::from_i64(n)
    }

    /// Multiplies by a `u128`.
    #[inline(always)]
    fn mul_u128(&self, n: u128) -> Self {
        *self * Self::from_u128(n)
    }

    /// Multiplies by an `i128`.
    #[inline(always)]
    fn mul_i128(&self, n: i128) -> Self {
        *self * Self::from_i128(n)
    }

    /// Multiplies this ring element by the integer `2^pow`.
    #[inline]
    fn mul_pow_2(&self, pow: usize) -> Self {
        assert!(pow <= 255, "pow > 255");
        let mut res = *self;
        let mut p = pow;
        while p >= 64 {
            res *= Self::from_u64(1 << 63);
            p -= 63;
        }
        res * Self::from_u64(1 << p)
    }
}

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
