#[cfg(feature = "allocative")]
use allocative::Allocative;
use ark_ff::BigInt;
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
    /// Byte length of a canonical (compressed) serialized element.
    const NUM_BYTES: usize;

    /// Serializes to compressed canonical form (little-endian, `NUM_BYTES` long).
    fn to_bytes(&self) -> Vec<u8>;

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

/// Exposes unreduced (wider-than-field) multiplication results.
///
/// In Montgomery form, a product of two 4-limb elements produces up to 8 limbs
/// before reduction. This trait gives access to those raw limbs so that
/// [`FMAdd`](crate::FMAdd) accumulators can defer reduction across many
/// multiply-add steps, amortizing the cost.
#[allow(dead_code)]
pub(crate) trait UnreducedOps: Field {
    /// Direct reference to the inner Montgomery-form limbs as `BigInt<4>`.
    fn as_unreduced_ref(&self) -> &BigInt<4>;

    /// Full Montgomery-form multiplication without reduction, returning `L` limbs.
    fn mul_unreduced<const L: usize>(self, other: Self) -> BigInt<L>;

    /// Multiply by a `u64` without reduction, returning 5 limbs.
    fn mul_u64_unreduced(self, other: u64) -> BigInt<5>;

    /// Multiply by a `u128` without reduction, returning 6 limbs.
    fn mul_u128_unreduced(self, other: u128) -> BigInt<6>;
}

/// Converts wide (unreduced) limb representations back to field elements.
///
/// Two reduction strategies are available:
/// - **Montgomery reduction** — standard REDC; used when the accumulator is
///   already in Montgomery form.
/// - **Barrett reduction** — uses a precomputed approximate inverse of `p`;
///   faster when the accumulator has more than `2N` limbs because it avoids
///   the sequential carry chain of REDC.
#[allow(dead_code)]
pub(crate) trait ReductionOps: UnreducedOps {
    /// Montgomery constant $R = 2^{256} \mod p$ (in Montgomery form).
    const MONTGOMERY_R: Self;
    /// Montgomery constant $R^2 = 2^{512} \mod p$ (in Montgomery form).
    const MONTGOMERY_R_SQUARE: Self;

    /// Reduces a wide `BigInt<L>` to a field element via Montgomery REDC.
    fn from_montgomery_reduce<const L: usize>(unreduced: BigInt<L>) -> Self;
    /// Reduces a wide `BigInt<L>` to a field element via Barrett reduction.
    fn from_barrett_reduce<const L: usize>(unreduced: BigInt<L>) -> Self;
}

/// A Fiat-Shamir challenge value that can be combined with field elements.
///
/// Challenges are drawn from a transcript and used as random scalars in
/// batching (RLC) and sumcheck. Two implementations exist:
/// - [`MontU128Challenge`](crate::challenge::MontU128Challenge) — 125-bit
///   range, avoids full-width Montgomery reduction (default).
/// - [`Mont254BitChallenge`](crate::challenge::Mont254BitChallenge) — full
///   254-bit field element, used when the wider range is needed.
pub trait Challenge<F: Field>:
    Copy
    + Send
    + Sync
    + From<u128>
    + Into<F>
    + Add<F, Output = F>
    + Sub<F, Output = F>
    + Mul<F, Output = F>
{
    fn rand<R: RngCore>(rng: &mut R) -> Self;
}

/// Associates a [`Field`] with its default [`Challenge`] type.
///
/// The associated type is selected at compile time via the
/// `challenge-254-bit` feature flag.
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
