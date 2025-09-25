use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

#[cfg(feature = "allocative")]
use allocative::Allocative;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{One, Zero};

pub trait FieldOps<Rhs = Self, Output = Self>:
    Add<Rhs, Output = Output>
    + Sub<Rhs, Output = Output>
    + Mul<Rhs, Output = Output>
    + Div<Rhs, Output = Output>
{
}

pub trait JoltField:
    'static
    + Sized
    + Zero
    + One
    + Neg<Output = Self>
    + FieldOps<Self, Self>
    + for<'a> FieldOps<&'a Self, Self>
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
    /// Number of bytes occupied by a single field element.
    const NUM_BYTES: usize;
    /// The Montgomery factor R = 2^(64*N) mod p
    const MONTGOMERY_R: Self;
    /// The squared Montgomery factor R^2 = 2^(128*N) mod p
    const MONTGOMERY_R_SQUARE: Self;

    /// An implementation of `JoltField` may use some precomputed lookup tables to speed up the
    /// conversion of small primitive integers (e.g. `u16` values) into field elements. For example,
    /// the arkworks BN254 scalar field requires a conversion into Montgomery form, which naively
    /// requires a field multiplication, but can instead be looked up.
    type SmallValueLookupTables: Clone + Default + CanonicalSerialize + CanonicalDeserialize;

    type Challenge: 'static
        + Sized
        + Copy
        + Clone
        + Send
        + Sync
        + Debug
        + Display
        + Default
        + Eq
        + Hash
        + Add<Self::Challenge, Output = Self>
        + Sub<Self::Challenge, Output = Self>
        + Mul<Self::Challenge, Output = Self>
        + Mul<Self, Output = Self>
        + for<'a> Mul<&'a Self, Output = Self>
        + CanonicalSerialize
        + CanonicalDeserialize
        + MaybeAllocative
        + From<u128>;

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self;
    /// Computes the small-value lookup tables.
    fn compute_lookup_tables() -> Self::SmallValueLookupTables {
        unimplemented!("Small-value lookup tables are unimplemented")
    }
    /// Conversion from primitive integers to field elements in Montgomery form.
    fn from_bool(val: bool) -> Self;
    fn from_u8(n: u8) -> Self;
    fn from_u16(n: u16) -> Self;
    fn from_u32(n: u32) -> Self;
    fn from_u64(n: u64) -> Self;
    fn from_i64(val: i64) -> Self;
    fn from_i128(val: i128) -> Self;
    fn from_u128(val: u128) -> Self;
    fn square(&self) -> Self;
    fn from_bytes(bytes: &[u8]) -> Self;
    fn inverse(&self) -> Option<Self>;
    fn to_u64(&self) -> Option<u64> {
        unimplemented!("conversion to u64 not implemented");
    }
    fn num_bits(&self) -> u32 {
        unimplemented!("num_bits is not implemented");
    }

    /// Does a field multiplication with a `u64`.
    /// The result will be in Montgomery form (if BN254)
    #[inline(always)]
    fn mul_u64(&self, n: u64) -> Self {
        *self * Self::from_u64(n)
    }
    /// Does a field multiplication with a `i64`.
    /// The result will be in Montgomery form (if BN254)
    #[inline(always)]
    fn mul_i64(&self, n: i64) -> Self {
        *self * Self::from_i64(n)
    }
    /// Does a field multiplication with a `u128`.
    /// Implementations may override with an intrinsic.
    #[inline(always)]
    fn mul_u128(&self, n: u128) -> Self {
        *self * Self::from_u128(n)
    }
    #[inline(always)]
    fn mul_i128(&self, n: i128) -> Self {
        *self * Self::from_i128(n)
    }

    fn mul_pow_2(&self, mut pow: usize) -> Self {
        if pow > 255 {
            panic!("pow > 255");
        }
        let mut res = *self;
        while pow >= 64 {
            res = res.mul_u64(1 << 63);
            pow -= 63;
        }
        res.mul_u64(1 << pow)
    }

    /// Get reference to the underlying BigInt<4> representation without copying
    fn as_bigint_ref(&self) -> &ark_ff::BigInt<4>;

    /// Montgomery reduction from 8-limb unreduced product to field element
    /// Note: Result is in Montgomery form with extra R factor
    fn from_montgomery_reduce_2n(unreduced: ark_ff::BigInt<8>) -> Self;

    /// Compute a linear combination of field elements with u64 coefficients.
    /// Performs unreduced accumulation in BigInt<NPLUS1>, then one final reduction.
    /// This is more efficient than individual multiplications and additions.
    fn linear_combination_u64(pairs: &[(Self, u64)], add_terms: &[Self]) -> Self;

    /// Compute a linear combination with separate positive and negative terms.
    /// Each term is multiplied by a u64 coefficient, then positive and negative
    /// sums are computed separately and subtracted. One final reduction is performed.
    fn linear_combination_i64(
        pos: &[(Self, u64)],
        neg: &[(Self, u64)],
        pos_add: &[Self],
        neg_add: &[Self],
    ) -> Self;
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

impl<T> OptimizedMul<T, T> for T
where
    T: JoltField,
{
    #[inline(always)]
    fn mul_0_optimized(self, other: T) -> T {
        if self.is_zero() || other.is_zero() {
            Self::zero()
        } else {
            self * other
        }
    }

    #[inline(always)]
    fn mul_1_optimized(self, other: T) -> T {
        if self.is_one() {
            other
        } else if other.is_one() {
            self
        } else {
            self * other
        }
    }

    #[inline(always)]
    fn mul_01_optimized(self, other: T) -> T {
        if self.is_zero() || other.is_zero() {
            Self::zero()
        } else {
            self.mul_1_optimized(other)
        }
    }
}

pub trait OptimizedMulI128<Output>: Sized {
    fn mul_i128_0_optimized(self, other: i128) -> Output;
    fn mul_i128_1_optimized(self, other: i128) -> Output;
    fn mul_i128_01_optimized(self, other: i128) -> Output;
}

/// Implement `OptimizedMul` for `JoltField` with `i128`
impl<T> OptimizedMulI128<T> for T
where
    T: JoltField,
{
    #[inline(always)]
    fn mul_i128_0_optimized(self, other: i128) -> T {
        if other.is_zero() {
            Self::zero()
        } else {
            self.mul_i128(other)
        }
    }

    #[inline(always)]
    fn mul_i128_1_optimized(self, other: i128) -> T {
        if other.is_one() {
            self
        } else {
            self.mul_i128(other)
        }
    }

    #[inline(always)]
    fn mul_i128_01_optimized(self, other: i128) -> T {
        if other.is_zero() {
            Self::zero()
        } else {
            self.mul_i128_1_optimized(other)
        }
    }
}

pub mod ark;
pub mod challenge;
//pub mod tracked_ark;
