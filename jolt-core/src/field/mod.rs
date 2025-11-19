use allocative::Allocative;
use ark_ff::{BigInt, UniformRand};
use num_traits::{One, Zero};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

pub trait FieldOps<Rhs = Self, Output = Self>:
    Add<Rhs, Output = Output>
    + Sub<Rhs, Output = Output>
    + Mul<Rhs, Output = Output>
    + Div<Rhs, Output = Output>
{
}

pub trait ChallengeFieldOps<F>:
    Copy
    + Send
    + Sync
    + Into<F>
    + Add<F, Output = F>
    + for<'a> Add<&'a F, Output = F>
    + Sub<F, Output = F>
    + for<'a> Sub<&'a F, Output = F>
    + Mul<F, Output = F>
    + for<'a> Mul<&'a F, Output = F>
    + Add<Self, Output = F>
    + for<'a> Add<&'a Self, Output = F>
    + Sub<Self, Output = F>
    + for<'a> Sub<&'a Self, Output = F>
    + Mul<Self, Output = F>
    + for<'a> Mul<&'a Self, Output = F>
{
}

pub trait FieldChallengeOps<C>:
    Add<C, Output = Self>
    + for<'a> Add<&'a C, Output = Self>
    + Sub<C, Output = Self>
    + for<'a> Sub<&'a C, Output = Self>
    + Mul<C, Output = Self>
    + for<'a> Mul<&'a C, Output = Self>
{
}

impl<F, C> ChallengeFieldOps<F> for C where
    C: Copy
        + Send
        + Sync
        + Into<F>
        + Add<F, Output = F>
        + for<'a> Add<&'a F, Output = F>
        + Sub<F, Output = F>
        + for<'a> Sub<&'a F, Output = F>
        + Mul<F, Output = F>
        + for<'a> Mul<&'a F, Output = F>
        + Add<C, Output = F>
        + for<'a> Add<&'a C, Output = F>
        + Sub<C, Output = F>
        + for<'a> Sub<&'a C, Output = F>
        + Mul<C, Output = F>
        + for<'a> Mul<&'a C, Output = F>
{
}

impl<F, C> FieldChallengeOps<C> for F where
    F: JoltField
        + Add<C, Output = F>
        + for<'a> Add<&'a C, Output = F>
        + Sub<C, Output = F>
        + for<'a> Sub<&'a C, Output = F>
        + Mul<C, Output = F>
        + for<'a> Mul<&'a C, Output = F>
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
    + FieldChallengeOps<Self::Challenge>
{
    /// Number of bytes occupied by a single field element.
    const NUM_BYTES: usize;
    /// The Montgomery factor R = 2^(64*N) mod p
    const MONTGOMERY_R: Self;
    /// The squared Montgomery factor R^2 = 2^(128*N) mod p
    const MONTGOMERY_R_SQUARE: Self;

    /// Unreduced field element representation with N 64 bit limbs
    type Unreduced<const N: usize>: Clone
        + Copy
        + Debug
        + Display
        + Send
        + Sync
        + Default
        + Eq
        + PartialEq
        + Ord
        + From<u128>
        + From<[u64; N]>
        + From<BigInt<N>>
        + Zero
        // Truncated multiplication variants used by accumulators
        + MulTrunc<Other<3> = Self::Unreduced<3>, Output<7> = Self::Unreduced<7>>
        + MulTrunc<Other<3> = Self::Unreduced<3>, Output<8> = Self::Unreduced<8>>
        + MulTrunc<Other<4> = Self::Unreduced<4>, Output<8> = Self::Unreduced<8>>
        + MulTrunc<Other<4> = Self::Unreduced<4>, Output<9> = Self::Unreduced<9>>
        + MulU64WithCarry<Output<5> = Self::Unreduced<5>>
        + Add<Output = Self::Unreduced<N>>
        + Add<Self::Unreduced<4>, Output = Self::Unreduced<N>>
        + for<'a> Add<&'a Self::Unreduced<N>, Output = Self::Unreduced<N>>
        + for<'a> Add<&'a Self::Unreduced<4>, Output = Self::Unreduced<N>>
        + Sub<Output = Self::Unreduced<N>>
        + for<'a> Sub<&'a Self::Unreduced<N>, Output = Self::Unreduced<N>>
        + AddAssign
        + for<'a> AddAssign<&'a Self::Unreduced<N>>
        + AddAssign<Self::Unreduced<4>>
        + AddAssign<Self::Unreduced<5>>
        + AddAssign<Self::Unreduced<6>>
        + AddAssign<Self::Unreduced<7>>
        + AddAssign<Self::Unreduced<8>>
        + SubAssign
        + for<'a> SubAssign<&'a Self::Unreduced<N>>;

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
        + CanonicalSerialize
        + CanonicalDeserialize
        + Allocative
        + From<u128>
        + Into<Self>
        + ChallengeFieldOps<Self>
        + UniformRand
        + OptimizedMul<Self, Self>;

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self;
    /// Computes the small-value lookup tables.
    fn compute_lookup_tables() -> Self::SmallValueLookupTables {
        unimplemented!("Small-value lookup tables are unimplemented")
    }
    // Conversion from primitive integers to field elements in Montgomery form.

    /// Conversion from a boolean to a field element.
    fn from_bool(val: bool) -> Self;
    /// Conversion from a 8-bit unsigned integer to a field element.
    fn from_u8(n: u8) -> Self;
    /// Conversion from a 16-bit unsigned integer to a field element.
    fn from_u16(n: u16) -> Self;
    /// Conversion from a 32-bit unsigned integer to a field element.
    fn from_u32(n: u32) -> Self;
    /// Conversion from a 64-bit unsigned integer to a field element.
    fn from_u64(n: u64) -> Self;
    /// Conversion from a 64-bit signed integer to a field element.
    fn from_i64(val: i64) -> Self;
    /// Conversion from a 128-bit signed integer to a field element.
    fn from_i128(val: i128) -> Self;
    /// Conversion from a 128-bit unsigned integer to a field element.
    fn from_u128(val: u128) -> Self;
    /// Squares a field element.
    fn square(&self) -> Self;
    /// Conversion from a byte array to a field element.
    fn from_bytes(bytes: &[u8]) -> Self;
    /// Inverts a field element.
    fn inverse(&self) -> Option<Self>;
    /// Conversion from a field element to a 64-bit unsigned integer.
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
    /// The result will be in Montgomery form (if BN254)
    #[inline(always)]
    fn mul_u128(&self, n: u128) -> Self {
        *self * Self::from_u128(n)
    }
    /// Does a field multiplication with a `i128`.
    /// The result will be in Montgomery form (if BN254)
    #[inline(always)]
    fn mul_i128(&self, n: i128) -> Self {
        *self * Self::from_i128(n)
    }

    /// Multiplication of a field element and a power of 2.
    /// Split into chunks of 64 bits, then multiply and accumulate.
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

    /// Get reference to the underlying Unreduced<4> representation without copying.
    fn as_unreduced_ref(&self) -> &Self::Unreduced<4>;

    /// Multiplication of two field elements without Montgomery reduction, returning a
    /// L-limb Unreduced (each limb is 64 bits).
    /// L = 8 or 9 depending on what the code needs.
    fn mul_unreduced<const N: usize>(self, other: Self) -> Self::Unreduced<N>;

    /// Multiplication of a field element and a u64 without Barrett reduction, returning a
    /// 5-limb Unreduced (each limb is 64 bits)
    fn mul_u64_unreduced(self, other: u64) -> Self::Unreduced<5>;

    /// Multiplication of a field element and a u128 without Barrett reduction, returning a
    /// 6-limb Unreduced (each limb is 64 bits)
    fn mul_u128_unreduced(self, other: u128) -> Self::Unreduced<6>;

    /// Montgomery reduction of an Unreduced to a field element (compute a * R^{-1} mod p).
    ///
    /// Need to specify the number of limbs in the Unreduced (at least 8)
    fn from_montgomery_reduce<const N: usize>(unreduced: Self::Unreduced<N>) -> Self;

    /// Barrett reduction of an Unreduced to a field element (compute a mod p).
    ///
    /// Need to specify the number of limbs in the Unreduced (at least 5, usually up to 7)
    fn from_barrett_reduce<const N: usize>(unreduced: Self::Unreduced<N>) -> Self;
}

pub trait MulU64WithCarry {
    type Output<const NPLUS1: usize>;

    /// Multiply by u64 with carry, returning
    fn mul_u64_w_carry<const NPLUS1: usize>(&self, other: u64) -> Self::Output<NPLUS1>;
}

pub trait MulTrunc {
    type Other<const M: usize>;
    type Output<const P: usize>;

    fn mul_trunc<const M: usize, const P: usize>(&self, other: &Self::Other<M>) -> Self::Output<P>;
}

/// Unified fused-multiply-add trait for accumulators.
/// Perform: acc += left * right.
pub trait FMAdd<Left, Right>: Sized {
    fn fmadd(&mut self, left: &Left, right: &Right);
}

/// Trait for accumulators that finish with Barrett reduction to a field element
pub trait BarrettReduce<F: JoltField> {
    fn barrett_reduce(&self) -> F;
}

/// Trait for accumulators that finish with Montgomery reduction to a field element
pub trait MontgomeryReduce<F: JoltField> {
    fn montgomery_reduce(&self) -> F;
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
    F: JoltField,
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

pub mod ark;
pub mod challenge;
pub mod tracked_ark;
