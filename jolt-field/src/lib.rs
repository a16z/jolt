use ark_ff::biginteger::S224;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use num_traits::{One, Zero};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

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
        + From<u128>
        + From<[u64; N]>
        + From<S224>
        + Zero
        + FmaddTrunc<Other<2> = Self::Unreduced<2>, Acc<8> = Self::Unreduced<8>>
        + FmaddTrunc<Other<3> = Self::Unreduced<3>, Acc<8> = Self::Unreduced<8>>
        + FmaddTrunc<Other<4> = Self::Unreduced<4>, Acc<8> = Self::Unreduced<8>>
        + MulU64WithCarry<Output<5> = Self::Unreduced<5>>
        + Add<Output = Self::Unreduced<N>>
        + for<'a> Add<&'a Self::Unreduced<N>, Output = Self::Unreduced<N>>
        + AddAssign
        + for<'a> AddAssign<&'a Self::Unreduced<N>>
        + AddAssign<Self::Unreduced<4>>
        + AddAssign<Self::Unreduced<5>>
        + AddAssign<Self::Unreduced<6>>
        + AddAssign<Self::Unreduced<7>>
        + AddAssign<Self::Unreduced<8>>;

    /// An implementation of `JoltField` may use some precomputed lookup tables to speed up the
    /// conversion of small primitive integers (e.g. `u16` values) into field elements. For example,
    /// the arkworks BN254 scalar field requires a conversion into Montgomery form, which naively
    /// requires a field multiplication, but can instead be looked up.
    type SmallValueLookupTables: Clone + Default + CanonicalSerialize + CanonicalDeserialize;

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

pub trait FmaddTrunc {
    type Other<const M: usize>;
    type Acc<const P: usize>;

    fn fmadd_trunc<const M: usize, const P: usize>(
        &self,
        other: &Self::Other<M>,
        acc: &mut Self::Acc<P>,
    );
}

#[cfg(feature = "allocative")]
pub trait MaybeAllocative: allocative::Allocative {}
#[cfg(feature = "allocative")]
impl<T: allocative::Allocative> MaybeAllocative for T {}
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

pub mod ark;
pub mod counters;
pub mod tracked_ark;
