use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

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
{
    /// Number of bytes occupied by a single field element.
    const NUM_BYTES: usize;
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
    /// Initializes the static lookup tables using the provided values.
    fn initialize_lookup_tables(_init: Self::SmallValueLookupTables) {
        unimplemented!("Small-value lookup tables are unimplemented")
    }
    /// Conversion from primitive integers to field elements in Montgomery form.
    fn from_u8(n: u8) -> Self;
    fn from_u16(n: u16) -> Self;
    fn from_u32(n: u32) -> Self;
    fn from_u64(n: u64) -> Self;
    fn from_i64(val: i64) -> Self;
    fn from_i128(val: i128) -> Self;
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
    #[inline(always)]
    fn mul_i128(&self, n: i128) -> Self {
        *self * Self::from_i128(n)
    }
}

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
