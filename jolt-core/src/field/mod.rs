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
    const NUM_BYTES: usize;
    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self;

    fn from_u64(n: u64) -> Option<Self>;
    fn from_i64(val: i64) -> Self;
    fn square(&self) -> Self;
    fn from_bytes(bytes: &[u8]) -> Self;
    fn inverse(&self) -> Option<Self>;
    fn to_u64(&self) -> Option<u64> {
        unimplemented!("conversion to u64 not implemented");
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
pub mod binius;
