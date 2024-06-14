use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

pub trait JoltField:
    'static
    + Sized
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
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
    + Debug
    + Default
    + CanonicalSerialize
    + CanonicalDeserialize
{
    const NUM_BYTES: usize;
    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self;

    fn is_zero(&self) -> bool;
    fn is_one(&self) -> bool;
    fn zero() -> Self;
    fn one() -> Self;
    fn from_u64(n: u64) -> Option<Self>;
    fn from_i64(val: i64) -> Self;
    fn square(&self) -> Self;
    fn from_bytes(bytes: &[u8]) -> Self;
    fn inverse(&self) -> Option<Self>;

    #[inline(always)]
    fn mul_0_optimized(self, other: Self) -> Self {
        if self.is_zero() || other.is_zero() {
            Self::zero()
        } else {
            self * other
        }
    }

    #[inline(always)]
    fn mul_1_optimized(self, other: Self) -> Self {
        if self.is_one() {
            other
        } else if other.is_one() {
            self
        } else {
            self * other
        }
    }

    #[inline(always)]
    fn mul_01_optimized(self, other: Self) -> Self {
        if self.is_zero() || other.is_zero() {
            Self::zero()
        } else if self.is_one() {
            other
        } else if other.is_one() {
            self
        } else {
            self * other
        }
    }
}

pub mod ark;
pub mod binius;
