use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use allocative::Allocative;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::UniformRand;

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
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + for<'a> Add<&'a mut Self, Output = Self>
    + for<'a> Sub<&'a mut Self, Output = Self>
    + for<'a> Mul<&'a mut Self, Output = Self>
    + for<'a> AddAssign<&'a mut Self>
    + for<'a> SubAssign<&'a mut Self>
    + for<'a> MulAssign<&'a mut Self>
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
    fn square(&self) -> Self;
    fn from_bytes(bytes: &[u8]) -> Self;
}

pub trait Final {
    fn finalise(&self) -> Self;
}

// impl Allocative for ark_bn254::Fr {
//     fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
//         visitor.visit_simple_sized::<Self>();
//     }
// }

impl Final for ark_bn254::Fr {
    fn finalise(&self) -> Self {
        *self
    }
}

impl JoltField for ark_bn254::Fr {
    const NUM_BYTES: usize = 32;

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        <Self as UniformRand>::rand(rng)
    }

    fn is_zero(&self) -> bool {
        <Self as ark_std::Zero>::is_zero(self)
    }

    fn is_one(&self) -> bool {
        <Self as ark_std::One>::is_one(self)
    }

    fn zero() -> Self {
        <Self as ark_std::Zero>::zero()
    }

    fn one() -> Self {
        <Self as ark_std::One>::one()
    }

    fn from_u64(n: u64) -> Option<Self> {
        <Self as ark_ff::PrimeField>::from_u64(n)
    }

    fn square(&self) -> Self {
        <Self as ark_ff::Field>::square(self)
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), Self::NUM_BYTES);
        ark_bn254::Fr::from_le_bytes_mod_order(bytes)
    }
}
