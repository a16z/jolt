//! Native `num_traits`/`std` supertrait impls for the wide unreduced
//! accumulator types and the generic [`AccumPair`].
//!
//! These are the Jolt-free supertrait obligations of the native
//! [`AdditiveGroup`] hierarchy: `Zero` plus the `Add`/`Sub` by-reference
//! forwarders that `AdditiveGroup` requires.

use std::ops::{Add, Sub};

use num_traits::Zero;

use super::{
    AccumPair, Fp128MulU64Accum, Fp128ProductAccum, Fp128x8i32, Fp32ProductAccum, Fp32x2i32,
    Fp64ProductAccum, Fp64x4i32, FpExt2Fp64ProductAccum, FpExt4Fp32ProductAccum,
};
use crate::AdditiveGroup;

macro_rules! impl_wide_native_additive {
    ($ty:ty, $zero:expr) => {
        impl Zero for $ty {
            #[inline]
            fn zero() -> Self {
                $zero
            }

            #[inline]
            fn is_zero(&self) -> bool {
                *self == Self::zero()
            }
        }

        impl<'a> Add<&'a Self> for $ty {
            type Output = Self;

            #[inline]
            fn add(self, rhs: &'a Self) -> Self::Output {
                self + *rhs
            }
        }

        impl<'a> Sub<&'a Self> for $ty {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: &'a Self) -> Self::Output {
                self - *rhs
            }
        }

        impl AdditiveGroup for $ty {}
    };
}

impl_wide_native_additive!(Fp32x2i32, Fp32x2i32([0; 2]));
impl_wide_native_additive!(Fp64x4i32, Fp64x4i32([0; 4]));
impl_wide_native_additive!(Fp128x8i32, Fp128x8i32([0; 8]));
impl_wide_native_additive!(Fp32ProductAccum, Fp32ProductAccum([0; 2]));
impl_wide_native_additive!(Fp64ProductAccum, Fp64ProductAccum([0; 2]));
impl_wide_native_additive!(Fp128MulU64Accum, Fp128MulU64Accum([0; 3]));
impl_wide_native_additive!(Fp128ProductAccum, Fp128ProductAccum([0; 4]));
impl_wide_native_additive!(FpExt4Fp32ProductAccum, FpExt4Fp32ProductAccum([0; 4]));
impl_wide_native_additive!(FpExt2Fp64ProductAccum, FpExt2Fp64ProductAccum([0; 4]));

impl<A: AdditiveGroup> Zero for AccumPair<A> {
    #[inline]
    fn zero() -> Self {
        Self(A::zero(), A::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.is_zero() && self.1.is_zero()
    }
}

impl<'a, A: AdditiveGroup> Add<&'a Self> for AccumPair<A> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}

impl<'a, A: AdditiveGroup> Sub<&'a Self> for AccumPair<A> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}

impl<A: AdditiveGroup> AdditiveGroup for AccumPair<A> {}
