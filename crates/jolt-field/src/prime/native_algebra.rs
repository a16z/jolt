//! Native `num_traits`/`std` supertrait impls and core-algebra markers for the
//! concrete prime fields (`Fp32`/`Fp64`/`Fp128`).
//!
//! These are the Jolt-free supertrait obligations of the native
//! [`AdditiveGroup`]/[`RingCore`]/[`FieldCore`] hierarchy:
//! `Zero`/`One`/`Display`/`Hash`/`Sum`/`Product` plus the empty algebra markers.
//! The non-trivial `FieldCore::inverse`/`FieldCore::random` impls stay
//! co-located with each prime type.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter::{Product, Sum};

use num_traits::{One, Zero};

use super::{Fp128, Fp32, Fp64};
use crate::{AdditiveGroup, CanonicalField, RingCore};

macro_rules! impl_prime_native_algebra {
    ($ty:ident<$p:ident: $p_ty:ty>, $canon:ident) => {
        impl<const $p: $p_ty> Zero for $ty<$p> {
            #[inline]
            fn zero() -> Self {
                Self::default()
            }

            #[inline]
            fn is_zero(&self) -> bool {
                self.to_canonical_u128() == 0
            }
        }

        impl<const $p: $p_ty> One for $ty<$p> {
            #[inline]
            fn one() -> Self {
                if $p > 1 {
                    Self::$canon(1)
                } else {
                    Self::zero()
                }
            }
        }

        impl<const $p: $p_ty> fmt::Display for $ty<$p> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.to_canonical_u128())
            }
        }

        impl<const $p: $p_ty> Hash for $ty<$p> {
            fn hash<H: Hasher>(&self, state: &mut H) {
                self.to_canonical_u128().hash(state);
            }
        }

        impl<const $p: $p_ty> Sum for $ty<$p> {
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Self::zero(), |acc, x| acc + x)
            }
        }

        impl<'a, const $p: $p_ty> Sum<&'a Self> for $ty<$p> {
            fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
                iter.fold(Self::zero(), |acc, x| acc + *x)
            }
        }

        impl<const $p: $p_ty> Product for $ty<$p> {
            fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Self::one(), |acc, x| acc * x)
            }
        }

        impl<'a, const $p: $p_ty> Product<&'a Self> for $ty<$p> {
            fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
                iter.fold(Self::one(), |acc, x| acc * *x)
            }
        }

        impl<const $p: $p_ty> AdditiveGroup for $ty<$p> {}
        impl<const $p: $p_ty> RingCore for $ty<$p> {}
    };
}

impl_prime_native_algebra!(Fp32<P: u32>, from_canonical_u32);
impl_prime_native_algebra!(Fp64<P: u64>, from_canonical_u64);
impl_prime_native_algebra!(Fp128<P: u128>, from_canonical_u128);
