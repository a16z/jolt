//! Optimized Challenge field for faster polynomial operations
//!
//! This module implements a specialized Challenge type that is a 125-bit subset of JoltField
//! with the two least significant bits zeroed out. This constraint enables ~1.6x faster
//! multiplication with ark_bn254::Fr elements, resulting in ~1.3x speedup for polynomial
//! binding operations.
//!
//! For implementation details and benchmarks, see: *TODO: LINK*

use crate::field::{tracked_ark::TrackedFr, JoltField};
use allocative::Allocative;
use ark_ff::{BigInt, PrimeField, UniformRand};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::{Rng, RngCore};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Sub};
#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    PartialEq,
    Eq,
    Hash,
    CanonicalSerialize,
    CanonicalDeserialize,
    Allocative,
)]
pub struct MontU128Challenge<F: JoltField> {
    value: [u64; 4],
    _marker: PhantomData<F>,
}

impl<F: JoltField> Display for MontU128Challenge<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MontU128Challenge([{}, {}, {}, {}]",
            self.value[0], self.value[1], self.value[2], self.value[3]
        )
    }
}

impl<F: JoltField> From<u128> for MontU128Challenge<F> {
    fn from(value: u128) -> Self {
        Self::new(value)
    }
}

impl<F: JoltField> MontU128Challenge<F> {
    pub fn new(value: u128) -> Self {
        // MontU128 can always be represented by 125 bits.
        // This guarantees that the big integer is never greater than the
        // bn254 modulus
        let val_masked = value & (u128::MAX >> 3);
        let low = val_masked as u64;
        let high = (val_masked >> 64) as u64;
        Self {
            value: [0, 0, low, high],
            _marker: PhantomData,
        }
    }

    pub fn value(&self) -> [u64; 4] {
        self.value
    }

    pub fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self::from(rng.gen::<u128>())
    }
}

impl<F: JoltField> UniformRand for MontU128Challenge<F> {
    fn rand<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        Self::from(rng.gen::<u128>())
    }
}

impl Into<ark_bn254::Fr> for MontU128Challenge<ark_bn254::Fr> {
    #[inline(always)]
    fn into(self) -> ark_bn254::Fr {
        ark_bn254::Fr::from_bigint_unchecked(BigInt::new(self.value())).unwrap()
    }
}
impl Into<ark_bn254::Fr> for &MontU128Challenge<ark_bn254::Fr> {
    #[inline(always)]
    fn into(self) -> ark_bn254::Fr {
        ark_bn254::Fr::from_bigint_unchecked(BigInt::new(self.value())).unwrap()
    }
}

/// Implements standard arithmetic operators (+, -, *) for F as JoltField types
///
/// This macro generates inline operator implementations between `F::Challenge`
/// and `F` types, as well as `Challenge` with itself, enabling efficient field
/// arithmetic without repeated boilerplate.
///
/// # Generated implementations
///
/// **Challenge with Field:**
/// - `F::Challenge + F` and `F + F::Challenge`
/// - `F::Challenge - F` and `F - F::Challenge`
/// - `F::Challenge * F` and `F * F::Challenge`
/// - Similar for `&F::Challenge` (reference types)
///
/// **Challenge with Challenge:**
/// - `F::Challenge + F::Challenge`
/// - `F::Challenge - F::Challenge`
/// - `F::Challenge * F::Challenge`
/// - Reference variants
///
/// All operations are marked `#[inline(always)]` for optimal performance in
/// hot path polynomial operations.
macro_rules! impl_field_ops_inline {
    ($t:ty, $f:ty) => {
        impl Add<$t> for $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: $t) -> $f {
                Into::<$f>::into(self) + Into::<$f>::into(rhs)
            }
        }
        impl<'a> Add<&'a $t> for $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: &'a $t) -> $f {
                Into::<$f>::into(self) + Into::<$f>::into(*rhs)
            }
        }
        impl<'a> Add<$t> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: $t) -> $f {
                Into::<$f>::into(*self) + Into::<$f>::into(rhs)
            }
        }
        impl<'a, 'b> Add<&'b $t> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: &'b $t) -> $f {
                Into::<$f>::into(*self) + Into::<$f>::into(*rhs)
            }
        }

        impl Sub<$t> for $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: $t) -> $f {
                Into::<$f>::into(self) - Into::<$f>::into(rhs)
            }
        }
        impl<'a> Sub<&'a $t> for $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: &'a $t) -> $f {
                Into::<$f>::into(self) - Into::<$f>::into(*rhs)
            }
        }
        impl<'a> Sub<$t> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: $t) -> $f {
                Into::<$f>::into(*self) - Into::<$f>::into(rhs)
            }
        }
        impl<'a, 'b> Sub<&'b $t> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: &'b $t) -> $f {
                Into::<$f>::into(*self) - Into::<$f>::into(*rhs)
            }
        }

        impl Mul<$t> for $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: $t) -> $f {
                Into::<$f>::into(self) * Into::<$f>::into(rhs)
            }
        }
        impl<'a> Mul<&'a $t> for $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: &'a $t) -> $f {
                Into::<$f>::into(self) * Into::<$f>::into(*rhs)
            }
        }
        impl<'a> Mul<$t> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: $t) -> $f {
                Into::<$f>::into(*self) * Into::<$f>::into(rhs)
            }
        }
        impl<'a, 'b> Mul<&'b $t> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: &'b $t) -> $f {
                Into::<$f>::into(*self) * Into::<$f>::into(*rhs)
            }
        }

        impl Add<$f> for $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: $f) -> $f {
                Into::<$f>::into(self) + rhs
            }
        }
        impl<'a> Add<&'a $f> for $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: &'a $f) -> $f {
                Into::<$f>::into(self) + rhs
            }
        }
        impl<'a> Add<$f> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: $f) -> $f {
                Into::<$f>::into(*self) + rhs
            }
        }
        impl<'a, 'b> Add<&'b $f> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: &'b $f) -> $f {
                Into::<$f>::into(*self) + rhs
            }
        }

        impl Sub<$f> for $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: $f) -> $f {
                Into::<$f>::into(self) - rhs
            }
        }
        impl<'a> Sub<&'a $f> for $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: &'a $f) -> $f {
                Into::<$f>::into(self) - rhs
            }
        }
        impl<'a> Sub<$f> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: $f) -> $f {
                Into::<$f>::into(*self) - rhs
            }
        }
        impl<'a, 'b> Sub<&'b $f> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: &'b $f) -> $f {
                Into::<$f>::into(*self) - rhs
            }
        }

        impl Mul<$f> for $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: $f) -> $f {
                rhs.mul_hi_bigint_u128(self.value())
            }
        }
        impl<'a> Mul<&'a $f> for $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: &'a $f) -> $f {
                rhs.mul_hi_bigint_u128(self.value())
            }
        }
        impl<'a> Mul<$f> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: $f) -> $f {
                rhs.mul_hi_bigint_u128(self.value())
            }
        }
        impl<'a, 'b> Mul<&'b $f> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: &'b $f) -> $f {
                rhs.mul_hi_bigint_u128(self.value())
            }
        }

        impl Add<$t> for $f {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: $t) -> $f {
                self + Into::<$f>::into(rhs)
            }
        }
        impl<'a> Add<&'a $t> for $f {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: &'a $t) -> $f {
                self + Into::<$f>::into(*rhs)
            }
        }
        impl<'a> Add<$t> for &'a $f {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: $t) -> $f {
                *self + Into::<$f>::into(rhs)
            }
        }
        impl<'a, 'b> Add<&'b $t> for &'a $f {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: &'b $t) -> $f {
                *self + Into::<$f>::into(*rhs)
            }
        }

        impl Sub<$t> for $f {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: $t) -> $f {
                self - Into::<$f>::into(rhs)
            }
        }
        impl<'a> Sub<&'a $t> for $f {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: &'a $t) -> $f {
                self - Into::<$f>::into(*rhs)
            }
        }
        impl<'a> Sub<$t> for &'a $f {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: $t) -> $f {
                *self - Into::<$f>::into(rhs)
            }
        }
        impl<'a, 'b> Sub<&'b $t> for &'a $f {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: &'b $t) -> $f {
                *self - Into::<$f>::into(*rhs)
            }
        }

        impl Mul<$t> for $f {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: $t) -> $f {
                self.mul_hi_bigint_u128(rhs.value())
            }
        }
        impl<'a> Mul<&'a $t> for $f {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: &'a $t) -> $f {
                self.mul_hi_bigint_u128(rhs.value())
            }
        }
        impl<'a> Mul<$t> for &'a $f {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: $t) -> $f {
                self.mul_hi_bigint_u128(rhs.value())
            }
        }
        impl<'a, 'b> Mul<&'b $t> for &'a $f {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: &'b $t) -> $f {
                self.mul_hi_bigint_u128(rhs.value())
            }
        }
    };
}

impl_field_ops_inline!(MontU128Challenge<ark_bn254::Fr>, ark_bn254::Fr);

impl Into<TrackedFr> for MontU128Challenge<TrackedFr> {
    #[inline(always)]
    fn into(self) -> TrackedFr {
        TrackedFr(ark_bn254::Fr::from_bigint_unchecked(BigInt::new(self.value())).unwrap())
    }
}

impl Into<TrackedFr> for &MontU128Challenge<TrackedFr> {
    #[inline(always)]
    fn into(self) -> TrackedFr {
        TrackedFr(ark_bn254::Fr::from_bigint_unchecked(BigInt::new(self.value())).unwrap())
    }
}

impl_field_ops_inline!(MontU128Challenge<TrackedFr>, TrackedFr);
