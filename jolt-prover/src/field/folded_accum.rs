//! Folded unreduced accumulators for field arithmetic.
//!
//! These types store partial products in positional `u128` slots to avoid
//! carry propagation in hot accumulation loops. Carries are deferred to
//! `normalize()` right before Barrett/Montgomery reduction.

use crate::field::UnreducedInteger;
use ark_ff::BigInt;
use num_traits::Zero;
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, AddAssign, Sub, SubAssign};

/// Implements the common `UnreducedInteger` surface for folded slot types:
/// arithmetic, ordering, zero/default, conversion from `u128`, and normalization.
macro_rules! impl_folded_core_traits {
    ($name:ident, $slots:literal, $out_limbs:literal) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct $name(pub [u128; $slots]);

        impl $name {
            #[inline(always)]
            pub fn from_bigint<const N: usize>(value: BigInt<N>) -> Self {
                let mut out = [0u128; $slots];
                let limit = if N < $slots { N } else { $slots };
                let mut i = 0;
                while i < limit {
                    out[i] = value.0[i] as u128;
                    i += 1;
                }
                Self(out)
            }

            #[inline]
            pub fn normalize(self) -> BigInt<$out_limbs> {
                let mut out = [0u64; $out_limbs];
                let mut carry: u128 = 0;

                let mut i = 0;
                while i < $slots {
                    let (sum, overflow) = self.0[i].overflowing_add(carry);
                    if i < $out_limbs {
                        out[i] = sum as u64;
                    }
                    carry = (sum >> 64) + ((overflow as u128) << 64);
                    i += 1;
                }

                let mut j = $slots;
                while j < $out_limbs {
                    out[j] = carry as u64;
                    carry >>= 64;
                    j += 1;
                }

                BigInt::new(out)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{:?}", self.normalize().0)
            }
        }

        impl Ord for $name {
            fn cmp(&self, other: &Self) -> Ordering {
                let a = self.normalize();
                let b = other.normalize();
                let mut i = $out_limbs;
                while i > 0 {
                    i -= 1;
                    match a.0[i].cmp(&b.0[i]) {
                        Ordering::Equal => continue,
                        non_eq => return non_eq,
                    }
                }
                Ordering::Equal
            }
        }

        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Zero for $name {
            #[inline(always)]
            fn zero() -> Self {
                Self([0u128; $slots])
            }

            #[inline(always)]
            fn is_zero(&self) -> bool {
                self.0.iter().all(|v| *v == 0)
            }
        }

        impl From<u128> for $name {
            #[inline(always)]
            fn from(value: u128) -> Self {
                let mut out = [0u128; $slots];
                out[0] = value;
                Self(out)
            }
        }

        impl Add for $name {
            type Output = Self;

            #[inline(always)]
            fn add(self, rhs: Self) -> Self::Output {
                let mut out = self;
                let mut i = 0;
                while i < $slots {
                    out.0[i] += rhs.0[i];
                    i += 1;
                }
                out
            }
        }

        impl<'a> Add<&'a Self> for $name {
            type Output = Self;

            #[inline(always)]
            fn add(self, rhs: &'a Self) -> Self::Output {
                self + *rhs
            }
        }

        impl Sub for $name {
            type Output = Self;

            #[inline(always)]
            fn sub(self, rhs: Self) -> Self::Output {
                let lhs = self.normalize();
                let rhs = rhs.normalize();
                let mut out = Self([0u128; $slots]);
                let mut borrow: i128 = 0;
                let limit = if $slots < $out_limbs {
                    $slots
                } else {
                    $out_limbs
                };
                let mut i = 0;
                while i < limit {
                    let d = lhs.0[i] as i128 - rhs.0[i] as i128 + borrow;
                    out.0[i] = (d as u64) as u128;
                    borrow = d >> 64;
                    i += 1;
                }
                out
            }
        }

        impl<'a> Sub<&'a Self> for $name {
            type Output = Self;

            #[inline(always)]
            fn sub(self, rhs: &'a Self) -> Self::Output {
                self - *rhs
            }
        }

        impl AddAssign for $name {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                let mut i = 0;
                while i < $slots {
                    self.0[i] += rhs.0[i];
                    i += 1;
                }
            }
        }

        impl<'a> AddAssign<&'a Self> for $name {
            #[inline(always)]
            fn add_assign(&mut self, rhs: &'a Self) {
                *self += *rhs;
            }
        }

        impl SubAssign for $name {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl<'a> SubAssign<&'a Self> for $name {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: &'a Self) {
                *self -= *rhs;
            }
        }

        impl UnreducedInteger for $name {}
    };
}

/// Implements `AddAssign<BigInt<N>>` by lane-wise truncating into folded slots.
macro_rules! impl_addassign_bigint_trunc {
    ($target:ident, $n:literal) => {
        impl AddAssign<BigInt<$n>> for $target {
            #[inline(always)]
            fn add_assign(&mut self, rhs: BigInt<$n>) {
                let mut i = 0;
                while i < $n {
                    self.0[i] += rhs.0[i] as u128;
                    i += 1;
                }
            }
        }
    };
}

/// Implements `Add<BigInt<N>>` in terms of `AddAssign<BigInt<N>>`.
macro_rules! impl_add_bigint_trunc {
    ($target:ident, $n:literal) => {
        impl Add<BigInt<$n>> for $target {
            type Output = Self;

            #[inline(always)]
            fn add(mut self, rhs: BigInt<$n>) -> Self::Output {
                self += rhs;
                self
            }
        }
    };
}

/// Implements lane-wise `AddAssign` from a narrower folded type into a wider one.
macro_rules! impl_addassign_folded_trunc {
    ($target:ident, $rhs:ident, $lim:literal) => {
        impl AddAssign<$rhs> for $target {
            #[inline(always)]
            fn add_assign(&mut self, rhs: $rhs) {
                let mut i = 0;
                while i < $lim {
                    self.0[i] += rhs.0[i];
                    i += 1;
                }
            }
        }
    };
}

impl_folded_core_traits!(Folded256MulU64, 5, 5);
impl_folded_core_traits!(Folded256MulU128, 6, 6);
impl_folded_core_traits!(Folded256MulU128Accum, 7, 7);
impl_folded_core_traits!(Folded256Product, 8, 8);
impl_folded_core_traits!(Folded256ProductAccum, 8, 9);

impl_addassign_bigint_trunc!(Folded256MulU64, 4);
impl_addassign_bigint_trunc!(Folded256MulU128, 4);
impl_addassign_bigint_trunc!(Folded256MulU128, 5);
impl_addassign_bigint_trunc!(Folded256MulU128Accum, 4);
impl_addassign_bigint_trunc!(Folded256MulU128Accum, 5);
impl_addassign_bigint_trunc!(Folded256MulU128Accum, 6);
impl_addassign_bigint_trunc!(Folded256Product, 4);
impl_addassign_bigint_trunc!(Folded256Product, 5);
impl_addassign_bigint_trunc!(Folded256Product, 6);
impl_addassign_bigint_trunc!(Folded256ProductAccum, 4);

impl_add_bigint_trunc!(Folded256MulU64, 4);

impl_addassign_folded_trunc!(Folded256MulU128Accum, Folded256MulU64, 5);
impl_addassign_folded_trunc!(Folded256MulU128Accum, Folded256MulU128, 6);
impl_addassign_folded_trunc!(Folded256MulU128, Folded256MulU64, 5);
impl_addassign_folded_trunc!(Folded256Product, Folded256MulU64, 5);
impl_addassign_folded_trunc!(Folded256Product, Folded256MulU128, 6);
impl_addassign_folded_trunc!(Folded256ProductAccum, Folded256Product, 8);

/// Folded 4x4 limb multiplication into 8 positional slots.
///
/// This keeps all per-product operations independent (no inter-slot carry chain)
/// and defers carry propagation to normalization. It is a fast portable baseline,
/// but not necessarily globally optimal versus architecture-specific SIMD/asm.
#[inline(always)]
fn fold_mul_4x4(a: BigInt<4>, b: BigInt<4>) -> [u128; 8] {
    let mut slots = [0u128; 8];
    let mut i = 0;
    while i < 4 {
        let mut j = 0;
        while j < 4 {
            let p = (a.0[i] as u128) * (b.0[j] as u128);
            slots[i + j] += (p as u64) as u128;
            slots[i + j + 1] += ((p >> 64) as u64) as u128;
            j += 1;
        }
        i += 1;
    }
    slots
}

impl Folded256Product {
    #[inline(always)]
    pub fn from_mul(a: BigInt<4>, b: BigInt<4>) -> Self {
        Self(fold_mul_4x4(a, b))
    }
}

impl Folded256ProductAccum {
    #[inline(always)]
    pub fn from_mul(a: BigInt<4>, b: BigInt<4>) -> Self {
        Self(fold_mul_4x4(a, b))
    }
}
