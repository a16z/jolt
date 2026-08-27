//! Sign-magnitude big integers.
//!
//! Two families share one sign state machine (stamped by
//! `impl_signed_family!` over normalized magnitude ops):
//!
//! - [`SignedBigInt<N>`]: magnitude `Limbs<N>` (width `N * 64` bits);
//!   aliases [`S64`], [`S128`], [`S192`], [`S256`].
//! - [`SignedBigIntHi32<N>`]: magnitude `[u64; N]` + `u32` tail (width
//!   `N * 64 + 32` bits) — 4 bytes smaller than `N + 1` full limbs, which
//!   matters when millions are stored in witness polynomials; aliases
//!   [`S96`], [`S160`], [`S224`].
//!
//! The surface is the consumer-used subset of jolt-field's signed API
//! (workspace + Akita audited); unconsumed constructors and truncating
//! combinators were dropped.
//!
//! Zero is not canonicalized: a zero magnitude may carry either sign.
//! Equality and ordering treat `+0` and `-0` as equal.

use crate::Limbs;
use core::cmp::Ordering;
use num_traits::Zero;

/// A signed big integer using `Limbs<N>` for magnitude and a sign bit.
#[derive(Clone, Copy, Debug)]
pub struct SignedBigInt<const N: usize> {
    pub magnitude: Limbs<N>,
    pub is_positive: bool,
}

pub type S64 = SignedBigInt<1>;
pub type S128 = SignedBigInt<2>;
pub type S192 = SignedBigInt<3>;
pub type S256 = SignedBigInt<4>;

/// Compact signed big integer with a `u32` top limb.
#[derive(Clone, Copy, Debug)]
pub struct SignedBigIntHi32<const N: usize> {
    magnitude_lo: [u64; N],
    magnitude_hi: u32,
    is_positive: bool,
}

pub type S96 = SignedBigIntHi32<1>;
pub type S160 = SignedBigIntHi32<2>;
pub type S224 = SignedBigIntHi32<3>;

/// Stamps one binary operator (owned rhs) plus its assign form, delegating
/// to an in-place method.
macro_rules! signed_binop {
    ($T:ident, $Op:ident, $method:ident, $OpAssign:ident, $assign_method:ident, $apply:ident) => {
        impl<const N: usize> core::ops::$Op for $T<N> {
            type Output = Self;
            #[inline]
            fn $method(mut self, rhs: Self) -> Self {
                self.$apply(&rhs);
                self
            }
        }
        impl<const N: usize> core::ops::$OpAssign for $T<N> {
            #[inline]
            fn $assign_method(&mut self, rhs: Self) {
                self.$apply(&rhs);
            }
        }
    };
}

/// Stamps the shared sign-magnitude state machine over a family providing
/// normalized magnitude ops (`mag_is_zero`, `mag_cmp`, `mag_add`, `mag_sub`,
/// `mag_mul`), a `zero()` constructor, and an `is_positive` field: the
/// operator matrix, `Neg`, sign-aware `Eq`/`Ord`, `Zero`, `Default`, and
/// `allocative` support.
macro_rules! impl_signed_family {
    ($T:ident) => {
        impl<const N: usize> $T<N> {
            #[inline(always)]
            fn add_assign_in_place(&mut self, rhs: &Self) {
                if self.is_positive == rhs.is_positive {
                    self.mag_add(rhs);
                } else if self.mag_cmp(rhs) != Ordering::Less {
                    self.mag_sub(rhs);
                } else {
                    let old = core::mem::replace(self, *rhs);
                    self.mag_sub(&old);
                }
            }

            #[inline(always)]
            fn sub_assign_in_place(&mut self, rhs: &Self) {
                self.add_assign_in_place(&rhs.negate());
            }

            #[inline(always)]
            fn mul_assign_in_place(&mut self, rhs: &Self) {
                self.is_positive = self.is_positive == rhs.is_positive;
                self.mag_mul(rhs);
            }

            /// Flips this value's sign.
            #[inline]
            pub fn negate(mut self) -> Self {
                self.is_positive = !self.is_positive;
                self
            }

            /// Returns the sign (`true` = non-negative).
            #[inline]
            pub const fn sign(&self) -> bool {
                self.is_positive
            }
        }

        signed_binop!($T, Add, add, AddAssign, add_assign, add_assign_in_place);
        signed_binop!($T, Sub, sub, SubAssign, sub_assign, sub_assign_in_place);
        signed_binop!($T, Mul, mul, MulAssign, mul_assign, mul_assign_in_place);

        impl<const N: usize> core::ops::Neg for $T<N> {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                self.negate()
            }
        }

        impl<const N: usize> PartialEq for $T<N> {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                (self.mag_is_zero() && other.mag_is_zero())
                    || (self.is_positive == other.is_positive
                        && self.mag_cmp(other) == Ordering::Equal)
            }
        }

        impl<const N: usize> Eq for $T<N> {}

        impl<const N: usize> PartialOrd for $T<N> {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl<const N: usize> Ord for $T<N> {
            #[inline]
            fn cmp(&self, other: &Self) -> Ordering {
                if self.mag_is_zero() && other.mag_is_zero() {
                    return Ordering::Equal;
                }
                match (self.is_positive, other.is_positive) {
                    (true, false) => Ordering::Greater,
                    (false, true) => Ordering::Less,
                    (positive, _) => {
                        let ord = self.mag_cmp(other);
                        if positive {
                            ord
                        } else {
                            ord.reverse()
                        }
                    }
                }
            }
        }

        impl<const N: usize> Zero for $T<N> {
            #[inline]
            fn zero() -> Self {
                Self::zero()
            }
            #[inline]
            fn is_zero(&self) -> bool {
                self.mag_is_zero()
            }
        }

        impl<const N: usize> Default for $T<N> {
            #[inline]
            fn default() -> Self {
                Self::zero()
            }
        }

        #[cfg(feature = "allocative")]
        impl<const N: usize> allocative::Allocative for $T<N> {
            fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
                visitor.visit_simple_sized::<Self>();
            }
        }
    };
}

impl_signed_family!(SignedBigInt);
impl_signed_family!(SignedBigIntHi32);

impl<const N: usize> SignedBigInt<N> {
    #[inline(always)]
    fn mag_is_zero(&self) -> bool {
        self.magnitude.is_zero()
    }

    #[inline(always)]
    fn mag_cmp(&self, rhs: &Self) -> Ordering {
        self.magnitude.cmp(&rhs.magnitude)
    }

    #[inline(always)]
    fn mag_add(&mut self, rhs: &Self) {
        let _ = self.magnitude.add_with_carry(&rhs.magnitude);
    }

    #[inline(always)]
    fn mag_sub(&mut self, rhs: &Self) {
        let _ = self.magnitude.sub_with_borrow(&rhs.magnitude);
    }

    #[inline(always)]
    fn mag_mul(&mut self, rhs: &Self) {
        self.magnitude = self.magnitude.mul_low(&rhs.magnitude);
    }

    #[inline]
    pub fn new(limbs: [u64; N], is_positive: bool) -> Self {
        Self::from_limbs(Limbs::new(limbs), is_positive)
    }

    #[inline]
    pub fn from_limbs(magnitude: Limbs<N>, is_positive: bool) -> Self {
        Self {
            magnitude,
            is_positive,
        }
    }

    #[inline]
    pub fn zero() -> Self {
        Self::from_limbs(Limbs::zero(), true)
    }

    #[inline]
    pub fn magnitude_limbs(&self) -> [u64; N] {
        self.magnitude.0
    }

    /// Multiplies and truncates the result to `P` limbs.
    #[inline]
    pub fn mul_trunc<const M: usize, const P: usize>(
        &self,
        rhs: &SignedBigInt<M>,
    ) -> SignedBigInt<P> {
        SignedBigInt::from_limbs(
            self.magnitude.mul_trunc::<M, P>(&rhs.magnitude),
            self.is_positive == rhs.is_positive,
        )
    }

    /// Adds `self * rhs` to `acc`, truncating the product and sum to `P` limbs.
    #[inline]
    pub fn fmadd_trunc<const M: usize, const P: usize>(
        &self,
        rhs: &SignedBigInt<M>,
        acc: &mut SignedBigInt<P>,
    ) {
        let product = self.mul_trunc::<M, P>(rhs);
        acc.add_assign_in_place(&product);
    }

    #[inline]
    pub fn from_u64(value: u64) -> Self {
        Self::from_u64_with_sign(value, true)
    }

    #[inline]
    pub fn from_u64_with_sign(value: u64, is_positive: bool) -> Self {
        Self::from_limbs(Limbs::from_u64(value), is_positive)
    }

    #[inline]
    pub fn from_i64(value: i64) -> Self {
        Self::from_u64_with_sign(value.unsigned_abs(), value >= 0)
    }

    #[inline]
    fn from_u128_with_sign(value: u128, is_positive: bool) -> Self {
        const { assert!(N >= 2, "u128 conversion requires at least 2 limbs") }
        let mut limbs = [0u64; N];
        limbs[0] = value as u64;
        limbs[1] = (value >> 64) as u64;
        Self::new(limbs, is_positive)
    }

    #[inline]
    pub fn from_u128(value: u128) -> Self {
        Self::from_u128_with_sign(value, true)
    }

    #[inline]
    pub fn from_i128(value: i128) -> Self {
        Self::from_u128_with_sign(value.unsigned_abs(), value >= 0)
    }
}

impl S64 {
    #[inline]
    pub fn to_i128(&self) -> i128 {
        let magnitude = self.magnitude.0[0] as i128;
        if self.is_positive {
            magnitude
        } else {
            -magnitude
        }
    }

    #[inline]
    pub fn magnitude_as_u64(&self) -> u64 {
        self.magnitude.0[0]
    }
}

impl S128 {
    /// Returns the value if it fits in `i128` (`i128::MIN` included).
    #[inline]
    pub fn to_i128(&self) -> Option<i128> {
        let mag = self.magnitude_as_u128();
        if self.is_positive {
            (mag >> 127 == 0).then_some(mag as i128)
        } else if mag >> 127 == 0 {
            Some(-(mag as i128))
        } else {
            (mag == 1 << 127).then_some(i128::MIN)
        }
    }

    #[inline]
    pub fn magnitude_as_u128(&self) -> u128 {
        (self.magnitude.0[1] as u128) << 64 | (self.magnitude.0[0] as u128)
    }
}

impl<const N: usize> SignedBigIntHi32<N> {
    #[inline(always)]
    fn mag_is_zero(&self) -> bool {
        self.magnitude_hi == 0 && self.magnitude_lo.iter().all(|&l| l == 0)
    }

    #[inline(always)]
    fn mag_cmp(&self, rhs: &Self) -> Ordering {
        self.magnitude_hi.cmp(&rhs.magnitude_hi).then_with(|| {
            self.magnitude_lo
                .iter()
                .rev()
                .cmp(rhs.magnitude_lo.iter().rev())
        })
    }

    #[inline(always)]
    fn mag_add(&mut self, rhs: &Self) {
        let mut carry: u128 = 0;
        for i in 0..N {
            let sum = (self.magnitude_lo[i] as u128) + (rhs.magnitude_lo[i] as u128) + carry;
            self.magnitude_lo[i] = sum as u64;
            carry = sum >> 64;
        }
        // The u32 tail wraps at width, matching full-limb truncation semantics.
        self.magnitude_hi =
            ((self.magnitude_hi as u128) + (rhs.magnitude_hi as u128) + carry) as u32;
    }

    #[inline(always)]
    fn mag_sub(&mut self, rhs: &Self) {
        let mut borrow = false;
        for i in 0..N {
            let (d1, b1) = self.magnitude_lo[i].overflowing_sub(rhs.magnitude_lo[i]);
            let (d2, b2) = d1.overflowing_sub(u64::from(borrow));
            self.magnitude_lo[i] = d2;
            borrow = b1 || b2;
        }
        self.magnitude_hi = self
            .magnitude_hi
            .wrapping_sub(rhs.magnitude_hi)
            .wrapping_sub(u32::from(borrow));
    }

    /// General `(N+1)`-limb schoolbook multiply truncated to the type width.
    ///
    /// Carries are extracted per partial product, so intermediate sums never
    /// overflow `u128` on any input (the baseline's unrolled kernels wrap for
    /// large second limbs). Constant bounds let LLVM fully unroll this.
    #[inline(always)]
    fn mag_mul(&mut self, rhs: &Self) {
        const { assert!(2 * (N + 1) <= 16, "N too large for the product buffer") }
        let limb = |lo: &[u64; N], hi: u32, i: usize| {
            if i < N {
                lo[i] as u128
            } else {
                hi as u128
            }
        };
        let mut prod = [0u64; 16];
        for i in 0..=N {
            let a = limb(&self.magnitude_lo, self.magnitude_hi, i);
            let mut carry: u128 = 0;
            for j in 0..=N {
                let p = a * limb(&rhs.magnitude_lo, rhs.magnitude_hi, j)
                    + (prod[i + j] as u128)
                    + carry;
                prod[i + j] = p as u64;
                carry = p >> 64;
            }
        }
        self.magnitude_lo.copy_from_slice(&prod[..N]);
        self.magnitude_hi = prod[N] as u32;
    }

    #[inline]
    pub const fn new(magnitude_lo: [u64; N], magnitude_hi: u32, is_positive: bool) -> Self {
        Self {
            magnitude_lo,
            magnitude_hi,
            is_positive,
        }
    }

    #[inline]
    pub const fn zero() -> Self {
        Self::new([0; N], 0, true)
    }

    #[inline]
    pub const fn magnitude_lo(&self) -> &[u64; N] {
        &self.magnitude_lo
    }

    #[inline]
    pub const fn magnitude_hi(&self) -> u32 {
        self.magnitude_hi
    }

    #[inline]
    pub const fn is_positive(&self) -> bool {
        self.is_positive
    }

    /// Converts into a full-limb `SignedBigInt<NPLUS1>`; asserts `NPLUS1 == N + 1`.
    #[inline]
    pub fn to_signed_bigint_nplus1<const NPLUS1: usize>(&self) -> SignedBigInt<NPLUS1> {
        assert!(NPLUS1 == N + 1, "NPLUS1 must be N + 1");
        let mut limbs = [0u64; NPLUS1];
        limbs[..N].copy_from_slice(&self.magnitude_lo);
        limbs[N] = self.magnitude_hi as u64;
        SignedBigInt::from_limbs(Limbs::new(limbs), self.is_positive)
    }
}

impl From<u128> for S160 {
    #[inline]
    fn from(val: u128) -> Self {
        Self::new([val as u64, (val >> 64) as u64], 0, true)
    }
}
