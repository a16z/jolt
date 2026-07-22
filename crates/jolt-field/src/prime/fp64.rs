//! Prime field for primes of the form `p = 2^k − c` with `c` small, backed
//! by `u64` storage.
//!
//! Uses Solinas-style two-fold reduction.  For `c = 2^a ± 1` the fold
//! multiply is replaced by shift+add/sub, saving a u128 widening multiply.

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::{FieldCore, FromPrimitiveInt};
use rand_core::RngCore;

use crate::{CanonicalField, HalvingField, PseudoMersenneField};

use super::util::{is_pow2_u64, log2_pow2_u64, mul64_wide};

/// Prime field element for primes `p = 2^k − c` stored as `u64`.
///
/// The fold point `k` and offset `c = 2^k − p` are computed at compile time
/// from the const-generic `P`.  For `c = 2^a ± 1`, the fold multiply is
/// replaced by shift+add/sub.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Fp64<const P: u64>(pub(crate) u64);

impl<const P: u64> Fp64<P> {
    /// Fold point: smallest `k` such that `P ≤ 2^k`.
    const BITS: u32 = 64 - P.leading_zeros();

    /// Offset `c = 2^k − P`.
    pub const C: u64 = {
        let c = if Self::BITS == 64 {
            0u64.wrapping_sub(P)
        } else {
            (1u64 << Self::BITS) - P
        };
        assert!(P != 0, "modulus must be nonzero");
        assert!(P & 1 == 1, "modulus must be odd");
        assert!(
            (c as u128) * (c as u128 + 1) < P as u128,
            "C(C+1) < P required for fused canonicalize"
        );
        c
    };

    /// +1 means `C = 2^a + 1`, -1 means `C = 2^a - 1`, 0 means generic.
    const C_SHIFT_KIND: i8 = {
        let c = Self::C;
        if c > 1 && is_pow2_u64(c - 1) {
            1
        } else if c == u64::MAX || is_pow2_u64(c + 1) {
            -1
        } else {
            0
        }
    };

    const C_SHIFT: u32 = {
        let c = Self::C;
        if Self::C_SHIFT_KIND == 1 {
            log2_pow2_u64(c - 1)
        } else if Self::C_SHIFT_KIND == -1 {
            if c == u64::MAX {
                64
            } else {
                log2_pow2_u64(c + 1)
            }
        } else {
            0
        }
    };

    /// Mask for extracting the low `BITS` bits from a u128.
    const MASK: u128 = if Self::BITS == 64 {
        u64::MAX as u128
    } else {
        (1u128 << Self::BITS) - 1
    };

    /// u64-width mask (only valid when BITS < 64).
    const MASK64: u64 = if Self::BITS < 64 {
        (1u64 << Self::BITS) - 1
    } else {
        u64::MAX
    };

    /// Whether Solinas folding of a multiplication product can stay
    /// entirely in u64.  True when BITS < 64 and C·2^BITS < 2^64.
    const FOLD_IN_U64: bool = Self::BITS < 64 && (Self::C as u128) < (1u128 << (64 - Self::BITS));

    /// u64 multiply by C, split into u32-wide halves so LLVM emits
    /// `umull` (32×32→64) instead of promoting to u128.
    /// Only valid when C fits in u32 (always true: C < sqrt(P) < 2^32).
    #[inline(always)]
    fn mul_c_narrow(x: u64) -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            // x86_64 has fast scalar 64-bit multiply; use one multiply instead
            // of two widened 32-bit multiplies in the fold hot path.
            Self::C.wrapping_mul(x)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let c = Self::C as u32;
            let x_lo = x as u32;
            let x_hi = (x >> 32) as u32;
            (c as u64 * x_lo as u64).wrapping_add((c as u64 * x_hi as u64) << 32)
        }
    }

    /// Multiply `x` by `C`.  For `C = 2^a ± 1` uses shift+add/sub.
    #[inline(always)]
    fn mul_c(x: u64) -> u128 {
        if Self::C_SHIFT_KIND == 1 {
            ((x as u128) << Self::C_SHIFT) + x as u128
        } else if Self::C_SHIFT_KIND == -1 {
            ((x as u128) << Self::C_SHIFT) - x as u128
        } else {
            (Self::C as u128) * (x as u128)
        }
    }

    /// Create from a canonical representative in `[0, P)`.
    #[inline]
    pub fn from_canonical_u64(x: u64) -> Self {
        debug_assert!(x < P);
        Self(x)
    }

    /// Additive identity.
    #[inline]
    pub fn zero() -> Self {
        Self(0)
    }

    /// Multiplicative identity.
    #[inline]
    pub fn one() -> Self {
        Self(u64::from(P > 1))
    }

    /// Check whether this element is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0 == 0
    }

    /// Multiplicative inverse, or `None` for zero.
    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        <Self as FieldCore>::inverse(self)
    }

    /// Construct from a `u64` reduced modulo the field modulus.
    #[inline]
    pub fn from_u64(val: u64) -> Self {
        Self(Self::reduce_u128(val as u128))
    }

    /// Construct from an `i64` reduced modulo the field modulus.
    #[inline]
    pub fn from_i64(val: i64) -> Self {
        if val >= 0 {
            Self::from_u64(val as u64)
        } else {
            -Self::from_u64(val.unsigned_abs())
        }
    }

    /// Construct from an `i8` reduced modulo the field modulus.
    #[inline]
    pub fn from_i8(val: i8) -> Self {
        Self::from_i64(val as i64)
    }

    /// Return the canonical representative in `[0, P)`.
    #[inline]
    pub fn to_canonical_u64(self) -> u64 {
        self.0
    }

    /// Solinas reduction: fold a u128 at bit `BITS` until the value fits,
    /// then conditionally subtract `P`.
    ///
    /// For multiplication products (< 2^{2·BITS}) exactly 2 folds suffice;
    /// for arbitrary u128 inputs the loop runs at most `ceil(128 / BITS)`
    /// iterations.
    #[inline(always)]
    fn reduce_u128(x: u128) -> u64 {
        let mut v = x;
        while v >> Self::BITS != 0 {
            v = (v & Self::MASK) + Self::mul_c((v >> Self::BITS) as u64);
        }
        let reduced = v.wrapping_sub(P as u128);
        let borrow = reduced >> 127;
        reduced.wrapping_add(borrow.wrapping_neg() & (P as u128)) as u64
    }

    /// Two-fold Solinas reduction for multiplication products.
    ///
    /// Input must be < 2^{2·BITS} (guaranteed for `a*b` where `a,b < P`).
    /// Exactly 2 folds + conditional subtract, no loop.
    ///
    /// When `FOLD_IN_U64` is true the entire reduction stays in u64,
    /// avoiding expensive u128 mask/shift on sub-word primes.
    #[inline(always)]
    fn reduce_product(x: u128) -> u64 {
        if Self::FOLD_IN_U64 {
            let lo = x as u64;
            let hi = (x >> 64) as u64;
            let high = (lo >> Self::BITS) | (hi << (64 - Self::BITS));
            let f1 = (lo & Self::MASK64) + Self::mul_c_narrow(high);
            let f2 = (f1 & Self::MASK64) + Self::mul_c_narrow(f1 >> Self::BITS);
            let reduced = f2.wrapping_sub(P);
            let borrow = reduced >> 63;
            reduced.wrapping_add(borrow.wrapping_neg() & P)
        } else {
            let f1 = (x & Self::MASK) + Self::mul_c((x >> Self::BITS) as u64);
            let f2 = (f1 & Self::MASK) + Self::mul_c((f1 >> Self::BITS) as u64);
            let reduced = f2.wrapping_sub(P as u128);
            let borrow = reduced >> 127;
            reduced.wrapping_add(borrow.wrapping_neg() & (P as u128)) as u64
        }
    }

    /// BMI2 fast path: avoid re-materializing `u128` product in the common
    /// sub-word configuration where reduction stays in `u64`.
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    #[inline(always)]
    fn reduce_product_wide(lo: u64, hi: u64) -> u64 {
        if Self::FOLD_IN_U64 {
            let high = (lo >> Self::BITS) | (hi << (64 - Self::BITS));
            let f1 = (lo & Self::MASK64) + Self::mul_c_narrow(high);
            let f2 = (f1 & Self::MASK64) + Self::mul_c_narrow(f1 >> Self::BITS);
            let reduced = f2.wrapping_sub(P);
            let borrow = reduced >> 63;
            reduced.wrapping_add(borrow.wrapping_neg() & P)
        } else {
            Self::reduce_product(lo as u128 | ((hi as u128) << 64))
        }
    }

    #[inline(always)]
    fn add_raw(a: u64, b: u64) -> u64 {
        if Self::BITS == 64 {
            let (s, overflow) = a.overflowing_add(b);
            let folded = s.wrapping_add((overflow as u64).wrapping_neg() & Self::C);
            let reduced = folded.wrapping_sub(P);
            let borrow = (folded < P) as u64;
            reduced.wrapping_add(borrow.wrapping_neg() & P)
        } else if Self::BITS <= 62 {
            let s = a + b;
            let reduced = s.wrapping_sub(P);
            let borrow = reduced >> 63;
            reduced.wrapping_add(borrow.wrapping_neg() & P)
        } else {
            let s = (a as u128) + (b as u128);
            let reduced = s.wrapping_sub(P as u128);
            let borrow = reduced >> 127;
            reduced.wrapping_add(borrow.wrapping_neg() & (P as u128)) as u64
        }
    }

    #[inline(always)]
    fn sub_raw(a: u64, b: u64) -> u64 {
        if Self::BITS == 64 {
            let (diff, underflow) = a.overflowing_sub(b);
            diff.wrapping_sub((underflow as u64).wrapping_neg() & Self::C)
        } else if Self::BITS <= 62 {
            let diff = a.wrapping_sub(b);
            let borrow = diff >> 63;
            diff.wrapping_add(borrow.wrapping_neg() & P)
        } else {
            let diff = (a as u128).wrapping_sub(b as u128);
            let borrow = diff >> 127;
            diff.wrapping_add(borrow.wrapping_neg() & (P as u128)) as u64
        }
    }

    #[inline(always)]
    fn mul_raw(a: u64, b: u64) -> u64 {
        #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
        {
            let (lo, hi) = mul64_wide(a, b);
            Self::reduce_product_wide(lo, hi)
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
        {
            Self::reduce_product((a as u128) * (b as u128))
        }
    }

    #[inline(always)]
    fn sqr_raw(a: u64) -> u64 {
        Self::mul_raw(a, a)
    }

    /// Squaring, equivalent to `self * self`.
    #[inline(always)]
    pub fn square(self) -> Self {
        Self(Self::sqr_raw(self.0))
    }

    fn pow(self, mut exp: u64) -> Self {
        let mut base = self;
        let mut acc = Self::one();
        while exp > 0 {
            if (exp & 1) == 1 {
                acc *= base;
            }
            base = base.square();
            exp >>= 1;
        }
        acc
    }

    /// Extract the canonical value.
    #[inline(always)]
    pub fn to_limbs(self) -> u64 {
        self.0
    }

    /// 64×64 → 128-bit widening multiply, **no reduction**.
    #[inline(always)]
    pub fn mul_wide(self, other: Self) -> u128 {
        let (lo, hi) = mul64_wide(self.0, other.0);
        lo as u128 | ((hi as u128) << 64)
    }

    /// 64×64 → 128-bit widening multiply with a raw `u64` operand,
    /// **no reduction**.
    #[inline(always)]
    pub fn mul_wide_u64(self, other: u64) -> u128 {
        let (lo, hi) = mul64_wide(self.0, other);
        lo as u128 | ((hi as u128) << 64)
    }

    /// Reduce a u128 value via Solinas folding to a canonical field element.
    #[inline(always)]
    pub fn solinas_reduce(x: u128) -> Self {
        Self(Self::reduce_u128(x))
    }

    /// Reduce the integer sum `w0 + w1` of two products of canonical residues
    /// (each `a·b` with `a, b < P`, so each `< 2^{2·BITS}`) to a canonical
    /// field element.
    ///
    /// Used by the specialized `FpExt2<Fp64>` EOR fold, which forms each output
    /// coordinate as a sum of two base-field products before a single
    /// reduction.
    ///
    /// - Sub-word primes (`BITS < 64`): each product is `< 2^{2·BITS} ≤ 2^126`,
    ///   so the sum is `< 2^127` and never overflows `u128`; reduce directly.
    /// - Full-word primes (`BITS == 64`, `P = 2^64 − C` with `C < 2^32`): the
    ///   sum can reach `< 2^129`. Split it into 64-bit limbs and fold the high
    ///   parts with `2^64 ≡ C` and `2^128 ≡ C² (mod P)`. The folded value is
    ///   `< 2^97`, which `solinas_reduce` finishes. The result is congruent to
    ///   `w0 + w1 (mod P)`, hence byte-identical to the canonical reduction.
    #[inline(always)]
    pub(crate) fn reduce_sum_of_two_products(w0: u128, w1: u128) -> Self {
        if Self::BITS < 64 {
            Self::solinas_reduce(w0.wrapping_add(w1))
        } else {
            let (s, carry) = w0.overflowing_add(w1);
            let cc = Self::C as u128;
            let folded =
                (s as u64 as u128) + ((s >> 64) as u64 as u128) * cc + (carry as u128) * cc * cc;
            Self::solinas_reduce(folded)
        }
    }
}

impl<const P: u64> Add for Fp64<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(Self::add_raw(self.0, rhs.0))
    }
}

impl<const P: u64> Sub for Fp64<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(Self::sub_raw(self.0, rhs.0))
    }
}

impl<const P: u64> Mul for Fp64<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(Self::mul_raw(self.0, rhs.0))
    }
}

impl<const P: u64> Neg for Fp64<P> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self(Self::sub_raw(0, self.0))
    }
}

impl<const P: u64> AddAssign for Fp64<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u64> SubAssign for Fp64<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u64> MulAssign for Fp64<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<'a, const P: u64> Add<&'a Self> for Fp64<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}

impl<'a, const P: u64> Sub<&'a Self> for Fp64<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}

impl<'a, const P: u64> Mul<&'a Self> for Fp64<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &'a Self) -> Self::Output {
        self * *rhs
    }
}

impl<const P: u64> FieldCore for Fp64<P> {
    #[inline(always)]
    fn inverse(&self) -> Option<Self> {
        let inv = self.inv_or_zero();
        if self.is_zero() {
            None
        } else {
            Some(inv)
        }
    }

    #[inline(always)]
    fn inv_or_zero(self) -> Self {
        let candidate = self.pow(P.wrapping_sub(2));
        let nz = ((self.0 | self.0.wrapping_neg()) >> 63) & 1;
        let mask = 0u64.wrapping_sub(nz);
        Self(candidate.0 & mask)
    }

    #[inline(always)]
    fn random<R: RngCore>(rng: &mut R) -> Self {
        let lo = rng.next_u64() as u128;
        let hi = rng.next_u64() as u128;
        Self(Self::reduce_u128(lo | (hi << 64)))
    }
}

impl<const P: u64> HalvingField for Fp64<P> {
    #[inline]
    fn half(self) -> Self {
        let x = self.0 as u128;
        Self(((x + (x & 1) * P as u128) >> 1) as u64)
    }
}

impl<const P: u64> FromPrimitiveInt for Fp64<P> {
    #[inline(always)]
    fn from_u64(val: u64) -> Self {
        Self::from_u64(val)
    }

    #[inline(always)]
    fn from_i64(val: i64) -> Self {
        Self::from_i64(val)
    }

    #[inline(always)]
    fn from_u128(val: u128) -> Self {
        Self(Self::reduce_u128(val))
    }

    #[inline(always)]
    fn from_i128(val: i128) -> Self {
        if val >= 0 {
            Self::from_u128(val as u128)
        } else {
            -Self::from_u128(val.unsigned_abs())
        }
    }
}

impl<const P: u64> CanonicalField for Fp64<P> {
    fn to_canonical_u128(self) -> u128 {
        self.0 as u128
    }

    fn modulus_bits() -> u32 {
        Self::BITS
    }

    fn from_canonical_u128_checked(val: u128) -> Option<Self> {
        if val < P as u128 {
            Some(Self(val as u64))
        } else {
            None
        }
    }

    fn from_canonical_u128_reduced(val: u128) -> Self {
        Self(Self::reduce_u128(val))
    }
}

impl<const P: u64> PseudoMersenneField for Fp64<P> {
    const MODULUS_BITS: u32 = Self::BITS;
    const MODULUS_OFFSET: u128 = Self::C as u128;
}

impl<const P: u64> serde::Serialize for Fp64<P> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let buf = (self.to_canonical_u128() as u64).to_le_bytes();
        <[u8; 8]>::serialize(&buf, serializer)
    }
}

impl<'de, const P: u64> serde::Deserialize<'de> for Fp64<P> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let buf = <[u8; 8]>::deserialize(deserializer)?;
        Self::from_canonical_u128_checked(u64::from_le_bytes(buf) as u128)
            .ok_or_else(|| serde::de::Error::custom("non-canonical Fp64 encoding"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type F40 = Fp64<{ (1u64 << 40) - 195 }>; // 2^40 - 195
    type F64 = Fp64<{ u64::MAX - 58 }>; // 2^64 - 59

    #[test]
    fn solinas_constants() {
        assert_eq!(F40::BITS, 40);
        assert_eq!(F40::C, 195);

        assert_eq!(F64::BITS, 64);
        assert_eq!(F64::C, 59);
    }

    #[test]
    fn basic_arithmetic_sub_word() {
        let a = F40::from_u64(1_000_000);
        let b = F40::from_u64(2_000_000);
        let p = (1u64 << 40) - 195;
        assert_eq!((a + b).to_canonical_u64(), 3_000_000);
        assert_eq!(
            (a * b).to_canonical_u64(),
            (1_000_000u128 * 2_000_000u128 % p as u128) as u64
        );
    }

    #[test]
    fn basic_arithmetic_full_word() {
        let a = F64::from_u64(1_000_000_000);
        let b = F64::from_u64(2_000_000_000);
        let p = u64::MAX - 58;
        assert_eq!(
            (a * b).to_canonical_u64(),
            (1_000_000_000u128 * 2_000_000_000u128 % p as u128) as u64
        );
    }

    #[test]
    fn mul_wide_matches_full_mul() {
        let mut rng = StdRng::seed_from_u64(0xdead_beef);
        for _ in 0..1000 {
            let a: F40 = FieldCore::random(&mut rng);
            let b: F40 = FieldCore::random(&mut rng);
            let expected = a * b;
            let reduced = F40::solinas_reduce(a.mul_wide(b));
            assert_eq!(reduced, expected);
        }
    }

    #[test]
    fn mul_wide_u64_matches() {
        let mut rng = StdRng::seed_from_u64(0xcafe_d00d);
        for _ in 0..1000 {
            let a: F40 = FieldCore::random(&mut rng);
            let b = rng.next_u64() % ((1u64 << 40) - 195);
            let expected = a * F40::from_canonical_u64(b);
            let reduced = F40::solinas_reduce(a.mul_wide_u64(b));
            assert_eq!(reduced, expected);
        }
    }

    #[test]
    fn pseudo_mersenne_trait() {
        assert_eq!(<F40 as PseudoMersenneField>::MODULUS_BITS, 40);
        assert_eq!(<F40 as PseudoMersenneField>::MODULUS_OFFSET, 195);
        assert_eq!(<F64 as PseudoMersenneField>::MODULUS_BITS, 64);
        assert_eq!(<F64 as PseudoMersenneField>::MODULUS_OFFSET, 59);
    }

    #[test]
    fn shift_optimization_detected() {
        type G = Fp64<{ (1u64 << 56) - 27 }>; // C = 27, not 2^a±1
        assert_eq!(G::C_SHIFT_KIND, 0);

        type H = Fp64<{ u64::MAX - 58 }>; // C = 59, not 2^a±1
        assert_eq!(H::C_SHIFT_KIND, 0);
    }

    #[test]
    fn reduce_u128_large() {
        assert_eq!(F64::from_canonical_u128_reduced(u128::MAX), {
            let p = u64::MAX as u128 - 58;
            F64::from_canonical_u64((u128::MAX % p) as u64)
        });
    }
}
