//! Prime field for primes of the form `p = 2^k − c` with `c` small, backed
//! by `u32` storage.
//!
//! Uses Solinas-style two-fold reduction: the offset `c` and fold point `k`
//! are computed at compile time from the const-generic modulus `P`.

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::{FieldCore, FromPrimitiveInt};
use rand_core::RngCore;

use crate::{CanonicalField, HalvingField, PseudoMersenneField};

/// Prime field element for primes `p = 2^k − c` stored as `u32`.
///
/// The fold point `k` and offset `c = 2^k − p` are computed at compile time
/// from the const-generic `P`.  Instantiating with a modulus that does not
/// satisfy the prime Solinas conditions is a compile-time error.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Fp32<const P: u32>(pub(crate) u32);

impl<const P: u32> Fp32<P> {
    /// Fold point: smallest `k` such that `P ≤ 2^k`.
    const BITS: u32 = 32 - P.leading_zeros();

    /// Offset `c = 2^k − P`.
    pub const C: u32 = {
        let c = if Self::BITS == 32 {
            0u32.wrapping_sub(P)
        } else {
            (1u32 << Self::BITS) - P
        };
        assert!(P != 0, "modulus must be nonzero");
        assert!(P & 1 == 1, "modulus must be odd");
        assert!(Self::is_prime_modulus(P), "modulus must be prime");
        assert!(
            (c as u64) * (c as u64 + 1) < P as u64,
            "C(C+1) < P required for fused canonicalize"
        );
        c
    };

    const fn is_prime_modulus(n: u32) -> bool {
        if n < 2 {
            return false;
        }
        if n.is_multiple_of(2) {
            return n == 2;
        }
        let mut d = 3u32;
        while (d as u64) * (d as u64) <= n as u64 {
            if n.is_multiple_of(d) {
                return false;
            }
            d += 2;
        }
        true
    }

    /// Mask for extracting the low `BITS` bits from a u64.
    const MASK: u64 = if Self::BITS == 32 {
        u32::MAX as u64
    } else {
        (1u64 << Self::BITS) - 1
    };

    pub(crate) const SHIFT64_MOD_P: u32 = {
        let c = Self::C as u128;
        let bits = Self::BITS;
        let mask = if bits == 32 {
            u32::MAX as u128
        } else {
            (1u128 << bits) - 1
        };
        let mut v = 1u128 << 64;
        while v >> bits != 0 {
            v = (v & mask) + c * (v >> bits);
        }
        let reduced = (v as u64).wrapping_sub(P as u64);
        let borrow = reduced >> 63;
        reduced.wrapping_add(borrow.wrapping_neg() & (P as u64)) as u32
    };

    #[inline(always)]
    fn canonicalize_folded(v: u64) -> u32 {
        if Self::BITS <= 31 {
            let x = v as u32;
            x.min(x.wrapping_sub(P))
        } else {
            let reduced = v.wrapping_sub(P as u64);
            let borrow = reduced >> 63;
            reduced.wrapping_add(borrow.wrapping_neg() & (P as u64)) as u32
        }
    }

    /// Create from a canonical representative in `[0, P)`.
    #[inline]
    pub fn from_canonical_u32(x: u32) -> Self {
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
        Self(u32::from(P > 1))
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
        Self(Self::reduce_u64(val))
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
    pub fn to_canonical_u32(self) -> u32 {
        self.0
    }

    /// Solinas reduction: fold a u64 at bit `BITS` until the value fits,
    /// then conditionally subtract `P`.
    ///
    /// For multiplication products (< 2^{2·BITS}) exactly 2 folds suffice;
    /// for arbitrary u64 inputs (e.g. `from_u64`) the loop runs at most
    /// `ceil(64 / BITS)` iterations.
    #[inline(always)]
    fn reduce_u64(x: u64) -> u32 {
        let c = Self::C as u64;
        let mut v = x;
        while v >> Self::BITS != 0 {
            v = (v & Self::MASK) + c * (v >> Self::BITS);
        }
        Self::canonicalize_folded(v)
    }

    /// Reduce a `u128` to canonical form (for `from_canonical_u128_reduced`).
    #[inline(always)]
    fn reduce_u128(x: u128) -> u32 {
        let c = Self::C as u128;
        let bits = Self::BITS;
        let mask = if bits == 32 {
            u32::MAX as u128
        } else {
            (1u128 << bits) - 1
        };
        let mut v = x;
        while v >> bits != 0 {
            v = (v & mask) + c * (v >> bits);
        }
        Self::canonicalize_folded(v as u64)
    }

    /// Two-fold Solinas reduction for multiplication products.
    ///
    /// Input must be < 2^{2·BITS} (guaranteed for `a*b` where `a,b < P`).
    /// Exactly 2 folds + conditional subtract, no loop.
    #[inline(always)]
    fn reduce_product(x: u64) -> u32 {
        let c = Self::C as u64;
        let f1 = (x & Self::MASK) + c * (x >> Self::BITS);
        let f2 = (f1 & Self::MASK) + c * (f1 >> Self::BITS);
        Self::canonicalize_folded(f2)
    }

    #[inline(always)]
    fn add_raw(a: u32, b: u32) -> u32 {
        if Self::BITS <= 31 {
            let sum = a.wrapping_add(b);
            sum.min(sum.wrapping_sub(P))
        } else {
            let s = (a as u64) + (b as u64);
            let reduced = s.wrapping_sub(P as u64);
            let borrow = reduced >> 63;
            reduced.wrapping_add(borrow.wrapping_neg() & (P as u64)) as u32
        }
    }

    #[inline(always)]
    fn sub_raw(a: u32, b: u32) -> u32 {
        if Self::BITS <= 31 {
            let diff = a.wrapping_sub(b);
            diff.min(diff.wrapping_add(P))
        } else {
            let diff = (a as u64).wrapping_sub(b as u64);
            let borrow = diff >> 63;
            diff.wrapping_add(borrow.wrapping_neg() & (P as u64)) as u32
        }
    }

    #[inline(always)]
    fn mul_raw(a: u32, b: u32) -> u32 {
        Self::reduce_product((a as u64) * (b as u64))
    }

    #[inline(always)]
    fn sqr_raw(a: u32) -> u32 {
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
    pub fn to_limbs(self) -> u32 {
        self.0
    }

    /// 32×32 → 64-bit widening multiply, **no reduction**.
    #[inline(always)]
    pub fn mul_wide(self, other: Self) -> u64 {
        (self.0 as u64) * (other.0 as u64)
    }

    /// 32×32 → 64-bit widening multiply with a raw `u32` operand,
    /// **no reduction**.
    #[inline(always)]
    pub fn mul_wide_u32(self, other: u32) -> u64 {
        (self.0 as u64) * (other as u64)
    }

    /// Reduce a u64 value via Solinas folding to a canonical field element.
    #[inline(always)]
    pub fn solinas_reduce(x: u64) -> Self {
        Self(Self::reduce_u64(x))
    }
}

impl<const P: u32> Add for Fp32<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(Self::add_raw(self.0, rhs.0))
    }
}

impl<const P: u32> Sub for Fp32<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(Self::sub_raw(self.0, rhs.0))
    }
}

impl<const P: u32> Mul for Fp32<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(Self::mul_raw(self.0, rhs.0))
    }
}

impl<const P: u32> Neg for Fp32<P> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self(Self::sub_raw(0, self.0))
    }
}

impl<const P: u32> AddAssign for Fp32<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u32> SubAssign for Fp32<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u32> MulAssign for Fp32<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<'a, const P: u32> Add<&'a Self> for Fp32<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}

impl<'a, const P: u32> Sub<&'a Self> for Fp32<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}

impl<'a, const P: u32> Mul<&'a Self> for Fp32<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &'a Self) -> Self::Output {
        self * *rhs
    }
}

impl<const P: u32> FieldCore for Fp32<P> {
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
        let candidate = self.pow((P as u64).wrapping_sub(2));
        let nz = ((self.0 | self.0.wrapping_neg()) >> 31) & 1;
        let mask = 0u32.wrapping_sub(nz);
        Self(candidate.0 & mask)
    }

    #[inline(always)]
    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self(Self::reduce_u64(rng.next_u64()))
    }
}

impl<const P: u32> HalvingField for Fp32<P> {
    #[inline]
    fn half(self) -> Self {
        if Self::BITS == 31 && Self::C == 1 {
            Self((self.0 >> 1) | ((self.0 & 1) << 30))
        } else {
            let half_p_plus_one = (P >> 1) + 1;
            let correction = 0u32.wrapping_sub(self.0 & 1) & half_p_plus_one;
            Self((self.0 >> 1) + correction)
        }
    }
}

impl<const P: u32> FromPrimitiveInt for Fp32<P> {
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

impl<const P: u32> CanonicalField for Fp32<P> {
    fn to_canonical_u128(self) -> u128 {
        self.0 as u128
    }

    fn modulus_bits() -> u32 {
        Self::BITS
    }

    fn from_canonical_u128_checked(val: u128) -> Option<Self> {
        if val < P as u128 {
            Some(Self(val as u32))
        } else {
            None
        }
    }

    fn from_canonical_u128_reduced(val: u128) -> Self {
        Self(Self::reduce_u128(val))
    }
}

impl<const P: u32> PseudoMersenneField for Fp32<P> {
    const MODULUS_BITS: u32 = Self::BITS;
    const MODULUS_OFFSET: u128 = Self::C as u128;
}

impl<const P: u32> serde::Serialize for Fp32<P> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let buf = (self.to_canonical_u128() as u32).to_le_bytes();
        <[u8; 4]>::serialize(&buf, serializer)
    }
}

impl<'de, const P: u32> serde::Deserialize<'de> for Fp32<P> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let buf = <[u8; 4]>::deserialize(deserializer)?;
        Self::from_canonical_u128_checked(u32::from_le_bytes(buf) as u128)
            .ok_or_else(|| serde::de::Error::custom("non-canonical Fp32 encoding"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type F = Fp32<251>; // 2^8 - 5

    #[test]
    fn solinas_constants() {
        assert_eq!(F::BITS, 8);
        assert_eq!(F::C, 5);
        assert_eq!(F::MASK, 255);

        type G = Fp32<{ (1u32 << 24) - 3 }>; // 2^24 - 3
        assert_eq!(G::BITS, 24);
        assert_eq!(G::C, 3);
    }

    #[test]
    fn basic_arithmetic() {
        let a = F::from_u64(100);
        let b = F::from_u64(200);
        assert_eq!((a + b).to_canonical_u32(), (100 + 200) % 251);
        assert_eq!((a * b).to_canonical_u32(), (100 * 200) % 251);
        assert_eq!((b - a).to_canonical_u32(), 100);
        assert_eq!((-a).to_canonical_u32(), 251 - 100);
    }

    #[test]
    fn prime31_fast_path_edges() {
        const P31: u32 = (1u32 << 31) - 19;
        type G = Fp32<P31>;

        assert_eq!(G::BITS, 31);
        assert_eq!(G::C, 19);

        let zero = G::zero();
        let one = G::one();
        let p_minus_one = G::from_canonical_u32(P31 - 1);
        let p_minus_two = G::from_canonical_u32(P31 - 2);

        assert_eq!((p_minus_one + one).to_canonical_u32(), 0);
        assert_eq!((p_minus_one + p_minus_one).to_canonical_u32(), P31 - 2);
        assert_eq!((zero - one).to_canonical_u32(), P31 - 1);
        assert_eq!((one - p_minus_one).to_canonical_u32(), 2);
        assert_eq!((-zero).to_canonical_u32(), 0);
        assert_eq!((-one).to_canonical_u32(), P31 - 1);
        assert_eq!((p_minus_one * p_minus_one).to_canonical_u32(), 1);
        assert_eq!((p_minus_two * p_minus_two).to_canonical_u32(), 4);

        for x in [zero, one, p_minus_two, p_minus_one] {
            assert_eq!(x.half() + x.half(), x);
        }

        type M = Fp32<{ (1u32 << 31) - 1 }>;
        for x in [
            M::zero(),
            M::one(),
            M::from_canonical_u32((1u32 << 31) - 3),
            M::from_canonical_u32((1u32 << 31) - 2),
        ] {
            assert_eq!(x.half() + x.half(), x);
        }
    }

    #[test]
    fn prime31_random_arithmetic_matches_u64_modulus() {
        const P31: u32 = (1u32 << 31) - 19;
        type G = Fp32<P31>;

        let mut rng = StdRng::seed_from_u64(0x31_31_31_31);
        for _ in 0..1000 {
            let a_raw = rng.next_u32() & ((1u32 << 31) - 1);
            let b_raw = rng.next_u32() & ((1u32 << 31) - 1);
            let a = G::from_u64(a_raw as u64);
            let b = G::from_u64(b_raw as u64);
            let p = P31 as u64;
            let a_can = (a_raw as u64) % p;
            let b_can = (b_raw as u64) % p;

            assert_eq!((a + b).to_canonical_u32() as u64, (a_can + b_can) % p);
            assert_eq!((a - b).to_canonical_u32() as u64, (a_can + p - b_can) % p);
            assert_eq!((a * b).to_canonical_u32() as u64, (a_can * b_can) % p);
        }

        assert_eq!(
            G::from_u64(u64::MAX).to_canonical_u32() as u64,
            u64::MAX % (P31 as u64)
        );
    }

    #[test]
    fn fp31_u128_reduction_matches_modulus() {
        fn check<const P: u32>(inputs: &[u128]) {
            for &input in inputs {
                assert_eq!(
                    Fp32::<P>::from_canonical_u128_reduced(input).to_canonical_u32() as u128,
                    input % (P as u128),
                    "u128 reduction mismatch for P={P}, input={input}"
                );
            }
        }

        const PRIME31: u32 = (1u32 << 31) - 19;
        const MERSENNE31: u32 = (1u32 << 31) - 1;
        const GENERIC30: u32 = (1u32 << 30) - 16_397;
        const GENERIC31: u32 = (1u32 << 31) - 32_787;
        let inputs = [
            0,
            1,
            PRIME31 as u128 - 1,
            PRIME31 as u128,
            PRIME31 as u128 + 1,
            (PRIME31 as u128) * (PRIME31 as u128) - 1,
            1u128 << 63,
            (1u128 << 96) + 123_456_789,
            u128::MAX,
        ];

        check::<PRIME31>(&inputs);
        check::<MERSENNE31>(&inputs);
        check::<GENERIC30>(&inputs);
        check::<GENERIC31>(&inputs);
    }

    #[test]
    fn mul_wide_matches_full_mul() {
        let mut rng = StdRng::seed_from_u64(0x1234_5678);
        for _ in 0..1000 {
            let a: F = FieldCore::random(&mut rng);
            let b: F = FieldCore::random(&mut rng);
            let expected = a * b;
            let reduced = F::solinas_reduce(a.mul_wide(b));
            assert_eq!(reduced, expected);
        }
    }

    #[test]
    fn mul_wide_u32_matches() {
        let mut rng = StdRng::seed_from_u64(0xabcd_ef01);
        for _ in 0..1000 {
            let a: F = FieldCore::random(&mut rng);
            let b = rng.next_u32() % 251;
            let expected = a * F::from_canonical_u32(b);
            let reduced = F::solinas_reduce(a.mul_wide_u32(b));
            assert_eq!(reduced, expected);
        }
    }

    #[test]
    fn reduce_large_values() {
        assert_eq!(
            F::from_u64(u64::MAX).to_canonical_u32(),
            (u64::MAX % 251) as u32
        );
        assert_eq!(F::from_u64(0).to_canonical_u32(), 0);
        assert_eq!(F::from_u64(251).to_canonical_u32(), 0);
        assert_eq!(F::from_u64(252).to_canonical_u32(), 1);
    }

    #[test]
    fn pseudo_mersenne_trait() {
        assert_eq!(<F as PseudoMersenneField>::MODULUS_BITS, 8);
        assert_eq!(<F as PseudoMersenneField>::MODULUS_OFFSET, 5);
    }

    #[test]
    fn cross_prime_32bit() {
        type G = Fp32<{ u32::MAX - 98 }>; // 2^32 - 99
        assert_eq!(G::BITS, 32);
        assert_eq!(G::C, 99);

        let a = G::from_u64(1_000_000);
        let b = G::from_u64(2_000_000);
        let product = (1_000_000u64 * 2_000_000u64) % ((1u64 << 32) - 99);
        assert_eq!((a * b).to_canonical_u32(), product as u32);
    }
}
