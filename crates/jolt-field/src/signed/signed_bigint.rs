//! Sign-magnitude big integer with `N * 64`-bit width.

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid, Validate,
    Write,
};
use core::cmp::Ordering;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use num_traits::Zero;

use crate::Limbs;

/// A signed big integer using `Limbs<N>` for magnitude and a sign bit.
///
/// Zero is not canonicalized: a zero magnitude can be paired with either sign.
/// Structural equality distinguishes `+0` and `-0`, but ordering treats them
/// as equal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignedBigInt<const N: usize> {
    pub magnitude: Limbs<N>,
    pub is_positive: bool,
}

#[cfg(feature = "allocative")]
impl<const N: usize> allocative::Allocative for SignedBigInt<N> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        visitor.visit_simple_sized::<Self>();
    }
}

impl<const N: usize> Default for SignedBigInt<N> {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}

impl<const N: usize> Zero for SignedBigInt<N> {
    #[inline]
    fn zero() -> Self {
        Self::zero()
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.magnitude.is_zero()
    }
}

pub type S64 = SignedBigInt<1>;
pub type S128 = SignedBigInt<2>;
pub type S192 = SignedBigInt<3>;
pub type S256 = SignedBigInt<4>;

impl<const N: usize> SignedBigInt<N> {
    #[inline]
    fn cmp_magnitude_mixed<const M: usize>(&self, rhs: &SignedBigInt<M>) -> Ordering {
        let max_limbs = if N > M { N } else { M };
        let mut i = max_limbs;
        while i > 0 {
            let idx = i - 1;
            let a = if idx < N { self.magnitude.0[idx] } else { 0u64 };
            let b = if idx < M { rhs.magnitude.0[idx] } else { 0u64 };
            if a > b {
                return Ordering::Greater;
            }
            if a < b {
                return Ordering::Less;
            }
            i -= 1;
        }
        Ordering::Equal
    }

    #[inline]
    pub fn new(limbs: [u64; N], is_positive: bool) -> Self {
        Self {
            magnitude: Limbs::new(limbs),
            is_positive,
        }
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
        Self {
            magnitude: Limbs::from_u64(0),
            is_positive: true,
        }
    }

    #[inline]
    pub fn one() -> Self {
        Self {
            magnitude: Limbs::from_u64(1),
            is_positive: true,
        }
    }

    #[inline]
    pub fn as_magnitude(&self) -> &Limbs<N> {
        &self.magnitude
    }

    #[inline]
    pub fn magnitude_limbs(&self) -> [u64; N] {
        self.magnitude.0
    }

    #[inline]
    pub fn magnitude_slice(&self) -> &[u64] {
        self.magnitude.as_ref()
    }

    #[inline]
    pub fn sign(&self) -> bool {
        self.is_positive
    }

    #[inline]
    pub fn negate(self) -> Self {
        Self::from_limbs(self.magnitude, !self.is_positive)
    }

    #[inline(always)]
    fn add_assign_in_place(&mut self, rhs: &Self) {
        if self.is_positive == rhs.is_positive {
            let _carry = self.magnitude.add_with_carry(&rhs.magnitude);
        } else {
            match self.magnitude.cmp(&rhs.magnitude) {
                Ordering::Greater | Ordering::Equal => {
                    let _borrow = self.magnitude.sub_with_borrow(&rhs.magnitude);
                }
                Ordering::Less => {
                    let old = core::mem::replace(&mut self.magnitude, rhs.magnitude);
                    let _borrow = self.magnitude.sub_with_borrow(&old);
                    self.is_positive = rhs.is_positive;
                }
            }
        }
    }

    #[inline(always)]
    fn sub_assign_in_place(&mut self, rhs: &Self) {
        if self.is_positive != rhs.is_positive {
            let _carry = self.magnitude.add_with_carry(&rhs.magnitude);
        } else {
            match self.magnitude.cmp(&rhs.magnitude) {
                Ordering::Greater | Ordering::Equal => {
                    let _borrow = self.magnitude.sub_with_borrow(&rhs.magnitude);
                }
                Ordering::Less => {
                    let old = core::mem::replace(&mut self.magnitude, rhs.magnitude);
                    let _borrow = self.magnitude.sub_with_borrow(&old);
                    self.is_positive = !self.is_positive;
                }
            }
        }
    }

    #[inline(always)]
    fn mul_assign_in_place(&mut self, rhs: &Self) {
        let low = self.magnitude.mul_low(&rhs.magnitude);
        self.magnitude = low;
        self.is_positive = self.is_positive == rhs.is_positive;
    }

    #[inline]
    pub fn zero_extend_from<const M: usize>(smaller: &SignedBigInt<M>) -> SignedBigInt<N> {
        debug_assert!(
            M <= N,
            "cannot zero-extend: source has more limbs than destination"
        );
        let widened_mag = Limbs::<N>::zero_extend_from::<M>(&smaller.magnitude);
        SignedBigInt::from_limbs(widened_mag, smaller.is_positive)
    }
}

impl<const N: usize> SignedBigInt<N> {
    /// Adds two values and truncates the result to `M` limbs.
    #[inline]
    pub fn add_trunc<const M: usize>(&self, rhs: &SignedBigInt<N>) -> SignedBigInt<M> {
        if self.is_positive == rhs.is_positive {
            let mag = self.magnitude.add_trunc::<N, M>(&rhs.magnitude);
            return SignedBigInt::<M> {
                magnitude: mag,
                is_positive: self.is_positive,
            };
        }
        match self.magnitude.cmp(&rhs.magnitude) {
            Ordering::Greater | Ordering::Equal => {
                let mag = self.magnitude.sub_trunc::<N, M>(&rhs.magnitude);
                SignedBigInt::<M> {
                    magnitude: mag,
                    is_positive: self.is_positive,
                }
            }
            Ordering::Less => {
                let mag = rhs.magnitude.sub_trunc::<N, M>(&self.magnitude);
                SignedBigInt::<M> {
                    magnitude: mag,
                    is_positive: rhs.is_positive,
                }
            }
        }
    }

    /// Subtracts and truncates the result to `M` limbs.
    #[inline]
    pub fn sub_trunc<const M: usize>(&self, rhs: &SignedBigInt<N>) -> SignedBigInt<M> {
        if self.is_positive != rhs.is_positive {
            let mag = self.magnitude.add_trunc::<N, M>(&rhs.magnitude);
            return SignedBigInt::<M> {
                magnitude: mag,
                is_positive: self.is_positive,
            };
        }
        match self.magnitude.cmp(&rhs.magnitude) {
            Ordering::Greater | Ordering::Equal => {
                let mag = self.magnitude.sub_trunc::<N, M>(&rhs.magnitude);
                SignedBigInt::<M> {
                    magnitude: mag,
                    is_positive: self.is_positive,
                }
            }
            Ordering::Less => {
                let mag = rhs.magnitude.sub_trunc::<N, M>(&self.magnitude);
                SignedBigInt::<M> {
                    magnitude: mag,
                    is_positive: !self.is_positive,
                }
            }
        }
    }

    /// Adds values of different widths (`N` and `M` limbs) and truncates to `P` limbs.
    #[inline]
    pub fn add_trunc_mixed<const M: usize, const P: usize>(
        &self,
        rhs: &SignedBigInt<M>,
    ) -> SignedBigInt<P> {
        if self.is_positive == rhs.is_positive {
            let mag = self.magnitude.add_trunc::<M, P>(&rhs.magnitude);
            return SignedBigInt::<P> {
                magnitude: mag,
                is_positive: self.is_positive,
            };
        }
        match self.cmp_magnitude_mixed(rhs) {
            Ordering::Greater | Ordering::Equal => {
                let mag = self.magnitude.sub_trunc::<M, P>(&rhs.magnitude);
                SignedBigInt::<P> {
                    magnitude: mag,
                    is_positive: self.is_positive,
                }
            }
            Ordering::Less => {
                let mag = rhs.magnitude.sub_trunc::<N, P>(&self.magnitude);
                SignedBigInt::<P> {
                    magnitude: mag,
                    is_positive: rhs.is_positive,
                }
            }
        }
    }

    /// Subtracts values of different widths and truncates to `P` limbs.
    #[inline]
    pub fn sub_trunc_mixed<const M: usize, const P: usize>(
        &self,
        rhs: &SignedBigInt<M>,
    ) -> SignedBigInt<P> {
        if self.is_positive != rhs.is_positive {
            let mag = self.magnitude.add_trunc::<M, P>(&rhs.magnitude);
            return SignedBigInt::<P> {
                magnitude: mag,
                is_positive: self.is_positive,
            };
        }
        match self.cmp_magnitude_mixed(rhs) {
            Ordering::Greater | Ordering::Equal => {
                let mag = self.magnitude.sub_trunc::<M, P>(&rhs.magnitude);
                SignedBigInt::<P> {
                    magnitude: mag,
                    is_positive: self.is_positive,
                }
            }
            Ordering::Less => {
                let mag = rhs.magnitude.sub_trunc::<N, P>(&self.magnitude);
                SignedBigInt::<P> {
                    magnitude: mag,
                    is_positive: !self.is_positive,
                }
            }
        }
    }

    /// Multiplies and truncates the result to `P` limbs.
    #[inline]
    pub fn mul_trunc<const M: usize, const P: usize>(
        &self,
        rhs: &SignedBigInt<M>,
    ) -> SignedBigInt<P> {
        let mag = self.magnitude.mul_trunc::<M, P>(&rhs.magnitude);
        let sign = self.is_positive == rhs.is_positive;
        SignedBigInt::<P> {
            magnitude: mag,
            is_positive: sign,
        }
    }

    /// Fused multiply-add: `acc += self * rhs`, truncated to `P` limbs.
    #[inline]
    pub fn fmadd_trunc<const M: usize, const P: usize>(
        &self,
        rhs: &SignedBigInt<M>,
        acc: &mut SignedBigInt<P>,
    ) {
        let prod_mag = self.magnitude.mul_trunc::<M, P>(&rhs.magnitude);
        let prod_sign = self.is_positive == rhs.is_positive;
        if acc.is_positive == prod_sign {
            let _ = acc.magnitude.add_with_carry(&prod_mag);
        } else {
            match acc.magnitude.cmp(&prod_mag) {
                Ordering::Greater | Ordering::Equal => {
                    let _ = acc.magnitude.sub_with_borrow(&prod_mag);
                }
                Ordering::Less => {
                    let old = core::mem::replace(&mut acc.magnitude, prod_mag);
                    let _ = acc.magnitude.sub_with_borrow(&old);
                    acc.is_positive = prod_sign;
                }
            }
        }
    }
}

impl<const N: usize> SignedBigInt<N> {
    #[inline]
    pub fn from_u64(value: u64) -> Self {
        Self::from_limbs(Limbs::from_u64(value), true)
    }

    #[inline]
    pub fn from_u64_with_sign(value: u64, is_positive: bool) -> Self {
        Self::from_limbs(Limbs::from_u64(value), is_positive)
    }

    #[inline]
    pub fn from_i64(value: i64) -> Self {
        if value >= 0 {
            Self::from_limbs(Limbs::from_u64(value as u64), true)
        } else {
            Self::from_limbs(Limbs::from_u64(value.wrapping_neg() as u64), false)
        }
    }

    #[inline]
    pub fn from_u128(value: u128) -> Self {
        debug_assert!(N >= 2, "from_u128 requires at least 2 limbs");
        let mut limbs = [0u64; N];
        limbs[0] = value as u64;
        limbs[1] = (value >> 64) as u64;
        Self::from_limbs(Limbs::new(limbs), true)
    }

    #[inline]
    pub fn from_i128(value: i128) -> Self {
        debug_assert!(N >= 2, "from_i128 requires at least 2 limbs");
        if value >= 0 {
            let mut limbs = [0u64; N];
            let v = value as u128;
            limbs[0] = v as u64;
            limbs[1] = (v >> 64) as u64;
            Self::from_limbs(Limbs::new(limbs), true)
        } else {
            let mag = value.unsigned_abs();
            let mut limbs = [0u64; N];
            limbs[0] = mag as u64;
            limbs[1] = (mag >> 64) as u64;
            Self::from_limbs(Limbs::new(limbs), false)
        }
    }
}

impl<const N: usize> From<u64> for SignedBigInt<N> {
    #[inline]
    fn from(value: u64) -> Self {
        Self::from_u64(value)
    }
}

impl<const N: usize> From<i64> for SignedBigInt<N> {
    #[inline]
    fn from(value: i64) -> Self {
        Self::from_i64(value)
    }
}

impl<const N: usize> From<(u64, bool)> for SignedBigInt<N> {
    #[inline]
    fn from(value_and_sign: (u64, bool)) -> Self {
        Self::from_u64_with_sign(value_and_sign.0, value_and_sign.1)
    }
}

impl<const N: usize> From<u128> for SignedBigInt<N> {
    #[inline]
    fn from(value: u128) -> Self {
        debug_assert!(N >= 2, "From<u128> requires at least 2 limbs");
        Self::from_u128(value)
    }
}

impl<const N: usize> From<i128> for SignedBigInt<N> {
    #[inline]
    fn from(value: i128) -> Self {
        debug_assert!(N >= 2, "From<i128> requires at least 2 limbs");
        Self::from_i128(value)
    }
}

impl S64 {
    #[inline]
    pub fn to_i128(&self) -> i128 {
        let magnitude = self.magnitude.0[0];
        if self.is_positive {
            magnitude as i128
        } else {
            -(magnitude as i128)
        }
    }

    #[inline]
    pub fn magnitude_as_u64(&self) -> u64 {
        self.magnitude.0[0]
    }

    #[inline(always)]
    pub fn from_diff_u64s(a: u64, b: u64) -> Self {
        if a < b {
            Self::new([b - a], false)
        } else {
            Self::new([a - b], true)
        }
    }
}

impl S128 {
    #[inline]
    pub fn to_i128(&self) -> Option<i128> {
        let hi = self.magnitude.0[1];
        let lo = self.magnitude.0[0];
        let hi_top_bit = hi >> 63;
        if self.is_positive {
            if hi_top_bit != 0 {
                return None;
            }
            let mag = ((hi as u128) << 64) | (lo as u128);
            Some(mag as i128)
        } else if hi_top_bit == 0 {
            let mag = ((hi as u128) << 64) | (lo as u128);
            Some(-(mag as i128))
        } else if hi == (1u64 << 63) && lo == 0 {
            Some(i128::MIN)
        } else {
            None
        }
    }

    #[inline]
    pub fn magnitude_as_u128(&self) -> u128 {
        (self.magnitude.0[1] as u128) << 64 | (self.magnitude.0[0] as u128)
    }

    #[inline]
    pub fn from_u128_and_sign(value: u128, is_positive: bool) -> Self {
        Self::new([value as u64, (value >> 64) as u64], is_positive)
    }

    #[inline]
    pub fn from_u64_mul_i64(u: u64, s: i64) -> Self {
        let mag = (u as u128) * (s.unsigned_abs() as u128);
        Self::from_u128_and_sign(mag, s >= 0)
    }

    #[inline]
    pub fn from_i64_mul_u64(s: i64, u: u64) -> Self {
        Self::from_u64_mul_i64(u, s)
    }

    #[inline]
    pub fn from_u64_mul_u64(a: u64, b: u64) -> Self {
        let mag = (a as u128) * (b as u128);
        Self::from_u128_and_sign(mag, true)
    }

    #[inline]
    pub fn from_i64_mul_i64(a: i64, b: i64) -> Self {
        let mag = (a.unsigned_abs() as u128) * (b.unsigned_abs() as u128);
        let is_positive = (a >= 0) == (b >= 0);
        Self::from_u128_and_sign(mag, is_positive)
    }
}

super::impl_signed_assign_ops!(SignedBigInt {
    Add, AddAssign, add, add_assign => add_assign_in_place;
    Sub, SubAssign, sub, sub_assign => sub_assign_in_place;
    Mul, MulAssign, mul, mul_assign => mul_assign_in_place;
});

impl<const N: usize> Neg for SignedBigInt<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        self.negate()
    }
}

impl<const N: usize> PartialOrd for SignedBigInt<N> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const N: usize> Ord for SignedBigInt<N> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        if self.magnitude.is_zero() && other.magnitude.is_zero() {
            return Ordering::Equal;
        }
        match (self.is_positive, other.is_positive) {
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            _ => {
                let ord = self.magnitude.cmp(&other.magnitude);
                if self.is_positive {
                    ord
                } else {
                    ord.reverse()
                }
            }
        }
    }
}

impl<const N: usize> CanonicalSerialize for SignedBigInt<N> {
    #[inline]
    fn serialize_with_mode<W: Write>(
        &self,
        mut w: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        (self.is_positive as u8).serialize_with_mode(&mut w, compress)?;
        self.magnitude.serialize_with_mode(w, compress)
    }

    #[inline]
    fn serialized_size(&self, compress: Compress) -> usize {
        (self.is_positive as u8).serialized_size(compress)
            + self.magnitude.serialized_size(compress)
    }
}

impl<const N: usize> CanonicalDeserialize for SignedBigInt<N> {
    #[inline]
    fn deserialize_with_mode<R: Read>(
        mut r: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let sign_u8 = u8::deserialize_with_mode(&mut r, compress, validate)?;
        let mag = Limbs::<N>::deserialize_with_mode(r, compress, validate)?;
        Ok(SignedBigInt {
            magnitude: mag,
            is_positive: sign_u8 != 0,
        })
    }
}

impl<const N: usize> Valid for SignedBigInt<N> {
    #[inline]
    fn check(&self) -> Result<(), SerializationError> {
        self.magnitude.check()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s64_basic_arithmetic() {
        let a = S64::from_i64(10);
        let b = S64::from_i64(-3);
        let c = a + b;
        assert_eq!(c.to_i128(), 7);

        let d = a - b;
        assert_eq!(d.to_i128(), 13);
    }

    #[test]
    fn s64_from_diff() {
        let d = S64::from_diff_u64s(5, 10);
        assert!(!d.is_positive);
        assert_eq!(d.magnitude_as_u64(), 5);

        let d2 = S64::from_diff_u64s(10, 5);
        assert!(d2.is_positive);
        assert_eq!(d2.magnitude_as_u64(), 5);
    }

    #[test]
    fn s128_mul_i64() {
        let r = S128::from_i64_mul_i64(-3, 7);
        assert_eq!(r.to_i128(), Some(-21));

        let r2 = S128::from_u64_mul_u64(100, 200);
        assert_eq!(r2.to_i128(), Some(20000));
    }

    #[test]
    fn s128_magnitude() {
        let v = S128::from_i128(-12_345_678_901_234_567_890_i128);
        assert!(!v.is_positive);
        assert_eq!(v.magnitude_as_u128(), 12_345_678_901_234_567_890_u128);
    }

    #[test]
    fn mul_trunc_s64_to_s128() {
        let a = S64::from_i64(-5);
        let b = S64::from_i64(7);
        let c: S128 = a.mul_trunc::<1, 2>(&b);
        assert_eq!(c.to_i128(), Some(-35));
    }

    #[test]
    fn ordering() {
        let a = S64::from_i64(5);
        let b = S64::from_i64(-5);
        let z1 = S64::from_u64(0);
        let z2 = S64::new([0], false); // negative zero
        assert!(a > b);
        assert_eq!(z1.cmp(&z2), Ordering::Equal);
    }

    #[test]
    fn zero_extend() {
        let s = S64::from_i64(-42);
        let wide: S128 = SignedBigInt::zero_extend_from(&s);
        assert!(!wide.is_positive);
        assert_eq!(wide.magnitude.0[0], 42);
        assert_eq!(wide.magnitude.0[1], 0);
    }

    #[test]
    fn add_trunc_mixed() {
        let a = S64::from_i64(100);
        let b = S128::from_i128(200);
        let c: S128 = a.add_trunc_mixed::<2, 2>(&b);
        assert_eq!(c.to_i128(), Some(300));
    }

    #[test]
    fn fmadd_trunc() {
        let a = S64::from_i64(3);
        let b = S64::from_i64(4);
        let mut acc = S128::from_i128(10);
        a.fmadd_trunc::<1, 2>(&b, &mut acc);
        assert_eq!(acc.to_i128(), Some(22)); // 10 + 3*4
    }

    #[test]
    fn serialization_roundtrip() {
        use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

        let val = S128::from_i128(-999_999);
        let mut bytes = Vec::new();
        val.serialize_compressed(&mut bytes).unwrap();
        let restored = S128::deserialize_compressed(&bytes[..]).unwrap();
        assert_eq!(val, restored);
    }

    #[test]
    fn s64_serialization_roundtrip() {
        use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

        for &v in &[0i64, 1, -1, i64::MAX, i64::MIN] {
            let val = S64::from_i64(v);
            let mut bytes = Vec::new();
            val.serialize_compressed(&mut bytes).unwrap();
            let restored = S64::deserialize_compressed(&bytes[..]).unwrap();
            assert_eq!(val, restored);
        }
    }

    #[test]
    fn s128_to_i128_out_of_range() {
        // Magnitude exceeding i128::MAX for positive
        let big_positive = S128::new([0, 1u64 << 63], true);
        assert_eq!(big_positive.to_i128(), None);

        // Magnitude exceeding i128::MIN for negative (not exactly MIN)
        let big_negative = S128::new([1, 1u64 << 63], false);
        assert_eq!(big_negative.to_i128(), None);

        // Exactly i128::MIN is representable
        let min_val = S128::new([0, 1u64 << 63], false);
        assert_eq!(min_val.to_i128(), Some(i128::MIN));
    }

    #[test]
    fn fmadd_trunc_sign_flip() {
        // Positive accumulator, subtract larger product → sign flips
        let a = S64::from_i64(-10);
        let b = S64::from_i64(5);
        let mut acc = S128::from_i128(3);
        a.fmadd_trunc::<1, 2>(&b, &mut acc);
        // 3 + (-10 * 5) = 3 - 50 = -47
        assert_eq!(acc.to_i128(), Some(-47));
        assert!(!acc.is_positive);
    }

    #[test]
    fn s64_from_diff_u64s_zero_zero() {
        let d = S64::from_diff_u64s(0, 0);
        assert!(d.is_positive);
        assert!(d.is_zero());
        assert_eq!(d.magnitude_as_u64(), 0);
    }
}
