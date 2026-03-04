//! Sign-magnitude big integer with `N * 64 + 32`-bit width.

#[cfg(feature = "allocative")]
use allocative::Allocative;

use ark_ff::BigInt;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid, Validate,
    Write,
};
use core::cmp::Ordering;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::{SignedBigInt, S128, S64};

/// Compact signed big-integer with width `N * 64 + 32` bits.
///
/// Uses `[u64; N]` for the low limbs and a `u32` for the high tail.
/// This representation saves 4 bytes per value compared to using `N + 1`
/// full 64-bit limbs, which matters when millions of these are stored
/// in witness polynomials.
///
/// Zero is not normalized: a zero magnitude can have either sign.
/// Structural equality distinguishes `+0` and `-0`, but ordering treats
/// them as equal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct SignedBigIntHi32<const N: usize> {
    magnitude_lo: [u64; N],
    magnitude_hi: u32,
    is_positive: bool,
}

pub type S96 = SignedBigIntHi32<1>;
pub type S160 = SignedBigIntHi32<2>;
pub type S224 = SignedBigIntHi32<3>;

impl<const N: usize> SignedBigIntHi32<N> {
    pub const fn new(magnitude_lo: [u64; N], magnitude_hi: u32, is_positive: bool) -> Self {
        Self {
            magnitude_lo,
            magnitude_hi,
            is_positive,
        }
    }

    pub const fn zero() -> Self {
        Self {
            magnitude_lo: [0; N],
            magnitude_hi: 0,
            is_positive: true,
        }
    }

    pub fn one() -> Self {
        let mut magnitude_lo = [0; N];
        let magnitude_hi;
        if N == 0 {
            magnitude_hi = 1;
        } else {
            magnitude_lo[0] = 1;
            magnitude_hi = 0;
        }
        Self {
            magnitude_lo,
            magnitude_hi,
            is_positive: true,
        }
    }

    pub const fn magnitude_lo(&self) -> &[u64; N] {
        &self.magnitude_lo
    }

    pub const fn magnitude_hi(&self) -> u32 {
        self.magnitude_hi
    }

    pub const fn is_positive(&self) -> bool {
        self.is_positive
    }

    pub const fn is_zero(&self) -> bool {
        let mut lo_is_zero = true;
        let mut i = 0;
        while i < N {
            if self.magnitude_lo[i] != 0 {
                lo_is_zero = false;
                break;
            }
            i += 1;
        }
        self.magnitude_hi == 0 && lo_is_zero
    }

    fn compare_magnitudes(&self, other: &Self) -> Ordering {
        if self.magnitude_hi != other.magnitude_hi {
            return self.magnitude_hi.cmp(&other.magnitude_hi);
        }
        for i in (0..N).rev() {
            if self.magnitude_lo[i] != other.magnitude_lo[i] {
                return self.magnitude_lo[i].cmp(&other.magnitude_lo[i]);
            }
        }
        Ordering::Equal
    }

    fn add_assign_in_place(&mut self, rhs: &Self) {
        if self.is_positive == rhs.is_positive {
            let (lo, hi, _carry) = self.add_magnitudes_with_carry(rhs);
            self.magnitude_lo = lo;
            self.magnitude_hi = hi;
        } else {
            match self.compare_magnitudes(rhs) {
                Ordering::Greater | Ordering::Equal => {
                    let (lo, hi, _borrow) = self.sub_magnitudes_with_borrow(rhs);
                    self.magnitude_lo = lo;
                    self.magnitude_hi = hi;
                }
                Ordering::Less => {
                    let (lo, hi, _borrow) = rhs.sub_magnitudes_with_borrow(self);
                    self.magnitude_lo = lo;
                    self.magnitude_hi = hi;
                    self.is_positive = rhs.is_positive;
                }
            }
        }
    }

    fn sub_assign_in_place(&mut self, rhs: &Self) {
        let neg_rhs = -*rhs;
        self.add_assign_in_place(&neg_rhs);
    }

    fn mul_magnitudes(&self, other: &Self) -> ([u64; N], u32) {
        if N == 0 {
            let a2 = self.magnitude_hi as u64;
            let b2 = other.magnitude_hi as u64;
            let prod = a2.wrapping_mul(b2);
            let hi = (prod & 0xFFFF_FFFF) as u32;
            let lo: [u64; N] = [0u64; N];
            return (lo, hi);
        }

        if N == 1 {
            let a0 = self.magnitude_lo[0];
            let a1 = self.magnitude_hi as u64;
            let b0 = other.magnitude_lo[0];
            let b1 = other.magnitude_hi as u64;

            let t0 = (a0 as u128) * (b0 as u128);
            let lo0 = t0 as u64;
            let cross = (t0 >> 64) + (a0 as u128) * (b1 as u128) + (a1 as u128) * (b0 as u128);
            let hi = (cross as u64 & 0xFFFF_FFFF) as u32;
            let mut lo = [0u64; N];
            lo[0] = lo0;
            return (lo, hi);
        }

        if N == 2 {
            let a0 = self.magnitude_lo[0];
            let a1 = self.magnitude_lo[1];
            let a2 = self.magnitude_hi as u64;
            let b0 = other.magnitude_lo[0];
            let b1 = other.magnitude_lo[1];
            let b2 = other.magnitude_hi as u64;

            let t0 = (a0 as u128) * (b0 as u128);
            let r0 = t0 as u64;
            let carry0 = t0 >> 64;

            let sum1 = carry0 + (a0 as u128) * (b1 as u128) + (a1 as u128) * (b0 as u128);
            let r1 = sum1 as u64;
            let carry1 = sum1 >> 64;

            let sum2 = carry1
                + (a0 as u128) * (b2 as u128)
                + (a1 as u128) * (b1 as u128)
                + (a2 as u128) * (b0 as u128);
            let r2 = sum2 as u64;

            let hi = (r2 & 0xFFFF_FFFF) as u32;
            let mut lo = [0u64; N];
            lo[0] = r0;
            lo[1] = r1;
            return (lo, hi);
        }

        // General path
        let mut prod = vec![0u64; 2 * N + 2];

        let self_limbs: Vec<u64> = self
            .magnitude_lo
            .iter()
            .copied()
            .chain(core::iter::once(self.magnitude_hi as u64))
            .collect();

        let other_limbs: Vec<u64> = other
            .magnitude_lo
            .iter()
            .copied()
            .chain(core::iter::once(other.magnitude_hi as u64))
            .collect();

        for (i, &a_limb) in self_limbs.iter().enumerate() {
            let mut carry: u128 = 0;
            for (j, &b_limb) in other_limbs.iter().enumerate() {
                let idx = i + j;
                let p = (a_limb as u128) * (b_limb as u128) + (prod[idx] as u128) + carry;
                prod[idx] = p as u64;
                carry = p >> 64;
            }
            if carry > 0 {
                let spill = i + other_limbs.len();
                if spill < prod.len() {
                    prod[spill] = prod[spill].wrapping_add(carry as u64);
                }
            }
        }

        let mut magnitude_lo = [0u64; N];
        magnitude_lo[..N].copy_from_slice(&prod[..N]);
        let magnitude_hi = (prod[N] & 0xFFFF_FFFF) as u32;
        (magnitude_lo, magnitude_hi)
    }

    fn add_magnitudes_with_carry(&self, other: &Self) -> ([u64; N], u32, bool) {
        let mut magnitude_lo = [0; N];
        let mut carry: u128 = 0;
        for (i, out) in magnitude_lo.iter_mut().enumerate() {
            let sum = (self.magnitude_lo[i] as u128) + (other.magnitude_lo[i] as u128) + carry;
            *out = sum as u64;
            carry = sum >> 64;
        }
        let sum_hi = (self.magnitude_hi as u128) + (other.magnitude_hi as u128) + carry;
        let magnitude_hi = sum_hi as u32;
        let final_carry = (sum_hi >> 32) != 0;
        (magnitude_lo, magnitude_hi, final_carry)
    }

    fn sub_magnitudes_with_borrow(&self, other: &Self) -> ([u64; N], u32, bool) {
        let mut magnitude_lo = [0u64; N];
        let mut borrow = false;
        for (i, out) in magnitude_lo.iter_mut().enumerate() {
            let (d1, b1) = self.magnitude_lo[i].overflowing_sub(other.magnitude_lo[i]);
            let (d2, b2) = d1.overflowing_sub(u64::from(borrow));
            *out = d2;
            borrow = b1 || b2;
        }
        let (hi1, b1) = self.magnitude_hi.overflowing_sub(other.magnitude_hi);
        let (hi2, b2) = hi1.overflowing_sub(u32::from(borrow));
        let final_borrow = b1 || b2;
        (magnitude_lo, hi2, final_borrow)
    }

    /// Return the unsigned magnitude as a `BigInt<NPLUS1>`.
    /// Debug-asserts `NPLUS1 == N + 1`.
    #[inline]
    pub fn magnitude_as_bigint_nplus1<const NPLUS1: usize>(&self) -> BigInt<NPLUS1> {
        debug_assert!(
            NPLUS1 == N + 1,
            "NPLUS1 must be N+1 for SignedBigIntHi32 magnitude pack"
        );
        let mut limbs = [0u64; NPLUS1];
        if N > 0 {
            limbs[..N].copy_from_slice(&self.magnitude_lo);
        }
        limbs[N] = self.magnitude_hi as u64;
        BigInt::<NPLUS1>(limbs)
    }

    #[inline]
    pub fn zero_extend_from<const M: usize>(smaller: &SignedBigIntHi32<M>) -> SignedBigIntHi32<N> {
        debug_assert!(
            M <= N,
            "cannot zero-extend: source has more limbs than destination"
        );
        if N == M {
            let mut lo = [0u64; N];
            if N > 0 {
                lo.copy_from_slice(smaller.magnitude_lo());
            }
            return SignedBigIntHi32::<N>::new(lo, smaller.magnitude_hi(), smaller.is_positive());
        }
        // N > M: place hi32 into limb M
        let mut lo = [0u64; N];
        if M > 0 {
            lo[..M].copy_from_slice(smaller.magnitude_lo());
        }
        lo[M] = smaller.magnitude_hi() as u64;
        SignedBigIntHi32::<N>::new(lo, 0u32, smaller.is_positive())
    }

    /// Convert into a `SignedBigInt<NPLUS1>`.
    /// Debug-asserts `NPLUS1 == N + 1`.
    #[inline]
    pub fn to_signed_bigint_nplus1<const NPLUS1: usize>(&self) -> SignedBigInt<NPLUS1> {
        debug_assert!(
            NPLUS1 == N + 1,
            "to_signed_bigint_nplus1 requires NPLUS1 = N + 1"
        );
        let mut limbs = [0u64; NPLUS1];
        if N > 0 {
            limbs[..N].copy_from_slice(self.magnitude_lo());
        }
        limbs[N] = self.magnitude_hi() as u64;
        let mag = BigInt::<NPLUS1>(limbs);
        SignedBigInt::from_bigint(mag, self.is_positive())
    }
}

impl<const N: usize> Neg for SignedBigIntHi32<N> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::new(self.magnitude_lo, self.magnitude_hi, !self.is_positive)
    }
}

impl<const N: usize> Neg for &SignedBigIntHi32<N> {
    type Output = SignedBigIntHi32<N>;
    fn neg(self) -> Self::Output {
        SignedBigIntHi32::new(self.magnitude_lo, self.magnitude_hi, !self.is_positive)
    }
}

impl<const N: usize> Add for SignedBigIntHi32<N> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        self.add_assign_in_place(&rhs);
        self
    }
}

impl<const N: usize> AddAssign for SignedBigIntHi32<N> {
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign_in_place(&rhs);
    }
}

impl<const N: usize> Sub for SignedBigIntHi32<N> {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self::Output {
        self.sub_assign_in_place(&rhs);
        self
    }
}

impl<const N: usize> SubAssign for SignedBigIntHi32<N> {
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign_in_place(&rhs);
    }
}

impl<const N: usize> Mul for SignedBigIntHi32<N> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let (lo, hi) = self.mul_magnitudes(&rhs);
        let is_positive = self.is_positive == rhs.is_positive;
        Self::new(lo, hi, is_positive)
    }
}

impl<const N: usize> MulAssign for SignedBigIntHi32<N> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const N: usize> Add<&SignedBigIntHi32<N>> for SignedBigIntHi32<N> {
    type Output = SignedBigIntHi32<N>;
    #[inline]
    fn add(mut self, rhs: &SignedBigIntHi32<N>) -> Self::Output {
        self.add_assign_in_place(rhs);
        self
    }
}

impl<const N: usize> Sub<&SignedBigIntHi32<N>> for SignedBigIntHi32<N> {
    type Output = SignedBigIntHi32<N>;
    #[inline]
    fn sub(mut self, rhs: &SignedBigIntHi32<N>) -> Self::Output {
        self.sub_assign_in_place(rhs);
        self
    }
}

impl<const N: usize> Mul<&SignedBigIntHi32<N>> for SignedBigIntHi32<N> {
    type Output = SignedBigIntHi32<N>;
    #[inline]
    fn mul(self, rhs: &SignedBigIntHi32<N>) -> Self::Output {
        let (lo, hi) = self.mul_magnitudes(rhs);
        let is_positive = self.is_positive == rhs.is_positive;
        Self::new(lo, hi, is_positive)
    }
}

impl<const N: usize> AddAssign<&SignedBigIntHi32<N>> for SignedBigIntHi32<N> {
    #[inline]
    fn add_assign(&mut self, rhs: &SignedBigIntHi32<N>) {
        self.add_assign_in_place(rhs);
    }
}

impl<const N: usize> SubAssign<&SignedBigIntHi32<N>> for SignedBigIntHi32<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: &SignedBigIntHi32<N>) {
        self.sub_assign_in_place(rhs);
    }
}

impl<const N: usize> MulAssign<&SignedBigIntHi32<N>> for SignedBigIntHi32<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: &SignedBigIntHi32<N>) {
        *self = *self * rhs;
    }
}

impl<const N: usize> Add for &SignedBigIntHi32<N> {
    type Output = SignedBigIntHi32<N>;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = *self;
        out.add_assign_in_place(rhs);
        out
    }
}

impl<const N: usize> Sub for &SignedBigIntHi32<N> {
    type Output = SignedBigIntHi32<N>;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = *self;
        out.sub_assign_in_place(rhs);
        out
    }
}

impl<const N: usize> Mul for &SignedBigIntHi32<N> {
    type Output = SignedBigIntHi32<N>;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let (lo, hi) = self.mul_magnitudes(rhs);
        let is_positive = self.is_positive == rhs.is_positive;
        SignedBigIntHi32::new(lo, hi, is_positive)
    }
}

impl<const N: usize> PartialOrd for SignedBigIntHi32<N> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const N: usize> Ord for SignedBigIntHi32<N> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        if self.is_zero() && other.is_zero() {
            return Ordering::Equal;
        }
        match (self.is_positive, other.is_positive) {
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            _ => {
                let ord = self.compare_magnitudes(other);
                if self.is_positive {
                    ord
                } else {
                    ord.reverse()
                }
            }
        }
    }
}

impl<const N: usize> CanonicalSerialize for SignedBigIntHi32<N> {
    #[inline]
    fn serialize_with_mode<W: Write>(
        &self,
        mut w: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        (self.is_positive as u8).serialize_with_mode(&mut w, compress)?;
        (self.magnitude_hi as i32).serialize_with_mode(&mut w, compress)?;
        for i in 0..N {
            self.magnitude_lo[i].serialize_with_mode(&mut w, compress)?;
        }
        Ok(())
    }

    #[inline]
    fn serialized_size(&self, compress: Compress) -> usize {
        (self.is_positive as u8).serialized_size(compress)
            + (self.magnitude_hi as i32).serialized_size(compress)
            + (0u64).serialized_size(compress) * N
    }
}

impl<const N: usize> CanonicalDeserialize for SignedBigIntHi32<N> {
    #[inline]
    fn deserialize_with_mode<R: Read>(
        mut r: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let sign_u8 = u8::deserialize_with_mode(&mut r, compress, validate)?;
        let hi = i32::deserialize_with_mode(&mut r, compress, validate)?;
        let mut lo = [0u64; N];
        for limb in &mut lo {
            *limb = u64::deserialize_with_mode(&mut r, compress, validate)?;
        }
        Ok(SignedBigIntHi32::new(lo, hi as u32, sign_u8 != 0))
    }
}

impl<const N: usize> Valid for SignedBigIntHi32<N> {
    #[inline]
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl From<i64> for S96 {
    #[inline]
    fn from(val: i64) -> Self {
        Self::new([val.unsigned_abs()], 0, val.is_positive())
    }
}

impl From<u64> for S96 {
    #[inline]
    fn from(val: u64) -> Self {
        Self::new([val], 0, true)
    }
}

impl From<S64> for S96 {
    #[inline]
    fn from(val: S64) -> Self {
        Self::new([val.magnitude.0[0]], 0, val.is_positive)
    }
}

impl From<i64> for S160 {
    #[inline]
    fn from(val: i64) -> Self {
        Self::new([val.unsigned_abs(), 0], 0, val.is_positive())
    }
}

impl From<u64> for S160 {
    #[inline]
    fn from(val: u64) -> Self {
        Self::new([val, 0], 0, true)
    }
}

impl From<S64> for S160 {
    #[inline]
    fn from(val: S64) -> Self {
        Self::new([val.magnitude.0[0], 0], 0, val.is_positive)
    }
}

impl From<u128> for S160 {
    #[inline]
    fn from(val: u128) -> Self {
        let lo = val as u64;
        let hi = (val >> 64) as u64;
        Self::new([lo, hi], 0, true)
    }
}

impl From<i128> for S160 {
    #[inline]
    fn from(val: i128) -> Self {
        let is_positive = val.is_positive();
        let mag = val.unsigned_abs();
        let lo = mag as u64;
        let hi = (mag >> 64) as u64;
        Self::new([lo, hi], 0, is_positive)
    }
}

impl From<S128> for S160 {
    #[inline]
    fn from(val: S128) -> Self {
        Self::new([val.magnitude.0[0], val.magnitude.0[1]], 0, val.is_positive)
    }
}

impl<const N: usize> From<S224> for BigInt<N> {
    #[inline]
    #[allow(unsafe_code)]
    fn from(val: S224) -> Self {
        assert!(
            N == 4,
            "FromS224 for BigInt<N> only supports N=4, got N={N}"
        );
        let lo = val.magnitude_lo();
        let hi = val.magnitude_hi() as u64;
        let bigint4 = BigInt::<4>([lo[0], lo[1], lo[2], hi]);

        // SAFETY: BigInt<4> and BigInt<N> have identical layout when N=4
        // (asserted above).
        unsafe { (&raw const bigint4).cast::<BigInt<N>>().read() }
    }
}

impl S160 {
    /// Computes the signed difference `a - b` as an `S160`.
    #[inline]
    pub fn from_diff_u64(a: u64, b: u64) -> Self {
        let mag = a.abs_diff(b);
        let is_positive = a >= b;
        S160::new([mag, 0], 0, is_positive)
    }

    /// Creates an `S160` from a `u128` magnitude and sign.
    #[inline]
    pub fn from_magnitude_u128(mag: u128, is_positive: bool) -> Self {
        let lo = mag as u64;
        let hi = (mag >> 64) as u64;
        S160::new([lo, hi], 0, is_positive)
    }

    /// Computes the signed difference `u1 - u2` as an `S160`.
    #[inline]
    pub fn from_diff_u128(u1: u128, u2: u128) -> Self {
        if u1 >= u2 {
            S160::from_magnitude_u128(u1 - u2, true)
        } else {
            S160::from_magnitude_u128(u2 - u1, false)
        }
    }

    /// Computes `u1 + u2` as an `S160`, handling carry into the hi32 limb.
    #[inline]
    pub fn from_sum_u128(u1: u128, u2: u128) -> Self {
        let u1_lo = u1 as u64;
        let u1_hi = (u1 >> 64) as u64;
        let u2_lo = u2 as u64;
        let u2_hi = (u2 >> 64) as u64;
        let (sum_lo, carry0) = u1_lo.overflowing_add(u2_lo);
        let (sum_hi1, carry1) = u1_hi.overflowing_add(u2_hi);
        let (sum_hi, carry2) = sum_hi1.overflowing_add(u64::from(carry0));
        let carry_out = carry1 || carry2;
        S160::new([sum_lo, sum_hi], u32::from(carry_out), true)
    }

    /// Computes `u - i` as an `S160`.
    #[inline]
    pub fn from_u128_minus_i128(u: u128, i: i128) -> Self {
        if i >= 0 {
            S160::from_diff_u128(u, i as u128)
        } else {
            let abs_i: u128 = i.unsigned_abs();
            S160::from_sum_u128(u, abs_i)
        }
    }
}

impl Default for S160 {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s160_from_diff_u64() {
        let d = S160::from_diff_u64(10, 3);
        assert!(d.is_positive());
        assert_eq!(d.magnitude_lo()[0], 7);

        let d2 = S160::from_diff_u64(3, 10);
        assert!(!d2.is_positive());
        assert_eq!(d2.magnitude_lo()[0], 7);
    }

    #[test]
    fn s160_addition() {
        let a = S160::from(100u64);
        let b = S160::from(200u64);
        let c = a + b;
        assert!(c.is_positive());
        assert_eq!(c.magnitude_lo()[0], 300);
    }

    #[test]
    fn s160_subtraction() {
        let a = S160::from(100u64);
        let b = S160::from(200u64);
        let c = a - b;
        assert!(!c.is_positive());
        assert_eq!(c.magnitude_lo()[0], 100);
    }

    #[test]
    fn s160_to_signed_bigint() {
        let v = S160::new([42, 0], 7, false);
        let sb: SignedBigInt<3> = v.to_signed_bigint_nplus1::<3>();
        assert!(!sb.is_positive);
        assert_eq!(sb.magnitude.0[0], 42);
        assert_eq!(sb.magnitude.0[1], 0);
        assert_eq!(sb.magnitude.0[2], 7);
    }

    #[test]
    fn serialization_roundtrip() {
        let val = S160::new([123_456_789, 987_654_321], 42, false);
        let mut bytes = Vec::new();
        val.serialize_compressed(&mut bytes).unwrap();
        let restored = S160::deserialize_compressed(&bytes[..]).unwrap();
        assert_eq!(val, restored);
    }

    #[test]
    fn s160_from_u128_minus_i128() {
        let v = S160::from_u128_minus_i128(100, -50);
        assert!(v.is_positive());
        assert_eq!(v.magnitude_lo()[0], 150);

        let v2 = S160::from_u128_minus_i128(100, 150);
        assert!(!v2.is_positive());
        assert_eq!(v2.magnitude_lo()[0], 50);
    }

    #[test]
    fn zero_extend() {
        let s = S96::from(42u64);
        let wide: S160 = SignedBigIntHi32::zero_extend_from(&s);
        assert!(wide.is_positive());
        assert_eq!(wide.magnitude_lo()[0], 42);
    }
}
