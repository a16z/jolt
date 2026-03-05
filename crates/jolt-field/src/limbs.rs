//! Fixed-width limb array for multi-precision arithmetic.
//!
//! [`Limbs<N>`] is a `#[repr(transparent)]` newtype over `[u64; N]` that
//! decouples the public API from `ark_ff::BigInt`. All truncated arithmetic
//! previously on `BigIntExt` lives here as inherent methods.

use ark_ff::BigInt;
use core::cmp::Ordering;

/// Fixed-width array of `N` 64-bit limbs in little-endian order.
///
/// Used as the magnitude type for [`SignedBigInt`](crate::signed::SignedBigInt)
/// and as the output of truncated multiplication in unreduced arithmetic.
/// Provides the same multi-precision operations that `BigIntExt` offered on
/// `ark_ff::BigInt`, without leaking arkworks types into the public API.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Limbs<const N: usize>(pub [u64; N]);

impl<const N: usize> Default for Limbs<N> {
    #[inline]
    fn default() -> Self {
        Self([0u64; N])
    }
}

impl<const N: usize> Limbs<N> {
    #[inline]
    pub const fn new(limbs: [u64; N]) -> Self {
        Self(limbs)
    }

    #[inline]
    pub const fn zero() -> Self {
        Self([0u64; N])
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&l| l == 0)
    }

    /// Number of significant bits in the value.
    #[inline]
    pub fn num_bits(&self) -> u32 {
        let mut i = N;
        while i > 0 {
            i -= 1;
            if self.0[i] != 0 {
                return (i as u32) * 64 + (64 - self.0[i].leading_zeros());
            }
        }
        0
    }

    /// Constructs from a single `u64`, placed in the lowest limb.
    #[inline]
    pub fn from_u64(val: u64) -> Self {
        let mut limbs = [0u64; N];
        if N > 0 {
            limbs[0] = val;
        }
        Self(limbs)
    }

    /// Converts to `BigInt<N>` by copying limbs.
    #[inline]
    pub fn to_bigint(self) -> BigInt<N> {
        BigInt(self.0)
    }

    /// In-place addition with carry propagation.
    /// Returns `true` if the final carry overflowed.
    #[inline]
    pub fn add_with_carry(&mut self, other: &Self) -> bool {
        let mut carry = 0u64;
        for i in 0..N {
            let sum = (self.0[i] as u128) + (other.0[i] as u128) + (carry as u128);
            self.0[i] = sum as u64;
            carry = (sum >> 64) as u64;
        }
        carry != 0
    }

    /// In-place subtraction with borrow propagation.
    /// Returns `true` if the final borrow underflowed.
    #[inline]
    pub fn sub_with_borrow(&mut self, other: &Self) -> bool {
        let mut borrow = false;
        for i in 0..N {
            let (d1, b1) = self.0[i].overflowing_sub(other.0[i]);
            let (d2, b2) = d1.overflowing_sub(u64::from(borrow));
            self.0[i] = d2;
            borrow = b1 || b2;
        }
        borrow
    }

    // -- Truncated arithmetic (migrated from BigIntExt) -----------------------

    /// Truncated multiplication: `self * other`, keeping the low `P` limbs.
    #[inline]
    pub fn mul_trunc<const M: usize, const P: usize>(&self, other: &Limbs<M>) -> Limbs<P> {
        let mut res = Limbs::<P>::zero();
        fm_limbs_into::<N, M, P>(&self.0, &other.0, &mut res.0);
        res
    }

    /// Truncated addition: `self + other`, keeping the low `P` limbs.
    #[inline]
    pub fn add_trunc<const M: usize, const P: usize>(&self, other: &Limbs<M>) -> Limbs<P> {
        let mut acc = Limbs::<P>::zero();
        let copy_len = if P < N { P } else { N };
        acc.0[..copy_len].copy_from_slice(&self.0[..copy_len]);
        acc.add_assign_trunc::<M>(other);
        acc
    }

    /// Truncated subtraction: `self - other`, keeping the low `P` limbs.
    #[inline]
    pub fn sub_trunc<const M: usize, const P: usize>(&self, other: &Limbs<M>) -> Limbs<P> {
        let mut acc = Limbs::<P>::zero();
        let copy_len = if P < N { P } else { N };
        acc.0[..copy_len].copy_from_slice(&self.0[..copy_len]);
        acc.sub_assign_trunc::<M>(other);
        acc
    }

    /// In-place truncated addition: `self += other`, keeping `N` limbs.
    #[inline]
    pub fn add_assign_trunc<const M: usize>(&mut self, other: &Limbs<M>) {
        debug_assert!(M <= N, "add_assign_trunc: right operand wider than self");
        let mut carry = 0u64;
        for i in 0..N {
            let rhs = if i < M { other.0[i] } else { 0 };
            let sum = (self.0[i] as u128) + (rhs as u128) + (carry as u128);
            self.0[i] = sum as u64;
            carry = (sum >> 64) as u64;
        }
    }

    /// In-place truncated subtraction: `self -= other`, keeping `N` limbs.
    #[inline]
    pub fn sub_assign_trunc<const M: usize>(&mut self, other: &Limbs<M>) {
        debug_assert!(M <= N, "sub_assign_trunc: right operand wider than self");
        let mut borrow = 0u64;
        for i in 0..N {
            let rhs = if i < M { other.0[i] } else { 0 };
            let diff = (self.0[i] as u128)
                .wrapping_sub(rhs as u128)
                .wrapping_sub(borrow as u128);
            self.0[i] = diff as u64;
            borrow = u64::from(diff > u64::MAX as u128);
        }
    }

    /// Truncated fused multiply-add: `self += a * b`, keeping `N` limbs.
    #[inline]
    pub fn fmadd_trunc<const A: usize, const B: usize>(&mut self, a: &Limbs<A>, b: &Limbs<B>) {
        let i_limit = if A < N { A } else { N };
        for i in 0..i_limit {
            let mut carry = 0u64;
            let j_limit = if B < (N - i) { B } else { N - i };
            for j in 0..j_limit {
                let idx = i + j;
                let prod =
                    (a.0[i] as u128) * (b.0[j] as u128) + (self.0[idx] as u128) + (carry as u128);
                self.0[idx] = prod as u64;
                carry = (prod >> 64) as u64;
            }
            let spill = i + j_limit;
            if spill < N {
                let (new_val, _) = self.0[spill].overflowing_add(carry);
                self.0[spill] = new_val;
            }
        }
    }

    /// Multiply and keep only the low `N` limbs (same width as self).
    #[inline]
    pub fn mul_low(&self, other: &Self) -> Self {
        let mut res = Limbs::<N>::zero();
        fm_limbs_into::<N, N, N>(&self.0, &other.0, &mut res.0);
        res
    }

    /// Zero-extend a narrower `Limbs<M>` into `Limbs<N>`.
    #[inline]
    pub fn zero_extend_from<const M: usize>(smaller: &Limbs<M>) -> Limbs<N> {
        debug_assert!(
            M <= N,
            "cannot zero-extend: source has more limbs than destination"
        );
        let mut limbs = [0u64; N];
        let copy_len = if M < N { M } else { N };
        limbs[..copy_len].copy_from_slice(&smaller.0[..copy_len]);
        Limbs(limbs)
    }
}

// -- Conversions --------------------------------------------------------------

impl<const N: usize> From<u64> for Limbs<N> {
    #[inline]
    fn from(val: u64) -> Self {
        Self::from_u64(val)
    }
}

impl<const N: usize> From<BigInt<N>> for Limbs<N> {
    #[inline]
    fn from(bigint: BigInt<N>) -> Self {
        Limbs(bigint.0)
    }
}

impl<const N: usize> From<Limbs<N>> for BigInt<N> {
    #[inline]
    fn from(limbs: Limbs<N>) -> Self {
        BigInt(limbs.0)
    }
}

impl<const N: usize> AsRef<[u64]> for Limbs<N> {
    #[inline]
    fn as_ref(&self) -> &[u64] {
        &self.0
    }
}

// -- Ordering -----------------------------------------------------------------

impl<const N: usize> PartialOrd for Limbs<N> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const N: usize> Ord for Limbs<N> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        let mut i = N;
        while i > 0 {
            i -= 1;
            match self.0[i].cmp(&other.0[i]) {
                Ordering::Equal => {}
                ord => return ord,
            }
        }
        Ordering::Equal
    }
}

// -- Formatting ---------------------------------------------------------------

impl<const N: usize> core::fmt::Debug for Limbs<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Limbs([")?;
        for (i, limb) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{limb:#018x}")?;
        }
        write!(f, "])")
    }
}

impl<const N: usize> core::fmt::Display for Limbs<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Display as hex starting from the most significant limb
        let mut started = false;
        for &limb in self.0.iter().rev() {
            if !started {
                if limb != 0 {
                    write!(f, "{limb:x}")?;
                    started = true;
                }
            } else {
                write!(f, "{limb:016x}")?;
            }
        }
        if !started {
            write!(f, "0")?;
        }
        Ok(())
    }
}

// -- Serialization ------------------------------------------------------------

impl<const N: usize> ark_serialize::CanonicalSerialize for Limbs<N> {
    #[inline]
    fn serialize_with_mode<W: ark_serialize::Write>(
        &self,
        writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        self.to_bigint().serialize_with_mode(writer, compress)
    }

    #[inline]
    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        self.to_bigint().serialized_size(compress)
    }
}

impl<const N: usize> ark_serialize::Valid for Limbs<N> {
    #[inline]
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        self.to_bigint().check()
    }
}

impl<const N: usize> ark_serialize::CanonicalDeserialize for Limbs<N> {
    #[inline]
    fn deserialize_with_mode<R: ark_serialize::Read>(
        reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        BigInt::<N>::deserialize_with_mode(reader, compress, validate).map(Limbs::from)
    }
}

// -- Internal helper ----------------------------------------------------------

/// Core schoolbook multiplication accumulator.
///
/// Computes `acc += a[0..N] * b[0..M]`, keeping only the low `P` limbs.
#[inline]
fn fm_limbs_into<const N: usize, const M: usize, const P: usize>(
    a: &[u64; N],
    b: &[u64; M],
    acc: &mut [u64; P],
) {
    for (j, &mul_limb) in b.iter().enumerate() {
        if mul_limb == 0 {
            continue;
        }
        let base = j;
        let mut carry = 0u64;
        for (i, &a_limb) in a.iter().enumerate() {
            let idx = base + i;
            if idx < P {
                let prod =
                    (a_limb as u128) * (mul_limb as u128) + (acc[idx] as u128) + (carry as u128);
                acc[idx] = prod as u64;
                carry = (prod >> 64) as u64;
            }
        }
        let next = base + N;
        if next < P {
            let (v, _) = acc[next].overflowing_add(carry);
            acc[next] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_trunc_small() {
        let a = Limbs::<1>([7u64]);
        let b = Limbs::<1>([6u64]);
        let c: Limbs<1> = a.mul_trunc::<1, 1>(&b);
        assert_eq!(c.0[0], 42);
    }

    #[test]
    fn mul_trunc_wider_output() {
        let a = Limbs::<1>([u64::MAX]);
        let b = Limbs::<1>([2u64]);
        let c: Limbs<2> = a.mul_trunc::<1, 2>(&b);
        let expected = (u64::MAX as u128) * 2;
        assert_eq!(c.0[0], expected as u64);
        assert_eq!(c.0[1], (expected >> 64) as u64);
    }

    #[test]
    fn add_trunc_basic() {
        let a = Limbs::<2>([u64::MAX, 0]);
        let b = Limbs::<2>([1u64, 0]);
        let c: Limbs<2> = a.add_trunc::<2, 2>(&b);
        assert_eq!(c.0[0], 0);
        assert_eq!(c.0[1], 1);
    }

    #[test]
    fn sub_trunc_basic() {
        let a = Limbs::<2>([0, 1]);
        let b = Limbs::<2>([1, 0]);
        let c: Limbs<2> = a.sub_trunc::<2, 2>(&b);
        assert_eq!(c.0[0], u64::MAX);
        assert_eq!(c.0[1], 0);
    }

    #[test]
    fn zero_extend() {
        let small = Limbs::<1>([42u64]);
        let big: Limbs<4> = Limbs::<4>::zero_extend_from::<1>(&small);
        assert_eq!(big.0[0], 42);
        assert_eq!(big.0[1], 0);
        assert_eq!(big.0[2], 0);
        assert_eq!(big.0[3], 0);
    }

    #[test]
    fn mul_low_basic() {
        let a = Limbs::<2>([3, 0]);
        let b = Limbs::<2>([5, 0]);
        let c = a.mul_low(&b);
        assert_eq!(c.0[0], 15);
        assert_eq!(c.0[1], 0);
    }

    #[test]
    fn fmadd_trunc_basic() {
        let a = Limbs::<1>([3u64]);
        let b = Limbs::<1>([4u64]);
        let mut acc = Limbs::<2>([10, 0]);
        acc.fmadd_trunc::<1, 1>(&a, &b);
        assert_eq!(acc.0[0], 22); // 10 + 3*4
    }

    #[test]
    fn add_sub_with_carry_borrow() {
        let mut a = Limbs::<2>([u64::MAX, 0]);
        let b = Limbs::<2>([1, 0]);
        let carry = a.add_with_carry(&b);
        assert!(!carry);
        assert_eq!(a.0[0], 0);
        assert_eq!(a.0[1], 1);

        let borrow = a.sub_with_borrow(&b);
        assert!(!borrow);
        assert_eq!(a.0[0], u64::MAX);
        assert_eq!(a.0[1], 0);
    }

    #[test]
    fn ordering() {
        let a = Limbs::<2>([0, 1]);
        let b = Limbs::<2>([u64::MAX, 0]);
        assert!(a > b);
        assert_eq!(a.cmp(&a), Ordering::Equal);
    }

    #[test]
    fn bigint_roundtrip() {
        let limbs = Limbs::<4>([1, 2, 3, 4]);
        let bigint: BigInt<4> = limbs.into();
        let back: Limbs<4> = bigint.into();
        assert_eq!(limbs, back);
    }

    #[test]
    fn display_formatting() {
        let z = Limbs::<2>([0, 0]);
        assert_eq!(format!("{z}"), "0");

        let one = Limbs::<1>([1]);
        assert_eq!(format!("{one}"), "1");

        let big = Limbs::<2>([0, 1]);
        assert_eq!(format!("{big}"), "10000000000000000");
    }
}
