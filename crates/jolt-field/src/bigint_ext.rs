//! Extension methods for `ark_ff::BigInt<N>` that were previously in the arkworks fork.
//!
//! These operations — truncated multiplication, truncated addition/subtraction,
//! zero-extension, and low-limb multiplication — are used extensively by the
//! signed big integer types and unreduced field arithmetic.

use ark_ff::BigInt;

/// Extension trait adding truncated-width arithmetic to `BigInt<N>`.
///
/// All "trunc" operations compute an exact result and then keep only the
/// low `P` limbs, silently discarding higher limbs.
pub trait BigIntExt<const N: usize> {
    /// Truncated multiplication: compute `self * other` and keep the low `P` limbs.
    fn mul_trunc<const M: usize, const P: usize>(&self, other: &BigInt<M>) -> BigInt<P>;

    /// Truncated addition: compute `self + other` and keep the low `P` limbs.
    fn add_trunc<const M: usize, const P: usize>(&self, other: &BigInt<M>) -> BigInt<P>;

    /// Truncated subtraction: compute `self - other` and keep the low `P` limbs.
    fn sub_trunc<const M: usize, const P: usize>(&self, other: &BigInt<M>) -> BigInt<P>;

    /// In-place truncated addition: `self += other`, keeping `N` limbs.
    fn add_assign_trunc<const M: usize>(&mut self, other: &BigInt<M>);

    /// In-place truncated subtraction: `self -= other`, keeping `N` limbs.
    fn sub_assign_trunc<const M: usize>(&mut self, other: &BigInt<M>);

    /// Truncated fused multiply-add: `acc += self * other`, keeping `P` limbs.
    fn fmadd_trunc<const M: usize, const P: usize>(&self, other: &BigInt<M>, acc: &mut BigInt<P>);

    /// Multiply and keep only the low `N` limbs (same width as self).
    fn mul_low(&self, other: &Self) -> Self;

    /// Zero-extend a `BigInt<M>` into a wider `BigInt<N>`.
    fn zero_extend_from<const M: usize>(smaller: &BigInt<M>) -> BigInt<N>;
}

impl<const N: usize> BigIntExt<N> for BigInt<N> {
    #[inline]
    fn mul_trunc<const M: usize, const P: usize>(&self, other: &BigInt<M>) -> BigInt<P> {
        let mut res = BigInt::<P>([0u64; P]);
        fm_limbs_into::<N, M, P>(&self.0, &other.0, &mut res, false);
        res
    }

    #[inline]
    fn add_trunc<const M: usize, const P: usize>(&self, other: &BigInt<M>) -> BigInt<P> {
        let mut acc = BigInt::<P>([0u64; P]);
        let copy_len = if P < N { P } else { N };
        acc.0[..copy_len].copy_from_slice(&self.0[..copy_len]);
        acc.add_assign_trunc::<M>(other);
        acc
    }

    #[inline]
    fn sub_trunc<const M: usize, const P: usize>(&self, other: &BigInt<M>) -> BigInt<P> {
        let mut acc = BigInt::<P>([0u64; P]);
        let copy_len = if P < N { P } else { N };
        acc.0[..copy_len].copy_from_slice(&self.0[..copy_len]);
        acc.sub_assign_trunc::<M>(other);
        acc
    }

    #[inline]
    fn add_assign_trunc<const M: usize>(&mut self, other: &BigInt<M>) {
        debug_assert!(M <= N, "add_assign_trunc: right operand wider than self");
        let mut carry = 0u64;
        for i in 0..N {
            let rhs = if i < M { other.0[i] } else { 0 };
            let sum = (self.0[i] as u128) + (rhs as u128) + (carry as u128);
            self.0[i] = sum as u64;
            carry = (sum >> 64) as u64;
        }
    }

    #[inline]
    fn sub_assign_trunc<const M: usize>(&mut self, other: &BigInt<M>) {
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

    #[inline]
    fn fmadd_trunc<const M: usize, const P: usize>(&self, other: &BigInt<M>, acc: &mut BigInt<P>) {
        let i_limit = if N < P { N } else { P };
        for i in 0..i_limit {
            let mut carry = 0u64;
            let j_limit = if M < (P - i) { M } else { P - i };
            for j in 0..j_limit {
                let idx = i + j;
                let prod = (self.0[i] as u128) * (other.0[j] as u128)
                    + (acc.0[idx] as u128)
                    + (carry as u128);
                acc.0[idx] = prod as u64;
                carry = (prod >> 64) as u64;
            }
            let spill = i + j_limit;
            if spill < P {
                let (new_val, _) = acc.0[spill].overflowing_add(carry);
                acc.0[spill] = new_val;
            }
        }
    }

    #[inline]
    fn mul_low(&self, other: &Self) -> Self {
        let mut res = BigInt::<N>([0u64; N]);
        fm_limbs_into::<N, N, N>(&self.0, &other.0, &mut res, false);
        res
    }

    #[inline]
    fn zero_extend_from<const M: usize>(smaller: &BigInt<M>) -> BigInt<N> {
        debug_assert!(
            M <= N,
            "cannot zero-extend: source has more limbs than destination"
        );
        let mut limbs = [0u64; N];
        let copy_len = if M < N { M } else { N };
        limbs[..copy_len].copy_from_slice(&smaller.0[..copy_len]);
        BigInt::<N>(limbs)
    }
}

/// Core schoolbook multiplication accumulator.
///
/// Computes `acc += a[0..N] * b[0..M]`, keeping only the low `P` limbs.
/// When `carry_propagate` is true, carries beyond `base + N` are propagated
/// through all remaining limbs; when false, only a single limb is updated.
#[inline]
fn fm_limbs_into<const N: usize, const M: usize, const P: usize>(
    a: &[u64; N],
    b: &[u64; M],
    acc: &mut BigInt<P>,
    carry_propagate: bool,
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
                    (a_limb as u128) * (mul_limb as u128) + (acc.0[idx] as u128) + (carry as u128);
                acc.0[idx] = prod as u64;
                carry = (prod >> 64) as u64;
            }
        }
        let next = base + N;
        if next < P {
            let (v, mut of) = acc.0[next].overflowing_add(carry);
            acc.0[next] = v;
            if carry_propagate && of {
                let mut k = next + 1;
                while of && k < P {
                    let (nv, nof) = acc.0[k].overflowing_add(1);
                    acc.0[k] = nv;
                    of = nof;
                    k += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_trunc_small() {
        let a = BigInt::<1>([7u64]);
        let b = BigInt::<1>([6u64]);
        let c: BigInt<1> = a.mul_trunc::<1, 1>(&b);
        assert_eq!(c.0[0], 42);
    }

    #[test]
    fn mul_trunc_wider_output() {
        let a = BigInt::<1>([u64::MAX]);
        let b = BigInt::<1>([2u64]);
        let c: BigInt<2> = a.mul_trunc::<1, 2>(&b);
        let expected = (u64::MAX as u128) * 2;
        assert_eq!(c.0[0], expected as u64);
        assert_eq!(c.0[1], (expected >> 64) as u64);
    }

    #[test]
    fn add_trunc_basic() {
        let a = BigInt::<2>([u64::MAX, 0]);
        let b = BigInt::<2>([1u64, 0]);
        let c: BigInt<2> = a.add_trunc::<2, 2>(&b);
        assert_eq!(c.0[0], 0);
        assert_eq!(c.0[1], 1);
    }

    #[test]
    fn sub_trunc_basic() {
        let a = BigInt::<2>([0, 1]);
        let b = BigInt::<2>([1, 0]);
        let c: BigInt<2> = a.sub_trunc::<2, 2>(&b);
        assert_eq!(c.0[0], u64::MAX);
        assert_eq!(c.0[1], 0);
    }

    #[test]
    fn zero_extend() {
        let small = BigInt::<1>([42u64]);
        let big: BigInt<4> = BigInt::<4>::zero_extend_from::<1>(&small);
        assert_eq!(big.0[0], 42);
        assert_eq!(big.0[1], 0);
        assert_eq!(big.0[2], 0);
        assert_eq!(big.0[3], 0);
    }

    #[test]
    fn mul_low_basic() {
        let a = BigInt::<2>([3, 0]);
        let b = BigInt::<2>([5, 0]);
        let c = a.mul_low(&b);
        assert_eq!(c.0[0], 15);
        assert_eq!(c.0[1], 0);
    }

    #[test]
    fn fmadd_trunc_basic() {
        let a = BigInt::<1>([3u64]);
        let b = BigInt::<1>([4u64]);
        let mut acc = BigInt::<2>([10, 0]);
        a.fmadd_trunc::<1, 2>(&b, &mut acc);
        assert_eq!(acc.0[0], 22); // 10 + 3*4
    }
}
