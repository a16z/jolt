//! Wide-integer accumulator for BN254 Fr deferred reduction.
//!
//! Accumulates `sum += a * b` as 9-limb (576-bit) schoolbook products,
//! deferring the Montgomery reduction to a single call at the end.
//!
//! # Capacity
//!
//! Each Fr element is 4 limbs (256 bits). The unreduced product of two
//! elements is 8 limbs (512 bits). A 9-limb accumulator (576 bits) can
//! hold up to 2^64 such products without overflow.

use crate::accumulator::FieldAccumulator;
use crate::arkworks::bn254::Fr;
use crate::Limbs;

use super::bn254_ops;

/// Wide 9-limb accumulator for BN254 Fr deferred reduction.
///
/// Stores the running sum of Montgomery-form products as a 576-bit integer.
/// Converting to a field element requires a single Montgomery reduction
/// via [`reduce`](FieldAccumulator::reduce).
#[derive(Clone, Copy)]
pub struct WideAccumulator {
    /// 9 limbs = 2×4 (product width) + 1 (addition headroom).
    limbs: Limbs<9>,
}

impl Default for WideAccumulator {
    #[inline]
    fn default() -> Self {
        Self {
            limbs: Limbs::zero(),
        }
    }
}

impl FieldAccumulator for WideAccumulator {
    type Field = Fr;

    #[inline(always)]
    fn fmadd(&mut self, a: Fr, b: Fr) {
        self.limbs.fmadd::<4, 4>(&a.inner_limbs(), &b.inner_limbs());
    }

    /// Accumulate `val` with unit weight, avoiding the 4×4 schoolbook multiply.
    ///
    /// Instead of `fmadd(one_mont, val_mont)` (which multiplies by R mod p),
    /// we add `val_mont << 256` (= val_mont × R) directly to the accumulator.
    /// After final Montgomery reduction (÷R), this contributes exactly `val`.
    ///
    /// Cost: 4 u64 additions with carry propagation vs 16 u64 multiplications.
    #[inline(always)]
    fn acc_add(&mut self, val: Fr) {
        let val_limbs = val.inner_limbs();
        // Add val_limbs[0..4] into self.limbs[4..8] with carry into limb 8.
        let mut carry = 0u64;
        for i in 0..4 {
            let sum = (self.limbs.0[4 + i] as u128) + (val_limbs.0[i] as u128) + (carry as u128);
            self.limbs.0[4 + i] = sum as u64;
            carry = (sum >> 64) as u64;
        }
        // Propagate final carry into limb 8.
        if carry != 0 {
            let sum = (self.limbs.0[8] as u128) + (carry as u128);
            self.limbs.0[8] = sum as u64;
        }
    }

    #[inline(always)]
    fn merge(&mut self, other: Self) {
        self.limbs.add_assign_trunc::<9>(&other.limbs);
    }

    fn reduce(self) -> Fr {
        // The accumulator holds sum_i (a_i_mont × b_i_mont).
        // Montgomery reduction divides by R, yielding the Montgomery form
        // of sum_i (a_i × b_i).
        let bigint = self.limbs.to_bigint();
        Fr::from_inner(bn254_ops::from_montgomery_reduce(bigint))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Field;

    #[test]
    fn single_fmadd() {
        let a = Fr::from_u64(7);
        let b = Fr::from_u64(6);
        let mut acc = WideAccumulator::default();
        acc.fmadd(a, b);
        assert_eq!(acc.reduce(), Fr::from_u64(42));
    }

    #[test]
    fn multiple_fmadd() {
        let mut acc = WideAccumulator::default();
        acc.fmadd(Fr::from_u64(3), Fr::from_u64(4));
        acc.fmadd(Fr::from_u64(5), Fr::from_u64(6));
        // 3*4 + 5*6 = 12 + 30 = 42
        assert_eq!(acc.reduce(), Fr::from_u64(42));
    }

    #[test]
    fn merge_two_accumulators() {
        let mut acc1 = WideAccumulator::default();
        acc1.fmadd(Fr::from_u64(10), Fr::from_u64(10));

        let mut acc2 = WideAccumulator::default();
        acc2.fmadd(Fr::from_u64(20), Fr::from_u64(20));

        acc1.merge(acc2);
        // 10*10 + 20*20 = 100 + 400 = 500
        assert_eq!(acc1.reduce(), Fr::from_u64(500));
    }

    #[test]
    fn empty_reduces_to_zero() {
        let acc = WideAccumulator::default();
        assert_eq!(acc.reduce(), Fr::from_u64(0));
    }

    #[test]
    fn large_accumulation() {
        let mut acc = WideAccumulator::default();
        let n = 10_000u64;
        let a = Fr::from_u64(1);
        let b = Fr::from_u64(1);
        for _ in 0..n {
            acc.fmadd(a, b);
        }
        assert_eq!(acc.reduce(), Fr::from_u64(n));
    }
}
