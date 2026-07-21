//! Wide-integer accumulator for BN254 Fr deferred reduction.
//!
//! Accumulates `sum += a * b` as folded 4x4 limb products, deferring
//! carry propagation and Montgomery reduction to a single call at the end.
//!
//! # Capacity
//!
//! Each Fr element is 4 limbs (256 bits). The product of two elements is
//! accumulated into eight positional `u128` slots. Carry headroom in each
//! slot lets the hot loop avoid carry propagation until reduction.

use crate::accumulator::Accumulator;
use crate::arkworks::bn254::Fr;
use ark_ff::BigInt;

use super::bn254_ops;

/// Folded 4x4 product accumulator for BN254 Fr deferred reduction.
///
/// Stores the running sum of Montgomery-form products in positional `u128`
/// slots. Converting to a field element requires one carry propagation pass
/// and one Montgomery reduction via [`Accumulator::reduce`].
#[derive(Clone, Copy)]
pub struct WideAccumulator {
    slots: [u128; 8],
}

impl Default for WideAccumulator {
    #[inline]
    fn default() -> Self {
        Self { slots: [0; 8] }
    }
}

impl Accumulator for WideAccumulator {
    type Element = Fr;

    #[inline(always)]
    fn add(&mut self, value: Fr) {
        self.fmadd(value, <Fr as num_traits::One>::one());
    }

    #[inline(always)]
    fn merge(&mut self, other: Self) {
        for (lhs, rhs) in self.slots.iter_mut().zip(other.slots) {
            *lhs += rhs;
        }
    }

    fn reduce(self) -> Fr {
        // The accumulator holds Montgomery-form products and/or elements.
        // Montgomery reduction divides product terms by R.
        Fr::from_inner(bn254_ops::from_montgomery_reduce(self.normalize()))
    }

    #[inline(always)]
    fn fmadd(&mut self, a: Fr, b: Fr) {
        let a = a.inner_limbs();
        let b = b.inner_limbs();
        for i in 0..4 {
            for j in 0..4 {
                let product = (a.0[i] as u128) * (b.0[j] as u128);
                self.slots[i + j] += (product as u64) as u128;
                self.slots[i + j + 1] += ((product >> 64) as u64) as u128;
            }
        }
    }
}

impl WideAccumulator {
    #[inline]
    fn normalize(self) -> BigInt<9> {
        let mut out = [0u64; 9];
        let mut carry = 0u128;
        for (index, slot) in self.slots.into_iter().enumerate() {
            let (sum, overflow) = slot.overflowing_add(carry);
            out[index] = sum as u64;
            carry = (sum >> 64) + ((overflow as u128) << 64);
        }
        out[8] = carry as u64;
        BigInt::new(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Accumulator, FromPrimitiveInt};

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
