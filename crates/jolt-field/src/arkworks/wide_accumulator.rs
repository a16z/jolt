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

use crate::accumulator::{AdditiveAccumulator, RingAccumulator};
use crate::arkworks::bn254::Fr;
use ark_ff::BigInt;

use super::bn254_ops;

/// Folded 4x4 product accumulator for BN254 Fr deferred reduction.
///
/// Stores the running sum of Montgomery-form products in positional `u128`
/// slots. Converting to a field element requires one carry propagation pass
/// and one Montgomery reduction via [`AdditiveAccumulator::reduce`].
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

impl AdditiveAccumulator for WideAccumulator {
    type Element = Fr;

    /// Adds `value` in four limb additions instead of a full 4×4 `fmadd` by
    /// `one()`: the slots hold products of Montgomery forms (each `â·b̂ =
    /// ab·R²`), so a plain element must enter as `â·R`. Since `2^256 ≡ R
    /// (mod p)`, placing `â`'s limbs at positions 4..8 contributes
    /// `â·2^256 ≡ â·R (mod p)` — exactly what `fmadd(value, one())` adds,
    /// and [`reduce`](AdditiveAccumulator::reduce) folds arbitrary high
    /// limbs mod `p`, so the reduced element is identical.
    #[inline(always)]
    fn add(&mut self, value: Fr) {
        let limbs = value.inner_limbs();
        for (slot, limb) in self.slots[4..].iter_mut().zip(limbs.0) {
            *slot += limb as u128;
        }
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
}

impl RingAccumulator for WideAccumulator {
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
    use crate::{AdditiveAccumulator, FromPrimitiveInt};
    use num_traits::One;

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

    /// Deterministic full-range field elements (products of large scalars
    /// wrap the modulus, exercising all limbs).
    fn spread(seed: u64) -> Fr {
        let a = Fr::from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let b = Fr::from_u64(seed.wrapping_mul(0xBF58_476D_1CE4_E5B9) | 1);
        a * b * b + a
    }

    /// The shifted-limb `add` must agree with `fmadd(value, one())` (the
    /// definitionally correct form) for arbitrary elements, including when
    /// mixed with products.
    #[test]
    fn add_matches_fmadd_by_one() {
        for seed in 0..100u64 {
            let value = spread(seed);
            let a = spread(seed + 1000);
            let b = spread(seed + 2000);

            let mut via_add = WideAccumulator::default();
            via_add.fmadd(a, b);
            via_add.add(value);

            let mut via_fmadd = WideAccumulator::default();
            via_fmadd.fmadd(a, b);
            via_fmadd.fmadd(value, <Fr as One>::one());

            assert_eq!(via_add.reduce(), via_fmadd.reduce());
            assert_eq!(via_add.reduce(), a * b + value);
        }
    }

    /// Many shifted adds stay within slot headroom and reduce exactly.
    #[test]
    fn repeated_add_reduces_exactly() {
        let mut acc = WideAccumulator::default();
        let mut expected = Fr::from_u64(0);
        for seed in 0..1000u64 {
            let value = spread(seed);
            acc.add(value);
            expected += value;
        }
        assert_eq!(acc.reduce(), expected);
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
