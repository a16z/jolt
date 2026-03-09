//! Wide-integer accumulator for BN254 Fr deferred reduction.
//!
//! Accumulates `sum += a * b` as 9-limb (576-bit) schoolbook products,
//! deferring the Montgomery reduction to a single call at the end. This
//! amortizes the ~40ns reduction cost across hundreds of multiply-add steps
//! in the sumcheck inner loop.
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

    #[inline]
    fn fmadd(&mut self, a: Fr, b: Fr) {
        self.limbs.fmadd::<4, 4>(&a.inner_limbs(), &b.inner_limbs());
    }

    #[inline]
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
