//! Deferred-reduction accumulator for fused multiply-add.
//!
//! In sumcheck inner loops, many products are summed before the final result
//! is needed. [`FieldAccumulator`] lets implementations defer modular reduction
//! by accumulating in wider integer types, reducing once at the end. This
//! amortizes the expensive reduction across hundreds of multiply-add steps.
//!
//! - [`NaiveAccumulator`] — fallback using standard field arithmetic.
//! - `WideAccumulator` (BN254, in `arkworks/`) — 9-limb wide integer accumulator
//!   that defers Montgomery reduction.

use crate::Field;
use num_traits::One;

/// Accumulates products with potentially deferred modular reduction.
///
/// The hot loop pattern `acc += a * b` repeated hundreds of times per output
/// slot dominates the CPU prover. Standard field arithmetic reduces mod p
/// after every multiply and every add. Implementations for specific fields
/// (e.g., BN254 Fr) can instead accumulate unreduced wide products and
/// reduce once at the end via [`reduce`](Self::reduce).
///
/// `Copy` is required so that accumulators can be used in const-generic
/// arrays (`[Self; N]`) for split-eq parallel folds.
///
/// # Contract
///
/// - [`fmadd`](Self::fmadd) must be equivalent to `result += a * b` in the field.
/// - [`merge`](Self::merge) must be equivalent to adding another accumulator's
///   partial result (used for parallel reduction).
/// - [`reduce`](Self::reduce) must return the field element equal to the
///   accumulated sum of products.
pub trait FieldAccumulator: Default + Copy + Send + Sync {
    /// The field type this accumulator operates over.
    type Field: crate::Field;

    /// Fused multiply-add: `self += a * b` without intermediate reduction.
    fn fmadd(&mut self, a: Self::Field, b: Self::Field);

    /// Fused multiply-add with a `u8` scalar: `self += a * F::from(b)`.
    ///
    /// Implementations may override for optimized small-scalar multiplication
    /// (e.g., 4×1 limb schoolbook instead of 4×4).
    #[inline]
    fn fmadd_u8(&mut self, a: Self::Field, b: u8) {
        self.fmadd(a, Self::Field::from_u8(b));
    }

    /// Fused multiply-add with a `u64` scalar: `self += a * F::from(b)`.
    #[inline]
    fn fmadd_u64(&mut self, a: Self::Field, b: u64) {
        self.fmadd(a, Self::Field::from_u64(b));
    }

    /// Fused multiply-add with an `i64` scalar: `self += a * F::from(b)`.
    #[inline]
    fn fmadd_i64(&mut self, a: Self::Field, b: i64) {
        self.fmadd(a, Self::Field::from_i64(b));
    }

    /// Fused multiply-add with a `bool` scalar: `self += a` when `b` is true.
    #[inline]
    fn fmadd_bool(&mut self, a: Self::Field, b: bool) {
        if b {
            self.fmadd(a, <Self::Field as One>::one());
        }
    }

    /// Merge another accumulator's partial sum into this one.
    ///
    /// Used in parallel reduction (e.g., Rayon fold+reduce) where each thread
    /// accumulates independently, then results are combined.
    fn merge(&mut self, other: Self);

    /// Finalize: reduce the accumulated value to a field element.
    fn reduce(self) -> Self::Field;
}

/// Naive accumulator using standard field arithmetic.
///
/// Every [`fmadd`](FieldAccumulator::fmadd) performs a full modular multiply
/// and add. Used as a fallback for fields without wide-integer optimization.
#[derive(Clone, Copy)]
pub struct NaiveAccumulator<F: Field>(F);

impl<F: Field> Default for NaiveAccumulator<F> {
    #[inline]
    fn default() -> Self {
        Self(F::zero())
    }
}

impl<F: Field> FieldAccumulator for NaiveAccumulator<F> {
    type Field = F;

    #[inline]
    fn fmadd(&mut self, a: F, b: F) {
        self.0 += a * b;
    }

    #[inline]
    fn merge(&mut self, other: Self) {
        self.0 += other.0;
    }

    #[inline]
    fn reduce(self) -> F {
        self.0
    }
}
