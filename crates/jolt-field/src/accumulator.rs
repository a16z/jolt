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

/// Accumulates products with potentially deferred modular reduction.
///
/// The hot loop pattern `acc += a * b` repeated hundreds of times per output
/// slot dominates the CPU prover. Standard field arithmetic reduces mod p
/// after every multiply and every add. Implementations for specific fields
/// (e.g., BN254 Fr) can instead accumulate unreduced wide products and
/// reduce once at the end via [`reduce`](Self::reduce).
///
/// # Contract
///
/// - [`fmadd`](Self::fmadd) must be equivalent to `result += a * b` in the field.
/// - [`merge`](Self::merge) must be equivalent to adding another accumulator's
///   partial result (used for parallel reduction).
/// - [`reduce`](Self::reduce) must return the field element equal to the
///   accumulated sum of products.
pub trait FieldAccumulator: Default + Send + Sync {
    /// The field type this accumulator operates over.
    type Field: crate::Field;

    /// Fused multiply-add: `self += a * b` without intermediate reduction.
    fn fmadd(&mut self, a: Self::Field, b: Self::Field);

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
