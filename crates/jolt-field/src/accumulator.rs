//! Deferred-reduction accumulators.
//!
//! In sumcheck inner loops, many products are summed before the final result
//! is needed. [`RingAccumulator`] lets implementations defer modular reduction
//! by accumulating in wider integer types, reducing once at the end. This
//! amortizes the expensive reduction across hundreds of multiply-add steps.
//!
//! - [`NaiveAccumulator`] — fallback using standard field arithmetic.
//! - `WideAccumulator` (BN254, in `arkworks/`) — 9-limb wide integer accumulator
//!   that defers Montgomery reduction.

use crate::{AdditiveGroup, FromPrimitiveInt, RingCore};
use num_traits::One;

/// Accumulates additive values with potentially deferred reduction.
pub trait AdditiveAccumulator: Default + Copy + Send + Sync {
    /// The element type this accumulator reduces to.
    type Element: AdditiveGroup;

    /// Adds one element into the accumulator.
    fn add(&mut self, value: Self::Element);

    /// Merge another accumulator's partial sum into this one.
    fn merge(&mut self, other: Self);

    /// Finalize: reduce the accumulated value to an element.
    fn reduce(self) -> Self::Element;
}

/// Accumulates products with potentially deferred modular reduction.
///
/// The hot loop pattern `acc += a * b` repeated hundreds of times per output
/// slot dominates the CPU prover. Standard field arithmetic reduces mod p
/// after every multiply and every add. Implementations for specific fields
/// (e.g., BN254 Fr) can instead accumulate unreduced wide products and
/// reduce once at the end via [`AdditiveAccumulator::reduce`].
///
/// # Invariants
///
/// - [`fmadd`](Self::fmadd) must be equivalent to `result += a * b` in the field.
/// - [`merge`](Self::merge) must be equivalent to adding another accumulator's
///   partial result (used for parallel reduction).
/// - [`reduce`](Self::reduce) must return the field element equal to the
///   accumulated sum of products.
pub trait RingAccumulator: AdditiveAccumulator
where
    Self::Element: RingCore + FromPrimitiveInt,
{
    /// Fused multiply-add: `self += a * b` without intermediate reduction.
    fn fmadd(&mut self, a: Self::Element, b: Self::Element);

    /// Fused multiply-add with a `u8` scalar: `self += a * F::from(b)`.
    ///
    /// Implementations may override for optimized small-scalar multiplication
    /// (e.g., 4×1 limb schoolbook instead of 4×4).
    #[inline]
    fn fmadd_u8(&mut self, a: Self::Element, b: u8) {
        self.fmadd(a, Self::Element::from_u8(b));
    }

    /// Fused multiply-add with a `u64` scalar: `self += a * F::from(b)`.
    #[inline]
    fn fmadd_u64(&mut self, a: Self::Element, b: u64) {
        self.fmadd(a, Self::Element::from_u64(b));
    }

    /// Fused multiply-add with an `i64` scalar: `self += a * F::from(b)`.
    #[inline]
    fn fmadd_i64(&mut self, a: Self::Element, b: i64) {
        self.fmadd(a, Self::Element::from_i64(b));
    }

    /// Fused multiply-add with a `bool` scalar: `self += a` when `b` is true.
    #[inline]
    fn fmadd_bool(&mut self, a: Self::Element, b: bool) {
        if b {
            self.fmadd(a, <Self::Element as One>::one());
        }
    }
}

/// Naive accumulator using standard field arithmetic.
///
/// Every [`fmadd`](RingAccumulator::fmadd) performs a full modular multiply
/// and add. Used as a fallback for fields without wide-integer optimization.
#[derive(Clone, Copy)]
pub struct NaiveAccumulator<R: RingCore + FromPrimitiveInt>(R);

impl<R: RingCore + FromPrimitiveInt> Default for NaiveAccumulator<R> {
    #[inline]
    fn default() -> Self {
        Self(R::zero())
    }
}

impl<R: RingCore + FromPrimitiveInt> AdditiveAccumulator for NaiveAccumulator<R> {
    type Element = R;

    #[inline]
    fn add(&mut self, value: R) {
        self.0 += value;
    }

    #[inline]
    fn merge(&mut self, other: Self) {
        self.0 += other.0;
    }

    #[inline]
    fn reduce(self) -> R {
        self.0
    }
}

impl<R: RingCore + FromPrimitiveInt> RingAccumulator for NaiveAccumulator<R> {
    #[inline]
    fn fmadd(&mut self, a: R, b: R) {
        self.0 += a * b;
    }
}
