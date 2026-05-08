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
/// # Invariants
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

    /// Accumulate a field element with unit weight: `self += val`.
    #[inline]
    fn acc_add(&mut self, val: Self::Field) {
        self.fmadd(<Self::Field as One>::one(), val);
    }

    /// Merge another accumulator's partial sum into this one.
    ///
    /// Used in parallel reduction (e.g., Rayon fold+reduce) where each thread
    /// accumulates independently, then results are combined.
    fn merge(&mut self, other: Self);

    /// Finalize: reduce the accumulated value to a field element.
    fn reduce(self) -> Self::Field;
}

/// Accumulates products of field elements with small integer scalars.
///
/// This is the raw-scalar analogue of [`FieldAccumulator`]. It is useful when a
/// hot loop repeatedly adds terms of the form `a * n`, where `n` is a `u64` or
/// `u128` known outside the field. Implementations may defer the modular
/// reduction across many bucketed additions.
pub trait FieldScalarAccumulator: Default + Copy + Send + Sync {
    /// The field type this accumulator operates over.
    type Field: crate::Field;

    /// Accumulate a field element with unit scalar.
    fn add(&mut self, value: Self::Field);

    /// Fused multiply-add with a `u64` scalar: `self += value * scalar`.
    fn add_mul_u64(&mut self, value: Self::Field, scalar: u64);

    /// Fused multiply-add with a `u128` scalar: `self += value * scalar`.
    fn add_mul_u128(&mut self, value: Self::Field, scalar: u128);

    /// Merge another accumulator's partial sum into this one.
    fn merge(&mut self, other: Self);

    /// Finalize to a field element.
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

/// Naive raw-scalar accumulator using ordinary field arithmetic.
#[derive(Clone, Copy)]
pub struct NaiveScalarAccumulator<F: Field>(F);

impl<F: Field> Default for NaiveScalarAccumulator<F> {
    #[inline]
    fn default() -> Self {
        Self(F::zero())
    }
}

impl<F: Field> FieldScalarAccumulator for NaiveScalarAccumulator<F> {
    type Field = F;

    #[inline]
    fn add(&mut self, value: F) {
        self.0 += value;
    }

    #[inline]
    fn add_mul_u64(&mut self, value: F, scalar: u64) {
        self.0 += value.mul_u64(scalar);
    }

    #[inline]
    fn add_mul_u128(&mut self, value: F, scalar: u128) {
        self.0 += value.mul_u128(scalar);
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
