//! Prover-side extension-opening-reduction sumcheck instance.
//!
//! Generic sumcheck proof containers and transcript drivers live in
//! `akita-sumcheck`. This module owns the Akita-specific EOR prover state over
//! witness and factor tables.

use akita_algebra::poly::fold_evals_in_place;
use akita_algebra::uni_poly::UniPoly;
use akita_algebra::EqPolynomial;
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
use akita_field::{AkitaError, ExtField, FieldCore, Zero};
use akita_sumcheck::SumcheckInstanceProver;
use akita_types::{
    checked_table_len, extension_opening_reduction_claim, num_rounds_from_table_len,
    project_tensor_factor_value, tensor_opening_split, validate_reduction_tables,
    EXTENSION_OPENING_REDUCTION_DEGREE,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Maximum number of sparse low-index rounds to keep in the lazy tensor factor.
///
/// The lazy factor caches one small state per low-bit assignment, avoiding a
/// full dense factor table while the sparse witness still has large support.
pub const SPARSE_TENSOR_FACTOR_MAX_LAZY_ROUNDS: usize = 12;

/// Degree-two round-message accumulator that honors a field's delayed-reduction
/// exactness contract (`HasUnreducedOps::DELAYED_PRODUCT_SUM_IS_EXACT`).
///
/// Every extension-opening reduction round needs two batch sums: the constant
/// coefficient `Σ c0·c1` and the quadratic coefficient `Σ q0·q1`. This trait
/// abstracts how those products are summed so the dense, fused-fold, and sparse
/// accumulation paths can all pick the right policy from one place:
///
/// - [`DelayedDeg2`] sums the wide `E::ProductAccum` products and reduces once
///   per round. This is only sound when `DELAYED_PRODUCT_SUM_IS_EXACT` is set,
///   i.e. the field's accumulator has been proven not to wrap for these batch
///   sizes (e.g. `FpExt4<Fp32>`, `FpExt2<Fp64>`).
/// - [`DirectDeg2`] reduces every product immediately via `Mul`, so the summed
///   coefficient is byte-identical to per-term reduction. This is the
///   contract-safe path for fields that leave `DELAYED_PRODUCT_SUM_IS_EXACT` at
///   its conservative `false` default.
///
/// Mirrors the same flag check already performed in
/// [`sparse::TensorEqualityFactor::factor_pair`], so the entire EOR prover
/// stays byte-identical to per-term `Mul` for any field whose delayed product
/// sum is not exact.
trait Deg2RoundAccum<E: FieldCore + HasUnreducedOps>: Sized + Send {
    /// Empty accumulator (both coefficients zero).
    fn zero() -> Self;
    /// Accumulate `lhs·rhs` into the constant coefficient.
    fn add_constant_product(&mut self, lhs: E, rhs: E);
    /// Accumulate `lhs·rhs` into the quadratic coefficient.
    fn add_quadratic_product(&mut self, lhs: E, rhs: E);
    /// Combine two partial accumulators (for parallel reduction).
    #[cfg(feature = "parallel")]
    fn merge(self, other: Self) -> Self;
    /// Reduce to the `(constant, quadratic)` round coefficients.
    fn finish(self) -> (E, E);
}

/// Delayed-reduction accumulator. Sound only when the field's
/// `DELAYED_PRODUCT_SUM_IS_EXACT` is `true`.
struct DelayedDeg2<E: HasUnreducedOps> {
    constant: E::ProductAccum,
    quadratic: E::ProductAccum,
}

impl<E: FieldCore + HasUnreducedOps> Deg2RoundAccum<E> for DelayedDeg2<E> {
    #[inline]
    fn zero() -> Self {
        Self {
            constant: E::ProductAccum::zero(),
            quadratic: E::ProductAccum::zero(),
        }
    }
    #[inline]
    fn add_constant_product(&mut self, lhs: E, rhs: E) {
        self.constant += lhs.mul_to_product_accum(rhs);
    }
    #[inline]
    fn add_quadratic_product(&mut self, lhs: E, rhs: E) {
        self.quadratic += lhs.mul_to_product_accum(rhs);
    }
    #[cfg(feature = "parallel")]
    #[inline]
    fn merge(self, other: Self) -> Self {
        Self {
            constant: self.constant + other.constant,
            quadratic: self.quadratic + other.quadratic,
        }
    }
    #[inline]
    fn finish(self) -> (E, E) {
        (
            E::reduce_product_accum(self.constant),
            E::reduce_product_accum(self.quadratic),
        )
    }
}

/// Per-term reduction accumulator: every product is reduced immediately, so the
/// summed coefficient is byte-identical to per-term `Mul`. Contract-safe
/// fallback for fields with `DELAYED_PRODUCT_SUM_IS_EXACT == false`.
struct DirectDeg2<E> {
    constant: E,
    quadratic: E,
}

impl<E: FieldCore + HasUnreducedOps> Deg2RoundAccum<E> for DirectDeg2<E> {
    #[inline]
    fn zero() -> Self {
        Self {
            constant: E::zero(),
            quadratic: E::zero(),
        }
    }
    #[inline]
    fn add_constant_product(&mut self, lhs: E, rhs: E) {
        self.constant += lhs * rhs;
    }
    #[inline]
    fn add_quadratic_product(&mut self, lhs: E, rhs: E) {
        self.quadratic += lhs * rhs;
    }
    #[cfg(feature = "parallel")]
    #[inline]
    fn merge(self, other: Self) -> Self {
        Self {
            constant: self.constant + other.constant,
            quadratic: self.quadratic + other.quadratic,
        }
    }
    #[inline]
    fn finish(self) -> (E, E) {
        (self.constant, self.quadratic)
    }
}

mod dense;
mod prover;
mod sparse;

pub use prover::ExtensionOpeningReductionProver;
pub use sparse::{ExtensionOpeningReductionTerm, SparseExtensionOpeningWitness};

pub(crate) use dense::{
    accumulate_dense_round, fold_dense_reduction_tables_in_place, fused_fold_and_accumulate,
};
