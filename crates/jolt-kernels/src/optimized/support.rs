//! Small shared primitives of the optimized kernels.

use jolt_field::Field;
use jolt_poly::{BindingOrder, EqPolynomial, Polynomial, UnivariatePoly};

/// `scale · eq(point, ·)` evaluations, big-endian (`point[0]` pairs the index
/// MSB) — the scaled variant of the reference tier's `eq_table`.
pub(crate) fn scaled_eq_table<F: Field>(point: &[F], scale: F) -> Vec<F> {
    EqPolynomial::<F>::evals(point, Some(scale))
}

/// `eq(point, ·)` evaluations, big-endian.
pub(crate) fn eq_table<F: Field>(point: &[F]) -> Vec<F> {
    EqPolynomial::<F>::evals(point, None)
}

/// The `(lo, hi)` sumcheck pair of a low-to-high-bound table at group `y`:
/// the two evaluations whose linear extension `lo + t·(hi − lo)` is the
/// table's per-round univariate restriction.
#[inline(always)]
pub(crate) fn pair<F: Field>(table: &Polynomial<F>, y: usize) -> (F, F) {
    let evals = table.evals();
    (evals[2 * y], evals[2 * y + 1])
}

/// Bind every table one round low-to-high, in place.
pub(crate) fn bind_all<'a, F: Field>(
    tables: impl IntoIterator<Item = &'a mut Polynomial<F>>,
    challenge: F,
) {
    for table in tables {
        table.bind_with_order(challenge, BindingOrder::LowToHigh);
    }
}

/// Assemble a round message from evaluations at `{0, 2, 3, .., degree}`,
/// recovering `s(1) = previous_claim − s(0)` — exactly the evaluation vector
/// the reference tier computes directly (its own round check pins
/// `s(0) + s(1) = previous_claim`), interpolated through the same
/// `UnivariatePoly::from_evals` path, so the coefficient vectors are
/// byte-identical on honest inputs.
pub(crate) fn round_poly_from_skipped_evals<F: Field>(
    evals_without_one: &[F],
    previous_claim: F,
) -> UnivariatePoly<F> {
    let mut evals = Vec::with_capacity(evals_without_one.len() + 1);
    evals.push(evals_without_one[0]);
    evals.push(previous_claim - evals_without_one[0]);
    evals.extend_from_slice(&evals_without_one[1..]);
    UnivariatePoly::from_evals(&evals)
}

/// Sum per-thread accumulator vectors elementwise.
#[cfg(feature = "parallel")]
pub(crate) fn merge_evals<F: Field>(mut left: Vec<F>, right: Vec<F>) -> Vec<F> {
    for (left, right) in left.iter_mut().zip(right) {
        *left += right;
    }
    left
}
