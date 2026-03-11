//! Compute kernel for the RA virtual sumcheck round polynomial.
//!
//! Computes `g(X) = Σ_j eq((r', X, j), r) · Π_i mle_i(X, j)` — the round
//! polynomial for the RA virtual sumcheck — using split-eq factored evaluation.
//!
//! Provides specialized kernels for arity `d ∈ {4, 8, 16, 32}` using
//! stack-allocated product evaluation, with a generic fallback for arbitrary `d`.

use jolt_field::{FieldAccumulator, WithChallenge};
use jolt_ir::toom_cook::{
    eval_linear_prod_assign, eval_prod_16_assign, eval_prod_32_assign, eval_prod_4_assign,
    eval_prod_8_assign,
};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::SplitEqEvaluator;

use super::ra_poly::RaPolynomial;

/// Computes the univariate polynomial `g(X) = Σ_j eq((r', X, j), r) · Π_i mle_i(X, j)`.
///
/// `claim` should equal `g(0) + g(1)`.
pub fn compute_mles_product_sum<F: WithChallenge>(
    mles: &[RaPolynomial<u8, F>],
    claim: F,
    eq_poly: &SplitEqEvaluator<F>,
) -> UnivariatePoly<F> {
    let d = mles.len();

    let sum_evals: Vec<F> = match d {
        16 => compute_mles_product_sum_evals_d16(mles, eq_poly),
        32 => compute_mles_product_sum_evals_d32(mles, eq_poly),
        _ => compute_mles_product_sum_evals_generic(mles, eq_poly),
    };

    finish_mles_product_sum_from_evals(&sum_evals, claim, eq_poly)
}

/// Computes `g(X) = Σ_j eq(·) · Σ_t weights[t] · Π_{k=0}^{m-1} mle_{t·m+k}(X, j)`.
///
/// Weighted sum-of-products variant for the RA virtual sumcheck with multiple
/// virtual polynomials. Each product group `t` contributes `weights[t] · Π_k mle_{t·m+k}`,
/// and the groups are summed before multiplication by the split-eq factor.
///
/// `claim` should equal `g(0) + g(1)`.
pub fn compute_mles_weighted_sop<F: WithChallenge>(
    mles: &[RaPolynomial<u8, F>],
    weights: &[F],
    n_products: usize,
    claim: F,
    eq_poly: &SplitEqEvaluator<F>,
) -> UnivariatePoly<F> {
    let m = mles.len() / n_products;
    debug_assert_eq!(mles.len(), n_products * m);
    debug_assert_eq!(weights.len(), n_products);

    let sum_evals = compute_mles_weighted_sop_evals_generic(mles, weights, m, eq_poly);
    finish_mles_product_sum_from_evals(&sum_evals, claim, eq_poly)
}

/// Inner evaluation kernel for the weighted sum-of-products.
///
/// Returns evaluations on the grid `[1, 2, ..., m-1, ∞]` where m is the
/// degree per product term.
#[inline]
fn compute_mles_weighted_sop_evals_generic<F: WithChallenge>(
    mles: &[RaPolynomial<u8, F>],
    weights: &[F],
    m: usize,
    eq_poly: &SplitEqEvaluator<F>,
) -> Vec<F> {
    let current_scalar = eq_poly.get_current_scalar();
    eq_poly
        .par_fold_out_in(
            || {
                (
                    vec![F::Accumulator::default(); m],
                    vec![(F::zero(), F::zero()); m],
                    vec![F::zero(); m],
                    vec![F::zero(); m],
                )
            },
            |(lanes, pairs, endpoints, sums), g, _x_in, e_in| {
                for s in sums.iter_mut() {
                    *s = F::zero();
                }

                for (t, &weight) in weights.iter().enumerate() {
                    let base = t * m;
                    for (idx, pair) in pairs.iter_mut().enumerate() {
                        let p0 = mles[base + idx].get_bound_coeff(2 * g);
                        let p1 = mles[base + idx].get_bound_coeff(2 * g + 1);
                        *pair = (p0, p1);
                    }
                    for e in endpoints.iter_mut() {
                        *e = F::zero();
                    }
                    eval_linear_prod_assign(pairs, endpoints);
                    for k in 0..m {
                        sums[k] += weight * endpoints[k];
                    }
                }

                for k in 0..m {
                    lanes[k].fmadd(e_in, sums[k]);
                }
            },
            |_x_out, e_out, (lanes, _, _, _)| {
                let mut outer = vec![F::Accumulator::default(); m];
                for (outer_k, inner_lane) in outer.iter_mut().zip(lanes.iter()) {
                    outer_k.fmadd(e_out, inner_lane.reduce());
                }
                outer
            },
            |mut a, b| {
                for k in 0..a.len() {
                    a[k].merge(b[k]);
                }
                a
            },
        )
        .into_iter()
        .map(|acc| acc.reduce() * current_scalar)
        .collect()
}

/// Generic split-eq fold computing evaluations of `g(X) / eq(X, r[round])`
/// on the grid `[1, 2, ..., d - 1, ∞]` for arbitrary `d`.
#[inline]
fn compute_mles_product_sum_evals_generic<F: WithChallenge>(
    mles: &[RaPolynomial<u8, F>],
    eq_poly: &SplitEqEvaluator<F>,
) -> Vec<F> {
    let d = mles.len();

    struct InnerAcc<F: WithChallenge> {
        lanes: Vec<F::Accumulator>,
        pairs: Vec<(F, F)>,
        endpoints: Vec<F>,
    }

    let current_scalar = eq_poly.get_current_scalar();
    eq_poly
        .par_fold_out_in(
            || InnerAcc {
                lanes: vec![F::Accumulator::default(); d],
                pairs: vec![(F::zero(), F::zero()); d],
                endpoints: vec![F::zero(); d],
            },
            |inner, g, _x_in, e_in| {
                for (idx, mle) in mles.iter().enumerate() {
                    let p0 = mle.get_bound_coeff(2 * g);
                    let p1 = mle.get_bound_coeff(2 * g + 1);
                    inner.pairs[idx] = (p0, p1);
                }

                eval_linear_prod_assign(&inner.pairs, &mut inner.endpoints);

                for k in 0..d {
                    inner.lanes[k].fmadd(e_in, inner.endpoints[k]);
                }
            },
            |_x_out, e_out, mut inner| {
                let mut outer = vec![F::Accumulator::default(); d];
                for (outer_k, inner_lane) in outer.iter_mut().zip(inner.lanes.iter()) {
                    outer_k.fmadd(e_out, inner_lane.reduce());
                }
                inner.lanes = outer;
                inner.lanes
            },
            |mut a, b| {
                for k in 0..a.len() {
                    a[k].merge(b[k]);
                }
                a
            },
        )
        .into_iter()
        .map(|acc| acc.reduce() * current_scalar)
        .collect()
}

macro_rules! impl_mles_product_sum_evals_d {
    ($fn_name:ident, $d:expr, $eval_prod:ident) => {
        #[inline]
        pub fn $fn_name<F: WithChallenge>(
            mles: &[RaPolynomial<u8, F>],
            eq_poly: &SplitEqEvaluator<F>,
        ) -> Vec<F> {
            debug_assert_eq!(mles.len(), $d);

            let current_scalar = eq_poly.get_current_scalar();

            let sum_evals_arr: [F; $d] = eq_poly.par_fold_out_in_unreduced::<$d>(&|g| {
                let pairs: [(F, F); $d] = core::array::from_fn(|i| {
                    let p0 = mles[i].get_bound_coeff(2 * g);
                    let p1 = mles[i].get_bound_coeff(2 * g + 1);
                    (p0, p1)
                });

                let mut endpoints = [F::zero(); $d];
                $eval_prod::<F>(&pairs, &mut endpoints);
                endpoints
            });

            sum_evals_arr
                .into_iter()
                .map(|x| x * current_scalar)
                .collect()
        }
    };
}

impl_mles_product_sum_evals_d!(compute_mles_product_sum_evals_d4, 4, eval_prod_4_assign);
impl_mles_product_sum_evals_d!(compute_mles_product_sum_evals_d8, 8, eval_prod_8_assign);
impl_mles_product_sum_evals_d!(compute_mles_product_sum_evals_d16, 16, eval_prod_16_assign);
impl_mles_product_sum_evals_d!(compute_mles_product_sum_evals_d32, 32, eval_prod_32_assign);

/// Sum-of-products variant: computes evaluations for
/// `q(X) = Σ_{t} Π_{i} mle_{t,i}(X, ·)` using a single `par_fold_out_in_unreduced` pass.
macro_rules! impl_mles_sum_of_products_evals_d {
    ($fn_name:ident, $d:expr, $eval_prod:ident) => {
        #[inline]
        pub fn $fn_name<F: WithChallenge>(
            mles: &[RaPolynomial<u8, F>],
            n_products: usize,
            eq_poly: &SplitEqEvaluator<F>,
        ) -> Vec<F> {
            debug_assert!(n_products > 0);
            debug_assert_eq!(mles.len(), n_products * $d);

            let current_scalar = eq_poly.get_current_scalar();

            let sum_evals_arr: [F; $d] = eq_poly.par_fold_out_in_unreduced::<$d>(&|g| {
                let mut sums = [F::zero(); $d];

                for t in 0..n_products {
                    let base = t * $d;

                    let pairs: [(F, F); $d] = core::array::from_fn(|i| {
                        let p0 = mles[base + i].get_bound_coeff(2 * g);
                        let p1 = mles[base + i].get_bound_coeff(2 * g + 1);
                        (p0, p1)
                    });

                    let mut endpoints = [F::zero(); $d];
                    $eval_prod::<F>(&pairs, &mut endpoints);

                    for k in 0..$d {
                        sums[k] += endpoints[k];
                    }
                }

                sums
            });

            sum_evals_arr
                .into_iter()
                .map(|x| x * current_scalar)
                .collect()
        }
    };
}

impl_mles_sum_of_products_evals_d!(
    compute_mles_product_sum_evals_sum_of_products_d4,
    4,
    eval_prod_4_assign
);
impl_mles_sum_of_products_evals_d!(
    compute_mles_product_sum_evals_sum_of_products_d8,
    8,
    eval_prod_8_assign
);
impl_mles_sum_of_products_evals_d!(
    compute_mles_product_sum_evals_sum_of_products_d16,
    16,
    eval_prod_16_assign
);

/// Recovers the full univariate polynomial `g(X) = eq(X, r[round]) · (interpolated quotient)`
/// from quotient evaluations on `[1, 2, ..., d - 1, ∞]`.
#[inline]
pub fn finish_mles_product_sum_from_evals<F: WithChallenge>(
    sum_evals: &[F],
    claim: F,
    eq_poly: &SplitEqEvaluator<F>,
) -> UnivariatePoly<F> {
    let r_round: F = eq_poly.get_current_w().into();
    let eq_eval_at_0 = F::one() - r_round;
    let eq_eval_at_1 = r_round;

    let eval_at_1 = sum_evals[0];
    let eval_at_0 = (claim - eq_eval_at_1 * eval_at_1) / eq_eval_at_0;

    let mut toom_evals = Vec::with_capacity(sum_evals.len() + 1);
    toom_evals.push(eval_at_0);
    toom_evals.extend_from_slice(sum_evals);
    let tmp_coeffs = UnivariatePoly::from_evals_toom(&toom_evals).into_coefficients();

    // Multiply by eq(X, r[round]) = (1 - r_round) + (2r_round - 1)·X
    let constant_coeff = F::one() - r_round;
    let x_coeff = r_round + r_round - F::one();
    let mut coeffs = vec![F::zero(); tmp_coeffs.len() + 1];
    for (i, coeff) in tmp_coeffs.into_iter().enumerate() {
        coeffs[i] += coeff * constant_coeff;
        coeffs[i + 1] += coeff * x_coeff;
    }

    UnivariatePoly::new(coeffs)
}
