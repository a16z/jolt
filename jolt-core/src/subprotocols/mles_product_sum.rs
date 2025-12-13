use crate::{
    field::{BarrettReduce, FMAdd, JoltField},
    poly::{ra_poly::RaPolynomial, split_eq_poly::GruenSplitEqPolynomial, unipoly::UniPoly},
    utils::accumulation::Acc5S,
};
use core::{mem::MaybeUninit, ptr};
use num_traits::Zero;

/// Computes the univariate polynomial `g(X) = sum_j eq((r', X, j), r) * prod_i mle_i(X, j)`.
///
/// Note `claim` should equal `g(0) + g(1)`.
pub fn compute_mles_product_sum<F: JoltField>(
    mles: &[RaPolynomial<u16, F>],
    claim: F,
    eq_poly: &GruenSplitEqPolynomial<F>,
) -> UniPoly<F> {
    let d = mles.len();

    // Evaluate g(X) / eq(X, r[round]) at [1, 2, ..., |mles| - 1, ∞] using split-eq fold.
    //
    // We dispatch on `d` to allow specialized fast paths (e.g., the fully
    // stack-allocated `d = 16` implementation) while keeping the final
    // interpolation logic centralized.
    let sum_evals: Vec<F> = match d {
        // Fully stack-allocated paths based on optimized interpolation kernels.
        13 => compute_mles_product_sum_evals_d13(mles, eq_poly),
        15 => compute_mles_product_sum_evals_d15(mles, eq_poly),
        16 => compute_mles_product_sum_evals_d16(mles, eq_poly),
        19 => compute_mles_product_sum_evals_d19(mles, eq_poly),
        22 => compute_mles_product_sum_evals_d22(mles, eq_poly),
        26 => compute_mles_product_sum_evals_d26(mles, eq_poly),
        32 => compute_mles_product_sum_evals_d32(mles, eq_poly),
        // Generic split-eq fold for all other arities.
        _ => compute_mles_product_sum_evals_generic(mles, eq_poly),
    };

    finish_mles_product_sum_from_evals(&sum_evals, claim, eq_poly)
}

/// Generic implementation of the split-eq fold that computes the evaluations
/// of `g(X) / eq(X, r[round])` on the grid `[1, 2, ..., d - 1, ∞]` for
/// arbitrary `d`.
#[inline]
fn compute_mles_product_sum_evals_generic<F: JoltField>(
    mles: &[RaPolynomial<u16, F>],
    eq_poly: &GruenSplitEqPolynomial<F>,
) -> Vec<F> {
    let d = mles.len();

    /// Per-`x_out` accumulator used inside `par_fold_out_in`.
    ///
    /// - `lanes[k]` accumulates the unreduced contribution for output lane `k`.
    /// - `pairs` and `endpoints` are scratch buffers reused across all `g`
    ///   values for a given `x_out`, to avoid repeated heap allocations.
    struct InnerAcc<F: JoltField> {
        lanes: Vec<F::Unreduced<9>>,
        pairs: Vec<(F, F)>,
        endpoints: Vec<F>,
    }

    let current_scalar = eq_poly.get_current_scalar();
    eq_poly
        .par_fold_out_in(
            // Allocate one InnerAcc per `x_out` lane; its scratch buffers are
            // reused across all inner `g` contributions for that lane.
            //
            // Note: `pairs` and `endpoints` are pre-sized to length `d` so that
            // we can update them by index without changing their length.
            || InnerAcc {
                lanes: vec![F::Unreduced::<9>::zero(); d],
                pairs: vec![(F::zero(), F::zero()); d],
                endpoints: vec![F::zero(); d],
            },
            |inner, g, _x_in, e_in| {
                // Build per-g pairs [(p0, p1); D] in-place, reusing `inner.pairs`.
                for (idx, mle) in mles.iter().enumerate() {
                    let p0 = mle.get_bound_coeff(2 * g);
                    let p1 = mle.get_bound_coeff(2 * g + 1);
                    inner.pairs[idx] = (p0, p1);
                }

                // Compute endpoints on the evaluation grid into the scratch
                // `endpoints` buffer. All entries are overwritten, so no need
                // to zero between uses.
                eval_linear_prod_assign(&inner.pairs, &mut inner.endpoints);

                // Accumulate with unreduced arithmetic
                for k in 0..d {
                    inner.lanes[k] += e_in.mul_unreduced::<9>(inner.endpoints[k]);
                }
            },
            |_x_out, e_out, mut inner| {
                // Reduce inner lanes, scale by e_out (unreduced), and reuse the
                // existing `lanes` allocation as the outer accumulator
                // `Vec<F::Unreduced<9>>`, avoiding an extra allocation.
                for k in 0..d {
                    let reduced_k = F::from_montgomery_reduce::<9>(inner.lanes[k]);
                    inner.lanes[k] = e_out.mul_unreduced::<9>(reduced_k);
                }
                inner.lanes
            },
            |mut a, b| {
                for k in 0..d {
                    a[k] += b[k];
                }
                a
            },
        )
        .into_iter()
        .map(|x| F::from_montgomery_reduce::<9>(x) * current_scalar)
        .collect()
}

macro_rules! impl_mles_product_sum_evals_d {
    ($fn_name:ident, $d:expr, $eval_prod:ident) => {
        #[inline]
        pub fn $fn_name<F: JoltField>(
            mles: &[RaPolynomial<u16, F>],
            eq_poly: &GruenSplitEqPolynomial<F>,
        ) -> Vec<F> {
            debug_assert_eq!(mles.len(), $d);

            let current_scalar = eq_poly.get_current_scalar();

            let sum_evals_arr: [F; $d] = eq_poly.par_fold_out_in_unreduced::<9, $d>(&|g| {
                // Build pairs[(p0, p1); D] on the stack.
                let pairs: [(F, F); $d] = core::array::from_fn(|i| {
                    let p0 = mles[i].get_bound_coeff(2 * g);
                    let p1 = mles[i].get_bound_coeff(2 * g + 1);
                    (p0, p1)
                });

                // Evaluate the product of the D linear polynomials on the
                // D-point grid [1, 2, ..., D - 1, ∞] using the specialized
                // kernel.
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
impl_mles_product_sum_evals_d!(compute_mles_product_sum_evals_d13, 13, eval_prod_13_assign);
impl_mles_product_sum_evals_d!(compute_mles_product_sum_evals_d15, 15, eval_prod_15_assign);
impl_mles_product_sum_evals_d!(compute_mles_product_sum_evals_d16, 16, eval_prod_16_assign);
impl_mles_product_sum_evals_d!(compute_mles_product_sum_evals_d19, 19, eval_prod_19_assign);
impl_mles_product_sum_evals_d!(compute_mles_product_sum_evals_d22, 22, eval_prod_22_assign);
impl_mles_product_sum_evals_d!(compute_mles_product_sum_evals_d26, 26, eval_prod_26_assign);
impl_mles_product_sum_evals_d!(compute_mles_product_sum_evals_d32, 32, eval_prod_32_assign);

/// Given the evaluations of `g(X) / eq(X, r[round])` on the grid
/// `[1, 2, ..., d - 1, ∞]`, recover the full univariate polynomial
/// `g(X) = eq(X, r[round]) * (interpolated quotient)` such that
/// `g(0) + g(1) = claim`.
#[inline]
pub fn finish_mles_product_sum_from_evals<F: JoltField>(
    sum_evals: &[F],
    claim: F,
    eq_poly: &GruenSplitEqPolynomial<F>,
) -> UniPoly<F> {
    // Get r[round] from the eq polynomial and compute the 1-bit equality polynomial
    // explicitly:
    //   eq(0, r_round) = 1 - r_round
    //   eq(1, r_round) = r_round
    let r_round = eq_poly.get_current_w();
    let eq_eval_at_0 = F::one() - r_round;
    let eq_eval_at_1 = r_round;

    // Obtain the eval at 0 from the claim.
    let eval_at_1 = sum_evals[0];
    let eval_at_0 = (claim - eq_eval_at_1 * eval_at_1) / eq_eval_at_0;

    // Interpolate the intermediate polynomial.
    let mut toom_evals = Vec::with_capacity(sum_evals.len() + 1);
    toom_evals.push(eval_at_0);
    toom_evals.extend_from_slice(sum_evals);
    let tmp_coeffs = UniPoly::from_evals_toom(&toom_evals).coeffs;

    // Add in the missing eq(X, r[round]) factor.
    // Note eq(X, r[round]) = (1 - r[round]) + (2r[round] - 1)X.
    let constant_coeff = F::one() - r_round;
    let x_coeff = r_round + r_round - F::one();
    let mut coeffs = vec![F::zero(); tmp_coeffs.len() + 1];
    for (i, coeff) in tmp_coeffs.into_iter().enumerate() {
        coeffs[i] += coeff * constant_coeff;
        coeffs[i + 1] += coeff * x_coeff;
    }

    UniPoly::from_coeff(coeffs)
}

/// Evaluate the product of linear polynomials on the grid `U_D = [1, 2, ..., D - 1, ∞]`.
///
/// - `D` is `pairs.len()`.
/// - Each pair satisfies `pairs[j] = (p_j(0), p_j(1))` for a linear `p_j`.
/// - This writes the evaluations of `P(x) = ∏_j p_j(x)` into `evals` in the
///   layout `[P(1), P(2), ..., P(D - 1), P(∞)]`.
pub fn eval_linear_prod_assign<F: JoltField>(pairs: &[(F, F)], evals: &mut [F]) {
    debug_assert_eq!(pairs.len(), evals.len());
    match pairs.len() {
        2 => {
            debug_assert!(evals.len() >= 2);
            // SAFETY: `pairs` has length 2 in this branch, so it is valid to
            // reinterpret the backing memory as `[(F, F); 2]`.
            let p = unsafe { &*(pairs.as_ptr() as *const [(F, F); 2]) };
            eval_prod_2_assign(p, evals)
        }
        3 => {
            debug_assert!(evals.len() >= 3);
            let p = unsafe { &*(pairs.as_ptr() as *const [(F, F); 3]) };
            eval_prod_3_assign(p, evals)
        }
        4 => {
            debug_assert!(evals.len() >= 4);
            let p = unsafe { &*(pairs.as_ptr() as *const [(F, F); 4]) };
            eval_prod_4_assign(p, evals)
        }
        5 => {
            debug_assert!(evals.len() >= 5);
            let p = unsafe { &*(pairs.as_ptr() as *const [(F, F); 5]) };
            eval_prod_5_assign(p, evals)
        }
        6 => {
            debug_assert!(evals.len() >= 6);
            let p = unsafe { &*(pairs.as_ptr() as *const [(F, F); 6]) };
            eval_prod_6_assign(p, evals)
        }
        7 => {
            debug_assert!(evals.len() >= 7);
            let p = unsafe { &*(pairs.as_ptr() as *const [(F, F); 7]) };
            eval_prod_7_assign(p, evals)
        }
        8 => {
            debug_assert!(evals.len() >= 8);
            let p = unsafe { &*(pairs.as_ptr() as *const [(F, F); 8]) };
            eval_prod_8_assign(p, evals)
        }
        13 => {
            debug_assert!(evals.len() >= 13);
            let p = unsafe { &*(pairs.as_ptr() as *const [(F, F); 13]) };
            eval_prod_13_assign(p, evals)
        }
        15 => {
            debug_assert!(evals.len() >= 15);
            let p = unsafe { &*(pairs.as_ptr() as *const [(F, F); 15]) };
            eval_prod_15_assign(p, evals)
        }
        16 => {
            debug_assert!(evals.len() >= 16);
            let p = unsafe { &*(pairs.as_ptr() as *const [(F, F); 16]) };
            eval_prod_16_assign(p, evals)
        }
        19 => {
            debug_assert!(evals.len() >= 19);
            let p = unsafe { &*(pairs.as_ptr() as *const [(F, F); 19]) };
            eval_prod_19_assign(p, evals)
        }
        22 => {
            debug_assert!(evals.len() >= 22);
            let p = unsafe { &*(pairs.as_ptr() as *const [(F, F); 22]) };
            eval_prod_22_assign(p, evals)
        }
        26 => {
            debug_assert!(evals.len() >= 26);
            let p = unsafe { &*(pairs.as_ptr() as *const [(F, F); 26]) };
            eval_prod_26_assign(p, evals)
        }
        32 => {
            debug_assert!(evals.len() >= 32);
            let p = unsafe { &*(pairs.as_ptr() as *const [(F, F); 32]) };
            eval_prod_32_assign(p, evals)
        }
        _ => eval_linear_prod_naive_assign(pairs, evals),
    }
}

/// Evaluate the product of linear polynomials on the grid `U_D = [1, 2, ..., D - 1, ∞]`.
///
/// - `D` is `pairs.len()`.
/// - Each pair satisfies `pairs[j] = (p_j(0), p_j(1))` for a linear `p_j`.
/// - This sums the evaluations of `P(x) = ∏_j p_j(x)` into `sums` in the
///   layout `[P(1), P(2), ..., P(D - 1), P(∞)]`.
pub fn eval_linear_prod_accumulate<F: JoltField>(pairs: &[(F, F)], sums: &mut [F::Unreduced<9>]) {
    debug_assert_eq!(pairs.len(), sums.len());
    match pairs.len() {
        2 => eval_prod_2_accumulate(pairs.try_into().unwrap(), sums),
        3 => eval_prod_3_accumulate(pairs.try_into().unwrap(), sums),
        5 => eval_prod_5_accumulate(pairs.try_into().unwrap(), sums),
        9 => eval_prod_9_accumulate(pairs.try_into().unwrap(), sums),
        // Implement more efficient than naive if these ever get used.
        4 | 6 | 7 | 8 => unimplemented!("n_pairs = {}", pairs.len()),
        _ => product_eval_univariate_naive_accumulate(pairs, sums),
    }
}

/// Evaluate the product of 2 linear polynomials at the small internal grid used
/// by the interpolation routines (returns values at 1, 2, and ∞).
///
/// This helper is only used inside higher-degree kernels and is not exposed
/// directly through `eval_linear_prod_assign`.
#[inline(always)]
fn eval_linear_prod_2_internal<F: JoltField>((p0, p1): (F, F), (q0, q1): (F, F)) -> (F, F, F) {
    let p_inf = p1 - p0;
    let p2 = p_inf + p1;
    let q_inf = q1 - q0;
    let q2 = q_inf + q1;
    let r1 = p1 * q1;
    let r2 = p2 * q2;
    let r_inf = p_inf * q_inf;
    (r1, r2, r_inf)
}

/// Evaluate the product of 2 linear polynomials on `U_2 = [1, ∞]`.
///
/// Given `p[j] = (p_j(0), p_j(1))` and `P(x) = ∏_j p_j(x)`, this writes:
/// - `outputs[0] = P(1)`
/// - `outputs[1] = P(∞) = ∏_j (p_j(1) - p_j(0))`
#[inline(always)]
fn eval_prod_2_assign<F: JoltField>(p: &[(F, F); 2], outputs: &mut [F]) {
    outputs[0] = p[0].1 * p[1].1; // 1
    outputs[1] = (p[0].1 - p[0].0) * (p[1].1 - p[1].0); // ∞
}

/// Evaluate the product of 2 linear polynomials on `U_2 = [1, ∞]`.
///
/// Given `p[j] = (p_j(0), p_j(1))` and `P(x) = ∏_j p_j(x)`, this writes:
/// - `outputs[0] += P(1)`
/// - `outputs[1] += P(∞) = ∏_j (p_j(1) - p_j(0))`
#[inline(always)]
fn eval_prod_2_accumulate<F: JoltField>(p: &[(F, F); 2], outputs: &mut [F::Unreduced<9>]) {
    outputs[0] += p[0].1.mul_unreduced::<9>(p[1].1); // 1
    outputs[1] += (p[0].1 - p[0].0).mul_unreduced::<9>(p[1].1 - p[1].0); // ∞
}

/// Evaluate the product of 3 linear polynomials on `U_3 = [1, 2, ∞]`.
///
/// Given `pairs[j] = (p_j(0), p_j(1))` and `P(x) = ∏_j p_j(x)`, this writes:
/// - `outputs[0] = P(1)`
/// - `outputs[1] = P(2)`
/// - `outputs[2] = P(∞)`
#[inline(always)]
fn eval_prod_3_assign<F: JoltField>(pairs: &[(F, F); 3], outputs: &mut [F]) {
    let (a1, a2, a_inf) = eval_linear_prod_2_internal(pairs[0], pairs[1]);
    let (b0, b1) = pairs[2];
    let b_inf = b1 - b0;
    let b2 = b1 + b_inf;
    outputs[0] = a1 * b1;
    outputs[1] = a2 * b2;
    outputs[2] = a_inf * b_inf;
}

/// Evaluate the product of 3 linear polynomials on `U_3 = [1, 2, ∞]`.
///
/// Given `pairs[j] = (p_j(0), p_j(1))` and `P(x) = ∏_j p_j(x)`, this writes:
/// - `outputs[0] += P(1)`
/// - `outputs[1] += P(2)`
/// - `outputs[2] += P(∞)`
#[inline(always)]
fn eval_prod_3_accumulate<F: JoltField>(pairs: &[(F, F); 3], outputs: &mut [F::Unreduced<9>]) {
    let (a1, a2, a_inf) = eval_linear_prod_2_internal(pairs[0], pairs[1]);
    let (b0, b1) = pairs[2];
    let b_inf = b1 - b0;
    let b2 = b1 + b_inf;
    outputs[0] += a1.mul_unreduced::<9>(b1);
    outputs[1] += a2.mul_unreduced::<9>(b2);
    outputs[2] += a_inf.mul_unreduced::<9>(b_inf);
}

/// Evaluate the product of 4 linear polynomials at the internal interpolation
/// grid used by the higher-degree kernels.
///
/// Returns 5 values corresponding to evaluations at points `[1, 2, 3, 4, ∞]`.
/// Only a subset of these points are exposed in `eval_prod_4_assign`, which
/// adheres to the public `U_4 = [1, 2, 3, ∞]` grid.
#[inline]
fn eval_linear_prod_4_internal<F: JoltField>(p: [(F, F); 4]) -> (F, F, F, F, F) {
    let (a1, a2, a_inf) = eval_linear_prod_2_internal(p[0], p[1]);
    let a3 = ex2(&[a1, a2], &a_inf);
    let a4 = ex2(&[a2, a3], &a_inf);
    let (b1, b2, b_inf) = eval_linear_prod_2_internal(p[2], p[3]);
    let b3 = ex2(&[b1, b2], &b_inf);
    let b4 = ex2(&[b2, b3], &b_inf);
    (a1 * b1, a2 * b2, a3 * b3, a4 * b4, a_inf * b_inf)
}

/// Evaluate the product of 4 linear polynomials on `U_4 = [1, 2, 3, ∞]`.
///
/// Given `p[j] = (p_j(0), p_j(1))` and `P(x) = ∏_j p_j(x)`, this writes:
/// - `outputs[0] = P(1)`
/// - `outputs[1] = P(2)`
/// - `outputs[2] = P(3)`
/// - `outputs[3] = P(∞)`
fn eval_prod_4_assign<F: JoltField>(p: &[(F, F); 4], outputs: &mut [F]) {
    let (a1, a2, a_inf) = eval_linear_prod_2_internal(p[0], p[1]);
    let a3 = ex2(&[a1, a2], &a_inf);
    let (b1, b2, b_inf) = eval_linear_prod_2_internal(p[2], p[3]);
    let b3 = ex2(&[b1, b2], &b_inf);
    outputs[0] = a1 * b1; // 1
    outputs[1] = a2 * b2; // 2
    outputs[2] = a3 * b3; // 3
    outputs[3] = a_inf * b_inf; // ∞
}

/// Evaluate the product of 5 linear polynomials on `U_5 = [1, 2, 3, 4, ∞]`.
///
/// The product is split into a size-2 prefix and size-3 suffix. The suffix uses
/// a single polynomial times a pair, so it reuses the existing `d = 2`
/// quadratic extrapolator.
fn eval_prod_5_assign<F: JoltField>(p: &[(F, F); 5], outputs: &mut [F]) {
    debug_assert!(outputs.len() >= 5);

    // Prefix: two polynomials → degree-2 product evaluated on 1..4 via `ex2`.
    let (a1, a2, a_inf) = eval_linear_prod_2_internal(p[0], p[1]);
    let a3 = ex2(&[a1, a2], &a_inf);
    let a4 = ex2(&[a2, a3], &a_inf);

    // Suffix: single polynomial times a pair.
    let (tail1, tail2) = (p[3], p[4]);
    let (r1, r2, r_inf) = eval_linear_prod_2_internal(tail1, tail2);
    let r3 = ex2(&[r1, r2], &r_inf);
    let r4 = ex2(&[r2, r3], &r_inf);

    let (lin0, lin1) = p[2];
    let delta = lin1 - lin0;
    let l1 = lin1;
    let l2 = l1 + delta;
    let l3 = l2 + delta;
    let l4 = l3 + delta;
    let l_inf = delta;

    let b1 = l1 * r1;
    let b2 = l2 * r2;
    let b3 = l3 * r3;
    let b4 = l4 * r4;
    let b_inf = l_inf * r_inf;

    outputs[0] = a1 * b1; // 1
    outputs[1] = a2 * b2; // 2
    outputs[2] = a3 * b3; // 3
    outputs[3] = a4 * b4; // 4
    outputs[4] = a_inf * b_inf; // ∞
}

/// Evaluate the product of 5 linear polynomials on `U_5 = [1, 2, 3, 4, ∞]`.
///
/// The product is split into a size-2 prefix and size-3 suffix. The suffix uses
/// a single polynomial times a pair, so it reuses the existing `d = 2`
/// quadratic extrapolator.
pub fn eval_prod_5_accumulate<F: JoltField>(p: &[(F, F); 5], outputs: &mut [F::Unreduced<9>]) {
    debug_assert!(outputs.len() >= 5);

    // Prefix: two polynomials → degree-2 product evaluated on 1..4 via `ex2`.
    let (a1, a2, a_inf) = eval_linear_prod_2_internal(p[0], p[1]);
    let a3 = ex2(&[a1, a2], &a_inf);
    let a4 = ex2(&[a2, a3], &a_inf);

    // Suffix: single polynomial times a pair.
    let (tail1, tail2) = (p[3], p[4]);
    let (r1, r2, r_inf) = eval_linear_prod_2_internal(tail1, tail2);
    let r3 = ex2(&[r1, r2], &r_inf);
    let r4 = ex2(&[r2, r3], &r_inf);

    let (lin0, lin1) = p[2];
    let delta = lin1 - lin0;
    let l1 = lin1;
    let l2 = l1 + delta;
    let l3 = l2 + delta;
    let l4 = l3 + delta;
    let l_inf = delta;

    let b1 = l1 * r1;
    let b2 = l2 * r2;
    let b3 = l3 * r3;
    let b4 = l4 * r4;
    let b_inf = l_inf * r_inf;

    outputs[0] += a1.mul_unreduced::<9>(b1); // 1
    outputs[1] += a2.mul_unreduced::<9>(b2); // 2
    outputs[2] += a3.mul_unreduced::<9>(b3); // 3
    outputs[3] += a4.mul_unreduced::<9>(b4); // 4
    outputs[4] += a_inf.mul_unreduced::<9>(b_inf); // ∞
}

/// Internal evaluator for the product of `d` linear polynomials on the grid
/// `[1, 2, ..., d, ∞]`.
///
/// Returns `[P(1), P(2), ..., P(d), P(∞)]` where
/// `P(x) = ∏_j (p_j(0) + (p_j(1) - p_j(0)) x)`.
macro_rules! impl_eval_linear_prod_internal {
    ($fn_name:ident, $d:expr) => {
        #[inline]
        fn $fn_name<F: JoltField>(pairs: [(F, F); $d]) -> [F; $d + 1] {
            let mut cur_vals_pinfs: [(F, F); $d] = [(F::zero(), F::zero()); $d];
            for (i, (p0, p1)) in pairs.into_iter().enumerate() {
                let pinf = p1 - p0;
                cur_vals_pinfs[i] = (p1, pinf);
            }

            let mut out = [F::zero(); $d + 1];

            // Evaluate at x = 1..D by sliding x ↦ x + 1 using the precomputed pinfs.
            for idx in 0..$d {
                let mut iter = cur_vals_pinfs.iter();
                let (first_val, _) = iter.next().expect("d > 0");
                let mut acc = *first_val;
                for (cur_val, _) in iter {
                    acc *= *cur_val;
                }
                out[idx] = acc;

                for (cur_val, pinf) in cur_vals_pinfs.iter_mut() {
                    *cur_val += *pinf;
                }
            }

            // Evaluate at infinity (product of leading coefficients).
            let mut iter = cur_vals_pinfs.iter();
            let (_, first_pinf) = iter.next().expect("d > 0");
            let mut acc_inf = *first_pinf;
            for (_, pinf) in iter {
                acc_inf *= *pinf;
            }
            out[$d] = acc_inf;

            out
        }
    };
}

impl_eval_linear_prod_internal!(eval_linear_prod_5_internal, 5);
impl_eval_linear_prod_internal!(eval_linear_prod_6_internal, 6);
impl_eval_linear_prod_internal!(eval_linear_prod_7_internal, 7);
impl_eval_linear_prod_internal!(eval_linear_prod_9_internal, 9);
impl_eval_linear_prod_internal!(eval_linear_prod_10_internal, 10);

/// Evaluate the product of 6 linear polynomials on `U_6 = [1, 2, 3, 4, 5, ∞]`.
///
/// This uses the internal 6-way sliding kernel to compute `[P(1), ..., P(5), P(∞)]`
/// without going through the generic naive evaluator.
#[inline]
fn eval_prod_6_assign<F: JoltField>(p: &[(F, F); 6], outputs: &mut [F]) {
    debug_assert!(outputs.len() >= 6);

    // Build (current value, leading coefficient) per polynomial.
    let mut cur_vals_pinfs: [(F, F); 6] = [(F::zero(), F::zero()); 6];
    for (i, (p0, p1)) in p.iter().copied().enumerate() {
        let pinf = p1 - p0;
        cur_vals_pinfs[i] = (p1, pinf);
    }

    // Evaluate at x = 1..5 by sliding x ↦ x + 1 using the precomputed pinfs.
    for idx in 0..5 {
        let mut iter = cur_vals_pinfs.iter();
        let (first_val, _) = iter.next().expect("d > 0");
        let mut acc = *first_val;
        for (cur_val, _) in iter {
            acc *= *cur_val;
        }
        outputs[idx] = acc;

        for (cur_val, pinf) in cur_vals_pinfs.iter_mut() {
            *cur_val += *pinf;
        }
    }

    // Evaluate at infinity (product of leading coefficients).
    let mut iter = cur_vals_pinfs.iter();
    let (_, first_pinf) = iter.next().expect("d > 0");
    let mut acc_inf = *first_pinf;
    for (_, pinf) in iter {
        acc_inf *= *pinf;
    }
    outputs[5] = acc_inf;
}

/// Evaluate the product of 7 linear polynomials on `U_7 = [1, 2, 3, 4, 5, 6, ∞]`.
///
/// This uses the internal 7-way sliding kernel to compute `[P(1), ..., P(6), P(∞)]`
/// without going through the generic naive evaluator.
#[inline]
fn eval_prod_7_assign<F: JoltField>(p: &[(F, F); 7], outputs: &mut [F]) {
    debug_assert!(outputs.len() >= 7);

    // Build (current value, leading coefficient) per polynomial.
    let mut cur_vals_pinfs: [(F, F); 7] = [(F::zero(), F::zero()); 7];
    for (i, (p0, p1)) in p.iter().copied().enumerate() {
        let pinf = p1 - p0;
        cur_vals_pinfs[i] = (p1, pinf);
    }

    // Evaluate at x = 1..6 by sliding x ↦ x + 1.
    for idx in 0..6 {
        let mut iter = cur_vals_pinfs.iter();
        let (first_val, _) = iter.next().expect("d > 0");
        let mut acc = *first_val;
        for (cur_val, _) in iter {
            acc *= *cur_val;
        }
        outputs[idx] = acc;

        for (cur_val, pinf) in cur_vals_pinfs.iter_mut() {
            *cur_val += *pinf;
        }
    }

    // Evaluate at infinity (product of leading coefficients).
    let mut iter = cur_vals_pinfs.iter();
    let (_, first_pinf) = iter.next().expect("d > 0");
    let mut acc_inf = *first_pinf;
    for (_, pinf) in iter {
        acc_inf *= *pinf;
    }
    outputs[6] = acc_inf;
}

/// Evaluate the product of 11 linear polynomials on `U_11 = [1, 2, ..., 10, ∞]`.
///
/// We split 11 = 5 + 6 into two halves:
///   - A(x): product of the first 5 polynomials (degree ≤ 5)
///   - B(x): product of the last 6 polynomials (degree ≤ 6)
///
/// The evaluation proceeds as:
///   1) Evaluate A on [1..5, ∞] with `eval_linear_prod_5_internal`.
///   2) Evaluate B on [1..6, ∞] with `eval_linear_prod_6_internal`.
///   3) Expand A to [1..10] using the degree-5 sliding kernel `ex5`.
///   4) Expand B to [1..10] using the degree-6 sliding kernel `ex6`.
///   5) Multiply pointwise and set P(∞) = A(∞) * B(∞).
#[inline]
fn eval_prod_11_assign<F: JoltField>(p: &[(F, F); 11], outputs: &mut [F]) {
    debug_assert!(outputs.len() >= 11);

    // SAFETY: `p[0..5]` and `p[5..11]` are disjoint slices of lengths 5 and 6.
    let a_vals =
        eval_linear_prod_5_internal::<F>(unsafe { *(p[0..5].as_ptr() as *const [(F, F); 5]) });
    let b_vals =
        eval_linear_prod_6_internal::<F>(unsafe { *(p[5..11].as_ptr() as *const [(F, F); 6]) });

    let a_inf = a_vals[5];
    let b_inf = b_vals[6];

    // Expand A to [1..10] using ex5 sliding from the base [1..5, ∞].
    let mut a_mu: MaybeUninit<[F; 10]> = MaybeUninit::uninit();
    let a_ptr = a_mu.as_mut_ptr();
    let a_slice_ptr = unsafe { (*a_ptr).as_mut_ptr() };

    // Seed A(1..5).
    unsafe {
        ptr::copy_nonoverlapping(a_vals.as_ptr(), a_slice_ptr, 5);
    }

    let a_inf5_fact = a_inf.mul_u64(120u64); // 5!
    for i in 0..5 {
        unsafe {
            let win_ptr = a_slice_ptr.add(i) as *const [F; 5];
            let next = ex5::<F>(&*win_ptr, a_inf5_fact);
            ptr::write(a_slice_ptr.add(5 + i), next);
        }
    }

    // Expand B to [1..10] using ex6 sliding from the base [1..6, ∞].
    let mut b_mu: MaybeUninit<[F; 10]> = MaybeUninit::uninit();
    let b_ptr = b_mu.as_mut_ptr();
    let b_slice_ptr = unsafe { (*b_ptr).as_mut_ptr() };

    unsafe {
        ptr::copy_nonoverlapping(b_vals.as_ptr(), b_slice_ptr, 6);
    }

    let b_inf6_fact = b_inf.mul_u64(720u64); // 6!
    for i in 0..4 {
        unsafe {
            let win_ptr = b_slice_ptr.add(i) as *const [F; 6];
            let next = ex6::<F>(&*win_ptr, b_inf6_fact);
            ptr::write(b_slice_ptr.add(6 + i), next);
        }
    }

    // SAFETY: all indices 0..9 have been written for both halves.
    let a_full = unsafe { a_mu.assume_init() }; // A(1..10)
    let b_full = unsafe { b_mu.assume_init() }; // B(1..10)

    // Pointwise product P(x) = A(x) * B(x) on [1..10].
    for i in 0..10 {
        outputs[i] = a_full[i] * b_full[i];
    }

    // ∞ evaluation is the product of leading coefficients.
    outputs[10] = a_inf * b_inf;
}

#[inline]
fn eval_prod_13_assign<F: JoltField>(p: &[(F, F); 13], outputs: &mut [F]) {
    debug_assert!(outputs.len() >= 13);

    // Recursive structure: split 13 = 6 + 7 and treat each half as a separate
    // product polynomial A(x), B(x). We:
    //   1) Evaluate A and B on their internal grids [1..6, ∞] and [1..7, ∞].
    //   2) Use degree-specific extrapolators `ex6` and `ex7` to slide each half
    //      forward so that we obtain A(x), B(x) for x = 1..12.
    //   3) Multiply pointwise to recover P(x) = A(x) * B(x) on U_13's finite
    //      grid [1..12].
    //   4) Compute P(∞) as the product of the halves' leading coefficients.

    // Step 1: internal half-evaluations.
    //
    // SAFETY: `p[0..6]` and `p[6..13]` are disjoint slices of lengths 6 and 7.
    let a_vals =
        eval_linear_prod_6_internal::<F>(unsafe { *(p[0..6].as_ptr() as *const [(F, F); 6]) });
    let b_vals =
        eval_linear_prod_7_internal::<F>(unsafe { *(p[6..13].as_ptr() as *const [(F, F); 7]) });

    let a_inf = a_vals[6];
    let b_inf = b_vals[7];

    // Step 2: expand A and B from their base grids to 1..12 using sliding
    // extrapolators ex6/ex7.
    let mut a_mu: MaybeUninit<[F; 12]> = MaybeUninit::uninit();
    let mut b_mu: MaybeUninit<[F; 12]> = MaybeUninit::uninit();

    let a_ptr = a_mu.as_mut_ptr();
    let b_ptr = b_mu.as_mut_ptr();

    let a_slice_ptr = unsafe { (*a_ptr).as_mut_ptr() };
    let b_slice_ptr = unsafe { (*b_ptr).as_mut_ptr() };

    unsafe {
        // Seed A with A(1..6).
        ptr::copy_nonoverlapping(a_vals.as_ptr(), a_slice_ptr, 6);
        // Seed B with B(1..7).
        ptr::copy_nonoverlapping(b_vals.as_ptr(), b_slice_ptr, 7);
    }

    let a_inf6_fact = a_inf.mul_u64(720u64); // 6!
    let b_inf7_fact = b_inf.mul_u64(5040u64); // 7!

    // A: compute A(7..12) using ex6 on sliding windows of length 6.
    for i in 0..6 {
        unsafe {
            let win_ptr = a_slice_ptr.add(i) as *const [F; 6];
            let next = ex6::<F>(&*win_ptr, a_inf6_fact);
            ptr::write(a_slice_ptr.add(6 + i), next);
        }
    }

    // B: compute B(8..12) using ex7 on sliding windows of length 7.
    for i in 0..5 {
        unsafe {
            let win_ptr = b_slice_ptr.add(i) as *const [F; 7];
            let next = ex7::<F>(&*win_ptr, b_inf7_fact);
            ptr::write(b_slice_ptr.add(7 + i), next);
        }
    }

    // SAFETY: all indices 0..12 for both halves have been written above.
    let a_full = unsafe { a_mu.assume_init() }; // A(1..12)
    let b_full = unsafe { b_mu.assume_init() }; // B(1..12)

    // Step 3: pointwise multiply to obtain P(x) on [1..12].
    for i in 0..12 {
        outputs[i] = a_full[i] * b_full[i];
    }

    // Step 4: ∞ evaluation is just the product of the halves' leading coeffs.
    outputs[12] = a_inf * b_inf;
}

/// Evaluate the product of 15 linear polynomials on `U_15 = [1, 2, ..., 14, ∞]`.
///
/// This degree corresponds to `d = ceil(128 / 9)` (K = 9) and is implemented via
/// a 7 + 8 split:
///   - A(x): product of the first 7 polynomials (degree ≤ 7)
///   - B(x): product of the last 8 polynomials (degree ≤ 8)
///
/// The evaluation proceeds as:
///   1) Evaluate A on `[1..7, ∞]` with `eval_linear_prod_7_internal`.
///   2) Evaluate B on `[1..8, ∞]` with `eval_linear_prod_8_internal`.
///   3) Expand A to `[1..14]` using the degree-7 sliding kernel `ex7`.
///   4) Expand B to `[1..14]` using the degree-8 sliding kernel `ex8`.
///   5) Multiply pointwise and set `P(∞) = A(∞) * B(∞)`.
fn eval_prod_15_assign<F: JoltField>(p: &[(F, F); 15], outputs: &mut [F]) {
    debug_assert!(outputs.len() >= 15);

    // SAFETY: `p[0..7]` and `p[7..15]` are disjoint slices of length 7 and 8.
    let a_vals =
        eval_linear_prod_7_internal::<F>(unsafe { *(p[0..7].as_ptr() as *const [(F, F); 7]) });
    let b_vals =
        eval_linear_prod_8_internal::<F>(unsafe { *(p[7..15].as_ptr() as *const [(F, F); 8]) });

    let a_inf = a_vals[7];
    let b_inf = b_vals[8];

    // Expand A from [1..7, ∞] to [1..14] using ex7 sliding.
    let mut a_mu: MaybeUninit<[F; 14]> = MaybeUninit::uninit();
    let a_ptr = a_mu.as_mut_ptr();
    let a_slice_ptr = unsafe { (*a_ptr).as_mut_ptr() };

    // Seed A(1..7).
    unsafe {
        ptr::copy_nonoverlapping(a_vals.as_ptr(), a_slice_ptr, 7);
    }

    let a_inf7_fact = a_inf.mul_u64(5040u64); // 7!
    for i in 0..7 {
        unsafe {
            let win_ptr = a_slice_ptr.add(i) as *const [F; 7];
            let next = ex7::<F>(&*win_ptr, a_inf7_fact);
            ptr::write(a_slice_ptr.add(7 + i), next);
        }
    }

    // Expand B from [1..8, ∞] to [1..14] using ex8 sliding.
    let mut b_mu: MaybeUninit<[F; 14]> = MaybeUninit::uninit();
    let b_ptr = b_mu.as_mut_ptr();
    let b_slice_ptr = unsafe { (*b_ptr).as_mut_ptr() };

    // Seed B(1..8).
    unsafe {
        ptr::copy_nonoverlapping(b_vals.as_ptr(), b_slice_ptr, 8);
    }

    let b_inf40320 = b_inf.mul_u64(40320u64); // 8!
    for i in 0..6 {
        unsafe {
            let win_ptr = b_slice_ptr.add(i) as *const [F; 8];
            let next = ex8::<F>(&*win_ptr, b_inf40320);
            ptr::write(b_slice_ptr.add(8 + i), next);
        }
    }

    // SAFETY: all indices 0..13 for both halves have been written above.
    let a_full = unsafe { a_mu.assume_init() }; // A(1..14)
    let b_full = unsafe { b_mu.assume_init() }; // B(1..14)

    // Pointwise product P(x) = A(x) * B(x) on [1..14].
    for i in 0..14 {
        outputs[i] = a_full[i] * b_full[i];
    }

    // ∞ evaluation is the product of leading coefficients.
    outputs[14] = a_inf * b_inf;
}

/// Evaluate the product of 19 linear polynomials on `U_19 = [1, 2, ..., 18, ∞]`.
///
/// This degree corresponds to `d = ceil(128 / 7)` (K = 7). It is currently a
/// direct wrapper around the generic `eval_linear_prod_naive_assign` and will
/// be specialized in follow-up work.
fn eval_prod_19_assign<F: JoltField>(p: &[(F, F); 19], outputs: &mut [F]) {
    debug_assert!(outputs.len() >= 19);

    // Factor 19 = 9 + 10 into two halves:
    //   - A(x): product of the first 9 polynomials (degree ≤ 9)
    //   - B(x): product of the last 10 polynomials (degree ≤ 10)
    //
    // We:
    //   1) Evaluate A on [1..9, ∞] via `eval_linear_prod_9_internal`.
    //   2) Evaluate B on [1..10, ∞] via `eval_linear_prod_10_internal`.
    //   3) Expand each half to [1..18, ∞] using degree-specific sliding kernels
    //      `ex9` / `ex10` via `expand9_to_u19` / `expand10_to_u19`.
    //   4) Multiply pointwise and set P(∞) = A(∞) * B(∞).

    // SAFETY: `p[0..9]` and `p[9..19]` are non-overlapping slices of length 9
    // and 10, respectively.
    let a_vals =
        eval_linear_prod_9_internal::<F>(unsafe { *(p[0..9].as_ptr() as *const [(F, F); 9]) });
    let b_vals =
        eval_linear_prod_10_internal::<F>(unsafe { *(p[9..19].as_ptr() as *const [(F, F); 10]) });

    let a_inf = a_vals[9];
    let b_inf = b_vals[10];

    // Build base grids for A and B on [1..9] and [1..10], respectively.
    let mut a_base: [F; 9] = [F::zero(); 9];
    a_base.copy_from_slice(&a_vals[0..9]);

    let mut b_base: [F; 10] = [F::zero(); 10];
    b_base.copy_from_slice(&b_vals[0..10]);

    let a_full = expand9_to_u19::<F>(&a_base, a_inf);
    let b_full = expand10_to_u19::<F>(&b_base, b_inf);

    // Multiply pointwise on [1..18] and set ∞ value.
    for i in 0..19 {
        let mut v = a_full[i];
        v *= b_full[i];
        outputs[i] = v;
    }
}

/// Evaluate the product of 22 linear polynomials on `U_22 = [1, 2, ..., 21, ∞]`.
///
/// This degree corresponds to `d = ceil(128 / 6)` (K = 6). We factor the
/// product evenly as 11 + 11 polynomials:
///   - A(x): product of the first 11 linears (degree ≤ 11)
///   - B(x): product of the last 11 linears (degree ≤ 11)
///
/// Each 11-way half is itself built from a 5 + 6 split (see
/// `eval_prod_11_assign`), and we:
///   1) Obtain A and B bases on [1..11, ∞] via `eval_half_11_base`.
///   2) Expand each half to [1..21, ∞] using the degree-11 sliding kernel
///      `ex11` via `expand11_to_u22`.
///   3) Multiply pointwise and set P(∞) = A(∞) * B(∞).
#[inline]
fn eval_prod_22_assign<F: JoltField>(p: &[(F, F); 22], outputs: &mut [F]) {
    debug_assert!(outputs.len() >= 22);

    // First 11 polynomials → half A.
    //
    // SAFETY: `p[0..11]` and `p[11..22]` are non-overlapping slices of length
    // 11, so the casts to `[(F, F); 11]` are valid.
    let (a11_base, a_inf) =
        eval_half_11_base::<F>(unsafe { *(p[0..11].as_ptr() as *const [(F, F); 11]) });
    let a_full = expand11_to_u22::<F>(&a11_base, a_inf);

    // Second 11 polynomials → half B.
    let (b11_base, b_inf) =
        eval_half_11_base::<F>(unsafe { *(p[11..22].as_ptr() as *const [(F, F); 11]) });
    let b_full = expand11_to_u22::<F>(&b11_base, b_inf);

    // Combine half A and half B pointwise to get the full product evaluated on
    // [1..21, ∞].
    for i in 0..22 {
        let mut v = a_full[i];
        v *= b_full[i];
        outputs[i] = v;
    }
}

/// Evaluate the product of 26 linear polynomials on `U_26 = [1, 2, ..., 25, ∞]`.
///
/// This degree corresponds to `d = ceil(128 / 5)` (K = 5). The implementation
/// will eventually mirror the recursive structure of the 16- and 32-way kernels;
/// for now we just forward to the generic naive evaluator.
#[inline]
fn eval_prod_26_assign<F: JoltField>(p: &[(F, F); 26], outputs: &mut [F]) {
    debug_assert!(outputs.len() >= 26);

    // First 13 polynomials → half A.
    //
    // SAFETY: `p[0..13]` and `p[13..26]` are non-overlapping slices of length
    // 13, so the casts to `[(F, F); 13]` are valid.
    let (a13_base, a_inf) =
        eval_half_13_base::<F>(unsafe { *(p[0..13].as_ptr() as *const [(F, F); 13]) });
    let a_full = expand13_to_u26::<F>(&a13_base, a_inf);

    // Second 13 polynomials → half B.
    let (b13_base, b_inf) =
        eval_half_13_base::<F>(unsafe { *(p[13..26].as_ptr() as *const [(F, F); 13]) });
    let b_full = expand13_to_u26::<F>(&b13_base, b_inf);

    // Combine half A and half B pointwise to get the full product evaluated on
    // [1..25, ∞].
    for i in 0..26 {
        let mut v = a_full[i];
        v *= b_full[i];
        outputs[i] = v;
    }
}

/// Evaluate the product of 8 linear polynomials on the internal interpolation
/// grid used by the higher-degree kernels.
///
/// Returns 9 values: the first 8 correspond to points `[1, 2, ..., 8]`, and the
/// 9th entry is the value at ∞. Only a subset of these points are exposed in
/// `eval_prod_8_assign`, which adheres to the public `U_8 = [1, 2, ..., 7, ∞]`
/// grid.
///
/// # Safety
///
/// Internally this function reinterprets disjoint 4-element slices of `p` as
/// `[(F, F); 4]` using pointer casts. This is sound because:
/// - `p` is a fixed-size `[ (F, F); 8 ]`, so `p[0..4]` and `p[4..8]` each have
///   length 4 and are properly aligned.
/// - The two halves are non-overlapping.
fn eval_linear_prod_8_internal<F: JoltField>(p: [(F, F); 8]) -> [F; 9] {
    #[inline]
    fn batch_helper<F: JoltField>(f0: F, f1: F, f2: F, f3: F, f_inf: F) -> (F, F, F, F) {
        let f_inf6 = f_inf.mul_u64(6);
        let (f4, f5) = ex4_2(&[f0, f1, f2, f3], &f_inf6);
        let (f6, f7) = ex4_2(&[f2, f3, f4, f5], &f_inf6);
        (f4, f5, f6, f7)
    }

    // SAFETY: `p[0..4]` and `p[4..8]` are disjoint slices of length 4 each.
    // We reinterpret them as `[ (F, F); 4 ]` to pass by value into
    // `eval_linear_prod_4_internal`, which avoids per-element copies while
    // preserving alignment.
    let (a1, a2, a3, a4, a_inf) =
        eval_linear_prod_4_internal(unsafe { *(p[0..4].as_ptr() as *const [(F, F); 4]) });
    let (a5, a6, a7, a8) = batch_helper(a1, a2, a3, a4, a_inf);
    let (b1, b2, b3, b4, b_inf) =
        eval_linear_prod_4_internal(unsafe { *(p[4..8].as_ptr() as *const [(F, F); 4]) });

    let (b5, b6, b7, b8) = batch_helper(b1, b2, b3, b4, b_inf);

    [
        a1 * b1,
        a2 * b2,
        a3 * b3,
        a4 * b4,
        a5 * b5,
        a6 * b6,
        a7 * b7,
        a8 * b8,
        a_inf * b_inf,
    ]
}

/// Evaluate the product of 8 linear polynomials on `U_8 = [1, 2, ..., 7, ∞]`.
///
/// Given `p[j] = (p_j(0), p_j(1))` and `P(x) = ∏_j p_j(x)`, this writes:
/// - `outputs[0..7] = [P(1), P(2), ..., P(7), P(∞)]`.
///
/// # Safety
///
/// As in `eval_linear_prod_8_internal`, this function reinterprets
/// `p[0..4]` and `p[4..8]` as `[(F, F); 4]`. The invariants are:
/// - `p` is a fixed-size `[ (F, F); 8 ]`, so both sub-slices have length 4 and
///   correct alignment.
/// - The sub-slices are non-overlapping.
fn eval_prod_8_assign<F: JoltField>(p: &[(F, F); 8], outputs: &mut [F]) {
    #[inline]
    fn batch_helper<F: JoltField>(f0: F, f1: F, f2: F, f3: F, f_inf: F) -> (F, F, F) {
        let f_inf6 = f_inf.mul_u64(6);
        let (f4, f5) = ex4_2(&[f0, f1, f2, f3], &f_inf6);
        let f6 = ex4(&[f2, f3, f4, f5], &f_inf6);
        (f4, f5, f6)
    }

    // SAFETY: as in `eval_linear_prod_8_internal`, reinterpreting `p[0..4]` and
    // `p[4..8]` as fixed-size arrays is sound because those sub-slices are
    // exactly length 4 and properly aligned.
    let (a1, a2, a3, a4, a_inf) =
        eval_linear_prod_4_internal(unsafe { *(p[0..4].as_ptr() as *const [(F, F); 4]) });
    let (a5, a6, a7) = batch_helper(a1, a2, a3, a4, a_inf);
    let (b1, b2, b3, b4, b_inf) =
        eval_linear_prod_4_internal(unsafe { *(p[4..8].as_ptr() as *const [(F, F); 4]) });
    let (b5, b6, b7) = batch_helper(b1, b2, b3, b4, b_inf);

    outputs[0] = a1 * b1;
    outputs[1] = a2 * b2;
    outputs[2] = a3 * b3;
    outputs[3] = a4 * b4;
    outputs[4] = a5 * b5;
    outputs[5] = a6 * b6;
    outputs[6] = a7 * b7;
    outputs[7] = a_inf * b_inf;
}

/// Evaluate the product of 8 linear polynomials on `U_8 = [1, 2, ..., 7, ∞]`.
///
/// Given `p[j] = (p_j(0), p_j(1))` and `P(x) = ∏_j p_j(x)`, this writes:
/// - `outputs[0..7] = [P(1), P(2), ..., P(7), P(∞)]`.
///
/// # Safety
///
/// As in `eval_linear_prod_8_internal`, this function reinterprets
/// `p[0..4]` and `p[4..8]` as `[(F, F); 4]`. The invariants are:
/// - `p` is a fixed-size `[ (F, F); 8 ]`, so both sub-slices have length 4 and
///   correct alignment.
/// - The sub-slices are non-overlapping.
fn eval_prod_9_accumulate<F: JoltField>(p: &[(F, F); 9], outputs: &mut [F::Unreduced<9>]) {
    // TODO: Implement more optimal way to do this.
    // 5x4 split probably better than current 8x1 split.
    let p8 = p[0..8].try_into().unwrap();
    let [a1, a2, a3, a4, a5, a6, a7, a8, a_inf] = eval_linear_prod_8_internal(p8);

    let (lin0, lin1) = p[8];
    let delta = lin1 - lin0;
    let l1 = lin1;
    let l2 = l1 + delta;
    let l3 = l2 + delta;
    let l4 = l3 + delta;
    let l5 = l4 + delta;
    let l6 = l5 + delta;
    let l7 = l6 + delta;
    let l8 = l7 + delta;
    let l_inf = delta;

    outputs[0] += a1.mul_unreduced::<9>(l1);
    outputs[1] += a2.mul_unreduced::<9>(l2);
    outputs[2] += a3.mul_unreduced::<9>(l3);
    outputs[3] += a4.mul_unreduced::<9>(l4);
    outputs[4] += a5.mul_unreduced::<9>(l5);
    outputs[5] += a6.mul_unreduced::<9>(l6);
    outputs[6] += a7.mul_unreduced::<9>(l7);
    outputs[7] += a8.mul_unreduced::<9>(l8);
    outputs[8] += a_inf.mul_unreduced::<9>(l_inf);
}

/// Evaluate the product of 16 linear polynomials on `U_16 = [1, 2, ..., 15, ∞]`.
///
/// This kernel first evaluates each half (8 polynomials) on an internal grid,
/// then uses `ex8` to slide the window and reconstruct the remaining points.
/// The final `outputs` slice has layout `[P(1), ..., P(15), P(∞)]`.
///
/// # Safety
///
/// This function uses pointer casts and `MaybeUninit` internally with the
/// following invariants:
/// - `p[0..8]` and `p[8..16]` are non-overlapping slices of length 8 and are
///   reinterpreted as `[(F, F); 8]` for `eval_linear_prod_8_internal`.
/// - The scratch buffers backed by `MaybeUninit<[F; 16]>` are fully written
///   at all indices that are later read, before `assume_init` is called.
/// - `outputs` must have length at least 16 (checked via `debug_assert!`).
fn eval_prod_16_assign<F: JoltField>(p: &[(F, F); 16], outputs: &mut [F]) {
    debug_assert!(outputs.len() >= 16);

    // First, split the 16 polynomials into two groups of 8 and evaluate each group
    // on the internal 8-point grid using the `eval_linear_prod_8_internal` kernel.
    //
    // SAFETY: `p[0..8]` and `p[8..16]` are non-overlapping slices of exactly
    // 8 elements, so reinterpreting them as `[(F, F); 8]` is valid.
    let a = eval_linear_prod_8_internal(unsafe { *(p[0..8].as_ptr() as *const [(F, F); 8]) });
    let b = eval_linear_prod_8_internal(unsafe { *(p[8..16].as_ptr() as *const [(F, F); 8]) });

    // Emit indices 1..8 directly by multiplying the corresponding values from
    // the two halves.
    for i in 0..8 {
        let v = a[i] * b[i];
        outputs[i] = v;
    }

    // Slide both 8-wide windows using pointer windows over a scratch buffer
    // (no per-iteration shifts). `a[8]`/`b[8]` are the ∞ evaluations for each
    // half; they are scaled so that `ex8` can reconstruct the next point using
    // factorial weights (see `ex8` comment).
    let a_inf40320 = a[8].mul_u64(40320);
    let b_inf40320 = b[8].mul_u64(40320);

    // Scratch buffers: seed first 8 entries with the currently-known values,
    // and pre-write slot 15 with the ∞ value for the final window.
    let mut aw_mu: MaybeUninit<[F; 16]> = MaybeUninit::uninit();
    let mut bw_mu: MaybeUninit<[F; 16]> = MaybeUninit::uninit();

    let aw_ptr = aw_mu.as_mut_ptr();
    let bw_ptr = bw_mu.as_mut_ptr();

    // SAFETY: `aw_ptr`/`bw_ptr` point to uninitialized `[F; 16]` storage. We
    // immediately treat them as mutable slices of length 16 and initialize all
    // indices we later read (0..8 + 8..15 and the pre-written 15th entry).
    let aw_slice_ptr = unsafe { (*aw_ptr).as_mut_ptr() };
    let bw_slice_ptr = unsafe { (*bw_ptr).as_mut_ptr() };

    unsafe {
        ptr::copy_nonoverlapping(a.as_ptr(), aw_slice_ptr, 8);
        ptr::write(aw_slice_ptr.add(15), a[8]);

        ptr::copy_nonoverlapping(b.as_ptr(), bw_slice_ptr, 8);
        ptr::write(bw_slice_ptr.add(15), b[8]);
    }

    for i in 0..7 {
        // Window over `aw[i..i+8]` and `bw[i..i+8]` without bounds checks.
        // Each window represents 8 consecutive evaluations for the half-product,
        // and `ex8` extrapolates the next value from that window and the
        // pre-scaled ∞ value.
        let na = unsafe {
            let win_a_ptr = aw_slice_ptr.add(i) as *const [F; 8];
            ex8::<F>(&*win_a_ptr, a_inf40320)
        };
        let nb = unsafe {
            let win_b_ptr = bw_slice_ptr.add(i) as *const [F; 8];
            ex8::<F>(&*win_b_ptr, b_inf40320)
        };

        let v = na * nb;
        outputs[8 + i] = v;

        // Append newly computed elements for subsequent windows. This grows the
        // scratch buffer so that `aw[i+1..i+9]` / `bw[i+1..i+9]` are always
        // valid windows for the next iteration.
        unsafe {
            ptr::write(aw_slice_ptr.add(8 + i), na);
            ptr::write(bw_slice_ptr.add(8 + i), nb);
        }
    }

    // The ∞ evaluation for the full product is just the product of the ∞
    // evaluations of each half.
    let v_inf = a[8] * b[8];
    outputs[15] = v_inf;
}

#[inline(always)]
fn ex2<F: JoltField>(f: &[F; 2], f_inf: &F) -> F {
    // Given a quadratic `P` with:
    //   f[0] = P(1), f[1] = P(2), f_inf = P(∞) = leading coefficient,
    // this extrapolates `P(3)` on the natural integer grid.
    dbl(f[1] + f_inf) - f[0]
}

#[inline]
fn ex4<F: JoltField>(f: &[F; 4], f_inf6: &F) -> F {
    // Extrapolate the next value of a degree-4 polynomial on the natural grid.
    //
    // Inputs:
    //   f[i]   = P(x + i) for i = 0..3
    //   f_inf6 = 6 * P(∞) = 6 * a4 where a4 is the leading coefficient
    //
    // This implements the natural-grid coefficients for target x+4:
    //   [1, -4, 6, -4] and uses 4! * a4 = 24 * a4 encoded in `f_inf6`.
    let mut t = *f_inf6;
    t += f[3];
    t -= f[2];
    t += f[1];
    dbl_assign(&mut t);
    t -= f[2];
    dbl_assign(&mut t);
    t -= f[0];
    t
}

#[inline]
fn ex4_2<F: JoltField>(f: &[F; 4], f_inf6: &F) -> (F, F) {
    // Variant of `ex4` that jointly extrapolates the next two values of a
    // degree-4 polynomial on the natural grid, reusing intermediate work.
    //
    // Inputs have the same interpretation as in `ex4`.
    let f3m2 = f[3] - f[2];
    let mut f4 = *f_inf6;
    f4 += f3m2;
    f4 += f[1];
    dbl_assign(&mut f4);
    f4 -= f[2];
    dbl_assign(&mut f4);
    f4 -= f[0];

    let mut f5 = f4 - f3m2 + f_inf6;
    dbl_assign(&mut f5);
    f5 -= f[3];
    dbl_assign(&mut f5);
    f5 -= f[1];

    (f4, f5)
}

#[inline]
fn ex5<F: JoltField>(f: &[F; 5], f_inf5_fact: F) -> F {
    // Extrapolate the next value of a degree-5 polynomial on the natural grid.
    //
    // Inputs:
    //   f[i]        = P(x + i) for i = 0..4
    //   f_inf5_fact = 5! * P(∞) = 120 * a5 where a5 is the leading coefficient
    //
    // Coefficients derived from the 6th-row binomial weights with alternating
    // signs give:
    //
    //   P(x + 5) =  1  * P(x + 0)
    //            -  5  * P(x + 1)
    //            + 10  * P(x + 2)
    //            - 10  * P(x + 3)
    //            +  5  * P(x + 4)
    //            +  5! * P(∞).
    let mut acc: Acc5S<F> = Acc5S::zero();

    acc.fmadd(&f[0], &1u64);
    acc.fmadd(&f[1], &(-5i64));
    acc.fmadd(&f[2], &10u64);
    acc.fmadd(&f[3], &(-10i64));
    acc.fmadd(&f[4], &5u64);

    // Coefficient +1 on 5! * P(∞).
    acc.pos += *f_inf5_fact.as_unreduced_ref();

    acc.barrett_reduce()
}

#[inline]
fn ex6<F: JoltField>(f: &[F; 6], f_inf6_fact: F) -> F {
    // Extrapolate the next value of a degree-6 polynomial on the natural grid.
    //
    // Inputs:
    //   f[i]        = P(x + i) for i = 0..5
    //   f_inf6_fact = 6! * P(∞) = 720 * a6 where a6 is the leading coefficient
    //
    // Coefficients derived from the 7th-row binomial weights with alternating
    // signs give:
    //
    //   P(x + 6) = -P(x + 0)
    //            +  6 P(x + 1)
    //            - 15 P(x + 2)
    //            + 20 P(x + 3)
    //            - 15 P(x + 4)
    //            +  6 P(x + 5)
    //            +  6! * P(∞).
    //
    // We use a signed accumulator to defer Montgomery reduction.
    let mut acc: Acc5S<F> = Acc5S::zero();

    let s6 = f[1] + f[5];
    acc.fmadd(&s6, &6u64);
    let s15 = f[2] + f[4];
    acc.fmadd(&s15, &(-15i64));
    acc.fmadd(&f[3], &20u64);

    // Coefficient -1 on f[0].
    acc.neg += *f[0].as_unreduced_ref();
    // Coefficient +1 on 6! * P(∞).
    acc.pos += *f_inf6_fact.as_unreduced_ref();

    acc.barrett_reduce()
}

#[inline]
fn ex7<F: JoltField>(f: &[F; 7], f_inf7_fact: F) -> F {
    // Extrapolate the next value of a degree-7 polynomial on the natural grid.
    //
    // Inputs:
    //   f[i]        = P(x + i) for i = 0..6
    //   f_inf7_fact = 7! * P(∞) = 5040 * a7 where a7 is the leading coefficient
    //
    // Coefficients obtained from binomial weights yield:
    //
    //   P(x + 7) =  1 P(x + 0)
    //            -  7 P(x + 1)
    //            + 21 P(x + 2)
    //            - 35 P(x + 3)
    //            + 35 P(x + 4)
    //            - 21 P(x + 5)
    //            +  7 P(x + 6)
    //            +  7! * P(∞).
    //
    // Again we employ a signed accumulator for fewer reductions.
    let mut acc: Acc5S<F> = Acc5S::zero();

    acc.fmadd(&f[0], &1u64);
    acc.fmadd(&f[1], &(-7i64));
    acc.fmadd(&f[2], &21u64);
    acc.fmadd(&f[3], &(-35i64));
    acc.fmadd(&f[4], &35u64);
    acc.fmadd(&f[5], &(-21i64));
    acc.fmadd(&f[6], &7u64);

    // Coefficient +1 on 7! * P(∞).
    acc.pos += *f_inf7_fact.as_unreduced_ref();

    acc.barrett_reduce()
}

#[inline(always)]
fn ex8<F: JoltField>(f: &[F; 8], f_inf40320: F) -> F {
    // Extrapolate `P(9)` from `f[i] = P(i+1)` for a degree-8 polynomial.
    //
    // The coefficients correspond to the 9th-row binomial weights with
    // alternating signs, grouped to minimize fmadd calls:
    //
    //   P(9) = 8(f[1] + f[7])
    //        + 56(f[3] + f[5])
    //        - 28(f[2] + f[6])
    //        - 70 f[4]
    //        - f[0]
    //        + f_inf40320
    //
    // where `f_inf40320 = 8! * a8` and `a8` is the leading coefficient.
    // We use a signed accumulator in Montgomery form to reduce only once.
    let mut acc: Acc5S<F> = Acc5S::zero();
    let t1 = f[1] + f[7];
    acc.fmadd(&t1, &8u64);
    let t2 = f[3] + f[5];
    acc.fmadd(&t2, &56u64);
    // Coefficient +1: add the unreduced representation directly to the positive
    // accumulator instead of going through `mul_u64_unreduced(1)`.
    acc.pos += *f_inf40320.as_unreduced_ref();

    let t3 = f[2] + f[6];
    acc.fmadd(&t3, &(-28i64));
    acc.fmadd(&f[4], &(-70i64));
    // Coefficient -1: add the unreduced representation directly to the negative
    // accumulator instead of going through `mul_u64_unreduced(1)` with a sign.
    acc.neg += *f[0].as_unreduced_ref();

    acc.barrett_reduce()
}

#[inline]
fn ex9<F: JoltField>(f: &[F; 9], f_inf9_fact: F) -> F {
    // Extrapolate the next value of a degree-9 polynomial on the natural grid.
    //
    // Inputs:
    //   f[i]        = P(x + i) for i = 0..8
    //   f_inf9_fact = 9! * P(∞) = 362880 * a9 where a9 is the leading coefficient
    //
    // Coefficients derived from binomial weights with alternating signs give:
    //
    //   P(x + 9) =  1  * P(x + 0)
    //            -  9  * P(x + 1)
    //            + 36  * P(x + 2)
    //            - 84  * P(x + 3)
    //            + 126 * P(x + 4)
    //            - 126 * P(x + 5)
    //            + 84  * P(x + 6)
    //            - 36  * P(x + 7)
    //            +  9  * P(x + 8)
    //            +  9! * P(∞).
    //
    // Group symmetric terms:
    //   9  (f[8] - f[1])
    //   36 (f[2] - f[7])
    //   84 (f[6] - f[3])
    //   126(f[4] - f[5])
    // and handle coefficient +1 on f[0] and on 9! * P(∞) via direct accumulator
    // updates.
    let mut acc: Acc5S<F> = Acc5S::zero();

    // +1 * f[0]
    acc.pos += *f[0].as_unreduced_ref();

    let t9 = f[8] - f[1];
    acc.fmadd(&t9, &9u64);

    let t36 = f[2] - f[7];
    acc.fmadd(&t36, &36u64);

    let t84 = f[6] - f[3];
    acc.fmadd(&t84, &84u64);

    let t126 = f[4] - f[5];
    acc.fmadd(&t126, &126u64);

    // +1 * (9! * P(∞))
    acc.pos += *f_inf9_fact.as_unreduced_ref();

    acc.barrett_reduce()
}

#[inline]
fn ex10<F: JoltField>(f: &[F; 10], f_inf10_fact: F) -> F {
    // Extrapolate the next value of a degree-10 polynomial on the natural grid.
    //
    // Inputs:
    //   f[i]         = P(x + i) for i = 0..9
    //   f_inf10_fact = 10! * P(∞) = 3628800 * a10 where a10 is the leading coefficient
    //
    // Coefficients derived from binomial weights with alternating signs give:
    //
    //   P(x + 10) = -1  * P(x + 0)
    //              +10  * P(x + 1)
    //              -45  * P(x + 2)
    //             +120  * P(x + 3)
    //             -210  * P(x + 4)
    //             +252  * P(x + 5)
    //             -210  * P(x + 6)
    //             +120  * P(x + 7)
    //              -45  * P(x + 8)
    //              +10  * P(x + 9)
    //              +10! * P(∞).
    //
    // Group symmetric terms where coefficients match:
    //   10 (f[1] + f[9])
    //   -45(f[2] + f[8])
    //   120(f[3] + f[7])
    //   -210(f[4] + f[6])
    //   252 f[5]
    // and handle the -1 on f[0] and +1 on 10! * P(∞) via direct accumulator
    // updates.
    let mut acc: Acc5S<F> = Acc5S::zero();

    // -1 * f[0]
    acc.neg += *f[0].as_unreduced_ref();

    let s10 = f[1] + f[9];
    acc.fmadd(&s10, &10u64);

    let s45 = f[2] + f[8];
    acc.fmadd(&s45, &(-45i64));

    let s120 = f[3] + f[7];
    acc.fmadd(&s120, &120u64);

    let s210 = f[4] + f[6];
    acc.fmadd(&s210, &(-210i64));

    acc.fmadd(&f[5], &252u64);

    // +1 * (10! * P(∞))
    acc.pos += *f_inf10_fact.as_unreduced_ref();

    acc.barrett_reduce()
}

#[inline]
fn ex11<F: JoltField>(f: &[F; 11], f_inf11_fact: F) -> F {
    // Extrapolate the next value of a degree-11 polynomial on the natural grid.
    //
    // Inputs:
    //   f[i]         = P(x + i) for i = 0..10
    //   f_inf11_fact = 11! * P(∞) = 39916800 * a11 where a11 is the leading coefficient
    //
    // Coefficients obtained from binomial weights yield:
    //
    //   P(x + 11) =  1  * P(x + 0)
    //             - 11  * P(x + 1)
    //             + 55  * P(x + 2)
    //             - 165 * P(x + 3)
    //             + 330 * P(x + 4)
    //             - 462 * P(x + 5)
    //             + 462 * P(x + 6)
    //             - 330 * P(x + 7)
    //             + 165 * P(x + 8)
    //             - 55  * P(x + 9)
    //             + 11  * P(x + 10)
    //             + 11! * P(∞).
    //
    // We group symmetric terms with opposite signs where possible:
    //   11 (f[10] - f[1])
    //   55 (f[2]  - f[9])
    //   165(f[8]  - f[3])
    //   330(f[4]  - f[7])
    //   462(f[6]  - f[5])
    // and handle coefficient +1 on f[0] and on 11! * P(∞) via direct
    // accumulator updates to avoid redundant scalar multiplies.
    let mut acc: Acc5S<F> = Acc5S::zero();

    // Coefficient +1 on f[0].
    acc.pos += *f[0].as_unreduced_ref();

    let t11 = f[10] - f[1];
    acc.fmadd(&t11, &11u64);

    let t55 = f[2] - f[9];
    acc.fmadd(&t55, &55u64);

    let t165 = f[8] - f[3];
    acc.fmadd(&t165, &165u64);

    let t330 = f[4] - f[7];
    acc.fmadd(&t330, &330u64);

    let t462 = f[6] - f[5];
    acc.fmadd(&t462, &462u64);

    // Coefficient +1 on 11! * P(∞).
    acc.pos += *f_inf11_fact.as_unreduced_ref();

    acc.barrett_reduce()
}

#[inline]
fn ex13<F: JoltField>(f: &[F; 13], f_inf13_fact: F) -> F {
    // Extrapolate the next value of a degree-13 polynomial on the natural grid.
    //
    // Inputs:
    //   f[i]         = P(x + i) for i = 0..12
    //   f_inf13_fact = 13! * P(∞) = 6227020800 * a13 where a13 is the leading coefficient
    //
    // Coefficients obtained from solving the interpolation system yield:
    //
    //   P(x + 13) =  1  * P(x + 0)
    //             - 13  * P(x + 1)
    //             + 78  * P(x + 2)
    //             - 286 * P(x + 3)
    //             + 715 * P(x + 4)
    //             - 1287* P(x + 5)
    //             + 1716* P(x + 6)
    //             - 1716* P(x + 7)
    //             + 1287* P(x + 8)
    //             - 715 * P(x + 9)
    //             + 286 * P(x + 10)
    //             - 78  * P(x + 11)
    //             + 13  * P(x + 12)
    //             + 13! * P(∞).
    //
    // We use a signed accumulator to combine all terms before a single reduction.
    let mut acc: Acc5S<F> = Acc5S::zero();

    // Coefficient +1 on f[0]: add directly to positive accumulator.
    acc.pos += *f[0].as_unreduced_ref();

    acc.fmadd(&f[1], &(-13i64));
    acc.fmadd(&f[2], &78u64);
    acc.fmadd(&f[3], &(-286i64));
    acc.fmadd(&f[4], &715u64);
    acc.fmadd(&f[5], &(-1287i64));
    acc.fmadd(&f[6], &1716u64);
    acc.fmadd(&f[7], &(-1716i64));
    acc.fmadd(&f[8], &1287u64);
    acc.fmadd(&f[9], &(-715i64));
    acc.fmadd(&f[10], &286u64);
    acc.fmadd(&f[11], &(-78i64));
    acc.fmadd(&f[12], &13u64);

    // Coefficient +1 on 13! * P(∞).
    acc.pos += *f_inf13_fact.as_unreduced_ref();

    acc.barrett_reduce()
}

#[inline]
fn ex16<F: JoltField>(f: &[F; 16], f_inf16_fact: F) -> F {
    // P(17) from f[i]=P(i+1), i=0..15, using 16th-row binomial weights with alternating signs:
    // Coeffs on f[0..15]: [-1, +16, -120, +560, -1820, +4368, -8008, +11440, -12870, +11440, -8008, +4368, -1820, +560, -120, +16]
    // Plus + 16! * a16 (passed as f_inf16_fact).
    //
    // Group symmetric terms with equal coefficients and signs to minimize fmadd calls:
    // +16  : (f[1] + f[15])
    // -120 : (f[2] + f[14])
    // +560 : (f[3] + f[13])
    // -1820: (f[4] + f[12])
    // +4368: (f[5] + f[11])
    // -8008: (f[6] + f[10])
    // +11440: (f[7] + f[9])
    // Center and edges:
    // -12870 f[8], -1 f[0], + f_inf16_fact
    //
    // We again use a signed accumulator to defer reduction.
    let mut acc: Acc5S<F> = Acc5S::zero();
    let s16 = f[1] + f[15];
    acc.fmadd(&s16, &16u64);
    let s120 = f[2] + f[14];
    acc.fmadd(&s120, &(-120i64));
    let s560 = f[3] + f[13];
    acc.fmadd(&s560, &560u64);
    let s1820 = f[4] + f[12];
    acc.fmadd(&s1820, &(-1820i64));
    let s4368 = f[5] + f[11];
    acc.fmadd(&s4368, &4368u64);
    let s8008 = f[6] + f[10];
    acc.fmadd(&s8008, &(-8008i64));
    let s11440 = f[7] + f[9];
    acc.fmadd(&s11440, &11440u64);
    acc.fmadd(&f[8], &(-12870i64));
    // Edge coefficient -1 and the +1 on `f_inf16_fact` can be handled by
    // direct accumulator updates, avoiding redundant scalar multiplies.
    acc.neg += *f[0].as_unreduced_ref();
    acc.pos += *f_inf16_fact.as_unreduced_ref();
    acc.barrett_reduce()
}

#[inline]
fn expand8_to16<F: JoltField>(vals: &[F; 9]) -> ([F; 16], F) {
    // Build `f[1..16]` for a degree-8 product polynomial from evaluations on
    // `[1..8, ∞]`, without zero-initialization; return also the ∞ value.
    let mut f_mu: MaybeUninit<[F; 16]> = MaybeUninit::uninit();
    let f_ptr = f_mu.as_mut_ptr();
    let f_slice_ptr = unsafe { (*f_ptr).as_mut_ptr() };

    // First 8 from vals: f[0..8] ← vals[0..8] = P(1..8)
    unsafe {
        ptr::copy_nonoverlapping(vals.as_ptr(), f_slice_ptr, 8);
    }

    let f_inf = vals[8];
    let f_inf40320 = f_inf.mul_u64(40320);

    // Compute positions 9..16 (indices 8..15) by sliding an 8-wide
    // window over the prefix of `f` and applying `ex8`.
    for i in 0..8 {
        unsafe {
            let win_ptr = f_slice_ptr.add(i) as *const [F; 8];
            let win_ref: &[F; 8] = &*win_ptr;
            let val: F = ex8(win_ref, f_inf40320);
            ptr::write(f_slice_ptr.add(8 + i), val);
        }
    }

    // SAFETY: all indices 0..15 are written above, so the array is fully
    // initialized before we call `assume_init`.
    let f = unsafe { f_mu.assume_init() };
    (f, f_inf)
}

#[inline]
fn eval_half_16_base<F: JoltField>(p: [(F, F); 16]) -> ([F; 16], F) {
    // Compute two 8-sized halves and evaluate each on the internal 8-point grid
    // with `eval_linear_prod_8_internal`.
    //
    // SAFETY: `p[0..8]` and `p[8..16]` are non-overlapping slices of length 8,
    // so reinterpreting them as `[(F, F); 8]` is sound.
    let a8 = eval_linear_prod_8_internal(unsafe { *(p[0..8].as_ptr() as *const [(F, F); 8]) });
    let b8 = eval_linear_prod_8_internal(unsafe { *(p[8..16].as_ptr() as *const [(F, F); 8]) });

    // Expand each 8-sized evaluation grid to 16 points using ex8 sliding
    // (computing positions 9..16). This gives us the half-product evaluated
    // at 1..16 plus its ∞ value.
    let (a16_vals, a_inf) = expand8_to16::<F>(&a8);
    let (b16_vals, b_inf) = expand8_to16::<F>(&b8);

    // Pointwise product to get the 16-base for the half and its inf
    // without zero-initialization. This yields the product of 16 polynomials
    // evaluated at 1..16 plus the ∞ value (returned separately as `a_inf * b_inf`).
    let mut base_mu: MaybeUninit<[F; 16]> = MaybeUninit::uninit();
    let base_ptr = base_mu.as_mut_ptr();
    let base_slice_ptr = unsafe { (*base_ptr).as_mut_ptr() };

    for i in 0..16 {
        unsafe {
            ptr::write(base_slice_ptr.add(i), a16_vals[i] * b16_vals[i]);
        }
    }

    // SAFETY: all indices 0..15 are written above, so the array is fully
    // initialized before we call `assume_init`.
    let base = unsafe { base_mu.assume_init() };
    (base, a_inf * b_inf)
}

#[inline]
fn eval_half_13_base<F: JoltField>(p: [(F, F); 13]) -> ([F; 13], F) {
    // Compute the half-product evaluated at 1..13 plus its ∞ value, without
    // zero-initialization.
    let mut tmp: [F; 13] = [F::zero(); 13];
    eval_prod_13_assign(&p, &mut tmp);

    let inf = tmp[12];

    // Compute the value at x = 13 directly: P(13) = ∏_j p_j(13).
    let mut p13 = F::one();
    for (p0, p1) in p.into_iter() {
        let pinf = p1 - p0;
        let v13 = p0 + pinf.mul_u64(13u64);
        p13 *= v13;
    }

    let mut base: [F; 13] = [F::zero(); 13];
    // Copy P(1..12).
    base[0..12].copy_from_slice(&tmp[0..12]);
    // Set P(13).
    base[12] = p13;

    (base, inf)
}

#[inline]
fn eval_half_11_base<F: JoltField>(p: [(F, F); 11]) -> ([F; 11], F) {
    // Compute the half-product evaluated at 1..11 plus its ∞ value, without
    // zero-initialization.
    let mut tmp: [F; 11] = [F::zero(); 11];
    eval_prod_11_assign(&p, &mut tmp);

    let inf = tmp[10];

    // Compute the value at x = 11 directly: P(11) = ∏_j p_j(11).
    let mut p11 = F::one();
    for (p0, p1) in p.into_iter() {
        let pinf = p1 - p0;
        let v11 = p0 + pinf.mul_u64(11u64);
        p11 *= v11;
    }

    let mut base: [F; 11] = [F::zero(); 11];
    // Copy P(1..10).
    base[0..10].copy_from_slice(&tmp[0..10]);
    // Set P(11).
    base[10] = p11;

    (base, inf)
}

#[inline]
fn expand11_to_u22<F: JoltField>(base11: &[F; 11], inf: F) -> [F; 22] {
    // Build [1..21, inf] for a degree-11 product using ex11 sliding without
    // zero-initialization.
    let mut f_mu: MaybeUninit<[F; 22]> = MaybeUninit::uninit();
    let f_ptr = f_mu.as_mut_ptr();
    let f_slice_ptr = unsafe { (*f_ptr).as_mut_ptr() };

    // Initialize first 11 with base11.
    unsafe {
        ptr::copy_nonoverlapping(base11.as_ptr(), f_slice_ptr, 11);
    }

    // Write inf at position 21 upfront (needed as the final output).
    unsafe {
        ptr::write(f_slice_ptr.add(21), inf);
    }

    let f_inf11_fact = inf.mul_u64(39916800u64); // 11!

    // Compute entries 12..21 (indices 11..20) by sliding an 11-wide window
    // and applying `ex11` with the pre-scaled ∞ value.
    for i in 0..10 {
        unsafe {
            let win_ptr = f_slice_ptr.add(i) as *const [F; 11];
            let win_ref: &[F; 11] = &*win_ptr;
            let val = ex11::<F>(win_ref, f_inf11_fact);
            ptr::write(f_slice_ptr.add(11 + i), val);
        }
    }

    // SAFETY: all indices 0..21 are written above, so the array is fully
    // initialized before we call `assume_init`.
    unsafe { f_mu.assume_init() }
}

#[inline]
fn expand9_to_u19<F: JoltField>(base9: &[F; 9], inf: F) -> [F; 19] {
    // Build [1..18, inf] for a degree-9 product using ex9 sliding without
    // zero-initialization.
    let mut f_mu: MaybeUninit<[F; 19]> = MaybeUninit::uninit();
    let f_ptr = f_mu.as_mut_ptr();
    let f_slice_ptr = unsafe { (*f_ptr).as_mut_ptr() };

    // Initialize first 9 with base9.
    unsafe {
        ptr::copy_nonoverlapping(base9.as_ptr(), f_slice_ptr, 9);
    }

    // Write inf at position 18 upfront (needed as the final output).
    unsafe {
        ptr::write(f_slice_ptr.add(18), inf);
    }

    let f_inf9_fact = inf.mul_u64(362880u64); // 9!

    // Compute entries 10..18 (indices 9..17) by sliding a 9-wide window
    // and applying `ex9` with the pre-scaled ∞ value.
    for i in 0..9 {
        unsafe {
            let win_ptr = f_slice_ptr.add(i) as *const [F; 9];
            let win_ref: &[F; 9] = &*win_ptr;
            let val = ex9::<F>(win_ref, f_inf9_fact);
            ptr::write(f_slice_ptr.add(9 + i), val);
        }
    }

    // SAFETY: all indices 0..18 are written above, so the array is fully
    // initialized before we call `assume_init`.
    unsafe { f_mu.assume_init() }
}

#[inline]
fn expand10_to_u19<F: JoltField>(base10: &[F; 10], inf: F) -> [F; 19] {
    // Build [1..18, inf] for a degree-10 product using ex10 sliding without
    // zero-initialization.
    let mut f_mu: MaybeUninit<[F; 19]> = MaybeUninit::uninit();
    let f_ptr = f_mu.as_mut_ptr();
    let f_slice_ptr = unsafe { (*f_ptr).as_mut_ptr() };

    // Initialize first 10 with base10.
    unsafe {
        ptr::copy_nonoverlapping(base10.as_ptr(), f_slice_ptr, 10);
    }

    // Write inf at position 18 upfront (needed as the final output).
    unsafe {
        ptr::write(f_slice_ptr.add(18), inf);
    }

    let f_inf10_fact = inf.mul_u64(3628800u64); // 10!

    // Compute entries 11..18 (indices 10..17) by sliding a 10-wide window
    // and applying `ex10` with the pre-scaled ∞ value.
    for i in 0..8 {
        unsafe {
            let win_ptr = f_slice_ptr.add(i) as *const [F; 10];
            let win_ref: &[F; 10] = &*win_ptr;
            let val = ex10::<F>(win_ref, f_inf10_fact);
            ptr::write(f_slice_ptr.add(10 + i), val);
        }
    }

    // SAFETY: all indices 0..18 are written above, so the array is fully
    // initialized before we call `assume_init`.
    unsafe { f_mu.assume_init() }
}

#[inline]
fn expand13_to_u26<F: JoltField>(base13: &[F; 13], inf: F) -> [F; 26] {
    // Build [1..25, inf] for a degree-13 product using ex13 sliding without
    // zero-initialization.
    let mut f_mu: MaybeUninit<[F; 26]> = MaybeUninit::uninit();
    let f_ptr = f_mu.as_mut_ptr();
    let f_slice_ptr = unsafe { (*f_ptr).as_mut_ptr() };

    // Initialize first 13 with base13.
    unsafe {
        ptr::copy_nonoverlapping(base13.as_ptr(), f_slice_ptr, 13);
    }

    // Write inf at position 25 upfront (needed by the last window).
    unsafe {
        ptr::write(f_slice_ptr.add(25), inf);
    }

    let f_inf13_fact = inf.mul_u64(6227020800u64); // 13!

    // Compute entries 14..25 (indices 13..24) by sliding a 13-wide window
    // and applying `ex13` with the pre-scaled ∞ value.
    for i in 0..12 {
        unsafe {
            let win_ptr = f_slice_ptr.add(i) as *const [F; 13];
            let win_ref: &[F; 13] = &*win_ptr;
            let val = ex13::<F>(win_ref, f_inf13_fact);
            ptr::write(f_slice_ptr.add(13 + i), val);
        }
    }

    // SAFETY: all indices 0..25 are written above, so the array is fully
    // initialized before we call `assume_init`.
    unsafe { f_mu.assume_init() }
}

#[inline]
fn expand16_to_u32<F: JoltField>(base16: &[F; 16], inf: F) -> [F; 32] {
    // Build [1..31, inf] for a degree-16 product using ex16 sliding
    // without zero-initialization.
    let mut f_mu: MaybeUninit<[F; 32]> = MaybeUninit::uninit();
    let f_ptr = f_mu.as_mut_ptr();
    let f_slice_ptr = unsafe { (*f_ptr).as_mut_ptr() };

    // Initialize first 16 with base16
    unsafe {
        ptr::copy_nonoverlapping(base16.as_ptr(), f_slice_ptr, 16);
    }

    // Write inf at position 31 upfront (needed by the last window).
    unsafe {
        ptr::write(f_slice_ptr.add(31), inf);
    }

    let f_inf16_fact = inf.mul_u64(20922789888000u64); // 16!

    // Compute entries 17..31 (indices 16..30) by sliding a 16-wide window
    // and applying `ex16` with the pre-scaled ∞ value.
    for i in 0..15 {
        unsafe {
            let win_ptr = f_slice_ptr.add(i) as *const [F; 16];
            let win_ref: &[F; 16] = &*win_ptr;
            let val = ex16::<F>(win_ref, f_inf16_fact);
            ptr::write(f_slice_ptr.add(16 + i), val);
        }
    }

    // SAFETY: all indices 0..31 are written above, so the array is fully
    // initialized before we call `assume_init`.
    unsafe { f_mu.assume_init() }
}

/// Evaluate the product of 32 linear polynomials on `U_32 = [1, 2, ..., 31, ∞]`.
///
/// This kernel factors the product into two halves of 16 polynomials, uses
/// `eval_half_16_base` and `expand16_to_u32` to obtain each half on
/// `[1..31, ∞]`, and then multiplies the halves pointwise.
/// The final `outputs` slice has layout `[P(1), ..., P(31), P(∞)]`.
///
/// # Safety
///
/// This function uses pointer casts to reinterpret `p[0..16]` and `p[16..32]`
/// as `[(F, F); 16]`. This is sound because:
/// - `p` is a fixed-size `[ (F, F); 32 ]`, so both sub-slices have length 16
///   and correct alignment.
/// - The sub-slices are non-overlapping.
fn eval_prod_32_assign<F: JoltField>(p: &[(F, F); 32], outputs: &mut [F]) {
    // First 16 polynomials → half A.
    //
    // SAFETY: `p[0..16]` and `p[16..32]` are non-overlapping slices of length
    // 16, so the casts to `[(F, F); 16]` are valid.
    let (a16_base, a_inf) =
        eval_half_16_base::<F>(unsafe { *(p[0..16].as_ptr() as *const [(F, F); 16]) });
    let a_full = expand16_to_u32::<F>(&a16_base, a_inf);

    // Second 16 polynomials → half B.
    let (b16_base, b_inf) =
        eval_half_16_base::<F>(unsafe { *(p[16..32].as_ptr() as *const [(F, F); 16]) });
    let b_full = expand16_to_u32::<F>(&b16_base, b_inf);

    // Combine half A and half B pointwise to get the full product evaluated on
    // [1..31, ∞].
    for i in 0..32 {
        let mut v = a_full[i];
        v *= b_full[i];
        outputs[i] = v;
    }
}

#[inline(always)]
fn dbl<F: JoltField>(x: F) -> F {
    x + x
}

#[inline(always)]
fn dbl_assign<F: JoltField>(x: &mut F) {
    *x += *x;
}

/// Naive evaluator for the product of `D` linear polynomials on `U_D = [1, 2, ..., D - 1, ∞]`.
///
/// The evaluations on `U_D` are assigned to `evals` in the layout
/// `[P(1), P(2), ..., P(D - 1), P(∞)]`, where `P(x) = ∏_j p_j(x)`.
///
/// Inputs:
/// - `pairs[j] = (p_j(0), p_j(1))`
/// - `evals`: output slice with layout `[1, 2, ..., D - 1, ∞]`
///
/// Complexity is `O(D^2)` field multiplications and is intended only as a
/// fallback for unsupported `D`; hot paths should go through the specialized
/// kernels above.
pub fn eval_linear_prod_naive_assign<F: JoltField>(pairs: &[(F, F)], evals: &mut [F]) {
    let d = pairs.len();
    debug_assert_eq!(evals.len(), d);
    if d == 0 {
        return;
    }

    // Single allocation: store (current value, leading coefficient) per polynomial.
    let mut cur_vals_pinfs: Vec<(F, F)> = Vec::with_capacity(d);
    for &(p0, p1) in pairs.iter() {
        let pinf = p1 - p0;
        cur_vals_pinfs.push((p1, pinf));
    }

    // Evaluate at x = 1..(d-1) by sliding x ↦ x+1 using the precomputed pinfs.
    //
    // Micro-optimization: initialize the accumulator from the first element so
    // that each product uses (d - 1) multiplications instead of d.
    for idx in 0..(d - 1) {
        let mut iter = cur_vals_pinfs.iter();
        let (first_val, _) = iter.next().expect("d > 0 is enforced above");
        let mut acc = *first_val;
        for (cur_val, _) in iter {
            acc *= *cur_val;
        }
        evals[idx] = acc;

        for (cur_val, pinf) in cur_vals_pinfs.iter_mut() {
            *cur_val += *pinf;
        }
    }

    // Evaluate at infinity (product of leading coefficients).
    let mut iter = cur_vals_pinfs.iter();
    let (_, first_pinf) = iter
        .next()
        .expect("d > 0 is enforced above; there is at least one leading coefficient");
    let mut acc_inf = *first_pinf;
    for (_, pinf) in iter {
        acc_inf *= *pinf;
    }
    evals[d - 1] = acc_inf;
}

/// Naive evaluator for the product of `D` linear polynomials on `U_D = [1, 2, ..., D - 1, ∞]`.
///
/// The evaluations on `U_D` are accumulated into `sums`.
///
/// Inputs:
/// - `pairs[j] = (p_j(0), p_j(1))`
/// - `sums`: accumulator with layout `[1, 2, ..., D - 1, ∞]`
fn product_eval_univariate_naive_accumulate<F: JoltField>(
    pairs: &[(F, F)],
    sums: &mut [F::Unreduced<9>],
) {
    let d = pairs.len();
    debug_assert_eq!(sums.len(), d);
    if d == 0 {
        return;
    }
    // Memoize p(1)=p1, then p(2)=p(1)+pinf, p(3)=p(2)+pinf, ...
    let mut cur_vals = Vec::with_capacity(d);
    let mut pinfs = Vec::with_capacity(d);
    for &(p0, p1) in pairs.iter() {
        let pinf = p1 - p0;
        cur_vals.push(p1);
        pinfs.push(pinf);
    }
    // Evaluate at x = 1..(d-1)
    for idx in 0..(d - 1) {
        let mut acc = F::one();
        for v in cur_vals.iter() {
            acc *= *v;
        }
        sums[idx] += *acc.as_unreduced_ref();
        // advance all to next x
        for i in 0..d {
            cur_vals[i] += pinfs[i];
        }
    }
    // Evaluate at infinity (product of leading coefficients)
    let mut acc_inf = F::one();
    for pinf in pinfs.iter() {
        acc_inf *= *pinf;
    }
    sums[d - 1] += *acc_inf.as_unreduced_ref();
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_std::{test_rng, UniformRand};
    use std::array::from_fn;

    use crate::{
        field::JoltField,
        poly::{
            dense_mlpoly::DensePolynomial,
            eq_poly::EqPolynomial,
            multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialEvaluation},
            ra_poly::RaPolynomial,
            split_eq_poly::GruenSplitEqPolynomial,
        },
        subprotocols::mles_product_sum::compute_mles_product_sum,
    };

    fn random_mle(n_vars: usize, rng: &mut impl rand::Rng) -> MultilinearPolynomial<Fr> {
        let values: Vec<Fr> = (0..(1 << n_vars)).map(|_| Fr::random(rng)).collect();
        MultilinearPolynomial::LargeScalars(DensePolynomial::new(values))
    }

    /// Generates MLE `p(x) = sum_j eq(j, x) * prod_i mle_i(j)`.
    fn gen_product_mle(mles: &[MultilinearPolynomial<Fr>]) -> MultilinearPolynomial<Fr> {
        let n_vars = mles[0].get_num_vars();
        assert!(mles.iter().all(|mle| mle.get_num_vars() == n_vars));
        let res = (0..1 << n_vars)
            .map(|i| mles.iter().map(|mle| mle.get_bound_coeff(i)).product())
            .collect::<Vec<Fr>>();
        res.into()
    }

    /// Checks that the optimized `compute_mles_product_sum` matches the naive
    /// polynomial
    /// `p(x) = ∑_j eq(j, x) * ∏_i mle_i(j)` constructed in `gen_product_mle`,
    /// thereby exercising all specialized product kernels for a given `N_MLE`.
    fn check_optimized_product_sum_matches_naive<const N_MLE: usize>() {
        let mut rng = &mut test_rng();
        let r_whole = [<Fr as JoltField>::Challenge::rand(&mut rng)];
        let r: &[<Fr as JoltField>::Challenge; 1] = &r_whole;
        let mles: [_; N_MLE] = from_fn(|_| random_mle(1, rng));
        let claim = gen_product_mle(&mles).evaluate(r);

        let challenge_whole = [<Fr as JoltField>::Challenge::rand(&mut rng)];
        let challenge: &[<Fr as JoltField>::Challenge; 1] = &challenge_whole;
        let mle_challenge_product = mles.iter().map(|p| p.evaluate(challenge)).product::<Fr>();
        let eval = EqPolynomial::mle(challenge, r) * mle_challenge_product;
        let mles = mles.map(RaPolynomial::RoundN);

        let eq_poly = GruenSplitEqPolynomial::new(r, BindingOrder::LowToHigh);
        let sum_poly = compute_mles_product_sum(&mles, claim, &eq_poly);
        let lhs = sum_poly.evaluate(&challenge[0]);

        assert_eq!(lhs, eval);
    }

    // Keep tests ordered by increasing `N_MLE` for readability.
    #[test]
    fn optimized_product_sum_matches_naive_2_mles() {
        check_optimized_product_sum_matches_naive::<2>();
    }

    #[test]
    fn optimized_product_sum_matches_naive_4_mles() {
        check_optimized_product_sum_matches_naive::<4>();
    }

    #[test]
    fn optimized_product_sum_matches_naive_5_mles() {
        check_optimized_product_sum_matches_naive::<5>();
    }

    #[test]
    fn optimized_product_sum_matches_naive_8_mles() {
        check_optimized_product_sum_matches_naive::<8>();
    }

    #[test]
    fn optimized_product_sum_matches_naive_13_mles() {
        check_optimized_product_sum_matches_naive::<13>();
    }

    #[test]
    fn optimized_product_sum_matches_naive_15_mles() {
        check_optimized_product_sum_matches_naive::<15>();
    }

    #[test]
    fn optimized_product_sum_matches_naive_16_mles() {
        check_optimized_product_sum_matches_naive::<16>();
    }

    #[test]
    fn optimized_product_sum_matches_naive_19_mles() {
        check_optimized_product_sum_matches_naive::<19>();
    }

    #[test]
    fn optimized_product_sum_matches_naive_22_mles() {
        check_optimized_product_sum_matches_naive::<22>();
    }

    #[test]
    fn optimized_product_sum_matches_naive_26_mles() {
        check_optimized_product_sum_matches_naive::<26>();
    }

    #[test]
    fn optimized_product_sum_matches_naive_32_mles() {
        check_optimized_product_sum_matches_naive::<32>();
    }
}
