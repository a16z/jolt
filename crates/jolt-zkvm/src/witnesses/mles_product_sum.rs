//! Compute kernel for the RA virtual sumcheck round polynomial.
//!
//! Computes `g(X) = Σ_j eq((r', X, j), r) · Π_i mle_i(X, j)` — the round
//! polynomial for the RA virtual sumcheck — using split-eq factored evaluation.
//!
//! Provides specialized kernels for arity `d ∈ {4, 8, 16, 32}` using
//! stack-allocated product evaluation, with a generic fallback for arbitrary `d`.

use core::{mem::MaybeUninit, ptr};

use jolt_field::{FieldAccumulator, WithChallenge};
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

// ─── Product evaluation kernels ──────────────────────────────────────

/// Evaluate the product of linear polynomials on `U_D = [1, 2, ..., D - 1, ∞]`.
///
/// Each `pair = (p(0), p(1))` defines a linear polynomial `p(x) = p(0) + (p(1) - p(0))·x`.
/// The output slice has layout `[P(1), P(2), ..., P(D-1), P(∞)]` where `P(x) = Π p_j(x)`.
pub fn eval_linear_prod_assign<F: jolt_field::Field>(pairs: &[(F, F)], evals: &mut [F]) {
    debug_assert_eq!(pairs.len(), evals.len());
    match pairs.len() {
        2 => {
            // SAFETY: `pairs` has length 2, so reinterpret as `[(F, F); 2]`.
            let p = unsafe { &*pairs.as_ptr().cast::<[(F, F); 2]>() };
            eval_prod_2_assign(p, evals);
        }
        3 => {
            // SAFETY: `pairs` has length 3, layout-compatible with `[(F, F); 3]`.
            let p = unsafe { &*pairs.as_ptr().cast::<[(F, F); 3]>() };
            eval_prod_3_assign(p, evals);
        }
        4 => {
            // SAFETY: `pairs` has length 4, layout-compatible with `[(F, F); 4]`.
            let p = unsafe { &*pairs.as_ptr().cast::<[(F, F); 4]>() };
            eval_prod_4_assign(p, evals);
        }
        5 => {
            // SAFETY: `pairs` has length 5, layout-compatible with `[(F, F); 5]`.
            let p = unsafe { &*pairs.as_ptr().cast::<[(F, F); 5]>() };
            eval_prod_5_assign(p, evals);
        }
        6 => {
            // SAFETY: `pairs` has length 6, layout-compatible with `[(F, F); 6]`.
            let p = unsafe { &*pairs.as_ptr().cast::<[(F, F); 6]>() };
            eval_prod_6_assign(p, evals);
        }
        7 => {
            // SAFETY: `pairs` has length 7, layout-compatible with `[(F, F); 7]`.
            let p = unsafe { &*pairs.as_ptr().cast::<[(F, F); 7]>() };
            eval_prod_7_assign(p, evals);
        }
        8 => {
            // SAFETY: `pairs` has length 8, layout-compatible with `[(F, F); 8]`.
            let p = unsafe { &*pairs.as_ptr().cast::<[(F, F); 8]>() };
            eval_prod_8_assign(p, evals);
        }
        16 => {
            // SAFETY: `pairs` has length 16, layout-compatible with `[(F, F); 16]`.
            let p = unsafe { &*pairs.as_ptr().cast::<[(F, F); 16]>() };
            eval_prod_16_assign(p, evals);
        }
        32 => {
            // SAFETY: `pairs` has length 32, layout-compatible with `[(F, F); 32]`.
            let p = unsafe { &*pairs.as_ptr().cast::<[(F, F); 32]>() };
            eval_prod_32_assign(p, evals);
        }
        _ => eval_linear_prod_naive_assign(pairs, evals),
    }
}

// --- Degree-2 through Degree-8 kernels ---

#[inline(always)]
fn eval_linear_prod_2_internal<F: jolt_field::Field>(
    (p0, p1): (F, F),
    (q0, q1): (F, F),
) -> (F, F, F) {
    let p_inf = p1 - p0;
    let p2 = p_inf + p1;
    let q_inf = q1 - q0;
    let q2 = q_inf + q1;
    let r1 = p1 * q1;
    let r2 = p2 * q2;
    let r_inf = p_inf * q_inf;
    (r1, r2, r_inf)
}

#[inline(always)]
fn eval_prod_2_assign<F: jolt_field::Field>(p: &[(F, F); 2], outputs: &mut [F]) {
    outputs[0] = p[0].1 * p[1].1;
    outputs[1] = (p[0].1 - p[0].0) * (p[1].1 - p[1].0);
}

#[inline(always)]
fn eval_prod_3_assign<F: jolt_field::Field>(pairs: &[(F, F); 3], outputs: &mut [F]) {
    let (a1, a2, a_inf) = eval_linear_prod_2_internal(pairs[0], pairs[1]);
    let (b0, b1) = pairs[2];
    let b_inf = b1 - b0;
    let b2 = b1 + b_inf;
    outputs[0] = a1 * b1;
    outputs[1] = a2 * b2;
    outputs[2] = a_inf * b_inf;
}

#[inline]
fn eval_linear_prod_4_internal<F: jolt_field::Field>(p: [(F, F); 4]) -> (F, F, F, F, F) {
    let (a1, a2, a_inf) = eval_linear_prod_2_internal(p[0], p[1]);
    let a3 = ex2(&[a1, a2], &a_inf);
    let a4 = ex2(&[a2, a3], &a_inf);
    let (b1, b2, b_inf) = eval_linear_prod_2_internal(p[2], p[3]);
    let b3 = ex2(&[b1, b2], &b_inf);
    let b4 = ex2(&[b2, b3], &b_inf);
    (a1 * b1, a2 * b2, a3 * b3, a4 * b4, a_inf * b_inf)
}

fn eval_prod_4_assign<F: jolt_field::Field>(p: &[(F, F); 4], outputs: &mut [F]) {
    let (a1, a2, a_inf) = eval_linear_prod_2_internal(p[0], p[1]);
    let a3 = ex2(&[a1, a2], &a_inf);
    let (b1, b2, b_inf) = eval_linear_prod_2_internal(p[2], p[3]);
    let b3 = ex2(&[b1, b2], &b_inf);
    outputs[0] = a1 * b1;
    outputs[1] = a2 * b2;
    outputs[2] = a3 * b3;
    outputs[3] = a_inf * b_inf;
}

fn eval_prod_5_assign<F: jolt_field::Field>(p: &[(F, F); 5], outputs: &mut [F]) {
    let (a1, a2, a_inf) = eval_linear_prod_2_internal(p[0], p[1]);
    let a3 = ex2(&[a1, a2], &a_inf);
    let a4 = ex2(&[a2, a3], &a_inf);

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

    outputs[0] = a1 * (l1 * r1);
    outputs[1] = a2 * (l2 * r2);
    outputs[2] = a3 * (l3 * r3);
    outputs[3] = a4 * (l4 * r4);
    outputs[4] = a_inf * (l_inf * r_inf);
}

fn eval_prod_6_assign<F: jolt_field::Field>(p: &[(F, F); 6], outputs: &mut [F]) {
    let mut cur_vals_pinfs: [(F, F); 6] = [(F::zero(), F::zero()); 6];
    for (i, (p0, p1)) in p.iter().copied().enumerate() {
        cur_vals_pinfs[i] = (p1, p1 - p0);
    }

    for output in &mut outputs[..5] {
        let mut iter = cur_vals_pinfs.iter();
        let (first_val, _) = iter.next().expect("d > 0");
        let mut acc = *first_val;
        for (cur_val, _) in iter {
            acc *= *cur_val;
        }
        *output = acc;

        for (cur_val, pinf) in &mut cur_vals_pinfs {
            *cur_val += *pinf;
        }
    }

    let mut iter = cur_vals_pinfs.iter();
    let (_, first_pinf) = iter.next().expect("d > 0");
    let mut acc_inf = *first_pinf;
    for (_, pinf) in iter {
        acc_inf *= *pinf;
    }
    outputs[5] = acc_inf;
}

fn eval_prod_7_assign<F: jolt_field::Field>(p: &[(F, F); 7], outputs: &mut [F]) {
    let mut cur_vals_pinfs: [(F, F); 7] = [(F::zero(), F::zero()); 7];
    for (i, (p0, p1)) in p.iter().copied().enumerate() {
        cur_vals_pinfs[i] = (p1, p1 - p0);
    }

    for output in &mut outputs[..6] {
        let mut iter = cur_vals_pinfs.iter();
        let (first_val, _) = iter.next().expect("d > 0");
        let mut acc = *first_val;
        for (cur_val, _) in iter {
            acc *= *cur_val;
        }
        *output = acc;

        for (cur_val, pinf) in &mut cur_vals_pinfs {
            *cur_val += *pinf;
        }
    }

    let mut iter = cur_vals_pinfs.iter();
    let (_, first_pinf) = iter.next().expect("d > 0");
    let mut acc_inf = *first_pinf;
    for (_, pinf) in iter {
        acc_inf *= *pinf;
    }
    outputs[6] = acc_inf;
}

// --- Degree-8 kernel ---

/// Evaluate 8 linear polynomials on internal 9-point grid `[1..8, ∞]`.
fn eval_linear_prod_8_internal<F: jolt_field::Field>(p: [(F, F); 8]) -> [F; 9] {
    #[inline]
    fn batch_helper<F: jolt_field::Field>(f0: F, f1: F, f2: F, f3: F, f_inf: F) -> (F, F, F, F) {
        let f_inf6 = f_inf.mul_u64(6);
        let (f4, f5) = ex4_2(&[f0, f1, f2, f3], &f_inf6);
        let (f6, f7) = ex4_2(&[f2, f3, f4, f5], &f_inf6);
        (f4, f5, f6, f7)
    }

    // SAFETY: `p[0..4]` is a valid 4-element subslice, reinterpreted as `[(F, F); 4]`.
    let (a1, a2, a3, a4, a_inf) =
        eval_linear_prod_4_internal(unsafe { *p[0..4].as_ptr().cast::<[(F, F); 4]>() });
    let (a5, a6, a7, a8) = batch_helper(a1, a2, a3, a4, a_inf);
    // SAFETY: `p[4..8]` is a valid 4-element subslice, reinterpreted as `[(F, F); 4]`.
    let (b1, b2, b3, b4, b_inf) =
        eval_linear_prod_4_internal(unsafe { *p[4..8].as_ptr().cast::<[(F, F); 4]>() });
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

fn eval_prod_8_assign<F: jolt_field::Field>(p: &[(F, F); 8], outputs: &mut [F]) {
    #[inline]
    fn batch_helper<F: jolt_field::Field>(f0: F, f1: F, f2: F, f3: F, f_inf: F) -> (F, F, F) {
        let f_inf6 = f_inf.mul_u64(6);
        let (f4, f5) = ex4_2(&[f0, f1, f2, f3], &f_inf6);
        let f6 = ex4(&[f2, f3, f4, f5], &f_inf6);
        (f4, f5, f6)
    }

    // SAFETY: `p[0..4]` is a valid 4-element subslice, reinterpreted as `[(F, F); 4]`.
    let (a1, a2, a3, a4, a_inf) =
        eval_linear_prod_4_internal(unsafe { *p[0..4].as_ptr().cast::<[(F, F); 4]>() });
    let (a5, a6, a7) = batch_helper(a1, a2, a3, a4, a_inf);
    // SAFETY: `p[4..8]` is a valid 4-element subslice, reinterpreted as `[(F, F); 4]`.
    let (b1, b2, b3, b4, b_inf) =
        eval_linear_prod_4_internal(unsafe { *p[4..8].as_ptr().cast::<[(F, F); 4]>() });
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

// --- Degree-16 kernel ---

fn eval_prod_16_assign<F: jolt_field::Field>(p: &[(F, F); 16], outputs: &mut [F]) {
    debug_assert!(outputs.len() >= 16);

    // SAFETY: `p[0..8]` is a valid 8-element subslice, reinterpreted as `[(F, F); 8]`.
    let a = eval_linear_prod_8_internal(unsafe { *p[0..8].as_ptr().cast::<[(F, F); 8]>() });
    // SAFETY: `p[8..16]` is a valid 8-element subslice, reinterpreted as `[(F, F); 8]`.
    let b = eval_linear_prod_8_internal(unsafe { *p[8..16].as_ptr().cast::<[(F, F); 8]>() });

    for i in 0..8 {
        outputs[i] = a[i] * b[i];
    }

    let a_inf40320 = a[8].mul_u64(40320);
    let b_inf40320 = b[8].mul_u64(40320);

    let mut aw_mu: MaybeUninit<[F; 16]> = MaybeUninit::uninit();
    let mut bw_mu: MaybeUninit<[F; 16]> = MaybeUninit::uninit();

    let aw_ptr = aw_mu.as_mut_ptr();
    let bw_ptr = bw_mu.as_mut_ptr();

    // SAFETY: Writing to uninitialized storage; all read indices are written first.
    let aw_slice_ptr = unsafe { (*aw_ptr).as_mut_ptr() };
    // SAFETY: Same as above for `bw`.
    let bw_slice_ptr = unsafe { (*bw_ptr).as_mut_ptr() };

    // SAFETY: Copying 8 values from `a` and `b` into the first 8 slots,
    // and writing `a[8]`/`b[8]` at slot 15.
    unsafe {
        ptr::copy_nonoverlapping(a.as_ptr(), aw_slice_ptr, 8);
        ptr::write(aw_slice_ptr.add(15), a[8]);
        ptr::copy_nonoverlapping(b.as_ptr(), bw_slice_ptr, 8);
        ptr::write(bw_slice_ptr.add(15), b[8]);
    }

    for i in 0..7 {
        // SAFETY: Sliding window `[i..i+8]` over initialized data.
        let na = unsafe {
            let win_a_ptr = aw_slice_ptr.add(i) as *const [F; 8];
            ex8::<F>(&*win_a_ptr, a_inf40320)
        };
        // SAFETY: Same sliding window for `b`.
        let nb = unsafe {
            let win_b_ptr = bw_slice_ptr.add(i) as *const [F; 8];
            ex8::<F>(&*win_b_ptr, b_inf40320)
        };

        outputs[8 + i] = na * nb;

        // SAFETY: Writing newly computed values into slots 8..15.
        unsafe {
            ptr::write(aw_slice_ptr.add(8 + i), na);
            ptr::write(bw_slice_ptr.add(8 + i), nb);
        }
    }

    outputs[15] = a[8] * b[8];
}

// --- Degree-32 kernel ---

fn eval_prod_32_assign<F: jolt_field::Field>(p: &[(F, F); 32], outputs: &mut [F]) {
    // SAFETY: `p[0..16]` is a valid 16-element subslice, reinterpreted as `[(F, F); 16]`.
    let (a16_base, a_inf) =
        eval_half_16_base::<F>(unsafe { *p[0..16].as_ptr().cast::<[(F, F); 16]>() });
    let a_full = expand16_to_u32::<F>(&a16_base, a_inf);

    // SAFETY: `p[16..32]` is a valid 16-element subslice, reinterpreted as `[(F, F); 16]`.
    let (b16_base, b_inf) =
        eval_half_16_base::<F>(unsafe { *p[16..32].as_ptr().cast::<[(F, F); 16]>() });
    let b_full = expand16_to_u32::<F>(&b16_base, b_inf);

    for i in 0..32 {
        outputs[i] = a_full[i] * b_full[i];
    }
}

// --- Extrapolation helpers ---

#[inline(always)]
fn dbl<F: jolt_field::Field>(x: F) -> F {
    x + x
}

#[inline(always)]
fn dbl_assign<F: jolt_field::Field>(x: &mut F) {
    *x += *x;
}

/// Extrapolate the next value of a degree-2 polynomial on the natural grid.
#[inline(always)]
fn ex2<F: jolt_field::Field>(f: &[F; 2], f_inf: &F) -> F {
    dbl(f[1] + f_inf) - f[0]
}

/// Extrapolate the next value of a degree-4 polynomial on the natural grid.
#[inline]
fn ex4<F: jolt_field::Field>(f: &[F; 4], f_inf6: &F) -> F {
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

/// Extrapolate the next two values of a degree-4 polynomial on the natural grid.
#[inline]
fn ex4_2<F: jolt_field::Field>(f: &[F; 4], f_inf6: &F) -> (F, F) {
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

/// Extrapolate `P(9)` from 8 consecutive evaluations of a degree-8 polynomial.
///
/// Uses field arithmetic with small-scalar multiplications instead of
/// unreduced accumulators.
#[inline(always)]
fn ex8<F: jolt_field::Field>(f: &[F; 8], f_inf40320: F) -> F {
    // P(9) = 8(f[1] + f[7]) + 56(f[3] + f[5]) - 28(f[2] + f[6]) - 70·f[4] - f[0] + f_inf40320
    let t1 = f[1] + f[7];
    let t2 = f[3] + f[5];
    let t3 = f[2] + f[6];

    t1.mul_u64(8) + t2.mul_u64(56) + f_inf40320 - t3.mul_u64(28) - f[4].mul_u64(70) - f[0]
}

/// Extrapolate `P(17)` from 16 consecutive evaluations of a degree-16 polynomial.
#[inline]
fn ex16<F: jolt_field::Field>(f: &[F; 16], f_inf16_fact: F) -> F {
    // Binomial coefficients with alternating signs:
    // [-1, +16, -120, +560, -1820, +4368, -8008, +11440,
    //  -12870, +11440, -8008, +4368, -1820, +560, -120, +16]
    let s16 = f[1] + f[15];
    let s120 = f[2] + f[14];
    let s560 = f[3] + f[13];
    let s1820 = f[4] + f[12];
    let s4368 = f[5] + f[11];
    let s8008 = f[6] + f[10];
    let s11440 = f[7] + f[9];

    s16.mul_u64(16) + s560.mul_u64(560) + s4368.mul_u64(4368) + s11440.mul_u64(11440)
        + f_inf16_fact
        - s120.mul_u64(120)
        - s1820.mul_u64(1820)
        - s8008.mul_u64(8008)
        - f[8].mul_u64(12870)
        - f[0]
}

#[inline]
fn expand8_to16<F: jolt_field::Field>(vals: &[F; 9]) -> ([F; 16], F) {
    let mut f_mu: MaybeUninit<[F; 16]> = MaybeUninit::uninit();
    let f_ptr = f_mu.as_mut_ptr();
    // SAFETY: Writing to uninitialized storage; all 16 slots are filled before `assume_init`.
    let f_slice_ptr = unsafe { (*f_ptr).as_mut_ptr() };

    // SAFETY: Copying first 8 values from `vals` into slots 0..8.
    unsafe {
        ptr::copy_nonoverlapping(vals.as_ptr(), f_slice_ptr, 8);
    }

    let f_inf = vals[8];
    let f_inf40320 = f_inf.mul_u64(40320);

    for i in 0..8 {
        // SAFETY: Sliding window `[i..i+8]` reads from already-initialized data,
        // result written at slot `8+i`.
        unsafe {
            let win_ptr = f_slice_ptr.add(i) as *const [F; 8];
            let val: F = ex8(&*win_ptr, f_inf40320);
            ptr::write(f_slice_ptr.add(8 + i), val);
        }
    }

    // SAFETY: All 16 slots written (0..8 from copy, 8..16 from loop).
    let f = unsafe { f_mu.assume_init() };
    (f, f_inf)
}

#[inline]
fn eval_half_16_base<F: jolt_field::Field>(p: [(F, F); 16]) -> ([F; 16], F) {
    // SAFETY: `p[0..8]` is a valid 8-element subslice, reinterpreted as `[(F, F); 8]`.
    let a8 = eval_linear_prod_8_internal(unsafe { *p[0..8].as_ptr().cast::<[(F, F); 8]>() });
    // SAFETY: `p[8..16]` is a valid 8-element subslice, reinterpreted as `[(F, F); 8]`.
    let b8 = eval_linear_prod_8_internal(unsafe { *p[8..16].as_ptr().cast::<[(F, F); 8]>() });

    let (a16_vals, a_inf) = expand8_to16::<F>(&a8);
    let (b16_vals, b_inf) = expand8_to16::<F>(&b8);

    let mut base_mu: MaybeUninit<[F; 16]> = MaybeUninit::uninit();
    let base_ptr = base_mu.as_mut_ptr();
    // SAFETY: Writing to uninitialized storage; all 16 slots filled before `assume_init`.
    let base_slice_ptr = unsafe { (*base_ptr).as_mut_ptr() };

    for i in 0..16 {
        // SAFETY: Writing product at each initialized slot.
        unsafe {
            ptr::write(base_slice_ptr.add(i), a16_vals[i] * b16_vals[i]);
        }
    }

    // SAFETY: All 16 slots written.
    let base = unsafe { base_mu.assume_init() };
    (base, a_inf * b_inf)
}

#[inline]
fn expand16_to_u32<F: jolt_field::Field>(base16: &[F; 16], inf: F) -> [F; 32] {
    let mut f_mu: MaybeUninit<[F; 32]> = MaybeUninit::uninit();
    let f_ptr = f_mu.as_mut_ptr();
    // SAFETY: Writing to uninitialized storage; all 32 slots filled before `assume_init`.
    let f_slice_ptr = unsafe { (*f_ptr).as_mut_ptr() };

    // SAFETY: Copy first 16 values and pre-write slot 31.
    unsafe {
        ptr::copy_nonoverlapping(base16.as_ptr(), f_slice_ptr, 16);
        ptr::write(f_slice_ptr.add(31), inf);
    }

    let f_inf16_fact = inf.mul_u64(20_922_789_888_000_u64); // 16!

    for i in 0..15 {
        // SAFETY: Sliding window `[i..i+16]` reads from already-initialized data,
        // result written at slot `16+i`.
        unsafe {
            let win_ptr = f_slice_ptr.add(i) as *const [F; 16];
            let val = ex16::<F>(&*win_ptr, f_inf16_fact);
            ptr::write(f_slice_ptr.add(16 + i), val);
        }
    }

    // SAFETY: All 32 slots written (0..16 from copy, 16..31 from loop, 31 pre-written).
    unsafe { f_mu.assume_init() }
}

/// Naive fallback for products of `D` linear polynomials.
pub fn eval_linear_prod_naive_assign<F: jolt_field::Field>(pairs: &[(F, F)], evals: &mut [F]) {
    let d = pairs.len();
    debug_assert_eq!(evals.len(), d);
    if d == 0 {
        return;
    }

    let mut cur_vals_pinfs: Vec<(F, F)> = Vec::with_capacity(d);
    for &(p0, p1) in pairs {
        cur_vals_pinfs.push((p1, p1 - p0));
    }

    for output in &mut evals[..(d - 1)] {
        let mut iter = cur_vals_pinfs.iter();
        let (first_val, _) = iter.next().expect("d > 0");
        let mut acc = *first_val;
        for (cur_val, _) in iter {
            acc *= *cur_val;
        }
        *output = acc;

        for (cur_val, pinf) in &mut cur_vals_pinfs {
            *cur_val += *pinf;
        }
    }

    let mut iter = cur_vals_pinfs.iter();
    let (_, first_pinf) = iter.next().expect("d > 0");
    let mut acc_inf = *first_pinf;
    for (_, pinf) in iter {
        acc_inf *= *pinf;
    }
    evals[d - 1] = acc_inf;
}
