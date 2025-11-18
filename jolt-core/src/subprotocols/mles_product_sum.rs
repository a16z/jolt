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
    mles: &[RaPolynomial<u8, F>],
    claim: F,
    eq_poly: &GruenSplitEqPolynomial<F>,
) -> UniPoly<F> {
    // Evaluate g(X) / eq(X, r[round]) at [1, 2, ..., |mles| - 1, ∞] using split-eq fold.
    let d = mles.len();

    // Fast path for when d = 16.
    //
    // This uses a fully stack-allocated implementation based on the optimized
    // 16-way interpolation kernel, and avoids any heap allocations in the inner loop.
    if d == 16 {
        return compute_mles_product_sum_d16(mles, claim, eq_poly);
    }

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

    // Evaluate g(X) / eq(X, r[round]) at [1, 2, ..., |mles| - 1, ∞] using split-eq fold
    // for the generic (non-16) case.
    let current_scalar = eq_poly.get_current_scalar();
    let sum_evals: Vec<F> = eq_poly
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
        .collect();

    finish_mles_product_sum_from_evals(&sum_evals, claim, eq_poly)
}

/// Specialized implementation of `compute_mles_product_sum` for `d = 16`.
///
/// This is the main zkVM setting (see `instruction_lookups::D`) and benefits from
/// a fully stack-allocated path that uses the optimized 16-way interpolation
/// kernels via `par_fold_out_in_unreduced`.
#[inline]
fn compute_mles_product_sum_d16<F: JoltField>(
    mles: &[RaPolynomial<u8, F>],
    claim: F,
    eq_poly: &GruenSplitEqPolynomial<F>,
) -> UniPoly<F> {
    debug_assert_eq!(mles.len(), 16);

    let current_scalar = eq_poly.get_current_scalar();

    // For each group index `g`, build the 16 pairs (p_j(0), p_j(1)) and evaluate
    // the product over j on U_16 = [1..15, ∞] using the optimized kernel.
    //
    // `par_fold_out_in_unreduced` handles the split-eq structure and returns the
    // combined sum over (x_out, x_in) after Montgomery reduction.
    let sum_evals_arr: [F; 16] = eq_poly.par_fold_out_in_unreduced::<9, 16>(&|g| {
        // Build pairs[(p0, p1); 16] on the stack.
        let pairs: [(F, F); 16] = core::array::from_fn(|i| {
            let p0 = mles[i].get_bound_coeff(2 * g);
            let p1 = mles[i].get_bound_coeff(2 * g + 1);
            (p0, p1)
        });

        // Evaluate the product of the 16 linear polynomials on the 16-point grid.
        let mut endpoints = [F::zero(); 16];
        eval_linear_prod_assign(&pairs, &mut endpoints);
        endpoints
    });

    let sum_evals: Vec<F> = sum_evals_arr
        .into_iter()
        .map(|x| x * current_scalar)
        .collect();

    finish_mles_product_sum_from_evals(&sum_evals, claim, eq_poly)
}

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
        8 => {
            debug_assert!(evals.len() >= 8);
            let p = unsafe { &*(pairs.as_ptr() as *const [(F, F); 8]) };
            eval_prod_8_assign(p, evals)
        }
        16 => {
            debug_assert!(evals.len() >= 16);
            let p = unsafe { &*(pairs.as_ptr() as *const [(F, F); 16]) };
            eval_prod_16_assign(p, evals)
        }
        32 => {
            debug_assert!(evals.len() >= 32);
            let p = unsafe { &*(pairs.as_ptr() as *const [(F, F); 32]) };
            eval_prod_32_assign(p, evals)
        }
        _ => eval_linear_prod_naive_assign(pairs, evals),
    }
}

/// Evaluate the product of 2 linear polynomials at the small internal grid used
/// by the interpolation routines (returns values at 1, 2, and ∞).
///
/// This helper is only used inside higher-degree kernels and is not exposed
/// directly through `eval_linear_prod_assign`.
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
fn eval_prod_2_assign<F: JoltField>(p: &[(F, F); 2], outputs: &mut [F]) {
    outputs[0] = p[0].1 * p[1].1; // 1
    outputs[1] = (p[0].1 - p[0].0) * (p[1].1 - p[1].0); // ∞
}

/// Evaluate the product of 3 linear polynomials on `U_3 = [1, 2, ∞]`.
///
/// Given `pairs[j] = (p_j(0), p_j(1))` and `P(x) = ∏_j p_j(x)`, this writes:
/// - `outputs[0] = P(1)`
/// - `outputs[1] = P(2)`
/// - `outputs[2] = P(∞)`
fn eval_prod_3_assign<F: JoltField>(pairs: &[(F, F); 3], outputs: &mut [F]) {
    let (a1, a2, a_inf) = eval_linear_prod_2_internal(pairs[0], pairs[1]);
    let (b0, b1) = pairs[2];
    let b_inf = b1 - b0;
    let b2 = b1 + b_inf;
    outputs[0] = a1 * b1;
    outputs[1] = a2 * b2;
    outputs[2] = a_inf * b_inf;
}

/// Evaluate the product of 4 linear polynomials at the internal interpolation
/// grid used by the higher-degree kernels.
///
/// Returns 5 values corresponding to evaluations at points `[1, 2, 3, 4, ∞]`.
/// Only a subset of these points are exposed in `eval_prod_4_assign`, which
/// adheres to the public `U_4 = [1, 2, 3, ∞]` grid.
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

    #[test]
    fn optimized_product_sum_matches_naive_2_mles() {
        check_optimized_product_sum_matches_naive::<2>();
    }

    #[test]
    fn optimized_product_sum_matches_naive_4_mles() {
        check_optimized_product_sum_matches_naive::<4>();
    }

    #[test]
    fn optimized_product_sum_matches_naive_8_mles() {
        check_optimized_product_sum_matches_naive::<8>();
    }

    #[test]
    fn optimized_product_sum_matches_naive_16_mles() {
        check_optimized_product_sum_matches_naive::<16>();
    }

    #[test]
    fn optimized_product_sum_matches_naive_32_mles() {
        check_optimized_product_sum_matches_naive::<32>();
    }

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
}
