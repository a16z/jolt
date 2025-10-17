use crate::field::JoltField;
use crate::poly::lagrange_poly::LagrangePolynomial;
use crate::poly::unipoly::UniPoly;
use crate::utils::accumulation as accum;
use crate::zkvm::r1cs::constraints::{
    eval_az_first_group, eval_az_second_group, eval_bz_first_group, eval_bz_second_group,
    NUM_REMAINING_R1CS_CONSTRAINTS, UNIVARIATE_SKIP_DOMAIN_SIZE,
};
use crate::zkvm::r1cs::inputs::R1CSCycleInputs;

/// Shared handoff state from a univariate-skip first round.
///
/// This bundles the claim after s1, the uni-skip challenge r0, and the tau vector
/// used to parameterize the Lagrange kernel and cycle eq polynomial.
#[derive(Clone, Debug)]
pub struct UniSkipState<F: JoltField> {
    pub claim_after_first: F,
    pub r0: F::Challenge,
    pub tau: Vec<F::Challenge>,
}

#[inline]
pub fn compute_az_r_group0<F: JoltField>(row: &R1CSCycleInputs, lagrange_evals_r: &[F]) -> F {
    // Group 0 Az are booleans; accumulate field elements unreduced, then Barrett-reduce
    let az_flags = eval_az_first_group(row);
    let mut acc: accum::Acc5Unsigned<F> = accum::acc5u_new::<F>();
    let mut i = 0;
    while i < UNIVARIATE_SKIP_DOMAIN_SIZE {
        if az_flags[i] {
            accum::acc5u_add_field(&mut acc, &lagrange_evals_r[i]);
        }
        i += 1;
    }
    accum::acc5u_reduce(&acc)
}

#[inline]
pub fn compute_bz_r_group0<F: JoltField>(row: &R1CSCycleInputs, lagrange_evals_r: &[F]) -> F {
    // Group 0 Bz are i128; accumulate field * i128 (converted) unreduced, then Barrett-reduce
    let bz_vals = eval_bz_first_group(row);
    let mut acc: accum::Acc6Signed<F> = accum::acc6s_new::<F>();
    let mut i = 0;
    while i < UNIVARIATE_SKIP_DOMAIN_SIZE {
        accum::acc6s_fmadd_i128(&mut acc, &lagrange_evals_r[i], bz_vals[i]);
        i += 1;
    }
    accum::acc6s_reduce(&acc)
}

#[inline]
pub fn compute_az_r_group1<F: JoltField>(row: &R1CSCycleInputs, lagrange_evals_r: &[F]) -> F {
    // Group 1 Az are u8 (nonnegative); accumulate field * i32 unreduced, then Barrett-reduce
    let az_vals_u8 = eval_az_second_group(row);
    let mut acc: accum::Acc6Signed<F> = accum::acc6s_new::<F>();
    let mut i = 0;
    while i < NUM_REMAINING_R1CS_CONSTRAINTS {
        let v_i32: i128 = (az_vals_u8[i] as i32) as i128;
        accum::acc6s_fmadd_i128(&mut acc, &lagrange_evals_r[i], v_i32);
        i += 1;
    }
    accum::acc6s_reduce(&acc)
}

#[inline]
pub fn compute_bz_r_group1<F: JoltField>(row: &R1CSCycleInputs, lagrange_evals_r: &[F]) -> F {
    // Group 1 Bz are S160; accumulate field * S160 in 7-limb signed accumulators, then Barrett-reduce once
    let bz_vals = eval_bz_second_group(row);
    let mut acc: accum::Acc7Signed<F> = accum::acc7s_new::<F>();
    let mut i = 0;
    while i < NUM_REMAINING_R1CS_CONSTRAINTS {
        accum::acc7s_fmadd_s160(&mut acc, &lagrange_evals_r[i], bz_vals[i]);
        i += 1;
    }
    accum::acc7s_reduce(&acc)
}

/// Returns the interleaved symmetric univariate-skip target indices outside the base window.
///
/// Domain is assumed to be the canonical symmetric window of size DOMAIN_SIZE with
/// base indices from start = -((DOMAIN_SIZE-1)/2) to end = start + DOMAIN_SIZE - 1.
///
/// Targets are the extended points z ∈ {−DEGREE..−1} ∪ {1..DEGREE}, interleaved as
/// [start-1, end+1, start-2, end+2, ...] until DEGREE points are produced.
#[inline]
pub fn uniskip_targets<const DOMAIN_SIZE: usize, const DEGREE: usize>() -> [i64; DEGREE] {
    let d: i64 = DEGREE as i64;
    let ext_left: i64 = -d;
    let ext_right: i64 = d;
    let base_left: i64 = -((DOMAIN_SIZE as i64 - 1) / 2);
    let base_right: i64 = base_left + (DOMAIN_SIZE as i64) - 1;

    let mut targets: [i64; DEGREE] = [0; DEGREE];
    let mut idx = 0usize;
    let mut n = base_left - 1;
    let mut p = base_right + 1;

    while n >= ext_left && p <= ext_right && idx < DEGREE {
        targets[idx] = n;
        idx += 1;
        if idx >= DEGREE {
            break;
        }
        targets[idx] = p;
        idx += 1;
        n -= 1;
        p += 1;
    }

    while idx < DEGREE && n >= ext_left {
        targets[idx] = n;
        idx += 1;
        n -= 1;
    }

    while idx < DEGREE && p <= ext_right {
        targets[idx] = p;
        idx += 1;
        p += 1;
    }

    debug_assert_eq!(idx, DEGREE);
    targets
}

/// Builds the uni-skip first-round polynomial s1 from base and extended evaluations of t1.
///
/// SPECIFIC: This helper targets the setting where s1(Y) = L(τ_high, Y) · t1(Y), with L the
/// degree-(DOMAIN_SIZE-1) Lagrange kernel over the base window and t1 a univariate of degree
/// at most 2·DEGREE (extended symmetric window size EXTENDED_SIZE = 2·DEGREE + 1).
/// Consequently, the resulting s1 has degree at most 3·DEGREE (NUM_COEFFS = 3·DEGREE + 1).
///
/// Inputs:
/// - base_evals: t1 evaluated on the base window (symmetric grid of size DOMAIN_SIZE).
/// - extended_evals: t1 evaluated on the extended symmetric grid outside the base window,
///   in the order given by `uniskip_targets::<DOMAIN_SIZE, DEGREE>()`.
/// - tau_high: the challenge used in the Lagrange kernel L(τ_high, ·) over the base window.
///
/// Generic parameters:
/// - BASE_EVALS_ARE_ZERO: const bool optimization flag. When true, skips filling base_evals
///   (assumes they are all zero), saving a loop iteration.
///
/// Returns: UniPoly s1 with exactly NUM_COEFFS coefficients.
#[inline]
pub fn build_uniskip_first_round_poly<
    F: JoltField,
    const DOMAIN_SIZE: usize,
    const DEGREE: usize,
    const EXTENDED_SIZE: usize,
    const NUM_COEFFS: usize,
    const BASE_EVALS_ARE_ZERO: bool,
>(
    base_evals: &[F; DOMAIN_SIZE],
    extended_evals: &[F; DEGREE],
    tau_high: F::Challenge,
) -> UniPoly<F> {
    debug_assert_eq!(EXTENDED_SIZE, 2 * DEGREE + 1);
    debug_assert_eq!(NUM_COEFFS, 3 * DEGREE + 1);

    // Rebuild t1 on the full extended symmetric window
    let targets: [i64; DEGREE] = uniskip_targets::<DOMAIN_SIZE, DEGREE>();
    let mut t1_vals: [F; EXTENDED_SIZE] = [F::zero(); EXTENDED_SIZE];

    // Fill in base window evaluations (skip if all zero)
    if !BASE_EVALS_ARE_ZERO {
        let base_left: i64 = -((DOMAIN_SIZE as i64 - 1) / 2);
        for (i, &val) in base_evals.iter().enumerate() {
            let z = base_left + (i as i64);
            let pos = (z + (DEGREE as i64)) as usize;
            t1_vals[pos] = val;
        }
    }

    // Fill in extended evaluations (outside base window)
    for (idx, &val) in extended_evals.iter().enumerate() {
        let z = targets[idx];
        let pos = (z + (DEGREE as i64)) as usize;
        t1_vals[pos] = val;
    }

    let t1_coeffs = LagrangePolynomial::<F>::interpolate_coeffs::<EXTENDED_SIZE>(&t1_vals);
    let lagrange_values = LagrangePolynomial::<F>::evals::<F::Challenge, DOMAIN_SIZE>(&tau_high);
    let lagrange_coeffs =
        LagrangePolynomial::<F>::interpolate_coeffs::<DOMAIN_SIZE>(&lagrange_values);

    let mut s1_coeffs: [F; NUM_COEFFS] = [F::zero(); NUM_COEFFS];
    for (i, &a) in lagrange_coeffs.iter().enumerate() {
        for (j, &b) in t1_coeffs.iter().enumerate() {
            s1_coeffs[i + j] += a * b;
        }
    }

    UniPoly::from_coeff(s1_coeffs.to_vec())
}
