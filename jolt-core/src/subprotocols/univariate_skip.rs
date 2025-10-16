use crate::field::JoltField;
use crate::poly::lagrange_poly::LagrangePolynomial;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::univariate_skip::accum::{
    acc5_add_field, acc5_new, acc5_reduce, acc6_fmadd_i128, acc6_new, acc6_reduce,
    accs160_fmadd_s160, accs160_new, accs160_reduce, Acc5, Acc6, AccS160,
};
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

// Accumulation primitives (unreduced (signed) accumulation + final reduction)
pub mod accum {
    use crate::field::{FmaddTrunc, JoltField};
    use ark_ff::biginteger::{I8OrI96, S160};

    /// Local helper to convert `S160` to field without using `.to_field()`
    #[inline]
    pub fn s160_to_field<F: JoltField>(bz: &S160) -> F {
        if bz.is_zero() {
            return F::zero();
        }
        let lo = bz.magnitude_lo();
        let hi = bz.magnitude_hi() as u64;
        let r64 = F::from_u128(1u128 << 64);
        let r128 = r64 * r64;
        let acc = F::from_u64(lo[0]) + F::from_u64(lo[1]) * r64 + F::from_u64(hi) * r128;
        if bz.is_positive() {
            acc
        } else {
            -acc
        }
    }

    // Accumulator for boolean-weighted field sums using 5-limb Barrett reduction
    // Booleans are non-negative (0/1), so we only need a positive accumulator.
    pub type Acc5<F> = <F as JoltField>::Unreduced<5>;

    #[inline(always)]
    pub fn acc5_new<F: JoltField>() -> Acc5<F> {
        <F as JoltField>::Unreduced::<5>::from([0u64; 5])
    }

    #[inline(always)]
    pub fn acc5_add_field<F: JoltField>(acc: &mut Acc5<F>, val: &F) {
        *acc += *val.as_unreduced_ref();
    }

    #[inline(always)]
    pub fn acc5_reduce<F: JoltField>(acc: &Acc5<F>) -> F {
        F::from_barrett_reduce(*acc)
    }

    // Accumulator for field * I8OrI96 using 6-limb Barrett reduction
    pub type Acc6<F> = (
        <F as JoltField>::Unreduced<6>,
        <F as JoltField>::Unreduced<6>,
    );

    #[inline(always)]
    pub fn acc6_new<F: JoltField>() -> Acc6<F> {
        (
            <F as JoltField>::Unreduced::<6>::from([0u64; 6]),
            <F as JoltField>::Unreduced::<6>::from([0u64; 6]),
        )
    }

    /// fmadd with I8OrI96 by converting to i128 and using mul_u128_unreduced (6 limbs)
    #[inline(always)]
    pub fn acc6_fmadd_i8ori96<F: JoltField>(acc: &mut Acc6<F>, field: &F, az: I8OrI96) {
        let v = az.to_i128();
        if v == 0 {
            return;
        }
        let abs = v.unsigned_abs();
        let term = (*field).mul_u128_unreduced(abs);
        if v > 0 {
            acc.0 += term;
        } else {
            acc.1 += term;
        }
    }

    #[inline(always)]
    pub fn acc6_reduce<F: JoltField>(acc: &Acc6<F>) -> F {
        F::from_barrett_reduce(acc.0) - F::from_barrett_reduce(acc.1)
    }

    /// fmadd with i128 using mul_u128_unreduced (6 limbs) for signed accumulation
    #[inline(always)]
    pub fn acc6_fmadd_i128<F: JoltField>(acc: &mut Acc6<F>, field: &F, v: i128) {
        if v == 0 {
            return;
        }
        let abs = v.unsigned_abs();
        let term = (*field).mul_u128_unreduced(abs);
        if v > 0 {
            acc.0 += term;
        } else {
            acc.1 += term;
        }
    }

    // Accumulator for field * S160 using 3-limb fmadd into 7-limb accumulators
    // Use the implementor-associated Acc<7> type to match fmadd_trunc binding exactly.
    // Reduce with Barrett to avoid Montgomery factors.
    type Acc7<F> = <<F as JoltField>::Unreduced<4> as FmaddTrunc>::Acc<7>;
    pub type AccS160<F> = (Acc7<F>, Acc7<F>);

    #[inline(always)]
    pub fn accs160_new<F: JoltField>() -> AccS160<F> {
        (
            <F as JoltField>::Unreduced::<7>::from([0u64; 7]),
            <F as JoltField>::Unreduced::<7>::from([0u64; 7]),
        )
    }

    /// fmadd with S160 (3-limb magnitude) using fmadd_trunc into 7-limb accumulators
    #[inline(always)]
    pub fn accs160_fmadd_s160<F: JoltField>(acc: &mut AccS160<F>, field: &F, bz: S160) {
        if bz.is_zero() {
            return;
        }
        let lo = bz.magnitude_lo();
        let hi = bz.magnitude_hi() as u64;
        let mag = <F as JoltField>::Unreduced::from([lo[0], lo[1], hi]);
        let field_bigint = field.as_unreduced_ref();
        if bz.is_positive() {
            field_bigint.fmadd_trunc::<3, 7>(&mag, &mut acc.0);
        } else {
            field_bigint.fmadd_trunc::<3, 7>(&mag, &mut acc.1);
        }
    }

    #[inline(always)]
    pub fn accs160_reduce<F: JoltField>(acc: &AccS160<F>) -> F {
        F::from_barrett_reduce(acc.0) - F::from_barrett_reduce(acc.1)
    }
}

#[inline]
pub fn compute_az_r_group0<F: JoltField>(row: &R1CSCycleInputs, lagrange_evals_r: &[F]) -> F {
    // Group 0 Az are booleans; accumulate field elements unreduced, then Barrett-reduce
    let az_flags = eval_az_first_group(row);
    let mut acc: Acc5<F> = acc5_new::<F>();
    let mut i = 0;
    while i < UNIVARIATE_SKIP_DOMAIN_SIZE {
        if az_flags[i] {
            acc5_add_field(&mut acc, &lagrange_evals_r[i]);
        }
        i += 1;
    }
    acc5_reduce(&acc)
}

#[inline]
pub fn compute_bz_r_group0<F: JoltField>(row: &R1CSCycleInputs, lagrange_evals_r: &[F]) -> F {
    // Group 0 Bz are i128; accumulate field * i128 (converted) unreduced, then Barrett-reduce
    let bz_vals = eval_bz_first_group(row);
    let mut acc: Acc6<F> = acc6_new::<F>();
    let mut i = 0;
    while i < UNIVARIATE_SKIP_DOMAIN_SIZE {
        acc6_fmadd_i128(&mut acc, &lagrange_evals_r[i], bz_vals[i]);
        i += 1;
    }
    acc6_reduce(&acc)
}

#[inline]
pub fn compute_az_r_group1<F: JoltField>(row: &R1CSCycleInputs, lagrange_evals_r: &[F]) -> F {
    // Group 1 Az are u8 (nonnegative); accumulate field * i32 unreduced, then Barrett-reduce
    let az_vals_u8 = eval_az_second_group(row);
    let mut acc: Acc6<F> = acc6_new::<F>();
    let mut i = 0;
    while i < NUM_REMAINING_R1CS_CONSTRAINTS {
        let v_i32: i128 = (az_vals_u8[i] as i32) as i128;
        acc6_fmadd_i128(&mut acc, &lagrange_evals_r[i], v_i32);
        i += 1;
    }
    acc6_reduce(&acc)
}

#[inline]
pub fn compute_bz_r_group1<F: JoltField>(row: &R1CSCycleInputs, lagrange_evals_r: &[F]) -> F {
    // Group 1 Bz are S160; accumulate field * S160 unreduced, then reduce once
    let bz_vals = eval_bz_second_group(row);
    let mut acc: AccS160<F> = accs160_new::<F>();
    let mut i = 0;
    while i < NUM_REMAINING_R1CS_CONSTRAINTS {
        accs160_fmadd_s160(&mut acc, &lagrange_evals_r[i], bz_vals[i]);
        i += 1;
    }
    accs160_reduce(&acc)
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

/// Builds the uni-skip first-round polynomial s1 from extended evaluations of t1.
///
/// SPECIFIC: This helper targets the setting where s1(Y) = L(τ_high, Y) · t1(Y), with L the
/// degree-(DOMAIN_SIZE-1) Lagrange kernel over the base window and t1 a univariate of degree
/// at most 2·DEGREE (extended symmetric window size EXTENDED_SIZE = 2·DEGREE + 1).
/// Consequently, the resulting s1 has degree at most 3·DEGREE (NUM_COEFFS = 3·DEGREE + 1).
///
/// Inputs:
/// - extended_evals: t1 evaluated on the extended symmetric grid outside the base window,
///   in the order given by `uniskip_targets::<DOMAIN_SIZE, DEGREE>()`.
/// - tau_high: the challenge used in the Lagrange kernel L(τ_high, ·) over the base window.
///
/// Returns: UniPoly s1 with exactly NUM_COEFFS coefficients.
#[inline]
pub fn build_uniskip_first_round_poly<
    F: JoltField,
    const DOMAIN_SIZE: usize,
    const DEGREE: usize,
    const EXTENDED_SIZE: usize,
    const NUM_COEFFS: usize,
>(
    extended_evals: &[F; DEGREE],
    tau_high: F::Challenge,
) -> UniPoly<F> {
    debug_assert_eq!(EXTENDED_SIZE, 2 * DEGREE + 1);
    debug_assert_eq!(NUM_COEFFS, 3 * DEGREE + 1);

    // Rebuild t1 on the full extended symmetric window
    let targets: [i64; DEGREE] = uniskip_targets::<DOMAIN_SIZE, DEGREE>();
    let mut t1_vals: [F; EXTENDED_SIZE] = [F::zero(); EXTENDED_SIZE];
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
