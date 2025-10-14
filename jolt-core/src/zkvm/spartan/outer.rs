use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_ff::biginteger::S160;
use ark_std::Zero;
use rayon::prelude::*;
use std::sync::Arc;
use tracer::instruction::Cycle;

use crate::field::{JoltField, OptimizedMul};
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::lagrange_poly::{LagrangeHelper, LagrangePolynomial};
use crate::poly::multilinear_polynomial::BindingOrder;
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck::{SumcheckInstance, UniSkipFirstRoundInstance};
use crate::transcripts::Transcript;
use crate::utils::math::Math;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
// use crate::utils::thread::drop_in_background_thread;
use crate::poly::opening_proof::BIG_ENDIAN;
use crate::poly::opening_proof::{
    OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
};
use crate::utils::univariate_skip::accum::{accs160_fmadd_s160, accs160_new, accs160_reduce};
use crate::utils::univariate_skip::{
    compute_az_r_group0, compute_az_r_group1, compute_bz_r_group0, compute_bz_r_group1,
    compute_cz_r_group1,
};
use crate::zkvm::r1cs::inputs::{
    compute_claimed_witness_evals, ALL_R1CS_INPUTS, COMMITTED_R1CS_INPUTS,
};
use crate::zkvm::r1cs::{
    constraints::{
        eval_az_first_group, eval_az_second_group, eval_bz_first_group, eval_bz_second_group,
        eval_cz_second_group, FIRST_ROUND_POLY_NUM_COEFFS, UNIVARIATE_SKIP_DEGREE,
        UNIVARIATE_SKIP_DOMAIN_SIZE, UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE,
    },
    inputs::R1CSCycleInputs,
};
use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use crate::zkvm::JoltSharedPreprocessing;
use crate::zkvm::dag::state_manager::StateManager;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;

#[inline]
fn outer_debug_enabled() -> bool {
    std::env::var("JOLT_DEBUG_OUTER").is_ok()
}

// Spartan Outer sumcheck (with univariate-skip first round on Z)
//
// We define a univariate in Z first-round polynomial
//   s1(Z) := L_Z(τ_high) · Σ_{x_out ∈ {0,1}^{m_out}} Σ_{x_in ∈ {0,1}^{m_in}}
//              E_out(r_out, x_out) · E_in(r_in, x_in) ·
//              [ Az(x_out, x_in, Z) · Bz(x_out, x_in, Z) − Cz(x_out, x_in, Z) ],
// where L_Z(τ_high) is the Lagrange basis polynomial over the univariate-skip
// base domain evaluated at τ_high, and Az(·,·,Z), Bz(·,·,Z), Cz(·,·,Z) are the
// per-row univariate polynomials in Z induced by the R1CS row (split into two
// internal groups in code, but algebraically composing to Az·Bz − Cz at Z).
// The prover sends s1(Z) via univariate-skip by evaluating t1(Z) := Σ Σ E_out·E_in·(Az·Bz−Cz)
// on an extended grid Z ∈ {−D..D} outside the base window, interpolating t1,
// multiplying by L_Z(τ_high) to obtain s1, and the verifier samples r0.
//
// Subsequent outer rounds bind the cycle variables r_tail = (r1, r2, …) using
// a streaming first cycle-bit round followed by linear-time rounds:
//   • Streaming round (after r0): compute
//       t(0)  = Σ_{x_out} E_out · Σ_{x_in} E_in · (Az(0)·Bz(0) − Cz(0))
//       t(∞)  = Σ_{x_out} E_out · Σ_{x_in} E_in · ((Az(1)−Az(0))·(Bz(1)−Bz(0)))
//     send a cubic built from these endpoints, and bind cached coefficients by r1.
//   • Remaining rounds: reuse bound coefficients to compute the same endpoints
//     in linear time for each subsequent bit and bind by r_i.
//
// Final check (verifier): with r = [r0 || r_tail] and outer binding order from
// the top, evaluate Eq_τ(τ, r) and verify
//   Eq_τ(τ, r) · (Az(r) · Bz(r) − Cz(r)).
//
// Notation and layout:
// - E_out/E_in are the split-eq factors from GruenSplitEqPolynomial (LowToHigh).
// - D = UNIVARIATE_SKIP_DEGREE; base = UNIVARIATE_SKIP_DOMAIN_SIZE.
// - Coefficients are interleaved per (x_out, x_in) in blocks of 6:
//   [Az(0), Bz(0), Cz(0), Az(1), Bz(1), Cz(1)].

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct SparseCoefficient<T> {
    pub(crate) index: usize,
    pub(crate) value: T,
}

impl<T> Allocative for SparseCoefficient<T> {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

impl<T> From<(usize, T)> for SparseCoefficient<T> {
    fn from(x: (usize, T)) -> Self {
        Self {
            index: x.0,
            value: x.1,
        }
    }
}

#[derive(Clone, Debug, Allocative, Default)]
pub struct SpartanInterleavedPoly<F: JoltField> {
    /// The bound coefficients for the Az, Bz, Cz polynomials.
    /// Will be populated in the streaming round (after SVO rounds)
    pub(crate) bound_coeffs: Vec<SparseCoefficient<F>>,

    binding_scratch_space: Vec<SparseCoefficient<F>>,
}

impl<F: JoltField> SpartanInterleavedPoly<F> {
    pub fn new() -> Self {
        Self {
            bound_coeffs: vec![],
            binding_scratch_space: vec![],
        }
    }
}

/// Uni-skip instance for Spartan outer sumcheck, computing the first-round polynomial only.
pub struct OuterUniSkipInstance<F: JoltField> {
    preprocess: Arc<JoltSharedPreprocessing>,
    trace: Arc<Vec<Cycle>>,
    tau: Vec<F::Challenge>,
}

impl<F: JoltField> OuterUniSkipInstance<F> {
    pub fn new(
        preprocess: &JoltSharedPreprocessing,
        trace: &[Cycle],
        tau: &[F::Challenge],
    ) -> Self {
        Self {
            preprocess: Arc::new(preprocess.clone()),
            trace: Arc::new(trace.to_vec()),
            tau: tau.to_vec(),
        }
    }

    #[inline]
    fn univariate_skip_targets() -> [i64; UNIVARIATE_SKIP_DEGREE] {
        let d: i64 = UNIVARIATE_SKIP_DEGREE as i64;
        let ext_left: i64 = -d;
        let ext_right: i64 = d;
        let base_left: i64 = -((UNIVARIATE_SKIP_DOMAIN_SIZE as i64 - 1) / 2);
        let base_right: i64 = base_left + (UNIVARIATE_SKIP_DOMAIN_SIZE as i64) - 1;

        let mut targets: [i64; UNIVARIATE_SKIP_DEGREE] = [0; UNIVARIATE_SKIP_DEGREE];
        let mut idx = 0usize;
        let mut n = base_left - 1; // first index just left of the base window
        let mut p = base_right + 1; // first index just right of the base window

        // Interleave negatives and positives while both sides have items
        while n >= ext_left && p <= ext_right && idx < UNIVARIATE_SKIP_DEGREE {
            targets[idx] = n;
            idx += 1;
            if idx >= UNIVARIATE_SKIP_DEGREE {
                break;
            }
            targets[idx] = p;
            idx += 1;
            n -= 1;
            p += 1;
        }

        // Append any remaining on the left side
        while idx < UNIVARIATE_SKIP_DEGREE && n >= ext_left {
            targets[idx] = n;
            idx += 1;
            n -= 1;
        }

        // Append any remaining on the right side
        while idx < UNIVARIATE_SKIP_DEGREE && p <= ext_right {
            targets[idx] = p;
            idx += 1;
            p += 1;
        }

        debug_assert_eq!(idx, UNIVARIATE_SKIP_DEGREE);
        targets
    }

    #[tracing::instrument(skip_all, name = "OuterUniSkipInstance::compute_univariate_skip_extended_evals")]
    fn compute_univariate_skip_extended_evals(
        &self,
        split_eq: &GruenSplitEqPolynomial<F>,
    ) -> [F; UNIVARIATE_SKIP_DEGREE] {
        // Precompute Lagrange coefficient vectors for target Z values outside the base window
        let base_left: i64 = -((UNIVARIATE_SKIP_DOMAIN_SIZE as i64 - 1) / 2);
        let d: i64 = UNIVARIATE_SKIP_DEGREE as i64;
        let ext_left: i64 = -d;
        let ext_right: i64 = d;
        let base_right: i64 = base_left + (UNIVARIATE_SKIP_DOMAIN_SIZE as i64) - 1;

        let mut targets: [i64; UNIVARIATE_SKIP_DEGREE] = [0; UNIVARIATE_SKIP_DEGREE];
        let mut idx = 0usize;
        let mut n = base_left - 1; // first index just left of the base window
        let mut p = base_right + 1; // first index just right of the base window
        while n >= ext_left && p <= ext_right && idx < UNIVARIATE_SKIP_DEGREE {
            targets[idx] = n;
            idx += 1;
            if idx >= UNIVARIATE_SKIP_DEGREE { break; }
            targets[idx] = p;
            idx += 1;
            n -= 1;
            p += 1;
        }
        while idx < UNIVARIATE_SKIP_DEGREE && n >= ext_left { targets[idx] = n; idx += 1; n -= 1; }
        while idx < UNIVARIATE_SKIP_DEGREE && p <= ext_right { targets[idx] = p; idx += 1; p += 1; }
        debug_assert_eq!(idx, UNIVARIATE_SKIP_DEGREE);

        let target_shifts: [i64; UNIVARIATE_SKIP_DEGREE] = core::array::from_fn(|j| targets[j] - base_left);
        let coeffs_per_j: [[i32; UNIVARIATE_SKIP_DOMAIN_SIZE]; UNIVARIATE_SKIP_DEGREE] =
            core::array::from_fn(|j| LagrangeHelper::shift_coeffs_i32::<UNIVARIATE_SKIP_DOMAIN_SIZE>(target_shifts[j]));

        let num_x_out_vals = split_eq.E_out_current_len();
        let num_x_in_vals = split_eq.E_in_current_len();
        let num_parallel_chunks = if num_x_out_vals > 0 {
            core::cmp::min(num_x_out_vals, rayon::current_num_threads().next_power_of_two() * 8)
        } else { 1 };
        let x_out_chunk_size = if num_x_out_vals > 0 { core::cmp::max(1, num_x_out_vals.div_ceil(num_parallel_chunks)) } else { 0 };
        let iter_num_x_in_vars = if num_x_in_vals > 0 { num_x_in_vals.log_2() } else { 0 };

        let extended: [F; UNIVARIATE_SKIP_DEGREE] = (0..num_parallel_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let x_out_start = chunk_idx * x_out_chunk_size;
                let x_out_end = core::cmp::min((chunk_idx + 1) * x_out_chunk_size, num_x_out_vals);
                let mut acc_field: [F; UNIVARIATE_SKIP_DEGREE] = [F::zero(); UNIVARIATE_SKIP_DEGREE];

                for x_out_val in x_out_start..x_out_end {
                    let mut inner_acc: [<F as JoltField>::Unreduced<9>; UNIVARIATE_SKIP_DEGREE] =
                        [<F as JoltField>::Unreduced::<9>::zero(); UNIVARIATE_SKIP_DEGREE];

                    for x_in_val in 0..num_x_in_vals {
                        let current_step_idx = (x_out_val << iter_num_x_in_vars) | x_in_val;
                        let row_inputs = R1CSCycleInputs::from_trace::<F>(&*self.preprocess, self.trace.as_slice(), current_step_idx);

                        let e_in = if num_x_in_vals == 0 { F::one() } else { split_eq.E_in_current()[x_in_val] };

                        let az1_bool = eval_az_first_group(&row_inputs);
                        let bz1_s160 = eval_bz_first_group(&row_inputs);
                        let az2_i96 = eval_az_second_group(&row_inputs);
                        let bz2 = eval_bz_second_group(&row_inputs);
                        let cz2 = eval_cz_second_group(&row_inputs);

                        let mut az2_i128_padded: [i128; UNIVARIATE_SKIP_DOMAIN_SIZE] = [0; UNIVARIATE_SKIP_DOMAIN_SIZE];
                        let mut bz2_s160_padded: [S160; UNIVARIATE_SKIP_DOMAIN_SIZE] = [S160::from(0i128); UNIVARIATE_SKIP_DOMAIN_SIZE];
                        let mut cz2_s160_padded: [S160; UNIVARIATE_SKIP_DOMAIN_SIZE] = [S160::from(0i128); UNIVARIATE_SKIP_DOMAIN_SIZE];
                        for i in 0..UNIVARIATE_SKIP_DOMAIN_SIZE {
                            az2_i128_padded[i] = az2_i96[i].to_i128();
                            bz2_s160_padded[i] = bz2[i];
                            cz2_s160_padded[i] = cz2[i];
                        }

                        for j in 0..UNIVARIATE_SKIP_DEGREE {
                            let coeffs = &coeffs_per_j[j];
                            let mut az1_csum: i64 = 0;
                            let mut bz1_acc = accs160_new::<F>();
                            for i in 0..UNIVARIATE_SKIP_DOMAIN_SIZE {
                                let c = coeffs[i] as i64;
                                if c == 0 { continue; }
                                if az1_bool[i] { az1_csum += c; }
                                accs160_fmadd_s160(&mut bz1_acc, &F::from_i64(c), bz1_s160[i]);
                            }
                            let bz1_ext: F = accs160_reduce::<F>(&bz1_acc);
                            inner_acc[j] += e_in.mul_unreduced::<9>(bz1_ext.mul_i64(az1_csum));

                            let mut az2_sum: i128 = 0;
                            let mut bz2_acc = accs160_new::<F>();
                            let mut cz2_acc = accs160_new::<F>();
                            for i in 0..UNIVARIATE_SKIP_DOMAIN_SIZE {
                                let c = coeffs[i] as i64;
                                if c == 0 { continue; }
                                az2_sum += az2_i128_padded[i] * (c as i128);
                                accs160_fmadd_s160(&mut bz2_acc, &F::from_i64(c), bz2_s160_padded[i]);
                                accs160_fmadd_s160(&mut cz2_acc, &F::from_i64(c), cz2_s160_padded[i]);
                            }
                            let az2_ext = F::from_i128(az2_sum);
                            let bz2_ext = accs160_reduce::<F>(&bz2_acc);
                            let cz2_ext = accs160_reduce::<F>(&cz2_acc);
                            inner_acc[j] += e_in.mul_unreduced::<9>(az2_ext * bz2_ext - cz2_ext);
                        }
                    }
                    let e_out = if num_x_out_vals > 0 { split_eq.E_out_current()[x_out_val] } else { F::zero() };
                    for j in 0..UNIVARIATE_SKIP_DEGREE {
                        let reduced = F::from_montgomery_reduce::<9>(inner_acc[j]);
                        acc_field[j] += e_out * reduced;
                    }
                }
                acc_field
            })
            .reduce(
                || [F::zero(); UNIVARIATE_SKIP_DEGREE],
                |mut a, b| { for j in 0..UNIVARIATE_SKIP_DEGREE { a[j] += b[j]; } a },
            )
        ;

        if outer_debug_enabled() {
            let s = if UNIVARIATE_SKIP_DEGREE < 4 { UNIVARIATE_SKIP_DEGREE } else { 4 };
            let mut sample = Vec::with_capacity(s);
            for i in 0..s { sample.push(extended[i]); }
            eprintln!("[outer/uniskip] extended sample (j=0..{}): {:?}", s - 1, sample);
        }

        extended
    }
}

impl<F: JoltField, T: Transcript> UniSkipFirstRoundInstance<F, T> for OuterUniSkipInstance<F> {
    const DEGREE_BOUND: usize = FIRST_ROUND_POLY_NUM_COEFFS - 1;
    const DOMAIN_SIZE: usize = UNIVARIATE_SKIP_DOMAIN_SIZE;

    fn input_claim(&self) -> F {
        F::zero()
    }

    fn compute_poly(&mut self) -> UniPoly<F> {
        // Compute extended univariate-skip evaluations directly without instantiating remainder
        let tau_low = &self.tau[0..self.tau.len() - 1];
        let split_eq = GruenSplitEqPolynomial::<F>::new(tau_low, BindingOrder::LowToHigh);
        let extended = self.compute_univariate_skip_extended_evals(&split_eq);

        // Rebuild s1(Z) without side effects on transcript
        let mut t1_vals: [F; 2 * UNIVARIATE_SKIP_DEGREE + 1] =
            [F::zero(); 2 * UNIVARIATE_SKIP_DEGREE + 1];
        let targets: [i64; UNIVARIATE_SKIP_DEGREE] = Self::univariate_skip_targets();
        for (idx, &val) in extended.iter().enumerate() {
            let z = targets[idx];
            let pos = (z + (UNIVARIATE_SKIP_DEGREE as i64)) as usize;
            t1_vals[pos] = val;
        }

        let t1_coeffs = LagrangePolynomial::<F>::interpolate_coeffs::<
            UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE,
        >(&t1_vals);

        let tau_high = self.tau[self.tau.len() - 1];
        let lagrange_poly_values = LagrangePolynomial::<F>::evals::<
            F::Challenge,
            UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&tau_high);
        if outer_debug_enabled() {
            let sum: F = lagrange_poly_values.iter().copied().sum();
            eprintln!("[outer/uniskip] sum L_i(r0) = {}", sum);
        }
        let lagrange_poly_coeffs =
            LagrangePolynomial::interpolate_coeffs::<UNIVARIATE_SKIP_DOMAIN_SIZE>(
                &lagrange_poly_values,
            );

        let mut s1_coeffs: [F; FIRST_ROUND_POLY_NUM_COEFFS] =
            [F::zero(); FIRST_ROUND_POLY_NUM_COEFFS];
        for (i, &a) in lagrange_poly_coeffs.iter().enumerate() {
            for (j, &b) in t1_coeffs.iter().enumerate() {
                s1_coeffs[i + j] += a * b;
            }
        }

        let poly = UniPoly::from_coeff(s1_coeffs.to_vec());
        if outer_debug_enabled() {
            let v0 = poly.evaluate::<F>(&F::from_u64(0));
            let v1 = poly.evaluate::<F>(&F::from_u64(1));
            eprintln!("[outer/uniskip] s1(0)={}, s1(1)={}", v0, v1);
        }
        poly
    }

    fn output_claim(&self, _r: &[F::Challenge]) -> F {
        // Not used by Spartan outer at this stage; verifier computes s1(r) directly from transcript
        F::zero()
    }
}

/// SumcheckInstance for Spartan outer rounds after the univariate-skip first round.
/// Round 0 in this instance corresponds to the "streaming" round; subsequent rounds
/// use the remaining linear-time algorithm over cycle variables.
pub struct OuterRemainingSumcheck<F: JoltField> {
    pub input_claim: F,
    pub interleaved_poly: SpartanInterleavedPoly<F>,
    pub split_eq_poly: GruenSplitEqPolynomial<F>,
    pub preprocess: Option<Arc<JoltSharedPreprocessing>>, // None on verifier
    pub trace: Option<Arc<Vec<Cycle>>>, // None on verifier
    pub r0_uniskip: F::Challenge,
    pub total_rounds: usize,
    /// Only used by verifier to compute expected_output_claim
    pub tau: Option<Vec<F::Challenge>>,
    /// Number of cycle bits for splitting opening points (consistent across prover/verifier)
    pub num_cycles_bits: usize,
    /// Cache for the streaming round (right after uniskip): t0, t_inf and bound6_at_r
    pub streaming_cache: Option<StreamingRoundCache<F>>,
}

pub struct StreamingRoundCache<F: JoltField> {
    pub t0: F,
    pub t_inf: F,
    pub bound6_at_r: Vec<SparseCoefficient<F>>,
}

impl<F: JoltField> OuterRemainingSumcheck<F> {
    /// Computes contiguous index ranges that emulate `par_chunk_by` grouping by `index / block_size`.
    /// Each range [start, end) contains coefficients belonging to the same block bucket.
    fn compute_block_ranges<T>(
        coeffs: &[SparseCoefficient<T>],
        block_size: usize,
    ) -> Vec<(usize, usize)> {
        if coeffs.is_empty() {
            return Vec::new();
        }
        // Safety/net: block_size must be a multiple of 6 to respect Az/Bz/Cz block layout
        debug_assert_eq!(block_size % 6, 0);

        let mut ranges = Vec::new();
        let mut start = 0usize;
        let mut current_bucket = coeffs[0].index / block_size;
        for (i, c) in coeffs.iter().enumerate().skip(1) {
            let bucket = c.index / block_size;
            if bucket != current_bucket {
                ranges.push((start, i));
                start = i;
                current_bucket = bucket;
            }
        }
        ranges.push((start, coeffs.len()));
        ranges
    }

    #[tracing::instrument(skip_all, name = "OuterRemainingSumcheck::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        input_claim: F,
        tau_low: &[F::Challenge],
        r0_uniskip: F::Challenge,
        total_rounds: usize,
        num_cycles_bits: usize,
    ) -> Self {
        let (preprocessing, trace, _program_io, _final_mem) = state_manager.get_prover_data();
        let split_eq_poly: GruenSplitEqPolynomial<F> =
            GruenSplitEqPolynomial::<F>::new(tau_low, BindingOrder::LowToHigh);

        // Prepare streaming round cache: compute t0, t_inf and bound6_at_r using r0
        let lagrange_evals_r = LagrangePolynomial::<F>::evals::<
            F::Challenge,
            UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&r0_uniskip);

        let num_x_out_vals = split_eq_poly.E_out_current_len();
        let num_x_in_vals = split_eq_poly.E_in_current_len();
        let iter_num_x_in_vars = if num_x_in_vals > 0 { num_x_in_vals.log_2() } else { 0 };

        let mut bound6_at_r: Vec<SparseCoefficient<F>> = Vec::new();
        // Reserve roughly 4 entries per block as in previous logic
        let mut reserve = num_x_out_vals.saturating_mul(core::cmp::max(1, num_x_in_vals)).saturating_mul(4);
        reserve = reserve.max(1024);
        bound6_at_r.reserve(reserve);

        let mut t0_acc = F::zero();
        let mut t_inf_acc = F::zero();

        for x_out_val in 0..num_x_out_vals {
            let mut inner_sum0 = F::Unreduced::<9>::zero();
            let mut inner_sumInf = F::Unreduced::<9>::zero();

            for x_in_val in 0..num_x_in_vals {
                let current_step_idx = (x_out_val << iter_num_x_in_vars) | x_in_val;
                let row_inputs = R1CSCycleInputs::from_trace::<F>(&preprocessing.shared, trace, current_step_idx);

                // reduce to field values at y=r for both x_next
                let az0 = compute_az_r_group0(&row_inputs, &lagrange_evals_r[..]);
                let bz0 = compute_bz_r_group0(&row_inputs, &lagrange_evals_r[..]);
                let az1 = compute_az_r_group1(&row_inputs, &lagrange_evals_r[..]);
                let bz1 = compute_bz_r_group1(&row_inputs, &lagrange_evals_r[..]);
                let cz1 = compute_cz_r_group1(&row_inputs, &lagrange_evals_r[..]);

                // sumcheck contributions
                let p0 = az0 * bz0;
                let slope = (az1 - az0) * (bz1 - bz0);

                // e_in per x_in
                let e_in = if num_x_in_vals == 0 {
                    F::one()
                } else if num_x_in_vals == 1 {
                    split_eq_poly.E_in_current()[0]
                } else {
                    split_eq_poly.E_in_current()[x_in_val]
                };

                inner_sum0 += e_in.mul_unreduced::<9>(p0);
                inner_sumInf += e_in.mul_unreduced::<9>(slope);

                // cache bound6 entries for later binding
                let block_id = current_step_idx;
                if !az0.is_zero() { bound6_at_r.push((6 * block_id, az0).into()); }
                if !bz0.is_zero() { bound6_at_r.push((6 * block_id + 1, bz0).into()); }
                if !az1.is_zero() { bound6_at_r.push((6 * block_id + 3, az1).into()); }
                if !bz1.is_zero() { bound6_at_r.push((6 * block_id + 4, bz1).into()); }
                if !cz1.is_zero() { bound6_at_r.push((6 * block_id + 5, cz1).into()); }
            }

            let e_out = if num_x_out_vals > 0 { split_eq_poly.E_out_current()[x_out_val] } else { F::zero() };
            let reduced0 = F::from_montgomery_reduce::<9>(inner_sum0);
            let reducedInf = F::from_montgomery_reduce::<9>(inner_sumInf);
            t0_acc += e_out * reduced0;
            t_inf_acc += e_out * reducedInf;
        }

        if outer_debug_enabled() {
            eprintln!("[outer/streaming-precompute] t0={}, t_inf={}, bound6_len={}", t0_acc, t_inf_acc, bound6_at_r.len());
        }
        let streaming_cache = StreamingRoundCache { t0: t0_acc, t_inf: t_inf_acc, bound6_at_r };

        Self {
            input_claim,
            interleaved_poly: SpartanInterleavedPoly::new(),
            split_eq_poly,
            preprocess: Some(Arc::new(preprocessing.shared.clone())),
            trace: Some(Arc::new(trace.to_vec())),
            r0_uniskip,
            total_rounds,
            tau: None,
            num_cycles_bits,
            streaming_cache: Some(streaming_cache),
        }
    }

    pub fn new_verifier(
        input_claim: F,
        split_eq_poly: GruenSplitEqPolynomial<F>,
        interleaved_poly: SpartanInterleavedPoly<F>,
        r0_uniskip: F::Challenge,
        total_rounds: usize,
        tau: Vec<F::Challenge>,
        num_cycles_bits: usize,
    ) -> Self {
        Self {
            input_claim,
            interleaved_poly,
            split_eq_poly,
            preprocess: None,
            trace: None,
            r0_uniskip,
            total_rounds,
            tau: Some(tau),
            num_cycles_bits,
            streaming_cache: None,
        }
    }

    #[inline]
    fn build_cubic_from_quadratic(
        eq_poly: &GruenSplitEqPolynomial<F>,
        t0: F,
        t_inf: F,
        previous_claim: F,
    ) -> UniPoly<F> {
        let scalar_times_w_i = eq_poly.current_scalar * eq_poly.w[eq_poly.current_index - 1];
        UniPoly::from_linear_times_quadratic_with_hint(
            [
                eq_poly.current_scalar - scalar_times_w_i,
                scalar_times_w_i + scalar_times_w_i - eq_poly.current_scalar,
            ],
            t0,
            t_inf,
            previous_claim,
        )
    }

    /// Compute the quadratic evaluations for the streaming round (right after univariate skip).
    ///
    /// This uses the streaming algorithm to compute the sum-check polynomial for the round
    /// right after the univariate skip round.
    ///
    /// Recall that we need to compute
    ///
    /// `t_i(0) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] * (unbound_coeffs_a(x_out, x_in,
    /// 0, r) * unbound_coeffs_b(x_out, x_in, 0, r) - unbound_coeffs_c(x_out, x_in, 0, r))`
    ///
    /// and
    ///
    /// `t_i(∞) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] * (unbound_coeffs_a(x_out,
    /// x_in, ∞, r) * unbound_coeffs_b(x_out, x_in, ∞, r))`
    ///
    /// Here the "_a,b,c" subscript indicates the coefficients of `unbound_coeffs` corresponding to
    /// Az, Bz, Cz respectively. Note that we index with x_out being the MSB here.
    ///
    /// Importantly, since the eval at `r` is not cached, we will need to recompute it via another
    /// sum
    ///
    /// `unbound_coeffs_{a,b,c}(x_out, x_in, {0,∞}, r) = \sum_{y in D} eq(r, y) *
    /// unbound_coeffs_{a,b,c}(x_out, x_in, {0,∞}, y)`
    ///
    /// (and the eval at ∞ is computed as (eval at 1) - (eval at 0))
    #[inline]
    fn streaming_quadratic_evals(&self) -> (F, F) {
        // Lagrange basis over the univariate-skip domain at r0
        let lagrange_evals_r = LagrangePolynomial::<F>::evals::<
            F::Challenge,
            UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&self.r0_uniskip);

        let eq_poly = &self.split_eq_poly;
        let num_x_out_vals = eq_poly.E_out_current_len();
        let num_x_in_vals = eq_poly.E_in_current_len();

        let num_parallel_chunks = if num_x_out_vals > 0 {
            core::cmp::min(
                num_x_out_vals,
                rayon::current_num_threads().next_power_of_two() * 8,
            )
        } else {
            1
        };
        let x_out_chunk_size = if num_x_out_vals > 0 {
            core::cmp::max(1, num_x_out_vals.div_ceil(num_parallel_chunks))
        } else {
            0
        };
        let iter_num_x_in_vars = if num_x_in_vals > 0 {
            num_x_in_vals.log_2()
        } else {
            0
        };

        let results: (F, F) = (0..num_parallel_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let x_out_start = chunk_idx * x_out_chunk_size;
                let x_out_end = core::cmp::min((chunk_idx + 1) * x_out_chunk_size, num_x_out_vals);
                let mut task_sum0 = F::zero();
                let mut task_sumInf = F::zero();

                for x_out_val in x_out_start..x_out_end {
                    let mut inner_sum0 = F::Unreduced::<9>::zero();
                    let mut inner_sumInf = F::Unreduced::<9>::zero();

                    for x_in_val in 0..num_x_in_vals {
                        let current_step_idx = (x_out_val << iter_num_x_in_vars) | x_in_val;

                        let row_inputs = R1CSCycleInputs::from_trace::<F>(
                            self.preprocess.as_ref().expect("prover preprocess missing"),
                            self.trace.as_ref().expect("prover trace missing").as_slice(),
                            current_step_idx,
                        );

                        // reduce to field values at y=r for both x_next
                        let az0 = compute_az_r_group0(&row_inputs, &lagrange_evals_r[..]);
                        let bz0 = compute_bz_r_group0(&row_inputs, &lagrange_evals_r[..]);
                        let az1 = compute_az_r_group1(&row_inputs, &lagrange_evals_r[..]);
                        let bz1 = compute_bz_r_group1(&row_inputs, &lagrange_evals_r[..]);
                        // cz1 only affects evaluation used later during binding, not t(0)/t(∞)
                        // let cz1 = compute_cz_r_group1(&row_inputs, &lagrange_evals_r[..]);

                        // sumcheck contributions
                        let p0 = az0 * bz0;
                        let slope = (az1 - az0) * (bz1 - bz0);

                        // e_in per x_in
                        let e_in = if num_x_in_vals == 0 {
                            F::one()
                        } else if num_x_in_vals == 1 {
                            eq_poly.E_in_current()[0]
                        } else {
                            eq_poly.E_in_current()[x_in_val]
                        };

                        inner_sum0 += e_in.mul_unreduced::<9>(p0);
                        inner_sumInf += e_in.mul_unreduced::<9>(slope);
                    }

                    let e_out = if num_x_out_vals > 0 {
                        eq_poly.E_out_current()[x_out_val]
                    } else {
                        F::zero()
                    };

                    let reduced0 = F::from_montgomery_reduce::<9>(inner_sum0);
                    let reducedInf = F::from_montgomery_reduce::<9>(inner_sumInf);
                    task_sum0 += e_out * reduced0;
                    task_sumInf += e_out * reducedInf;
                }

                (task_sum0, task_sumInf)
            })
            .reduce(|| (F::zero(), F::zero()), |a, b| (a.0 + b.0, a.1 + b.1));

        if outer_debug_enabled() {
            eprintln!("[outer/streaming] t0={}, t_inf={}", results.0, results.1);
        }
        results
    }

    /// Compute the polynomial for each of the remaining rounds, using the
    /// linear-time algorithm with split-eq optimizations.
    ///
    /// At this point, we have computed the `bound_coeffs` for the current round.
    /// We need to compute:
    ///
    /// `t_i(0) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] *
    /// (az_bound[x_out, x_in, 0] * bz_bound[x_out, x_in, 0] - cz_bound[x_out, x_in, 0])`
    ///
    /// and
    ///
    /// `t_i(∞) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] *
    /// az_bound[x_out, x_in, ∞] * bz_bound[x_out, x_in, ∞]`
    ///
    /// (ordering of indices is MSB to LSB, so x_out is the MSB and x_in is the LSB)
    #[inline]
    fn remaining_quadratic_evals(&self) -> (F, F) {
        let eq_poly = &self.split_eq_poly;

        let chunk_ranges = {
            let block_size = self
                .interleaved_poly
                .bound_coeffs
                .len()
                .div_ceil(rayon::current_num_threads())
                .next_multiple_of(6);
            Self::compute_block_ranges(
                &self.interleaved_poly.bound_coeffs,
                block_size,
            )
        };

        if eq_poly.E_in_current_len() == 1 {
            chunk_ranges
                .par_iter()
                .flat_map_iter(|&(start, end)| {
                    let chunk = &self.interleaved_poly.bound_coeffs[start..end];
                    chunk
                        .chunk_by(|x, y| x.index / 6 == y.index / 6)
                        .map(|sparse_block| {
                            let block_index = sparse_block[0].index / 6;
                            let mut block = [F::zero(); 6];
                            for coeff in sparse_block {
                                block[coeff.index % 6] = coeff.value;
                            }

                            let az = (block[0], block[3]);
                            let bz = (block[1], block[4]);
                            let cz0 = block[2];

                            let az_eval_infty = az.1 - az.0;
                            let bz_eval_infty = bz.1 - bz.0;

                            let eq_evals = eq_poly.E_out_current()[block_index];

                            (
                                eq_evals.mul_0_optimized(az.0.mul_0_optimized(bz.0) - cz0),
                                eq_evals
                                    .mul_0_optimized(az_eval_infty.mul_0_optimized(bz_eval_infty)),
                            )
                        })
                })
                .reduce_with(|sum, evals| (sum.0 + evals.0, sum.1 + evals.1))
                .unwrap_or((F::zero(), F::zero()))
        } else {
            let num_x1_bits = eq_poly.E_in_current_len().log_2();
            let x1_bitmask = (1 << num_x1_bits) - 1;

            chunk_ranges
                .par_iter()
                .map(|&(start, end)| {
                    let chunk = &self.interleaved_poly.bound_coeffs[start..end];
                    let mut eval_point_0 = F::zero();
                    let mut eval_point_infty = F::zero();

                    let mut inner_sums = (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero());
                    let mut prev_x2 = 0;

                    for sparse_block in chunk.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                        let block_index = sparse_block[0].index / 6;
                        let x1 = block_index & x1_bitmask;
                        let E_in_evals = eq_poly.E_in_current()[x1];
                        let x2 = block_index >> num_x1_bits;

                        if x2 != prev_x2 {
                            let reduced0 = F::from_montgomery_reduce::<9>(inner_sums.0);
                            let reducedInf = F::from_montgomery_reduce::<9>(inner_sums.1);
                            eval_point_0 += eq_poly.E_out_current()[prev_x2] * reduced0;
                            eval_point_infty += eq_poly.E_out_current()[prev_x2] * reducedInf;
                            inner_sums = (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero());
                            prev_x2 = x2;
                        }

                        let mut block = [F::zero(); 6];
                        for coeff in sparse_block {
                            block[coeff.index % 6] = coeff.value;
                        }

                        let az = (block[0], block[3]);
                        let bz = (block[1], block[4]);
                        let cz0 = block[2];

                        let az_eval_infty = az.1 - az.0;
                        let bz_eval_infty = bz.1 - bz.0;

                        inner_sums.0 +=
                            E_in_evals.mul_unreduced::<9>(az.0.mul_0_optimized(bz.0) - cz0);
                        inner_sums.1 += E_in_evals
                            .mul_unreduced::<9>(az_eval_infty.mul_0_optimized(bz_eval_infty));
                    }

                    let reduced0 = F::from_montgomery_reduce::<9>(inner_sums.0);
                    let reducedInf = F::from_montgomery_reduce::<9>(inner_sums.1);
                    eval_point_0 += eq_poly.E_out_current()[prev_x2] * reduced0;
                    eval_point_infty += eq_poly.E_out_current()[prev_x2] * reducedInf;

                    (eval_point_0, eval_point_infty)
                })
                .reduce_with(|sum, evals| (sum.0 + evals.0, sum.1 + evals.1))
                .unwrap_or((F::zero(), F::zero()))
        }
    }

    /// Bind the streaming round after deriving challenge r_i.
    ///
    /// As we compute each `{a/b/c}(x_out, x_in, {0,∞}, r)`, we will
    /// store them in `bound_coeffs` in sparse format (the eval at 1 will be eval
    /// at 0 + eval at ∞). We then bind these bound coeffs with r_i for the next round.
    fn bind_streaming_round(&mut self, r_i: F::Challenge) {
        if let Some(cache) = self.streaming_cache.take() {
            // Size output buffer from cached sparse entries
            let mut total_len: usize = 0;
            fn binding_output_length<T>(coeffs: &[SparseCoefficient<T>]) -> usize {
                let mut output_size = 0;
                for block in coeffs.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                    let mut Az_coeff_found = false;
                    let mut Bz_coeff_found = false;
                    let mut Cz_coeff_found = false;
                    for coeff in block {
                        match coeff.index % 3 {
                            0 => { if !Az_coeff_found { Az_coeff_found = true; output_size += 1; } }
                            1 => { if !Bz_coeff_found { Bz_coeff_found = true; output_size += 1; } }
                            2 => { if !Cz_coeff_found { Cz_coeff_found = true; output_size += 1; } }
                            _ => unreachable!(),
                        }
                    }
                }
                output_size
            }
            for block6 in cache.bound6_at_r.chunk_by(|a, b| a.index / 6 == b.index / 6) {
                total_len += binding_output_length(block6);
            }
            if self.interleaved_poly.binding_scratch_space.capacity() < total_len {
                self.interleaved_poly
                    .binding_scratch_space
                    .reserve_exact(total_len - self.interleaved_poly.binding_scratch_space.capacity());
            }
            unsafe {
                self.interleaved_poly
                    .binding_scratch_space
                    .set_len(total_len);
            }

            // Bind per block using round challenge r_i
            let mut output_index = 0usize;
            let out_slice = self.interleaved_poly.binding_scratch_space.as_mut_slice();
            for block6 in cache.bound6_at_r.chunk_by(|a, b| a.index / 6 == b.index / 6) {
                if block6.is_empty() { continue; }
                let blk = block6[0].index / 6;
                let mut az0 = F::zero();
                let mut bz0 = F::zero();
                let mut cz0 = F::zero();
                let mut az1 = F::zero();
                let mut bz1 = F::zero();
                let mut cz1 = F::zero();
                let mut has_az = false;
                let mut has_bz = false;
                let mut has_cz = false;
                for c in block6 {
                    match c.index % 6 {
                        0 => { az0 = c.value; has_az = true; }
                        1 => { bz0 = c.value; has_bz = true; }
                        2 => { cz0 = c.value; has_cz = true; }
                        3 => { az1 = c.value; has_az = true; }
                        4 => { bz1 = c.value; has_bz = true; }
                        5 => { cz1 = c.value; has_cz = true; }
                        _ => {}
                    }
                }
                let azb = az0 + r_i * (az1 - az0);
                if has_az { out_slice[output_index] = (3 * blk, azb).into(); output_index += 1; }
                let bzb = bz0 + r_i * (bz1 - bz0);
                if has_bz { out_slice[output_index] = (3 * blk + 1, bzb).into(); output_index += 1; }
                if has_cz { let czb = cz0 + r_i * (cz1 - cz0); out_slice[output_index] = (3 * blk + 2, czb).into(); output_index += 1; }
            }
            debug_assert_eq!(output_index, out_slice.len());

            core::mem::swap(
                &mut self.interleaved_poly.bound_coeffs,
                &mut self.interleaved_poly.binding_scratch_space,
            );
        } else {
            // Fallback: recompute bound6_at_r on the fly
            // Lagrange basis over the univariate-skip domain at r0
            let lagrange_evals_r = LagrangePolynomial::<F>::evals::<
                F::Challenge,
                UNIVARIATE_SKIP_DOMAIN_SIZE,
            >(&self.r0_uniskip);
            let eq_poly = &mut self.split_eq_poly;
            let num_x_out_vals = eq_poly.E_out_current_len();
            let num_x_in_vals = eq_poly.E_in_current_len();
            let iter_num_x_in_vars = if num_x_in_vals > 0 { num_x_in_vals.log_2() } else { 0 };

            let mut bound6_at_r: Vec<SparseCoefficient<F>> = Vec::new();
            let mut reserve = num_x_out_vals.saturating_mul(core::cmp::max(1, num_x_in_vals)).saturating_mul(4);
            reserve = reserve.max(1024);
            bound6_at_r.reserve(reserve);

            for x_out_val in 0..num_x_out_vals {
                for x_in_val in 0..num_x_in_vals {
                    let current_step_idx = (x_out_val << iter_num_x_in_vars) | x_in_val;
                    let row_inputs = R1CSCycleInputs::from_trace::<F>(
                        self.preprocess.as_ref().expect("prover preprocess missing"),
                        self.trace.as_ref().expect("prover trace missing").as_slice(),
                        current_step_idx,
                    );
                    let az0 = compute_az_r_group0(&row_inputs, &lagrange_evals_r[..]);
                    let bz0 = compute_bz_r_group0(&row_inputs, &lagrange_evals_r[..]);
                    let az1 = compute_az_r_group1(&row_inputs, &lagrange_evals_r[..]);
                    let bz1 = compute_bz_r_group1(&row_inputs, &lagrange_evals_r[..]);
                    let cz1 = compute_cz_r_group1(&row_inputs, &lagrange_evals_r[..]);

                    let block_id = current_step_idx;
                    if !az0.is_zero() { bound6_at_r.push((6 * block_id, az0).into()); }
                    if !bz0.is_zero() { bound6_at_r.push((6 * block_id + 1, bz0).into()); }
                    if !az1.is_zero() { bound6_at_r.push((6 * block_id + 3, az1).into()); }
                    if !bz1.is_zero() { bound6_at_r.push((6 * block_id + 4, bz1).into()); }
                    if !cz1.is_zero() { bound6_at_r.push((6 * block_id + 5, cz1).into()); }
                }
            }

            // Size output buffer
            let mut total_len: usize = 0;
            for block6 in bound6_at_r.chunk_by(|a, b| a.index / 6 == b.index / 6) {
                let mut output_size = 0;
                for block in block6.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                    let mut Az_coeff_found = false;
                    let mut Bz_coeff_found = false;
                    let mut Cz_coeff_found = false;
                    for coeff in block {
                        match coeff.index % 3 {
                            0 => { if !Az_coeff_found { Az_coeff_found = true; output_size += 1; } }
                            1 => { if !Bz_coeff_found { Bz_coeff_found = true; output_size += 1; } }
                            2 => { if !Cz_coeff_found { Cz_coeff_found = true; output_size += 1; } }
                            _ => unreachable!(),
                        }
                    }
                }
                total_len += output_size;
            }
            if self.interleaved_poly.binding_scratch_space.capacity() < total_len {
                self.interleaved_poly
                    .binding_scratch_space
                    .reserve_exact(total_len - self.interleaved_poly.binding_scratch_space.capacity());
            }
            unsafe {
                self.interleaved_poly
                    .binding_scratch_space
                    .set_len(total_len);
            }

            // Bind per block
            let mut output_index = 0usize;
            let out_slice = self.interleaved_poly.binding_scratch_space.as_mut_slice();
            for block6 in bound6_at_r.chunk_by(|a, b| a.index / 6 == b.index / 6) {
                if block6.is_empty() { continue; }
                let blk = block6[0].index / 6;
                let mut az0 = F::zero();
                let mut bz0 = F::zero();
                let mut cz0 = F::zero();
                let mut az1 = F::zero();
                let mut bz1 = F::zero();
                let mut cz1 = F::zero();
                let mut has_az = false;
                let mut has_bz = false;
                let mut has_cz = false;
                for c in block6 {
                    match c.index % 6 {
                        0 => { az0 = c.value; has_az = true; }
                        1 => { bz0 = c.value; has_bz = true; }
                        2 => { cz0 = c.value; has_cz = true; }
                        3 => { az1 = c.value; has_az = true; }
                        4 => { bz1 = c.value; has_bz = true; }
                        5 => { cz1 = c.value; has_cz = true; }
                        _ => {}
                    }
                }
                let azb = az0 + r_i * (az1 - az0);
                if has_az { out_slice[output_index] = (3 * blk, azb).into(); output_index += 1; }
                let bzb = bz0 + r_i * (bz1 - bz0);
                if has_bz { out_slice[output_index] = (3 * blk + 1, bzb).into(); output_index += 1; }
                if has_cz { let czb = cz0 + r_i * (cz1 - cz0); out_slice[output_index] = (3 * blk + 2, czb).into(); output_index += 1; }
            }
            debug_assert_eq!(output_index, out_slice.len());

            core::mem::swap(
                &mut self.interleaved_poly.bound_coeffs,
                &mut self.interleaved_poly.binding_scratch_space,
            );
        }
    }

    pub fn final_sumcheck_evals(&self) -> [F; 3] {
        let mut final_az_eval = F::zero();
        let mut final_bz_eval = F::zero();
        let mut final_cz_eval = F::zero();
        for coeff in &self.interleaved_poly.bound_coeffs {
            match coeff.index {
                0 => final_az_eval = coeff.value,
                1 => final_bz_eval = coeff.value,
                2 => final_cz_eval = coeff.value,
                _ => {}
            }
        }
        [final_az_eval, final_bz_eval, final_cz_eval]
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for OuterRemainingSumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.total_rounds - 1 /* exclude first already handled? */ + 1
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    fn compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F> {
        let (t0, t_inf) = if round == 0 {
            if let Some(cache) = self.streaming_cache.take() {
                // Load precomputed streaming results and also populate bound6_at_r into interleaved
                // Size output buffer for binding later rounds
                let mut total_len: usize = 0;
                for block6 in cache.bound6_at_r.chunk_by(|a, b| a.index / 6 == b.index / 6) {
                    let mut output_size = 0;
                    for block in block6.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                        let mut Az_coeff_found = false;
                        let mut Bz_coeff_found = false;
                        let mut Cz_coeff_found = false;
                        for coeff in block {
                            match coeff.index % 3 {
                                0 => { if !Az_coeff_found { Az_coeff_found = true; output_size += 1; } }
                                1 => { if !Bz_coeff_found { Bz_coeff_found = true; output_size += 1; } }
                                2 => { if !Cz_coeff_found { Cz_coeff_found = true; output_size += 1; } }
                                _ => unreachable!(),
                            }
                        }
                    }
                    total_len += output_size;
                }
                if self.interleaved_poly.binding_scratch_space.capacity() < total_len {
                    self.interleaved_poly
                        .binding_scratch_space
                        .reserve_exact(total_len - self.interleaved_poly.binding_scratch_space.capacity());
                }
                unsafe {
                    self.interleaved_poly
                        .binding_scratch_space
                        .set_len(total_len);
                }
                // Bind streaming round immediately using r0
                let mut output_index = 0usize;
                let out_slice = self.interleaved_poly.binding_scratch_space.as_mut_slice();
                for block6 in cache.bound6_at_r.chunk_by(|a, b| a.index / 6 == b.index / 6) {
                    if block6.is_empty() { continue; }
                    let blk = block6[0].index / 6;
                    let mut az0 = F::zero();
                    let mut bz0 = F::zero();
                    let mut cz0 = F::zero();
                    let mut az1 = F::zero();
                    let mut bz1 = F::zero();
                    let mut cz1 = F::zero();
                    let mut has_az = false;
                    let mut has_bz = false;
                    let mut has_cz = false;
                    for c in block6 {
                        match c.index % 6 {
                            0 => { az0 = c.value; has_az = true; }
                            1 => { bz0 = c.value; has_bz = true; }
                            2 => { cz0 = c.value; has_cz = true; }
                            3 => { az1 = c.value; has_az = true; }
                            4 => { bz1 = c.value; has_bz = true; }
                            5 => { cz1 = c.value; has_cz = true; }
                            _ => {}
                        }
                    }
                    let azb = az0 + self.r0_uniskip * (az1 - az0);
                    if has_az { out_slice[output_index] = (3 * blk, azb).into(); output_index += 1; }
                    let bzb = bz0 + self.r0_uniskip * (bz1 - bz0);
                    if has_bz { out_slice[output_index] = (3 * blk + 1, bzb).into(); output_index += 1; }
                    if has_cz { let czb = cz0 + self.r0_uniskip * (cz1 - cz0); out_slice[output_index] = (3 * blk + 2, czb).into(); output_index += 1; }
                }
                core::mem::swap(
                    &mut self.interleaved_poly.bound_coeffs,
                    &mut self.interleaved_poly.binding_scratch_space,
                );
                (cache.t0, cache.t_inf)
            } else {
                self.streaming_quadratic_evals()
            }
        } else {
            self.remaining_quadratic_evals()
        };
        let cubic = Self::build_cubic_from_quadratic(&self.split_eq_poly, t0, t_inf, previous_claim);
        let m0 = cubic.evaluate::<F>(&F::zero());
        let m2 = cubic.evaluate::<F>(&F::from_u64(2));
        let m3 = cubic.evaluate::<F>(&F::from_u64(3));
        if outer_debug_enabled() {
            eprintln!("[outer/round {}] cubic evals: m0={}, m2={}, m3={} (t0={}, t_inf={})", round, m0, m2, m3, t0, t_inf);
        }
        vec![m0, m2, m3]
    }

    fn bind(&mut self, r_j: F::Challenge, round: usize) {
        if round == 0 {
            self.bind_streaming_round(r_j);
        } else {
            // Remaining rounds binding (reuse existing logic)
            let block_size = self
                .interleaved_poly
                .bound_coeffs
                .len()
                .div_ceil(rayon::current_num_threads())
                .next_multiple_of(6);
            // Local helper to compute block ranges
            fn compute_block_ranges<T>(coeffs: &[SparseCoefficient<T>], block_size: usize) -> Vec<(usize, usize)> {
                if coeffs.is_empty() { return Vec::new(); }
                let mut ranges = Vec::new();
                let mut start = 0usize;
                let mut current_bucket = coeffs[0].index / block_size;
                for (i, c) in coeffs.iter().enumerate().skip(1) {
                    let bucket = c.index / block_size;
                    if bucket != current_bucket { ranges.push((start, i)); start = i; current_bucket = bucket; }
                }
                ranges.push((start, coeffs.len()));
                ranges
            }
            let chunk_ranges = compute_block_ranges(
                &self.interleaved_poly.bound_coeffs,
                block_size,
            );

            let output_sizes: Vec<_> = chunk_ranges
                .par_iter()
                .map(|&(start, end)| {
                    let coeffs = &self.interleaved_poly.bound_coeffs[start..end];
                    let mut output_size = 0;
                    for block in coeffs.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                        let mut Az_coeff_found = false;
                        let mut Bz_coeff_found = false;
                        let mut Cz_coeff_found = false;
                        for coeff in block {
                            match coeff.index % 3 {
                                0 => { if !Az_coeff_found { Az_coeff_found = true; output_size += 1; } }
                                1 => { if !Bz_coeff_found { Bz_coeff_found = true; output_size += 1; } }
                                2 => { if !Cz_coeff_found { Cz_coeff_found = true; output_size += 1; } }
                                _ => unreachable!(),
                            }
                        }
                    }
                    output_size
                })
                .collect();
            let total_output_len = output_sizes.iter().sum();
            if self.interleaved_poly.binding_scratch_space.is_empty() {
                self.interleaved_poly.binding_scratch_space = Vec::with_capacity(total_output_len);
            }
            unsafe {
                self.interleaved_poly
                    .binding_scratch_space
                    .set_len(total_output_len);
            }

            let mut output_slices: Vec<&mut [SparseCoefficient<F>]> =
                Vec::with_capacity(chunk_ranges.len());
            let mut remainder = self.interleaved_poly.binding_scratch_space.as_mut_slice();
            for slice_len in output_sizes {
                let (first, second) = remainder.split_at_mut(slice_len);
                output_slices.push(first);
                remainder = second;
            }

            chunk_ranges
                .par_iter()
                .zip_eq(output_slices.into_par_iter())
                .for_each(|(&(start, end), output_slice)| {
                    let coeffs = &self.interleaved_poly.bound_coeffs[start..end];
                    let mut output_index = 0;
                    for block in coeffs.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                        let block_index = block[0].index / 6;
                        let mut az_coeff: (Option<F>, Option<F>) = (None, None);
                        let mut bz_coeff: (Option<F>, Option<F>) = (None, None);
                        let mut cz_coeff: (Option<F>, Option<F>) = (None, None);
                        for coeff in block {
                            match coeff.index % 6 {
                                0 => az_coeff.0 = Some(coeff.value),
                                1 => bz_coeff.0 = Some(coeff.value),
                                2 => cz_coeff.0 = Some(coeff.value),
                                3 => az_coeff.1 = Some(coeff.value),
                                4 => bz_coeff.1 = Some(coeff.value),
                                5 => cz_coeff.1 = Some(coeff.value),
                                _ => unreachable!(),
                            }
                        }
                        if az_coeff != (None, None) {
                            let (low, high) = (
                                az_coeff.0.unwrap_or(F::zero()),
                                az_coeff.1.unwrap_or(F::zero()),
                            );
                            output_slice[output_index] =
                                (3 * block_index, low + r_j * (high - low)).into();
                            output_index += 1;
                        }
                        if bz_coeff != (None, None) {
                            let (low, high) = (
                                bz_coeff.0.unwrap_or(F::zero()),
                                bz_coeff.1.unwrap_or(F::zero()),
                            );
                            output_slice[output_index] =
                                (3 * block_index + 1, low + r_j * (high - low)).into();
                            output_index += 1;
                        }
                        if cz_coeff != (None, None) {
                            let (low, high) = (
                                cz_coeff.0.unwrap_or(F::zero()),
                                cz_coeff.1.unwrap_or(F::zero()),
                            );
                            output_slice[output_index] =
                                (3 * block_index + 2, low + r_j * (high - low)).into();
                            output_index += 1;
                        }
                    }
                    debug_assert_eq!(output_index, output_slice.len())
                });

            std::mem::swap(
                &mut self.interleaved_poly.bound_coeffs,
                &mut self.interleaved_poly.binding_scratch_space,
            );
        }

        // Bind eq_poly for next round
        self.split_eq_poly.bind(r_j);
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<std::rc::Rc<std::cell::RefCell<VerifierOpeningAccumulator<F>>>>,
        r_tail: &[F::Challenge],
    ) -> F {
        let tau = self
            .tau
            .as_ref()
            .expect("Verifier tau not set in OuterRemainingSumcheck");

        // Reconstruct full r = [r0] + r_tail, then reverse (outer is bound from the top)
        let mut r_full: Vec<F::Challenge> = Vec::with_capacity(1 + r_tail.len());
        r_full.push(self.r0_uniskip);
        r_full.extend_from_slice(r_tail);
        let r_reversed: Vec<F::Challenge> = r_full.into_iter().rev().collect();

        let acc_cell = accumulator.as_ref().expect("accumulator required");
        let acc_ref = acc_cell.borrow();
        let (_, claim_Az) = acc_ref
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter);
        let (_, claim_Bz) = acc_ref
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanBz, SumcheckId::SpartanOuter);
        let (_, claim_Cz) = acc_ref
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanCz, SumcheckId::SpartanOuter);

        let tau_bound_rx = EqPolynomial::mle(tau, &r_reversed);
        tau_bound_rx * (claim_Az * claim_Bz - claim_Cz)
    }

    fn normalize_opening_point(&self, r_tail: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        // Construct full r and reverse to match outer convention
        let mut r_full: Vec<F::Challenge> = Vec::with_capacity(1 + r_tail.len());
        r_full.push(self.r0_uniskip);
        r_full.extend_from_slice(r_tail);
        let r_reversed: Vec<F::Challenge> = r_full.into_iter().rev().collect();
        OpeningPoint::new(r_reversed)
    }

    fn cache_openings_prover(
        &self,
        accumulator: std::rc::Rc<std::cell::RefCell<ProverOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        // Append Az, Bz, Cz claims and corresponding opening point
        let claims = self.final_sumcheck_evals();
        let mut acc = accumulator.borrow_mut();
        acc.append_virtual(
            transcript,
            VirtualPolynomial::SpartanAz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
            claims[0],
        );
        acc.append_virtual(
            transcript,
            VirtualPolynomial::SpartanBz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
            claims[1],
        );
        acc.append_virtual(
            transcript,
            VirtualPolynomial::SpartanCz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
            claims[2],
        );

        // Handle witness openings at r_cycle (use consistent split length)
        let (r_cycle, _rx_var) = opening_point.r.split_at(self.num_cycles_bits);

        // Compute claimed witness evals and append commitments and virtuals
        let claimed_witness_evals = compute_claimed_witness_evals::<F>(
            self.preprocess.as_ref().expect("prover preprocess missing"),
            self.trace.as_ref().expect("prover trace missing").as_slice(),
            r_cycle,
        );
        let committed_polys: Vec<_> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| CommittedPolynomial::try_from(input).ok().unwrap())
            .collect();
        let committed_poly_claims: Vec<_> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| claimed_witness_evals[input.to_index()])
            .collect();
        acc.append_dense(
            transcript,
            committed_polys,
            SumcheckId::SpartanOuter,
            r_cycle.to_vec(),
            &committed_poly_claims,
        );
        for (input, eval) in ALL_R1CS_INPUTS.iter().zip(claimed_witness_evals.iter()) {
            if !COMMITTED_R1CS_INPUTS.contains(input) {
                acc.append_virtual(
                    transcript,
                    VirtualPolynomial::try_from(input).ok().unwrap(),
                    SumcheckId::SpartanOuter,
                    OpeningPoint::new(r_cycle.to_vec()),
                    *eval,
                );
            }
        }
    }

    fn cache_openings_verifier(
        &self,
        accumulator: std::rc::Rc<std::cell::RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let mut acc = accumulator.borrow_mut();
        // Populate Az, Bz, Cz openings at the full outer opening point
        acc.append_virtual(
            transcript,
            VirtualPolynomial::SpartanAz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
        );
        acc.append_virtual(
            transcript,
            VirtualPolynomial::SpartanBz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
        );
        acc.append_virtual(
            transcript,
            VirtualPolynomial::SpartanCz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
        );

        // Append witness openings at r_cycle (no claims at verifier)
        let (r_cycle, _rx_var) = opening_point.r.split_at(self.num_cycles_bits);
        let committed_polys: Vec<_> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| CommittedPolynomial::try_from(input).ok().unwrap())
            .collect();
        acc.append_dense(
            transcript,
            committed_polys,
            SumcheckId::SpartanOuter,
            r_cycle.to_vec(),
        );
        ALL_R1CS_INPUTS.iter().for_each(|input| {
            if !COMMITTED_R1CS_INPUTS.contains(input) {
                acc.append_virtual(
                    transcript,
                    VirtualPolynomial::try_from(input).ok().unwrap(),
                    SumcheckId::SpartanOuter,
                    OpeningPoint::new(r_cycle.to_vec()),
                );
            }
        });
    }
}
