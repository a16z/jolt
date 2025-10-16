#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_ff::biginteger::S160;
use ark_std::Zero;
use rayon::prelude::*;
use std::sync::Arc;
use tracer::instruction::Cycle;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::lagrange_poly::{LagrangeHelper, LagrangePolynomial};
use crate::poly::multilinear_polynomial::BindingOrder;
use crate::poly::opening_proof::{
    OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
};
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck::{SumcheckInstance, UniSkipFirstRoundInstance};
use crate::transcripts::Transcript;
use crate::utils::math::Math;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::utils::univariate_skip::{
    accum::{
        acc6_fmadd_i128, acc6_new, acc6_reduce, accs160_fmadd_s160, accs160_new, accs160_reduce,
    },
    build_uniskip_first_round_poly, compute_az_r_group0, compute_az_r_group1, compute_bz_r_group0,
    compute_bz_r_group1, uniskip_targets,
};
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::r1cs::{
    constraints::{
        eval_az_first_group, eval_az_second_group, eval_bz_first_group, eval_bz_second_group,
        FIRST_ROUND_POLY_NUM_COEFFS, NUM_REMAINING_R1CS_CONSTRAINTS, UNIVARIATE_SKIP_DEGREE,
        UNIVARIATE_SKIP_DOMAIN_SIZE, UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE,
    },
    inputs::{compute_claimed_witness_evals, R1CSCycleInputs, ALL_R1CS_INPUTS},
};
use crate::zkvm::witness::VirtualPolynomial;
use crate::zkvm::JoltSharedPreprocessing;

// Spartan Outer sumcheck
// (with univariate-skip first round on Z, and no Cz term given all eq conditional constraints)
//
// We define a univariate in Z first-round polynomial
//   s1(Y) := L(τ_high, Y) · Σ_{x_out ∈ {0,1}^{m_out}} Σ_{x_in ∈ {0,1}^{m_in}}
//              E_out(r_out, x_out) · E_in(r_in, x_in) ·
//              [ Az(x_out, x_in, Y) · Bz(x_out, x_in, Y) ],
// where L(τ_high, Y) is the Lagrange basis polynomial over the univariate-skip
// base domain evaluated at τ_high, and Az(·,·,Y), Bz(·,·,Y) are the
// per-row univariate polynomials in Y induced by the R1CS row (split into two
// internal groups in code, but algebraically composing to Az·Bz at Y).
// The prover sends s1(Y) via univariate-skip by evaluating t1(Y) := Σ Σ E_out·E_in·(Az·Bz)
// on an extended grid Y ∈ {−D..D} outside the base window, interpolating t1,
// multiplying by L(τ_high, Y) to obtain s1, and the verifier samples r0.
//
// Subsequent outer rounds bind the cycle variables r_tail = (r1, r2, …) using
// a streaming first cycle-bit round followed by linear-time rounds:
//   • Streaming round (after r0): compute
//       t(0)  = Σ_{x_out} E_out · Σ_{x_in} E_in · (Az(0)·Bz(0))
//       t(∞)  = Σ_{x_out} E_out · Σ_{x_in} E_in · ((Az(1)−Az(0))·(Bz(1)−Bz(0)))
//     send a cubic built from these endpoints, and bind cached coefficients by r1.
//   • Remaining rounds: reuse bound coefficients to compute the same endpoints
//     in linear time for each subsequent bit and bind by r_i.
//
// Final check (verifier): with r = [r0 || r_tail] and outer binding order from
// the top, evaluate Eq_τ(τ, r) and verify
//   Eq_τ(τ, r) · (Az(r) · Bz(r)).

/// Uni-skip instance for Spartan outer sumcheck, computing the first-round polynomial only.
pub struct OuterUniSkipInstance<F: JoltField> {
    tau: Vec<F::Challenge>,
    /// Prover-only state (None on verifier)
    prover_state: Option<OuterUniSkipProverState<F>>,
}

#[derive(Clone, Debug)]
struct OuterUniSkipProverState<F: JoltField> {
    /// Evaluations of t1(Z) at the extended univariate-skip targets (outside base window)
    extended_evals: [F; UNIVARIATE_SKIP_DEGREE],
}

impl<F: JoltField> OuterUniSkipInstance<F> {
    #[tracing::instrument(skip_all, name = "OuterUniSkipInstance::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        tau: &[F::Challenge],
    ) -> Self {
        let (preprocessing, trace, _program_io, _final_mem) = state_manager.get_prover_data();

        let tau_low = &tau[0..tau.len() - 1];

        let extended =
            Self::compute_univariate_skip_extended_evals(&preprocessing.shared, trace, tau_low);

        Self {
            tau: tau.to_vec(),
            prover_state: Some(OuterUniSkipProverState {
                extended_evals: extended,
            }),
        }
    }

    pub fn new_verifier(tau: &[F::Challenge]) -> Self {
        Self {
            tau: tau.to_vec(),
            prover_state: None,
        }
    }

    // local univariate_skip_targets removed in favor of utils::univariate_skip::uniskip_targets

    /// Compute the extended evaluations of the univariate skip polynomial, i.e.
    ///
    /// t_1(y) = \sum_{x_out} eq(tau_out, x_out) * \sum_{x_in} eq(tau_in, x_in) * Az(x_out, x_in, y) * Bz(x_out, x_in, y)
    ///
    /// for all y in the extended domain {−D..D} outside the base window
    /// (inside the base window, we have t_1(y) = 0)
    ///
    /// Note that the last of the x_in variables corresponds to the group index of the constraints
    /// (since we split the constraints in half, and y ranges over the number of constraints in each group)
    ///
    /// So we actually need to be careful and compute
    ///
    /// \sum_{x_in'} eq(tau_in, (x_in', 0)) * Az(x_out, x_in', 0, y) * Bz(x_out, x_in', 0, y)
    ///     + eq(tau_in, (x_in', 1)) * Az(x_out, x_in', 1, y) * Bz(x_out, x_in', 1, y)
    fn compute_univariate_skip_extended_evals(
        preprocess: &JoltSharedPreprocessing,
        trace: &[Cycle],
        tau_low: &[F::Challenge],
    ) -> [F; UNIVARIATE_SKIP_DEGREE] {
        // Precompute Lagrange coefficient vectors for target Z values outside the base window
        let base_left: i64 = -((UNIVARIATE_SKIP_DOMAIN_SIZE as i64 - 1) / 2);
        let targets: [i64; UNIVARIATE_SKIP_DEGREE] =
            uniskip_targets::<UNIVARIATE_SKIP_DOMAIN_SIZE, UNIVARIATE_SKIP_DEGREE>();

        let target_shifts: [i64; UNIVARIATE_SKIP_DEGREE] =
            core::array::from_fn(|j| targets[j] - base_left);
        let coeffs_per_j: [[i32; UNIVARIATE_SKIP_DOMAIN_SIZE]; UNIVARIATE_SKIP_DEGREE] =
            core::array::from_fn(|j| {
                LagrangeHelper::shift_coeffs_i32::<UNIVARIATE_SKIP_DOMAIN_SIZE>(target_shifts[j])
            });

        let m = tau_low.len() / 2;
        let (tau_out, tau_in) = tau_low.split_at(m);
        let (E_out, E_in) = rayon::join(
            || EqPolynomial::evals(tau_out),
            || EqPolynomial::evals(tau_in),
        );

        let num_x_out_vals = E_out.len();
        let num_x_in_vals = E_in.len();
        assert!(
            num_x_in_vals >= 2,
            "univariate skip expects at least 2 x_in values (last bit is group index)"
        );
        // The last x_in bit is the group selector: even indices -> group 0, odd -> group 1
        let num_x_in_half = num_x_in_vals >> 1;

        let num_parallel_chunks = core::cmp::min(
            num_x_out_vals,
            rayon::current_num_threads().next_power_of_two() * 8,
        );
        let x_out_chunk_size = if num_x_out_vals > 0 {
            core::cmp::max(1, num_x_out_vals.div_ceil(num_parallel_chunks))
        } else {
            0
        };
        let iter_num_x_in_vars = num_x_in_vals.log_2();
        let iter_num_x_in_prime_vars = iter_num_x_in_vars - 1; // ignore last bit (group index)

        (0..num_parallel_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let x_out_start = chunk_idx * x_out_chunk_size;
                let x_out_end = core::cmp::min((chunk_idx + 1) * x_out_chunk_size, num_x_out_vals);
                let mut acc_field: [F; UNIVARIATE_SKIP_DEGREE] =
                    [F::zero(); UNIVARIATE_SKIP_DEGREE];

                for x_out_val in x_out_start..x_out_end {
                    let mut inner_acc: [<F as JoltField>::Unreduced<9>; UNIVARIATE_SKIP_DEGREE] =
                        [<F as JoltField>::Unreduced::<9>::zero(); UNIVARIATE_SKIP_DEGREE];
                    for x_in_prime in 0..num_x_in_half {
                        // Materialize row once for both groups (ignores last bit)
                        let base_step_idx = (x_out_val << iter_num_x_in_prime_vars) | x_in_prime;
                        let row_inputs =
                            R1CSCycleInputs::from_trace::<F>(preprocess, trace, base_step_idx);

                        // Group 0 (even index)
                        let x_in_even = x_in_prime << 1;
                        let e_in_even = E_in[x_in_even];

                        let az0_bool = eval_az_first_group(&row_inputs);
                        let bz0_i128 = eval_bz_first_group(&row_inputs);

                        for j in 0..UNIVARIATE_SKIP_DEGREE {
                            let coeffs = &coeffs_per_j[j];
                            // (sum_i (Az0_i ? c_i : 0)) * (sum_i c_i * Bz0_i)
                            let mut sum_c_bz0_acc6 = acc6_new::<F>();
                            let mut sum_c_az0_i64: i64 = 0;
                            for i in 0..UNIVARIATE_SKIP_DOMAIN_SIZE {
                                let c = coeffs[i] as i64;
                                if c == 0 {
                                    continue;
                                }
                                if az0_bool[i] {
                                    sum_c_az0_i64 += c;
                                }
                                let term = (c as i128).saturating_mul(bz0_i128[i]);
                                acc6_fmadd_i128(&mut sum_c_bz0_acc6, &F::one(), term);
                            }
                            let sum_c_bz0: F = acc6_reduce::<F>(&sum_c_bz0_acc6);
                            let sum_c_az0: F = F::from_i64(sum_c_az0_i64);
                            let g0_term = sum_c_az0 * sum_c_bz0;
                            inner_acc[j] += e_in_even.mul_unreduced::<9>(g0_term);
                        }

                        // Group 1 (odd index) using same row inputs
                        let x_in_odd = x_in_even + 1;
                        let e_in_odd = E_in[x_in_odd];

                        let az1_u8 = eval_az_second_group(&row_inputs);
                        let bz1 = eval_bz_second_group(&row_inputs);

                        let g2_len = core::cmp::min(
                            NUM_REMAINING_R1CS_CONSTRAINTS,
                            UNIVARIATE_SKIP_DOMAIN_SIZE,
                        );
                        let mut az1_i32_padded: [i32; UNIVARIATE_SKIP_DOMAIN_SIZE] =
                            [0; UNIVARIATE_SKIP_DOMAIN_SIZE];
                        let mut bz1_s160_padded: [S160; UNIVARIATE_SKIP_DOMAIN_SIZE] =
                            [S160::from(0i128); UNIVARIATE_SKIP_DOMAIN_SIZE];
                        for i in 0..g2_len {
                            az1_i32_padded[i] = az1_u8[i] as i32;
                            bz1_s160_padded[i] = bz1[i];
                        }

                        for j in 0..UNIVARIATE_SKIP_DEGREE {
                            let coeffs = &coeffs_per_j[j];
                            // (sum_i c_i * Az1_i) * (sum_i c_i * Bz1_i)
                            let mut sum_c_bz1_acc = accs160_new::<F>();
                            let mut sum_c_az1_i64: i64 = 0;
                            for i in 0..UNIVARIATE_SKIP_DOMAIN_SIZE {
                                let c = coeffs[i] as i64;
                                if c == 0 {
                                    continue;
                                }
                                let cF = F::from_i64(c);
                                let az1_i = az1_i32_padded[i] as i64;
                                sum_c_az1_i64 += c * az1_i;
                                accs160_fmadd_s160(&mut sum_c_bz1_acc, &cF, bz1_s160_padded[i]);
                            }
                            let sum_c_bz1: F = accs160_reduce::<F>(&sum_c_bz1_acc);
                            let sum_c_az1: F = F::from_i64(sum_c_az1_i64);
                            let g1_term = sum_c_az1 * sum_c_bz1;
                            inner_acc[j] += e_in_odd.mul_unreduced::<9>(g1_term);
                        }
                    }
                    let e_out = E_out[x_out_val];
                    for j in 0..UNIVARIATE_SKIP_DEGREE {
                        let reduced = F::from_montgomery_reduce::<9>(inner_acc[j]);
                        acc_field[j] += e_out * reduced;
                    }
                }
                acc_field
            })
            .reduce(
                || [F::zero(); UNIVARIATE_SKIP_DEGREE],
                |mut a, b| {
                    for j in 0..UNIVARIATE_SKIP_DEGREE {
                        a[j] += b[j];
                    }
                    a
                },
            )
    }
}

impl<F: JoltField, T: Transcript> UniSkipFirstRoundInstance<F, T> for OuterUniSkipInstance<F> {
    const DEGREE_BOUND: usize = FIRST_ROUND_POLY_NUM_COEFFS - 1;
    const DOMAIN_SIZE: usize = UNIVARIATE_SKIP_DOMAIN_SIZE;

    fn input_claim(&self) -> F {
        F::zero()
    }

    fn compute_poly(&mut self) -> UniPoly<F> {
        // Load extended univariate-skip evaluations from prover state
        let extended = self.prover_state.as_ref().unwrap().extended_evals;

        let tau_high = self.tau[self.tau.len() - 1];

        // Compute the univariate-skip first round polynomial s1(Y) = L(τ_high, Y) · t1(Y)
        build_uniskip_first_round_poly::<
            F,
            UNIVARIATE_SKIP_DOMAIN_SIZE,
            UNIVARIATE_SKIP_DEGREE,
            UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE,
            FIRST_ROUND_POLY_NUM_COEFFS,
        >(&extended, tau_high)
    }
}

#[derive(Clone, Debug)]
pub struct StreamingRoundCache<F: JoltField> {
    pub t0: F,
    pub t_inf: F,
    /// Per (x_out, x_in) block values at y=r0 (lo := x_next=0, hi := x_next=1)
    pub az_lo: Vec<F>,
    pub az_hi: Vec<F>,
    pub bz_lo: Vec<F>,
    pub bz_hi: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct OuterProverState<F: JoltField> {
    pub split_eq_poly: GruenSplitEqPolynomial<F>,
    pub preprocess: Arc<JoltSharedPreprocessing>,
    pub trace: Arc<Vec<Cycle>>,
    pub az: DensePolynomial<F>,
    pub bz: DensePolynomial<F>,
    pub streaming_cache: Option<StreamingRoundCache<F>>,
}

/// SumcheckInstance for Spartan outer rounds after the univariate-skip first round.
/// Round 0 in this instance corresponds to the "streaming" round; subsequent rounds
/// use the remaining linear-time algorithm over cycle variables.
pub struct OuterRemainingSumcheck<F: JoltField> {
    /// Number of cycle bits for splitting opening points (consistent across prover/verifier)
    /// Total number of rounds is `1 + num_cycles_bits`
    pub num_cycles_bits: usize,
    /// The tau vector (length `2 + num_cycles_bits`, sampled at the beginning for Lagrange + eq poly)
    pub tau: Vec<F::Challenge>,
    /// The univariate-skip first round challenge
    pub r0_uniskip: F::Challenge,
    /// Claim after the univariate-skip first round, updated every round
    pub input_claim: F,
    /// Prover-only state (None on verifier)
    pub prover_state: Option<OuterProverState<F>>,
}

impl<F: JoltField> OuterRemainingSumcheck<F> {
    #[tracing::instrument(skip_all, name = "OuterRemainingSumcheck::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        num_cycles_bits: usize,
        tau: &[F::Challenge],
        r0_uniskip: F::Challenge,
        input_claim: F,
    ) -> Self {
        let (preprocessing, trace, _program_io, _final_mem) = state_manager.get_prover_data();

        let lagrange_evals_r = LagrangePolynomial::<F>::evals::<
            F::Challenge,
            UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&r0_uniskip);

        let tau_high = tau[tau.len() - 1];
        let tau_low = &tau[..tau.len() - 1];

        let lagrange_tau_r0 = LagrangePolynomial::<F>::lagrange_kernel::<
            F::Challenge,
            UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&r0_uniskip, &tau_high);

        let split_eq_poly: GruenSplitEqPolynomial<F> =
            GruenSplitEqPolynomial::<F>::new_with_scaling(
                tau_low,
                BindingOrder::LowToHigh,
                Some(lagrange_tau_r0),
            );

        let streaming_cache = Self::compute_streaming_round_cache(
            &preprocessing.shared,
            trace,
            &lagrange_evals_r,
            &split_eq_poly,
        );

        Self {
            num_cycles_bits,
            tau: tau.to_vec(),
            r0_uniskip,
            input_claim,
            prover_state: Some(OuterProverState {
                split_eq_poly,
                preprocess: Arc::new(preprocessing.shared.clone()),
                trace: Arc::new(trace.to_vec()),
                az: DensePolynomial::default(),
                bz: DensePolynomial::default(),
                streaming_cache: Some(streaming_cache),
            }),
        }
    }

    pub fn new_verifier(
        num_cycles_bits: usize,
        tau: Vec<F::Challenge>,
        r0_uniskip: F::Challenge,
        input_claim: F,
    ) -> Self {
        Self {
            num_cycles_bits,
            tau,
            r0_uniskip,
            input_claim,
            prover_state: None,
        }
    }

    /// Compute the quadratic evaluations for the streaming round (right after univariate skip).
    ///
    /// This uses the streaming algorithm to compute the sum-check polynomial for the round
    /// right after the univariate skip round.
    ///
    /// Recall that we need to compute
    ///
    /// `t_i(0) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] *
    ///       unbound_coeffs_a(x_out, x_in, 0, r) * unbound_coeffs_b(x_out, x_in, 0, r)`
    ///
    /// and
    ///
    /// `t_i(∞) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] * (unbound_coeffs_a(x_out,
    /// x_in, ∞, r) * unbound_coeffs_b(x_out, x_in, ∞, r))`
    ///
    /// Here the "_a,b" subscript indicates the coefficients of `unbound_coeffs` corresponding to
    /// Az and Bz respectively. Note that we index with x_out being the MSB here.
    ///
    /// Importantly, since the eval at `r` is not cached, we will need to recompute it via another
    /// sum
    ///
    /// `unbound_coeffs_{a,b}(x_out, x_in, {0,∞}, r) = \sum_{y in D} Lagrange(r, y) *
    /// unbound_coeffs_{a,b}(x_out, x_in, {0,∞}, y)`
    ///
    /// (and the eval at ∞ is computed as (eval at 1) - (eval at 0))
    #[inline]
    fn compute_streaming_round_cache(
        preprocess: &JoltSharedPreprocessing,
        trace: &[Cycle],
        lagrange_evals_r: &[F; UNIVARIATE_SKIP_DOMAIN_SIZE],
        split_eq_poly: &GruenSplitEqPolynomial<F>,
    ) -> StreamingRoundCache<F> {
        let num_x_out_vals = split_eq_poly.E_out_current_len();
        let num_x_in_vals = split_eq_poly.E_in_current_len();
        let iter_num_x_in_vars = num_x_in_vals.log_2();
        debug_assert!(
            num_x_out_vals > 0,
            "E_out_current_len() must be > 0 for outer streaming cache"
        );

        let groups_exact = num_x_out_vals
            .checked_mul(num_x_in_vals)
            .expect("overflow computing groups_exact");

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

        struct ChunkOut<F: JoltField> {
            base: usize,
            len: usize,
            az_lo: Vec<F>,
            az_hi: Vec<F>,
            bz_lo: Vec<F>,
            bz_hi: Vec<F>,
            sum0: F,
            sum_inf: F,
        }

        let chunk_results: Vec<ChunkOut<F>> = (0..num_parallel_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let x_out_start = chunk_idx * x_out_chunk_size;
                let x_out_end = core::cmp::min((chunk_idx + 1) * x_out_chunk_size, num_x_out_vals);
                let chunk_width = x_out_end.saturating_sub(x_out_start);
                let local_len = chunk_width.saturating_mul(num_x_in_vals);
                let mut local_az_lo: Vec<F> = Vec::with_capacity(local_len);
                let mut local_az_hi: Vec<F> = Vec::with_capacity(local_len);
                let mut local_bz_lo: Vec<F> = Vec::with_capacity(local_len);
                let mut local_bz_hi: Vec<F> = Vec::with_capacity(local_len);

                let mut task_sum0 = F::zero();
                let mut task_sum_inf = F::zero();
                for x_out_val in x_out_start..x_out_end {
                    let mut inner_sum0 = F::Unreduced::<9>::zero();
                    let mut inner_sum_inf = F::Unreduced::<9>::zero();
                    for x_in_val in 0..num_x_in_vals {
                        let current_step_idx = (x_out_val << iter_num_x_in_vars) | x_in_val;
                        let row_inputs =
                            R1CSCycleInputs::from_trace::<F>(preprocess, trace, current_step_idx);
                        let az0 = compute_az_r_group0(&row_inputs, &lagrange_evals_r[..]);
                        let bz0 = compute_bz_r_group0(&row_inputs, &lagrange_evals_r[..]);
                        let az1 = compute_az_r_group1(&row_inputs, &lagrange_evals_r[..]);
                        let bz1 = compute_bz_r_group1(&row_inputs, &lagrange_evals_r[..]);
                        let p0 = az0 * bz0;
                        let slope = (az1 - az0) * (bz1 - bz0);
                        let e_in = if num_x_in_vals == 0 {
                            F::one()
                        } else if num_x_in_vals == 1 {
                            split_eq_poly.E_in_current()[0]
                        } else {
                            split_eq_poly.E_in_current()[x_in_val]
                        };
                        inner_sum0 += e_in.mul_unreduced::<9>(p0);
                        inner_sum_inf += e_in.mul_unreduced::<9>(slope);
                        local_az_lo.push(az0);
                        local_bz_lo.push(bz0);
                        local_az_hi.push(az1);
                        local_bz_hi.push(bz1);
                    }
                    let e_out = if num_x_out_vals > 0 {
                        split_eq_poly.E_out_current()[x_out_val]
                    } else {
                        F::zero()
                    };
                    let reduced0 = F::from_montgomery_reduce::<9>(inner_sum0);
                    let reduced_inf = F::from_montgomery_reduce::<9>(inner_sum_inf);
                    task_sum0 += e_out * reduced0;
                    task_sum_inf += e_out * reduced_inf;
                }
                ChunkOut {
                    base: x_out_start.saturating_mul(num_x_in_vals),
                    len: local_az_lo.len(),
                    az_lo: local_az_lo,
                    az_hi: local_az_hi,
                    bz_lo: local_bz_lo,
                    bz_hi: local_bz_hi,
                    sum0: task_sum0,
                    sum_inf: task_sum_inf,
                }
            })
            .collect();

        // Assemble global buffers and sums
        let mut az_lo: Vec<F> = unsafe_allocate_zero_vec(groups_exact);
        let mut az_hi: Vec<F> = unsafe_allocate_zero_vec(groups_exact);
        let mut bz_lo: Vec<F> = unsafe_allocate_zero_vec(groups_exact);
        let mut bz_hi: Vec<F> = unsafe_allocate_zero_vec(groups_exact);
        let mut t0_acc = F::zero();
        let mut t_inf_acc = F::zero();
        for chunk in &chunk_results {
            t0_acc += chunk.sum0;
            t_inf_acc += chunk.sum_inf;
            let dst_base = chunk.base;
            let dst_end = dst_base + chunk.len;
            az_lo[dst_base..dst_end].copy_from_slice(&chunk.az_lo);
            az_hi[dst_base..dst_end].copy_from_slice(&chunk.az_hi);
            bz_lo[dst_base..dst_end].copy_from_slice(&chunk.bz_lo);
            bz_hi[dst_base..dst_end].copy_from_slice(&chunk.bz_hi);
        }

        StreamingRoundCache {
            t0: t0_acc,
            t_inf: t_inf_acc,
            az_lo,
            az_hi,
            bz_lo,
            bz_hi,
        }
    }

    /// Bind the streaming round after deriving the first challenge r_0.
    ///
    /// As we compute each `{a/b/c}(x_out, x_in, {0,∞}, r)`, we will
    /// store them in `bound_coeffs` in sparse format (the eval at 1 will be eval
    /// at 0 + eval at ∞). We then bind these bound coeffs with r_i for the next round.
    fn bind_streaming_round(&mut self, r_0: F::Challenge) {
        // Build dense bound arrays from streaming cache, or recompute on the fly
        if let Some(ps) = self.prover_state.as_mut() {
            if let Some(cache) = ps.streaming_cache.take() {
                let groups = cache.az_lo.len();
                let mut az_bound: Vec<F> = unsafe_allocate_zero_vec(groups);
                let mut bz_bound: Vec<F> = unsafe_allocate_zero_vec(groups);
                // Parallelize by x_out chunks; build local buffers, then assemble sequentially
                let num_x_out_vals = ps.split_eq_poly.E_out_current_len();
                let num_x_in_vals = ps.split_eq_poly.E_in_current_len();
                let chunk_threads = if num_x_out_vals > 0 {
                    core::cmp::min(
                        num_x_out_vals,
                        rayon::current_num_threads().next_power_of_two() * 8,
                    )
                } else {
                    1
                };
                let chunk_size = if num_x_out_vals > 0 {
                    core::cmp::max(1, num_x_out_vals.div_ceil(chunk_threads))
                } else {
                    0
                };
                struct BoundChunk<F: JoltField> {
                    base: usize,
                    az: Vec<F>,
                    bz: Vec<F>,
                }
                let chunks: Vec<BoundChunk<F>> = (0..chunk_threads)
                    .into_par_iter()
                    .map(|chunk_idx| {
                        let x_out_start = chunk_idx * chunk_size;
                        if x_out_start >= num_x_out_vals {
                            return BoundChunk {
                                base: 0,
                                az: Vec::new(),
                                bz: Vec::new(),
                            };
                        }
                        let x_out_end = core::cmp::min(x_out_start + chunk_size, num_x_out_vals);
                        let local_len = (x_out_end - x_out_start).saturating_mul(num_x_in_vals);
                        let mut local_az: Vec<F> = Vec::with_capacity(local_len);
                        let mut local_bz: Vec<F> = Vec::with_capacity(local_len);
                        for xo in x_out_start..x_out_end {
                            for xi in 0..num_x_in_vals {
                                let idx = xo * num_x_in_vals + xi;
                                let az0 = cache.az_lo[idx];
                                let az1 = cache.az_hi[idx];
                                let bz0 = cache.bz_lo[idx];
                                let bz1 = cache.bz_hi[idx];
                                local_az.push(az0 + r_0 * (az1 - az0));
                                local_bz.push(bz0 + r_0 * (bz1 - bz0));
                            }
                        }
                        BoundChunk {
                            base: x_out_start * num_x_in_vals,
                            az: local_az,
                            bz: local_bz,
                        }
                    })
                    .collect();
                for c in &chunks {
                    let end = c.base + c.az.len();
                    az_bound[c.base..end].copy_from_slice(&c.az);
                    bz_bound[c.base..end].copy_from_slice(&c.bz);
                }
                ps.az = DensePolynomial::new(az_bound);
                ps.bz = DensePolynomial::new(bz_bound);
                return;
            }
        }
        panic!("Streaming cache missing; fallback path removed as unreachable");
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
        let ps = self
            .prover_state
            .as_ref()
            .expect("prover state missing for remaining_quadratic_evals");
        let eq_poly = &ps.split_eq_poly;

        let n = ps.az.len();
        debug_assert_eq!(n, ps.bz.len());
        if eq_poly.E_in_current_len() == 1 {
            // groups are pairs (0,1)
            let groups = n / 2;
            (0..groups)
                .into_par_iter()
                .map(|g| {
                    let az0 = ps.az[2 * g];
                    let az1 = ps.az[2 * g + 1];
                    let bz0 = ps.bz[2 * g];
                    let bz1 = ps.bz[2 * g + 1];
                    let eq = eq_poly.E_out_current()[g];
                    let t0 = eq * az0 * bz0;
                    let tinf = eq * (az1 - az0) * (bz1 - bz0);
                    (t0, tinf)
                })
                .reduce(|| (F::zero(), F::zero()), |a, b| (a.0 + b.0, a.1 + b.1))
        } else {
            let num_x1_bits = eq_poly.E_in_current_len().log_2();
            let x1_len = eq_poly.E_in_current_len();
            let x2_len = eq_poly.E_out_current_len();
            (0..x2_len)
                .into_par_iter()
                .map(|x2| {
                    let mut inner0 = F::zero();
                    let mut inner_inf = F::zero();
                    for x1 in 0..x1_len {
                        let g = (x2 << num_x1_bits) | x1;
                        let az0 = ps.az[2 * g];
                        let az1 = ps.az[2 * g + 1];
                        let bz0 = ps.bz[2 * g];
                        let bz1 = ps.bz[2 * g + 1];
                        let e_in = eq_poly.E_in_current()[x1];
                        inner0 += e_in * az0 * bz0;
                        inner_inf += e_in * (az1 - az0) * (bz1 - bz0);
                    }
                    let e_out = eq_poly.E_out_current()[x2];
                    (e_out * inner0, e_out * inner_inf)
                })
                .reduce(|| (F::zero(), F::zero()), |a, b| (a.0 + b.0, a.1 + b.1))
        }
    }

    pub fn final_sumcheck_evals(&self) -> [F; 2] {
        let ps = self.prover_state.as_ref();
        let az0 = ps
            .and_then(|s| {
                if !s.az.is_empty() {
                    Some(s.az[0])
                } else {
                    None
                }
            })
            .unwrap_or(F::zero());
        let bz0 = ps
            .and_then(|s| {
                if !s.bz.is_empty() {
                    Some(s.bz[0])
                } else {
                    None
                }
            })
            .unwrap_or(F::zero());
        [az0, bz0]
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for OuterRemainingSumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        1 + self.num_cycles_bits
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    fn compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F> {
        let (t0, t_inf) = if round == 0 {
            let ps = self.prover_state.as_ref().expect("prover state missing");
            let cache = ps
                .streaming_cache
                .as_ref()
                .expect("streaming cache missing in round 0");
            (cache.t0, cache.t_inf)
        } else {
            self.remaining_quadratic_evals()
        };
        let eq_poly = &self
            .prover_state
            .as_ref()
            .expect("prover state missing")
            .split_eq_poly;
        let evals = eq_poly.gruen_evals_deg_3(t0, t_inf, previous_claim);
        vec![evals[0], evals[1], evals[2]]
    }

    fn bind(&mut self, r_j: F::Challenge, round: usize) {
        if round == 0 {
            self.bind_streaming_round(r_j);
        } else {
            let ps = self.prover_state.as_mut().expect("prover state missing");
            ps.az.bind_parallel(r_j, BindingOrder::LowToHigh);
            ps.bz.bind_parallel(r_j, BindingOrder::LowToHigh);
        }

        // Bind eq_poly for next round
        self.prover_state
            .as_mut()
            .expect("prover state missing")
            .split_eq_poly
            .bind(r_j);
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<std::rc::Rc<std::cell::RefCell<VerifierOpeningAccumulator<F>>>>,
        r_tail: &[F::Challenge],
    ) -> F {
        let acc_cell = accumulator.as_ref().expect("accumulator required");
        let acc_ref = acc_cell.borrow();
        let (_, claim_Az) = acc_ref
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter);
        let (_, claim_Bz) = acc_ref
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanBz, SumcheckId::SpartanOuter);

        let tau_high = &self.tau[self.tau.len() - 1];
        let tau_high_bound_r0 = LagrangePolynomial::<F>::lagrange_kernel::<
            F::Challenge,
            UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(tau_high, &self.r0_uniskip);

        let tau_low = &self.tau[..self.tau.len() - 1];
        let r_tail_reversed: Vec<F::Challenge> = r_tail.iter().rev().copied().collect();
        let tau_bound_r_tail_reversed = EqPolynomial::mle(tau_low, &r_tail_reversed);

        tau_high_bound_r0 * tau_bound_r_tail_reversed * claim_Az * claim_Bz
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
        // Append Az, Bz claims and corresponding opening point
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

        // Handle witness openings at r_cycle (use consistent split length)
        let (r_cycle, _rx_var) = opening_point.r.split_at(self.num_cycles_bits);

        // Compute claimed witness evals and append virtual openings for all R1CS inputs
        let ps = self.prover_state.as_ref().expect("prover state missing");
        let claimed_witness_evals =
            compute_claimed_witness_evals::<F>(&ps.preprocess, ps.trace.as_slice(), r_cycle);
        for (i, input) in ALL_R1CS_INPUTS.iter().enumerate() {
            acc.append_virtual(
                transcript,
                VirtualPolynomial::from(input),
                SumcheckId::SpartanOuter,
                OpeningPoint::new(r_cycle.to_vec()),
                claimed_witness_evals[i],
            );
        }
    }

    fn cache_openings_verifier(
        &self,
        accumulator: std::rc::Rc<std::cell::RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let mut acc = accumulator.borrow_mut();
        // Populate Az, Bz openings at the full outer opening point
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

        // Append witness openings at r_cycle (no claims at verifier) for all R1CS inputs
        let (r_cycle, _rx_var) = opening_point.r.split_at(self.num_cycles_bits);
        ALL_R1CS_INPUTS.iter().for_each(|input| {
            acc.append_virtual(
                transcript,
                VirtualPolynomial::from(input),
                SumcheckId::SpartanOuter,
                OpeningPoint::new(r_cycle.to_vec()),
            );
        });
    }
}
