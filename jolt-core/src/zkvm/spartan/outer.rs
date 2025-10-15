#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_ff::biginteger::S160;
use ark_std::Zero;
use rayon::prelude::*;
use std::sync::Arc;
use tracer::instruction::Cycle;

use crate::field::{JoltField, OptimizedMul};
use crate::poly::dense_mlpoly::DensePolynomial;
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
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::BIG_ENDIAN;
use crate::poly::opening_proof::{
    OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
};
use crate::utils::univariate_skip::accum::{
    accs160_fmadd_s160, accs160_new, accs160_reduce, s160_to_field,
};
use crate::utils::univariate_skip::{
    compute_az_r_group0, compute_az_r_group1, compute_bz_r_group0, compute_bz_r_group1,
};
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::r1cs::inputs::{
    compute_claimed_witness_evals, ALL_R1CS_INPUTS
};
use crate::zkvm::r1cs::{
    constraints::{
        eval_az_first_group, eval_az_second_group, eval_bz_first_group, eval_bz_second_group,
        FIRST_ROUND_POLY_NUM_COEFFS, UNIVARIATE_SKIP_DEGREE,
        UNIVARIATE_SKIP_DOMAIN_SIZE, UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE,
    },
    inputs::R1CSCycleInputs,
};
use crate::zkvm::witness::VirtualPolynomial;
use crate::zkvm::JoltSharedPreprocessing;

// Spartan Outer sumcheck
// (with univariate-skip first round on Z, and no Cz term given all eq conditional constraints)
//
// We define a univariate in Z first-round polynomial
//   s1(Z) := L_Z(τ_high) · Σ_{x_out ∈ {0,1}^{m_out}} Σ_{x_in ∈ {0,1}^{m_in}}
//              E_out(r_out, x_out) · E_in(r_in, x_in) ·
//              [ Az(x_out, x_in, Z) · Bz(x_out, x_in, Z) ],
// where L_Z(τ_high) is the Lagrange basis polynomial over the univariate-skip
// base domain evaluated at τ_high, and Az(·,·,Z), Bz(·,·,Z) are the
// per-row univariate polynomials in Z induced by the R1CS row (split into two
// internal groups in code, but algebraically composing to Az·Bz at Z).
// The prover sends s1(Z) via univariate-skip by evaluating t1(Z) := Σ Σ E_out·E_in·(Az·Bz)
// on an extended grid Z ∈ {−D..D} outside the base window, interpolating t1,
// multiplying by L_Z(τ_high) to obtain s1, and the verifier samples r0.
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

    #[tracing::instrument(
        skip_all,
        name = "OuterUniSkipInstance::compute_univariate_skip_extended_evals"
    )]
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
            if idx >= UNIVARIATE_SKIP_DEGREE {
                break;
            }
            targets[idx] = p;
            idx += 1;
            n -= 1;
            p += 1;
        }
        while idx < UNIVARIATE_SKIP_DEGREE && n >= ext_left {
            targets[idx] = n;
            idx += 1;
            n -= 1;
        }
        while idx < UNIVARIATE_SKIP_DEGREE && p <= ext_right {
            targets[idx] = p;
            idx += 1;
            p += 1;
        }
        debug_assert_eq!(idx, UNIVARIATE_SKIP_DEGREE);

        let target_shifts: [i64; UNIVARIATE_SKIP_DEGREE] =
            core::array::from_fn(|j| targets[j] - base_left);
        let coeffs_per_j: [[i32; UNIVARIATE_SKIP_DOMAIN_SIZE]; UNIVARIATE_SKIP_DEGREE] =
            core::array::from_fn(|j| {
                LagrangeHelper::shift_coeffs_i32::<UNIVARIATE_SKIP_DOMAIN_SIZE>(target_shifts[j])
            });

        let num_x_out_vals = split_eq.E_out_current_len();
        let num_x_in_vals = split_eq.E_in_current_len();
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

        let extended: [F; UNIVARIATE_SKIP_DEGREE] = (0..num_parallel_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let x_out_start = chunk_idx * x_out_chunk_size;
                let x_out_end = core::cmp::min((chunk_idx + 1) * x_out_chunk_size, num_x_out_vals);
                let mut acc_field: [F; UNIVARIATE_SKIP_DEGREE] =
                    [F::zero(); UNIVARIATE_SKIP_DEGREE];

                for x_out_val in x_out_start..x_out_end {
                    let mut inner_acc: [<F as JoltField>::Unreduced<9>; UNIVARIATE_SKIP_DEGREE] =
                        [<F as JoltField>::Unreduced::<9>::zero(); UNIVARIATE_SKIP_DEGREE];

                    for x_in_val in 0..num_x_in_vals {
                        let current_step_idx = (x_out_val << iter_num_x_in_vars) | x_in_val;
                        let row_inputs = R1CSCycleInputs::from_trace::<F>(
                            &*self.preprocess,
                            self.trace.as_slice(),
                            current_step_idx,
                        );

                        let e_in = if num_x_in_vals == 0 {
                            F::one()
                        } else {
                            split_eq.E_in_current()[x_in_val]
                        };

                        let az1_bool = eval_az_first_group(&row_inputs);
                        let bz1_s160 = eval_bz_first_group(&row_inputs);
                        let az2_i96 = eval_az_second_group(&row_inputs);
                        let bz2 = eval_bz_second_group(&row_inputs);

                        let mut az2_i128_padded: [i128; UNIVARIATE_SKIP_DOMAIN_SIZE] =
                            [0; UNIVARIATE_SKIP_DOMAIN_SIZE];
                        let mut bz2_s160_padded: [S160; UNIVARIATE_SKIP_DOMAIN_SIZE] =
                            [S160::from(0i128); UNIVARIATE_SKIP_DOMAIN_SIZE];
                        for i in 0..UNIVARIATE_SKIP_DOMAIN_SIZE {
                            az2_i128_padded[i] = az2_i96[i].to_i128();
                            bz2_s160_padded[i] = bz2[i];
                            // no Cz term
                        }

                        for j in 0..UNIVARIATE_SKIP_DEGREE {
                            let coeffs = &coeffs_per_j[j];
                            // Group 0: sum_i c_i * (Az1_i ? 1 : 0) * Bz1_i
                            let mut bz1_acc = accs160_new::<F>();
                            for i in 0..UNIVARIATE_SKIP_DOMAIN_SIZE {
                                let c = coeffs[i] as i64;
                                if c == 0 || !az1_bool[i] {
                                    continue;
                                }
                                accs160_fmadd_s160(&mut bz1_acc, &F::from_i64(c), bz1_s160[i]);
                            }
                            let bz1_ext: F = accs160_reduce::<F>(&bz1_acc);
                            inner_acc[j] += e_in.mul_unreduced::<9>(bz1_ext);

                            // Group 1: sum_i c_i * (Az2_i * Bz2_i)
                            for i in 0..UNIVARIATE_SKIP_DOMAIN_SIZE {
                                let c = coeffs[i] as i64;
                                if c == 0 {
                                    continue;
                                }
                                let cF = F::from_i64(c);
                                let az2 = F::from_i128(az2_i128_padded[i]);
                                let bz2 = s160_to_field::<F>(&bz2_s160_padded[i]);
                                let term = az2 * bz2;
                                inner_acc[j] += e_in.mul_unreduced::<9>(term * cF);
                            }
                        }
                    }
                    let e_out = if num_x_out_vals > 0 {
                        split_eq.E_out_current()[x_out_val]
                    } else {
                        F::zero()
                    };
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
            );

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

        println!("tau: {:?}", self.tau);
        let split_eq = GruenSplitEqPolynomial::<F>::new(tau_low, BindingOrder::LowToHigh);
        let extended = self.compute_univariate_skip_extended_evals(&split_eq);
        println!("extended: {:?}", extended);

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
        println!("t1_coeffs: {:?}", t1_coeffs);

        let tau_high = self.tau[self.tau.len() - 1];
        let lagrange_poly_values =
            LagrangePolynomial::<F>::evals::<F::Challenge, UNIVARIATE_SKIP_DOMAIN_SIZE>(&tau_high);
        let lagrange_poly_coeffs = LagrangePolynomial::interpolate_coeffs::<
            UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&lagrange_poly_values);

        let mut s1_coeffs: [F; FIRST_ROUND_POLY_NUM_COEFFS] =
            [F::zero(); FIRST_ROUND_POLY_NUM_COEFFS];
        for (i, &a) in lagrange_poly_coeffs.iter().enumerate() {
            for (j, &b) in t1_coeffs.iter().enumerate() {
                s1_coeffs[i + j] += a * b;
            }
        }

        println!("s1_coeffs: {:?}", s1_coeffs);

        UniPoly::from_coeff(s1_coeffs.to_vec())
    }

    fn output_claim(&self, _r: &[F::Challenge]) -> F {
        // Not used by Spartan outer at this stage; verifier computes s1(r) directly from transcript
        F::zero()
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
    pub input_claim: F,
    pub r0_uniskip: F::Challenge,
    pub total_rounds: usize,
    /// Only used by verifier to compute expected_output_claim
    pub tau: Option<Vec<F::Challenge>>,
    /// Number of cycle bits for splitting opening points (consistent across prover/verifier)
    pub num_cycles_bits: usize,
    /// Prover-only state (None on verifier)
    pub prover_state: Option<OuterProverState<F>>,
}

impl<F: JoltField> OuterRemainingSumcheck<F> {
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

        // Prepare streaming round cache: compute t0, t_inf and dense per-block arrays at y=r0
        let lagrange_evals_r = LagrangePolynomial::<F>::evals::<
            F::Challenge,
            UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&r0_uniskip);

        let num_x_out_vals = split_eq_poly.E_out_current_len();
        let num_x_in_vals = split_eq_poly.E_in_current_len();
        let iter_num_x_in_vars = if num_x_in_vals > 0 {
            num_x_in_vals.log_2()
        } else {
            0
        };

        let groups = num_x_out_vals.saturating_mul(core::cmp::max(1, num_x_in_vals));
        let mut az_lo: Vec<F> = Vec::with_capacity(groups);
        let mut az_hi: Vec<F> = Vec::with_capacity(groups);
        let mut bz_lo: Vec<F> = Vec::with_capacity(groups);
        let mut bz_hi: Vec<F> = Vec::with_capacity(groups);

        let mut t0_acc = F::zero();
        let mut t_inf_acc = F::zero();

        for x_out_val in 0..num_x_out_vals {
            let mut inner_sum0 = F::Unreduced::<9>::zero();
            let mut inner_sumInf = F::Unreduced::<9>::zero();

            for x_in_val in 0..num_x_in_vals {
                let current_step_idx = (x_out_val << iter_num_x_in_vars) | x_in_val;
                let row_inputs = R1CSCycleInputs::from_trace::<F>(
                    &preprocessing.shared,
                    trace,
                    current_step_idx,
                );

                // reduce to field values at y=r for both x_next
                let az0 = compute_az_r_group0(&row_inputs, &lagrange_evals_r[..]);
                let bz0 = compute_bz_r_group0(&row_inputs, &lagrange_evals_r[..]);
                let az1 = compute_az_r_group1(&row_inputs, &lagrange_evals_r[..]);
                let bz1 = compute_bz_r_group1(&row_inputs, &lagrange_evals_r[..]);

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

                // push dense per-block values in index order current_step_idx
                az_lo.push(az0);
                bz_lo.push(bz0);
                az_hi.push(az1);
                bz_hi.push(bz1);
            }

            let e_out = if num_x_out_vals > 0 {
                split_eq_poly.E_out_current()[x_out_val]
            } else {
                F::zero()
            };
            let reduced0 = F::from_montgomery_reduce::<9>(inner_sum0);
            let reducedInf = F::from_montgomery_reduce::<9>(inner_sumInf);
            t0_acc += e_out * reduced0;
            t_inf_acc += e_out * reducedInf;
        }

        let streaming_cache = StreamingRoundCache {
            t0: t0_acc,
            t_inf: t_inf_acc,
            az_lo,
            az_hi,
            bz_lo,
            bz_hi,
        };

        Self {
            input_claim,
            r0_uniskip,
            total_rounds,
            tau: None,
            num_cycles_bits,
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
        input_claim: F,
        r0_uniskip: F::Challenge,
        total_rounds: usize,
        tau: Vec<F::Challenge>,
        num_cycles_bits: usize,
    ) -> Self {
        Self {
            input_claim,
            r0_uniskip,
            total_rounds,
            tau: Some(tau),
            num_cycles_bits,
            prover_state: None,
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

        let ps = self
            .prover_state
            .as_ref()
            .expect("prover state missing for streaming_quadratic_evals");
        let eq_poly = &ps.split_eq_poly;
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
                            &ps.preprocess,
                            ps.trace.as_slice(),
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
                    let t0 = eq * (az0.mul_0_optimized(bz0));
                    let tinf = eq * ((az1 - az0).mul_0_optimized(bz1 - bz0));
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
                        inner0 += e_in * (az0.mul_0_optimized(bz0));
                        inner_inf += e_in * ((az1 - az0).mul_0_optimized(bz1 - bz0));
                    }
                    let e_out = eq_poly.E_out_current()[x2];
                    (e_out * inner0, e_out * inner_inf)
                })
                .reduce(|| (F::zero(), F::zero()), |a, b| (a.0 + b.0, a.1 + b.1))
        }
    }

    /// Bind the streaming round after deriving challenge r_i.
    ///
    /// As we compute each `{a/b/c}(x_out, x_in, {0,∞}, r)`, we will
    /// store them in `bound_coeffs` in sparse format (the eval at 1 will be eval
    /// at 0 + eval at ∞). We then bind these bound coeffs with r_i for the next round.
    fn bind_streaming_round(&mut self, r_i: F::Challenge) {
        // Build dense bound arrays from streaming cache, or recompute on the fly
        if let Some(ps) = self.prover_state.as_mut() {
            if let Some(cache) = ps.streaming_cache.take() {
                let groups = cache.az_lo.len();
                let mut az_bound: Vec<F> = Vec::with_capacity(groups);
                let mut bz_bound: Vec<F> = Vec::with_capacity(groups);
                for g in 0..groups {
                    let az0 = cache.az_lo[g];
                    let az1 = cache.az_hi[g];
                    let bz0 = cache.bz_lo[g];
                    let bz1 = cache.bz_hi[g];
                    az_bound.push(az0 + r_i * (az1 - az0));
                    bz_bound.push(bz0 + r_i * (bz1 - bz0));
                }
                ps.az = DensePolynomial::new(az_bound);
                ps.bz = DensePolynomial::new(bz_bound);
                return;
            }
        }
        // Fallback: recompute the per-block values and bind
        // Fallback: recompute the per-block values and bind
        let lagrange_evals_r = LagrangePolynomial::<F>::evals::<
            F::Challenge,
            UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&self.r0_uniskip);
        let ps = self.prover_state.as_ref().expect("prover state missing");
        let eq_poly = &ps.split_eq_poly;
        let num_x_out_vals = eq_poly.E_out_current_len();
        let num_x_in_vals = eq_poly.E_in_current_len();
        let iter_num_x_in_vars = if num_x_in_vals > 0 {
            num_x_in_vals.log_2()
        } else {
            0
        };
        let groups = num_x_out_vals.saturating_mul(core::cmp::max(1, num_x_in_vals));
        let mut az_bound: Vec<F> = Vec::with_capacity(groups);
        let mut bz_bound: Vec<F> = Vec::with_capacity(groups);
        for x_out_val in 0..num_x_out_vals {
            for x_in_val in 0..num_x_in_vals {
                let current_step_idx = (x_out_val << iter_num_x_in_vars) | x_in_val;
                let ps_inner = self.prover_state.as_ref().expect("prover state missing");
                let row_inputs = R1CSCycleInputs::from_trace::<F>(
                    &ps_inner.preprocess,
                    ps_inner.trace.as_slice(),
                    current_step_idx,
                );
                let az0 = compute_az_r_group0(&row_inputs, &lagrange_evals_r[..]);
                let bz0 = compute_bz_r_group0(&row_inputs, &lagrange_evals_r[..]);
                let az1 = compute_az_r_group1(&row_inputs, &lagrange_evals_r[..]);
                let bz1 = compute_bz_r_group1(&row_inputs, &lagrange_evals_r[..]);
                az_bound.push(az0 + r_i * (az1 - az0));
                bz_bound.push(bz0 + r_i * (bz1 - bz0));
            }
        }
        if let Some(ps) = self.prover_state.as_mut() {
            ps.az = DensePolynomial::new(az_bound);
            ps.bz = DensePolynomial::new(bz_bound);
        }
    }

    pub fn final_sumcheck_evals(&self) -> [F; 2] {
        let ps = self.prover_state.as_ref();
        let az0 = ps
            .and_then(|s| if s.az.len() > 0 { Some(s.az[0]) } else { None })
            .unwrap_or(F::zero());
        let bz0 = ps
            .and_then(|s| if s.bz.len() > 0 { Some(s.bz[0]) } else { None })
            .unwrap_or(F::zero());
        [az0, bz0]
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
            let ps = self.prover_state.as_ref().expect("prover state missing");
            if let Some(cache) = ps.streaming_cache.as_ref() {
                (cache.t0, cache.t_inf)
            } else {
                self.streaming_quadratic_evals()
            }
        } else {
            self.remaining_quadratic_evals()
        };
        let cubic = Self::build_cubic_from_quadratic(
            &self
                .prover_state
                .as_ref()
                .expect("prover state missing")
                .split_eq_poly,
            t0,
            t_inf,
            previous_claim,
        );
        let m0 = cubic.evaluate::<F>(&F::zero());
        let m2 = cubic.evaluate::<F>(&F::from_u64(2));
        let m3 = cubic.evaluate::<F>(&F::from_u64(3));
        vec![m0, m2, m3]
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

        let tau_bound_rx = EqPolynomial::mle(tau, &r_reversed);
        tau_bound_rx * (claim_Az * claim_Bz)
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
                VirtualPolynomial::try_from(input).ok().unwrap(),
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
                VirtualPolynomial::try_from(input).ok().unwrap(),
                SumcheckId::SpartanOuter,
                OpeningPoint::new(r_cycle.to_vec()),
            );
        });
    }
}
