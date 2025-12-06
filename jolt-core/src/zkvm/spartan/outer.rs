use std::sync::Arc;

use allocative::Allocative;
use ark_std::Zero;
use rayon::prelude::*;
use tracer::instruction::Cycle;
use tracer::LazyTraceIterator;

use crate::field::BarrettReduce;
use crate::field::{FMAdd, JoltField, MontgomeryReduce};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::lagrange_poly::LagrangePolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, PolynomialBinding};
use crate::poly::multiquadratic_poly::MultiquadraticPolynomial;
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::streaming_schedule::StreamingSchedule;
use crate::subprotocols::sumcheck_prover::{
    SumcheckInstanceProver, UniSkipFirstRoundInstanceProver,
};
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::subprotocols::univariate_skip::{build_uniskip_first_round_poly, UniSkipState};
use crate::transcripts::Transcript;
use crate::utils::accumulation::{Acc5U, Acc6S, Acc7S, Acc8S};
use crate::utils::expanding_table::ExpandingTable;
use crate::utils::math::Math;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::r1cs::{
    constraints::{
        OUTER_FIRST_ROUND_POLY_NUM_COEFFS, OUTER_UNIVARIATE_SKIP_DEGREE,
        OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE, OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE,
    },
    evaluation::R1CSEval,
    inputs::{R1CSCycleInputs, ALL_R1CS_INPUTS},
};
use crate::zkvm::witness::VirtualPolynomial;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;

/// Degree bound of the sumcheck round polynomials for [`OuterRemainingSumcheckVerifier`].
const OUTER_REMAINING_DEGREE_BOUND: usize = 3;
// this represents the index position in multi-quadratic poly array
// This should actually be d where degree is the degree of the streaming data structure
// For example : MultiQuadratic has d=2; for cubic this would be 3 etc.
const INFINITY: usize = 2;

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
//  L(τ_high, r_high) · Eq_τ(τ, r) · (Az(r) · Bz(r)).

/// Uni-skip instance for Spartan outer sumcheck, computing the first-round polynomial only.
#[derive(Allocative)]
pub struct OuterUniSkipInstanceProver<F: JoltField> {
    tau: Vec<F::Challenge>,
    /// Evaluations of t1(Z) at the extended univariate-skip targets (outside base window)
    extended_evals: [F; OUTER_UNIVARIATE_SKIP_DEGREE],
}

impl<F: JoltField> OuterUniSkipInstanceProver<F> {
    #[tracing::instrument(skip_all, name = "OuterUniSkipInstanceProver::gen")]
    pub fn gen(
        trace: &[Cycle],
        bytecode_preprocessing: &BytecodePreprocessing,
        tau: &[F::Challenge],
    ) -> Self {
        let extended =
            Self::compute_univariate_skip_extended_evals(bytecode_preprocessing, trace, tau);

        let instance = Self {
            tau: tau.to_vec(),
            extended_evals: extended,
        };

        #[cfg(feature = "allocative")]
        print_data_structure_heap_usage("OuterUniSkipInstance", &instance);
        instance
    }

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
        bytecode_preprocessing: &BytecodePreprocessing,
        trace: &[Cycle],
        tau: &[F::Challenge],
    ) -> [F; OUTER_UNIVARIATE_SKIP_DEGREE] {
        // Build split-eq over full τ; new_with_scaling drops the last variable (τ_high) for the split,
        // and we carry an outer scaling factor (R^2) via current_scalar.
        let split_eq = GruenSplitEqPolynomial::<F>::new_with_scaling(
            tau,
            BindingOrder::LowToHigh,
            Some(F::MONTGOMERY_R_SQUARE),
        );
        let outer_scale = split_eq.get_current_scalar(); // = R^2 at this stage

        let num_x_in_bits = split_eq.E_in_current_len().log_2();
        let num_x_in_prime_bits = num_x_in_bits.saturating_sub(1); // ignore last bit (group index)

        split_eq
            .par_fold_out_in(
                || [Acc8S::<F>::zero(); OUTER_UNIVARIATE_SKIP_DEGREE],
                |inner, g, x_in, e_in| {
                    // Decode (x_out, x_in') from g and choose group by the last x_in bit
                    let x_out = g >> num_x_in_bits;
                    let x_in_prime = x_in >> 1;
                    let base_step_idx = (x_out << num_x_in_prime_bits) | x_in_prime;

                    let row_inputs = R1CSCycleInputs::from_trace::<F>(
                        bytecode_preprocessing,
                        trace,
                        base_step_idx,
                    );
                    let eval = R1CSEval::<F>::from_cycle_inputs(&row_inputs);

                    let is_group1 = (x_in & 1) == 1;
                    for j in 0..OUTER_UNIVARIATE_SKIP_DEGREE {
                        let prod_s192 = if !is_group1 {
                            eval.extended_azbz_product_first_group(j)
                        } else {
                            eval.extended_azbz_product_second_group(j)
                        };
                        inner[j].fmadd(&e_in, &prod_s192);
                    }
                },
                |_x_out, e_out, inner| {
                    let mut out = [F::Unreduced::<9>::zero(); OUTER_UNIVARIATE_SKIP_DEGREE];
                    for j in 0..OUTER_UNIVARIATE_SKIP_DEGREE {
                        let reduced = inner[j].montgomery_reduce();
                        out[j] = e_out.mul_unreduced::<9>(reduced);
                    }
                    out
                },
                |mut a, b| {
                    for j in 0..OUTER_UNIVARIATE_SKIP_DEGREE {
                        a[j] += b[j];
                    }
                    a
                },
            )
            .map(|x| F::from_montgomery_reduce::<9>(x) * outer_scale)
    }
}

impl<F: JoltField, T: Transcript> UniSkipFirstRoundInstanceProver<F, T>
    for OuterUniSkipInstanceProver<F>
{
    fn input_claim(&self) -> F {
        F::zero()
    }

    #[tracing::instrument(skip_all, name = "OuterUniSkipInstanceProver::compute_poly")]
    fn compute_poly(&mut self) -> UniPoly<F> {
        // Load extended univariate-skip evaluations from prover state
        let extended_evals = &self.extended_evals;

        let tau_high = self.tau[self.tau.len() - 1];

        // Compute the univariate-skip first round polynomial s1(Y) = L(τ_high, Y) · t1(Y)
        build_uniskip_first_round_poly::<
            F,
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
            OUTER_UNIVARIATE_SKIP_DEGREE,
            OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE,
            OUTER_FIRST_ROUND_POLY_NUM_COEFFS,
        >(None, extended_evals, tau_high)
    }
}

/// SumcheckInstance for Spartan outer rounds after the univariate-skip first round.
/// Round 0 in this instance corresponds to the "streaming" round; subsequent rounds
/// use the remaining linear-time algorithm over cycle variables.
//#[derive(Allocative)]
//pub struct OuterRemainingSumcheckProver<F: JoltField> {
//    #[allocative(skip)]
//    bytecode_preprocessing: BytecodePreprocessing,
//    #[allocative(skip)]
//    trace: Arc<Vec<Cycle>>,
//    split_eq_poly: GruenSplitEqPolynomial<F>,
//    az: Option<DensePolynomial<F>>,
//    bz: Option<DensePolynomial<F>>,
//    t_prime_poly: Option<MultiquadraticPolynomial<F>>, // multiquadratic polynomial used to answer queries in a streaming window
//    /// The first round evals (t0, t_inf) computed from a streaming pass over the trace
//    first_round_evals: (F, F),
//    #[allocative(skip)]
//    params: OuterRemainingSumcheckParams<F>,
//}
//
#[derive(Allocative)]
pub struct OuterRemainingSumcheckProver<'a, F: JoltField, S: StreamingSchedule + Allocative> {
    #[allocative(skip)]
    bytecode_preprocessing: BytecodePreprocessing,
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    /// Split-eq instance used for both streaming and linear phases of the
    /// outer Spartan sumcheck over cycle variables.
    split_eq_poly: GruenSplitEqPolynomial<F>,
    az: Option<DensePolynomial<F>>,
    bz: Option<DensePolynomial<F>>,
    t_prime_poly: Option<MultiquadraticPolynomial<F>>, // multiquadratic polynomial used to answer queries in a streaming window
    r_grid: ExpandingTable<F>, // hadamard product of (1 - r_j, r_j) for bound variables so far to help with streaming
    /// The first round evals (t0, t_inf) computed from a streaming pass over the trace
    #[allocative(skip)]
    params: OuterRemainingSumcheckParams<F>,
    lagrange_evals_r0: [F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE],
    schedule: S,
    #[allocative(skip)]
    checkpoints: &'a [std::iter::Take<LazyTraceIterator>],
    checkpoint_interval: usize,
}

impl<'a, F: JoltField, S: StreamingSchedule + Allocative> OuterRemainingSumcheckProver<'a, F, S> {
    #[tracing::instrument(skip_all, name = "OuterRemainingSumcheckProver::gen")]
    pub fn gen(
        trace: Arc<Vec<Cycle>>,
        checkpoints: &'a [std::iter::Take<LazyTraceIterator>], // Add lifetime 'a, use slice
        checkpoint_interval: usize,
        bytecode_preprocessing: &BytecodePreprocessing,
        uni: &UniSkipState<F>,
        schedule: S,
    ) -> Self {
        let bytecode_preprocessing = bytecode_preprocessing.clone();
        let n_cycle_vars = trace.len().log_2();
        let outer_params = OuterRemainingSumcheckParams::new(n_cycle_vars, uni);

        let lagrange_evals_r = LagrangePolynomial::<F>::evals::<
            F::Challenge,
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&uni.r0);

        let tau_high = uni.tau[uni.tau.len() - 1];
        let tau_low = &uni.tau[..uni.tau.len() - 1];

        let lagrange_tau_r0 = LagrangePolynomial::<F>::lagrange_kernel::<
            F::Challenge,
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&uni.r0, &tau_high);

        let split_eq_poly: GruenSplitEqPolynomial<F> =
            GruenSplitEqPolynomial::<F>::new_with_scaling(
                tau_low,
                BindingOrder::LowToHigh,
                Some(lagrange_tau_r0),
            );

        // NOTE: The API changed recently: Both binding orders will technically pass
        // based on current implementation.
        let mut r_grid = ExpandingTable::new(1 << n_cycle_vars, BindingOrder::LowToHigh);
        r_grid.reset(F::one());

        Self {
            split_eq_poly,
            bytecode_preprocessing,
            trace,
            checkpoints,
            checkpoint_interval,
            az: None,
            bz: None,
            t_prime_poly: None,
            r_grid,
            params: outer_params,
            lagrange_evals_r0: lagrange_evals_r,
            schedule,
        }
    }

    // gets the evaluations of az(x, {0,1}^log(jlen), r) and bz(x, {0,1}^log(jlen), r)
    // where x is determined by the bit decomposition of offset
    // and r is log(klen) variables
    // this is used both in window computation (jlen is window size)
    // and in converting to linear time (offset is 0, log(jlen) is the number of unbound variables)
    // The caller must pass in `scaled_w`, the tensor product of the Lagrange weights
    // at r0 with the current `r_grid` weights:
    //   scaled_w[k][t] = lagrange_evals_r0[t] * r_grid[k]  (for klen > 1)
    // and scaled_w[0][t] = lagrange_evals_r0[t] when klen == 1 (no r_grid factor).
    #[allow(clippy::too_many_arguments)]
    fn extrapolate_from_binary_grid_to_tertiary_grid(
        &self,
        grid_az: &mut [F],
        grid_bz: &mut [F],
        jlen: usize,
        klen: usize,
        offset: usize,
        parallel: bool,
        scaled_w: &[[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE]],
    ) {
        let preprocess = &self.bytecode_preprocessing;
        let _checkpoints = &self.checkpoints;
        let _checkpoint_interval = self.checkpoint_interval;
        let trace = &self.trace;
        debug_assert_eq!(scaled_w.len(), klen);
        debug_assert_eq!(grid_az.len(), jlen);
        debug_assert_eq!(grid_bz.len(), jlen);

        // Unreduced accumulators per j for Az and the two Bz groups.
        let mut acc_az = vec![Acc5U::<F>::zero(); jlen];
        let mut acc_bz_first = vec![Acc6S::<F>::zero(); jlen];
        let mut acc_bz_second = vec![Acc7S::<F>::zero(); jlen];

        if !parallel {
            // Sequential traversal: iterate over j first and then k so that we
            // walk consecutive cycles in memory (full_idx increases by 1 inside
            // the inner loop).
            for j in 0..jlen {
                for k in 0..klen {
                    let full_idx = offset + j * klen + k;
                    let current_step_idx = full_idx >> 1;
                    let selector = (full_idx & 1) == 1;

                    // TODO: use the lazy trace iterator here instead of indexing directly into the
                    // trace that is all that needs to change for now
                    //let row_inputs_prime = R1CSCycleInputs::from_checkpoints::<F>(
                    //    preprocess,
                    //    checkpoints,
                    //    checkpoint_interval,
                    //    current_step_idx,
                    //);
                    let row_inputs =
                        R1CSCycleInputs::from_trace::<F>(preprocess, trace, current_step_idx);
                    let eval = R1CSEval::<F>::from_cycle_inputs(&row_inputs);
                    let w_k = &scaled_w[k];

                    if !selector {
                        eval.fmadd_first_group_at_r(w_k, &mut acc_az[j], &mut acc_bz_first[j]);
                    } else {
                        eval.fmadd_second_group_at_r(w_k, &mut acc_az[j], &mut acc_bz_second[j]);
                    }
                }
            }
        } else {
            // Parallel traversal over j for the linear-time prover.
            // Each worker owns disjoint accumulators for a fixed j, so there
            // are no data races. We reuse the precomputed scaled Lagrange weights
            // per k from `scaled_w`, avoiding redundant tensor products.
            acc_az
                .par_iter_mut()
                .zip(acc_bz_first.par_iter_mut())
                .zip(acc_bz_second.par_iter_mut())
                .enumerate()
                .for_each(|(j, ((acc_az_j, acc_bz_first_j), acc_bz_second_j))| {
                    for k in 0..klen {
                        let full_idx = offset + j * klen + k;
                        let current_step_idx = full_idx >> 1;
                        let selector = (full_idx & 1) == 1;

                        let row_inputs =
                            R1CSCycleInputs::from_trace::<F>(preprocess, trace, current_step_idx);
                        let eval = R1CSEval::<F>::from_cycle_inputs(&row_inputs);
                        let w_k = &scaled_w[k];

                        if !selector {
                            eval.fmadd_first_group_at_r(w_k, acc_az_j, acc_bz_first_j);
                        } else {
                            eval.fmadd_second_group_at_r(w_k, acc_az_j, acc_bz_second_j);
                        }
                    }
                });
        }

        // Final reductions: reduce accumulators and write to output slices.
        // Each chunk writes to disjoint indices, so parallel iteration is safe.
        const REDUCE_CHUNK_SIZE: usize = 4096;
        grid_az
            .par_chunks_mut(REDUCE_CHUNK_SIZE)
            .zip(grid_bz.par_chunks_mut(REDUCE_CHUNK_SIZE))
            .enumerate()
            .for_each(|(chunk_idx, (az_chunk, bz_chunk))| {
                let start = chunk_idx * REDUCE_CHUNK_SIZE;
                for (local_j, (az_out, bz_out)) in
                    az_chunk.iter_mut().zip(bz_chunk.iter_mut()).enumerate()
                {
                    let j = start + local_j;
                    *az_out = acc_az[j].barrett_reduce();
                    let bz_first_j = acc_bz_first[j].barrett_reduce();
                    let bz_second_j = acc_bz_second[j].barrett_reduce();
                    *bz_out = bz_first_j + bz_second_j;
                }
            });
    }

    // returns the grid of evaluations on {0,1,inf}^window_size
    // touches each cycle of the trace exactly once and in order!
    #[tracing::instrument(
        skip_all,
        name = "OuterRemainingSumcheckProver::compute_evaluation_grid_from_trace"
    )]
    fn compute_evaluation_grid_from_trace(&mut self, window_size: usize) {
        // semantics in one place (see `split_eq_poly::E_out_in_for_window`).
        let split_eq = &self.split_eq_poly;

        // helper constants
        let three_pow_dim = 3_usize.pow(window_size as u32);
        let jlen = 1 << window_size;
        let klen = 1 << split_eq.num_challenges();

        // Precompute the tensor product of the Lagrange weights at r0 with the
        // current r_grid weights so that all calls into `build_grids` can reuse
        // these scaled tables.
        let lagrange_evals_r = &self.lagrange_evals_r0;
        let r_grid = &self.r_grid;
        let scaled_w: Vec<[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE]> = if klen > 1 {
            debug_assert_eq!(klen, r_grid.len());
            (0..klen)
                .into_par_iter()
                .map(|k| {
                    let weight = r_grid[k];
                    let mut row = [F::zero(); OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE];
                    for t in 0..OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE {
                        row[t] = lagrange_evals_r[t] * weight;
                    }
                    row
                })
                .collect()
        } else {
            debug_assert_eq!(klen, 1);
            let mut row = [F::zero(); OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE];
            row.copy_from_slice(lagrange_evals_r);
            vec![row]
        };

        // Head-factor eq tables for this window.
        let (e_out, e_in) = split_eq.E_out_in_for_window(window_size);
        let e_in_len = e_in.len();

        // main logic: parallelize outer sum over E_out_current; for each x_out,
        // perform an inner unreduced accumulation over E_in_current and only
        // reduce once per grid cell, then multiply by E_out unreduced.
        let res_unr = e_out
            .par_iter()
            .enumerate()
            .map(|(out_idx, out_val)| {
                // Local unreduced accumulators and scratch buffers for this out_idx.
                let mut local_res_unr = vec![F::Unreduced::<9>::zero(); three_pow_dim];
                let mut buff_a: Vec<F> = vec![F::zero(); three_pow_dim];
                let mut buff_b = vec![F::zero(); three_pow_dim];
                let mut tmp = vec![F::zero(); three_pow_dim];
                let mut grid_a = vec![F::zero(); jlen];
                let mut grid_b = vec![F::zero(); jlen];

                for (in_idx, in_val) in e_in.iter().enumerate() {
                    let i = out_idx * e_in_len + in_idx;

                    // Reuse the same grid buffers across all x_in for this x_out.
                    grid_a.fill(F::zero());
                    grid_b.fill(F::zero());
                    // Keep this call sequential to avoid nested rayon parallelism.
                    self.extrapolate_from_binary_grid_to_tertiary_grid(
                        &mut grid_a,
                        &mut grid_b,
                        jlen,
                        klen,
                        i * jlen * klen,
                        false,
                        &scaled_w,
                    );

                    // Extrapolate grid_a and grid_b from {0,1}^window_size to {0,1,∞}^window_size.
                    MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                        &grid_a,
                        &mut buff_a,
                        &mut tmp,
                        window_size,
                    );
                    MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                        &grid_b,
                        &mut buff_b,
                        &mut tmp,
                        window_size,
                    );

                    let e_in_val = *in_val;
                    // For window_size == 1, we only need evals at 0 and ∞ (indices 0 and 2).
                    // The eval at 1 is unused by Gruen polynomial computation.
                    if window_size == 1 {
                        local_res_unr[0] += e_in_val.mul_unreduced::<9>(buff_a[0] * buff_b[0]);
                        local_res_unr[2] += e_in_val.mul_unreduced::<9>(buff_a[2] * buff_b[2]);
                    } else {
                        for idx in 0..three_pow_dim {
                            let val = buff_a[idx] * buff_b[idx];
                            local_res_unr[idx] += e_in_val.mul_unreduced::<9>(val);
                        }
                    }
                }

                // Fold in E_out for this x_out.
                let e_out_val = *out_val;
                for idx in 0..three_pow_dim {
                    let inner_red = F::from_montgomery_reduce::<9>(local_res_unr[idx]);
                    local_res_unr[idx] = e_out_val.mul_unreduced::<9>(inner_red);
                }
                local_res_unr
            })
            .reduce(
                || vec![F::Unreduced::<9>::zero(); three_pow_dim],
                |mut acc, local| {
                    for idx in 0..three_pow_dim {
                        acc[idx] += local[idx];
                    }
                    acc
                },
            );

        // Final reduction over all (x_out, x_in)
        let res: Vec<F> = res_unr
            .into_iter()
            .map(|unr| F::from_montgomery_reduce::<9>(unr))
            .collect();
        self.t_prime_poly = Some(MultiquadraticPolynomial::new(window_size, res));
    }

    #[tracing::instrument(
        skip_all,
        name = "OuterRemainingSumcheckProver::compute_evaluation_grid_from_poly_parallel"
    )]
    pub fn compute_evaluation_grid_from_polynomials_parallel(&mut self, num_vars: usize) {
        let eq_poly = &self.split_eq_poly;

        let n = self.az.as_ref().expect("az should be initialized").len();
        let az = self.az.as_ref().expect("az should be initialized");
        let bz = self.bz.as_ref().expect("bz should be initialized");
        debug_assert_eq!(n, bz.len());

        let three_pow_dim = 3_usize.pow(num_vars as u32);
        let grid_size = 1 << num_vars;
        let (E_out, E_in) = eq_poly.E_out_in_for_window(num_vars);
        let ans: Vec<F> = if E_in.len() == 1 {
            // Parallel version with reduction
            (0..E_out.len())
                .into_par_iter()
                .map(|i| {
                    let mut local_ans = vec![F::zero(); three_pow_dim];
                    let mut az_grid = vec![F::zero(); grid_size];
                    let mut bz_grid = vec![F::zero(); grid_size];
                    let mut buff_a = vec![F::zero(); three_pow_dim];
                    let mut buff_b = vec![F::zero(); three_pow_dim];
                    let mut tmp = vec![F::zero(); three_pow_dim];

                    // Fill grids
                    for j in 0..grid_size {
                        let index = grid_size * i + j;
                        az_grid[j] = az[index];
                        bz_grid[j] = bz[index];
                    }

                    MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                        &az_grid,
                        &mut buff_a,
                        &mut tmp,
                        num_vars,
                    );
                    MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                        &bz_grid,
                        &mut buff_b,
                        &mut tmp,
                        num_vars,
                    );

                    for idx in 0..three_pow_dim {
                        local_ans[idx] = buff_a[idx] * buff_b[idx] * E_out[i];
                    }

                    local_ans
                })
                .reduce(
                    || vec![F::zero(); three_pow_dim],
                    |mut acc, local_ans| {
                        for idx in 0..three_pow_dim {
                            acc[idx] += local_ans[idx];
                        }
                        acc
                    },
                )
        } else {
            let num_xin_bits = E_in.len().log_2();
            (0..E_out.len())
                .into_par_iter()
                .map(|x_out| {
                    let mut local_ans = vec![F::zero(); three_pow_dim];
                    let mut az_grid = vec![F::zero(); grid_size];
                    let mut bz_grid = vec![F::zero(); grid_size];
                    let mut buff_a = vec![F::zero(); three_pow_dim];
                    let mut buff_b = vec![F::zero(); three_pow_dim];
                    let mut tmp = vec![F::zero(); three_pow_dim];

                    for x_in in 0..E_in.len() {
                        let i = (x_out << num_xin_bits) | x_in;

                        //az_grid.fill(F::zero());
                        //bz_grid.fill(F::zero());
                        for j in 0..grid_size {
                            az_grid[j] = az[grid_size * i + j];
                            bz_grid[j] = bz[grid_size * i + j];
                        }

                        MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                            &az_grid,
                            &mut buff_a,
                            &mut tmp,
                            num_vars,
                        );
                        MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                            &bz_grid,
                            &mut buff_b,
                            &mut tmp,
                            num_vars,
                        );

                        for idx in 0..three_pow_dim {
                            local_ans[idx] += buff_a[idx] * buff_b[idx] * E_in[x_in];
                        }
                    }
                    for idx in 0..three_pow_dim {
                        local_ans[idx] *= E_out[x_out];
                    }

                    local_ans
                })
                .reduce(
                    || vec![F::zero(); three_pow_dim],
                    |mut acc, local_ans| {
                        for idx in 0..three_pow_dim {
                            acc[idx] += local_ans[idx];
                        }
                        acc
                    },
                )
        };
        self.t_prime_poly = Some(MultiquadraticPolynomial::new(num_vars, ans));
    }

    // Good to keep as a reference
    #[tracing::instrument(
        skip_all,
        name = "OuterRemainingSumcheckProver::compute_evaluation_grid_from_poly_serial"
    )]
    pub fn _compute_evaluation_grid_from_polynomials_serial(&mut self, num_vars: usize) {
        let eq_poly = &self.split_eq_poly;

        let n = self.az.as_ref().expect("az should be initialized").len();
        let az = self.az.as_ref().expect("az should be initialized");
        let bz = self.bz.as_ref().expect("bz should be initialized");
        debug_assert_eq!(n, bz.len());

        let three_pow_dim = 3_usize.pow(num_vars as u32);
        let grid_size = 1 << num_vars;
        let mut az_grid = vec![F::zero(); grid_size];
        let mut bz_grid = vec![F::zero(); grid_size];
        let mut buff_a: Vec<F> = vec![F::zero(); three_pow_dim];
        let mut buff_b = vec![F::zero(); three_pow_dim];
        let mut tmp = vec![F::zero(); three_pow_dim];
        let mut ans = vec![F::zero(); three_pow_dim];

        let (E_out, E_in) = eq_poly.E_out_in_for_window(num_vars);
        if E_in.len() == 1 {
            // this is a simple case of a linear loop
            for i in 0..E_out.len() {
                az_grid.fill(F::zero());
                bz_grid.fill(F::zero());
                for j in 0..grid_size {
                    let index = (grid_size) * i + j;
                    az_grid[j] = az[index];
                    bz_grid[j] = bz[index];
                }
                MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                    &az_grid,
                    &mut buff_a,
                    &mut tmp,
                    num_vars,
                );
                MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                    &bz_grid,
                    &mut buff_b,
                    &mut tmp,
                    num_vars,
                );
                for idx in 0..three_pow_dim {
                    ans[idx] += buff_a[idx] * buff_b[idx] * E_out[i];
                }
            }
        } else {
            let num_xin_bits = E_in.len().log_2();
            for x_out in 0..E_out.len() {
                for x_in in 0..E_in.len() {
                    let i = (x_out << num_xin_bits) | x_in;
                    az_grid.fill(F::zero());
                    bz_grid.fill(F::zero());
                    for j in 0..grid_size {
                        az_grid[j] = az[grid_size * i + j];
                        bz_grid[j] = bz[grid_size * i + j];
                    }

                    MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                        &az_grid,
                        &mut buff_a,
                        &mut tmp,
                        num_vars,
                    );
                    MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                        &bz_grid,
                        &mut buff_b,
                        &mut tmp,
                        num_vars,
                    );
                    for idx in 0..three_pow_dim {
                        ans[idx] += buff_a[idx] * buff_b[idx] * E_out[x_out] * E_in[x_in];
                    }
                }
            }
        }

        self.t_prime_poly = Some(MultiquadraticPolynomial::new(num_vars, ans));
    }

    #[tracing::instrument(
        skip_all,
        name = "OuterRemainingSumcheckProver::materialise_poly_from_trace_parallel_dim_one"
    )]
    fn materialise_polynomials_from_trace_parallel_dim_one(&mut self) {
        let num_x_out_vals = self.split_eq_poly.E_out_current_len();
        let num_x_in_vals = self.split_eq_poly.E_in_current_len();
        let r_grid = &self.r_grid;
        let num_r_vals = r_grid.len();

        // Output arrays are sized by (x_out, x_in) pairs
        let output_size = num_x_out_vals * num_x_in_vals;
        let mut az_bound: Vec<F> = unsafe_allocate_zero_vec(2 * output_size);
        let mut bz_bound: Vec<F> = unsafe_allocate_zero_vec(2 * output_size);

        let num_r_bits = num_r_vals.log_2();
        let num_x_in_bits = num_x_in_vals.log_2();

        // Precompute scaled Lagrange weights: scaled_w[r_idx][t] = lagrange_evals_r0[t] * r_grid[r_idx]
        // This avoids multiplying by r_eval inside the hot loop.
        let lagrange_evals_r = &self.lagrange_evals_r0;
        let scaled_w: Vec<[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE]> = (0..num_r_vals)
            .into_par_iter()
            .map(|r_idx| {
                let weight = r_grid[r_idx];
                let mut row = [F::zero(); OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE];
                for t in 0..OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE {
                    row[t] = lagrange_evals_r[t] * weight;
                }
                row
            })
            .collect();

        // Dynamic chunking for parallelization
        let num_threads = rayon::current_num_threads();
        let target_chunks = num_threads * 4;
        let min_chunk_pairs = 16;
        let pairs_per_chunk = output_size.div_ceil(target_chunks).max(min_chunk_pairs);
        let chunk_size = pairs_per_chunk * 2;

        // Parallel computation with reduction
        let (t0_acc, t_inf_acc) = az_bound
            .par_chunks_mut(chunk_size)
            .zip(bz_bound.par_chunks_mut(chunk_size))
            .enumerate()
            .fold(
                || (F::zero(), F::zero()),
                |(mut t0_local, mut t_inf_local), (chunk_idx, (az_chunk, bz_chunk))| {
                    let start_pair = chunk_idx * pairs_per_chunk;
                    let end_pair = (start_pair + pairs_per_chunk).min(output_size);

                    for pair_idx in start_pair..end_pair {
                        let x_in_val = pair_idx % num_x_in_vals;
                        let x_out_val = pair_idx / num_x_in_vals;

                        // Unreduced accumulators for X=0 and X=1
                        // Az uses Acc5U for both groups; Bz uses Acc6S (first) and Acc7S (second)
                        let mut acc_az0: Acc5U<F> = Acc5U::zero();
                        let mut acc_bz0_first: Acc6S<F> = Acc6S::zero();
                        let mut acc_bz0_second: Acc7S<F> = Acc7S::zero();
                        let mut acc_az1: Acc5U<F> = Acc5U::zero();
                        let mut acc_bz1_first: Acc6S<F> = Acc6S::zero();
                        let mut acc_bz1_second: Acc7S<F> = Acc7S::zero();

                        // Single loop over r values, computing both X=0 and X=1
                        for r_idx in 0..num_r_vals {
                            let w_r = &scaled_w[r_idx];

                            // Build indices for both X=0 and X=1
                            let base_idx = (x_out_val << (num_x_in_bits + 1 + num_r_bits))
                                | (x_in_val << (1 + num_r_bits));

                            let full_idx_x0 = base_idx | r_idx;
                            let full_idx_x1 = base_idx | (1 << num_r_bits) | r_idx;

                            // Process X=0
                            let step_idx_x0 = full_idx_x0 >> 1;
                            let selector_x0 = (full_idx_x0 & 1) == 1;

                            let row_inputs_x0 = R1CSCycleInputs::from_trace::<F>(
                                &self.bytecode_preprocessing,
                                &self.trace,
                                step_idx_x0,
                            );
                            let eval_x0 = R1CSEval::<F>::from_cycle_inputs(&row_inputs_x0);

                            if !selector_x0 {
                                eval_x0.fmadd_first_group_at_r(
                                    w_r,
                                    &mut acc_az0,
                                    &mut acc_bz0_first,
                                );
                            } else {
                                eval_x0.fmadd_second_group_at_r(
                                    w_r,
                                    &mut acc_az0,
                                    &mut acc_bz0_second,
                                );
                            }

                            // Process X=1
                            let step_idx_x1 = full_idx_x1 >> 1;
                            let selector_x1 = (full_idx_x1 & 1) == 1;

                            let row_inputs_x1 = R1CSCycleInputs::from_trace::<F>(
                                &self.bytecode_preprocessing,
                                &self.trace,
                                step_idx_x1,
                            );
                            let eval_x1 = R1CSEval::<F>::from_cycle_inputs(&row_inputs_x1);

                            if !selector_x1 {
                                eval_x1.fmadd_first_group_at_r(
                                    w_r,
                                    &mut acc_az1,
                                    &mut acc_bz1_first,
                                );
                            } else {
                                eval_x1.fmadd_second_group_at_r(
                                    w_r,
                                    &mut acc_az1,
                                    &mut acc_bz1_second,
                                );
                            }
                        }

                        // Reduce accumulators to field elements
                        let az0_sum = acc_az0.barrett_reduce();
                        let az1_sum = acc_az1.barrett_reduce();
                        let bz0_sum =
                            acc_bz0_first.barrett_reduce() + acc_bz0_second.barrett_reduce();
                        let bz1_sum =
                            acc_bz1_first.barrett_reduce() + acc_bz1_second.barrett_reduce();

                        // Store in chunk-relative position
                        let buffer_offset = 2 * (pair_idx - start_pair);
                        az_chunk[buffer_offset] = az0_sum;
                        az_chunk[buffer_offset + 1] = az1_sum;
                        bz_chunk[buffer_offset] = bz0_sum;
                        bz_chunk[buffer_offset + 1] = bz1_sum;

                        // Local accumulation for t_0 and t_inf
                        let e_in = self.split_eq_poly.E_in_current()[x_in_val];
                        let e_out = self.split_eq_poly.E_out_current()[x_out_val];
                        let p0 = az0_sum * bz0_sum;
                        let slope = (az1_sum - az0_sum) * (bz1_sum - bz0_sum);

                        t0_local += e_out * e_in * p0;
                        t_inf_local += e_out * e_in * slope;
                    }

                    (t0_local, t_inf_local)
                },
            )
            .reduce(
                || (F::zero(), F::zero()),
                |(t0_a, t_inf_a), (t0_b, t_inf_b)| (t0_a + t0_b, t_inf_a + t_inf_b),
            );

        self.az = Some(DensePolynomial::new(az_bound));
        self.bz = Some(DensePolynomial::new(bz_bound));
        self.t_prime_poly = Some(MultiquadraticPolynomial::new(
            1,
            vec![t0_acc, t0_acc, t_inf_acc],
        ));
    }

    // If the first round of the sumcheck is the switchover point
    // then materialisng Az and Bz is significantly simpler.
    // We do not need to deal with challenges.
    #[tracing::instrument(
        skip_all,
        name = "OuterRemainingSumcheckProver::materialise_poly_from_trace_round_zero_dim_one"
    )]
    fn materialise_polynomials_from_trace_round_zero_dim_one(&mut self) {
        let num_x_out_vals = self.split_eq_poly.E_out_current_len();
        let num_x_in_vals = self.split_eq_poly.E_in_current_len();
        let iter_num_x_in_vars = num_x_in_vals.log_2();

        let groups_exact = num_x_out_vals
            .checked_mul(num_x_in_vals)
            .expect("overflow computing groups_exact");

        // Preallocate interleaved buffers once ([lo, hi] per entry)
        let mut az_bound: Vec<F> = unsafe_allocate_zero_vec(2 * groups_exact);
        let mut bz_bound: Vec<F> = unsafe_allocate_zero_vec(2 * groups_exact);

        // Parallel over x_out groups using exact-sized mutable chunks, with per-worker fold
        let (t0_acc_unr, t_inf_acc_unr) = az_bound
            .par_chunks_exact_mut(2 * num_x_in_vals)
            .zip(bz_bound.par_chunks_exact_mut(2 * num_x_in_vals))
            .enumerate()
            .fold(
                || (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero()),
                |(mut acc0, mut acci), (x_out_val, (az_chunk, bz_chunk))| {
                    let mut inner_sum0 = F::Unreduced::<9>::zero();
                    let mut inner_sum_inf = F::Unreduced::<9>::zero();
                    for x_in_val in 0..num_x_in_vals {
                        let current_step_idx = (x_out_val << iter_num_x_in_vars) | x_in_val;

                        // Possibly re-design this
                        let row_inputs = R1CSCycleInputs::from_trace::<F>(
                            &self.bytecode_preprocessing,
                            &self.trace,
                            current_step_idx,
                        );
                        let eval = R1CSEval::<F>::from_cycle_inputs(&row_inputs);
                        let az0 = eval.az_at_r_first_group(&self.lagrange_evals_r0);
                        let bz0 = eval.bz_at_r_first_group(&self.lagrange_evals_r0);
                        let az1 = eval.az_at_r_second_group(&self.lagrange_evals_r0);
                        let bz1 = eval.bz_at_r_second_group(&self.lagrange_evals_r0);
                        let p0 = az0 * bz0;
                        let slope = (az1 - az0) * (bz1 - bz0);
                        let e_in = self.split_eq_poly.E_in_current()[x_in_val];
                        inner_sum0 += e_in.mul_unreduced::<9>(p0);
                        inner_sum_inf += e_in.mul_unreduced::<9>(slope);
                        let off = 2 * x_in_val;
                        az_chunk[off] = az0;
                        az_chunk[off + 1] = az1;
                        bz_chunk[off] = bz0;
                        bz_chunk[off + 1] = bz1;
                    }
                    let e_out = self.split_eq_poly.E_out_current()[x_out_val];
                    let reduced0 = F::from_montgomery_reduce::<9>(inner_sum0);
                    let reduced_inf = F::from_montgomery_reduce::<9>(inner_sum_inf);
                    acc0 += e_out.mul_unreduced::<9>(reduced0);
                    acci += e_out.mul_unreduced::<9>(reduced_inf);
                    (acc0, acci)
                },
            )
            .reduce(
                || (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero()),
                |a, b| (a.0 + b.0, a.1 + b.1),
            );

        self.az = Some(DensePolynomial::new(az_bound));
        self.bz = Some(DensePolynomial::new(bz_bound));
        let t_0 = F::from_montgomery_reduce::<9>(t0_acc_unr);
        let t_inf = F::from_montgomery_reduce::<9>(t_inf_acc_unr);
        self.t_prime_poly = Some(MultiquadraticPolynomial::new(1, vec![t_0, t_0, t_inf]));
    }
    fn fused_materialise_polynomials_general_with_multiquadratic(&mut self, window_size: usize) {
        let (E_out, E_in) = self.split_eq_poly.E_out_in_for_window(window_size);
        let num_x_out_vals = E_out.len();
        let num_x_in_vals = E_in.len();
        let r_grid = &self.r_grid;
        let num_r_vals = r_grid.len();

        let three_pow_dim = 3_usize.pow(window_size as u32);
        let grid_size = 1 << window_size;
        let num_evals_az = E_out.len() * E_in.len() * grid_size;

        let mut az_bound: Vec<F> = unsafe_allocate_zero_vec(num_evals_az);
        let mut bz_bound: Vec<F> = unsafe_allocate_zero_vec(num_evals_az);

        let num_r_bits = num_r_vals.log_2();
        let num_x_in_bits = num_x_in_vals.log_2();

        // Precompute scaled Lagrange weights: scaled_w[r_idx][t] = lagrange_evals_r0[t] * r_grid[r_idx]
        let lagrange_evals_r = &self.lagrange_evals_r0;
        let scaled_w: Vec<[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE]> = (0..num_r_vals)
            .into_par_iter()
            .map(|r_idx| {
                let weight = r_grid[r_idx];
                let mut row = [F::zero(); OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE];
                for t in 0..OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE {
                    row[t] = lagrange_evals_r[t] * weight;
                }
                row
            })
            .collect();

        let output_size = num_x_out_vals * num_x_in_vals;

        // Dynamic chunking for parallelization
        let num_threads = rayon::current_num_threads();
        let target_chunks = num_threads * 4;
        let min_chunk_pairs = 16;
        let pairs_per_chunk = output_size.div_ceil(target_chunks).max(min_chunk_pairs);
        let chunk_size = pairs_per_chunk * grid_size;

        // Parallel computation with reduction
        let ans = az_bound
            .par_chunks_mut(chunk_size)
            .zip(bz_bound.par_chunks_mut(chunk_size))
            .enumerate()
            .fold(
                || vec![F::zero(); three_pow_dim],
                |mut local_ans, (chunk_idx, (az_chunk, bz_chunk))| {
                    let start_pair = chunk_idx * pairs_per_chunk;
                    let end_pair = (start_pair + pairs_per_chunk).min(output_size);

                    // Thread-local buffers for multiquadratic expansion
                    let mut buff_a = vec![F::zero(); three_pow_dim];
                    let mut buff_b = vec![F::zero(); three_pow_dim];
                    let mut tmp = vec![F::zero(); three_pow_dim];
                    let mut az_grid = vec![F::zero(); grid_size];
                    let mut bz_grid = vec![F::zero(); grid_size];

                    // Unreduced accumulators for each grid point
                    let mut acc_az: Vec<Acc5U<F>> = vec![Acc5U::zero(); grid_size];
                    let mut acc_bz_first: Vec<Acc6S<F>> = vec![Acc6S::zero(); grid_size];
                    let mut acc_bz_second: Vec<Acc7S<F>> = vec![Acc7S::zero(); grid_size];

                    for pair_idx in start_pair..end_pair {
                        let x_in_val = pair_idx % num_x_in_vals;
                        let x_out_val = pair_idx / num_x_in_vals;

                        // Reset accumulators for this (x_out, x_in) pair
                        for x_val in 0..grid_size {
                            acc_az[x_val] = Acc5U::zero();
                            acc_bz_first[x_val] = Acc6S::zero();
                            acc_bz_second[x_val] = Acc7S::zero();
                        }

                        // Loop over r values and accumulate for each grid point using fmadd
                        for r_idx in 0..num_r_vals {
                            let w_r = &scaled_w[r_idx];

                            let base_idx = (x_out_val
                                << (num_x_in_bits + window_size + num_r_bits))
                                | (x_in_val << (window_size + num_r_bits));

                            // Process all grid_size points
                            for x_val in 0..grid_size {
                                let full_idx = base_idx | (x_val << num_r_bits) | r_idx;

                                let step_idx = full_idx >> 1;
                                let selector = (full_idx & 1) == 1;

                                let row_inputs = R1CSCycleInputs::from_trace::<F>(
                                    &self.bytecode_preprocessing,
                                    &self.trace,
                                    step_idx,
                                );
                                let eval = R1CSEval::<F>::from_cycle_inputs(&row_inputs);

                                if !selector {
                                    eval.fmadd_first_group_at_r(
                                        w_r,
                                        &mut acc_az[x_val],
                                        &mut acc_bz_first[x_val],
                                    );
                                } else {
                                    eval.fmadd_second_group_at_r(
                                        w_r,
                                        &mut acc_az[x_val],
                                        &mut acc_bz_second[x_val],
                                    );
                                }
                            }
                        }

                        // Reduce accumulators and fill grids
                        for x_val in 0..grid_size {
                            az_grid[x_val] = acc_az[x_val].barrett_reduce();
                            bz_grid[x_val] = acc_bz_first[x_val].barrett_reduce()
                                + acc_bz_second[x_val].barrett_reduce();
                        }

                        // Store the accumulated grid in chunk-relative position
                        let buffer_offset = grid_size * (pair_idx - start_pair);
                        let end = buffer_offset + grid_size;
                        az_chunk[buffer_offset..end].copy_from_slice(&az_grid[..grid_size]);
                        bz_chunk[buffer_offset..end].copy_from_slice(&bz_grid[..grid_size]);

                        // Expand to multiquadratic
                        MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                            &az_grid,
                            &mut buff_a,
                            &mut tmp,
                            window_size,
                        );
                        MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                            &bz_grid,
                            &mut buff_b,
                            &mut tmp,
                            window_size,
                        );

                        // Accumulate into local ans with E_out * E_in weight
                        let e_in = E_in[x_in_val];
                        let e_out = E_out[x_out_val];
                        let e_product = e_out * e_in;

                        for idx in 0..three_pow_dim {
                            local_ans[idx] += buff_a[idx] * buff_b[idx] * e_product;
                        }
                    }

                    local_ans
                },
            )
            .reduce(
                || vec![F::zero(); three_pow_dim],
                |mut acc, local_ans| {
                    for idx in 0..three_pow_dim {
                        acc[idx] += local_ans[idx];
                    }
                    acc
                },
            );

        self.az = Some(DensePolynomial::new(az_bound));
        self.bz = Some(DensePolynomial::new(bz_bound));
        self.t_prime_poly = Some(MultiquadraticPolynomial::new(window_size, ans));
    }
    fn fused_materialise_polynomials_round_zero(&mut self, num_vars: usize) {
        // Note: this is the simplest materialise as there are no challenges to deal with
        let eq_poly = &self.split_eq_poly;

        let three_pow_dim = 3_usize.pow(num_vars as u32);
        let grid_size = 1 << num_vars;
        let (E_out, E_in) = eq_poly.E_out_in_for_window(num_vars);

        let num_evals_az = E_out.len() * E_in.len() * grid_size;
        let mut az: Vec<F> = unsafe_allocate_zero_vec(num_evals_az);
        let mut bz: Vec<F> = unsafe_allocate_zero_vec(num_evals_az);

        let ans: Vec<F> = if E_in.len() == 1 {
            // Parallel version with reduction (E_in has only one element)
            az.par_chunks_exact_mut(grid_size)
                .zip(bz.par_chunks_exact_mut(grid_size))
                .enumerate()
                .map(|(i, (az_chunk, bz_chunk))| {
                    let mut local_ans = vec![F::zero(); three_pow_dim];
                    let mut az_grid = vec![F::zero(); grid_size];
                    let mut bz_grid = vec![F::zero(); grid_size];
                    let mut buff_a = vec![F::zero(); three_pow_dim];
                    let mut buff_b = vec![F::zero(); three_pow_dim];
                    let mut tmp = vec![F::zero(); three_pow_dim];

                    for j in 0..grid_size {
                        let full_idx = grid_size * i + j;
                        // Extract time_step_idx and selector from full_idx
                        let time_step_idx = full_idx >> 1;
                        let selector = (full_idx & 1) == 1;

                        let row_inputs = R1CSCycleInputs::from_trace::<F>(
                            &self.bytecode_preprocessing,
                            &self.trace,
                            time_step_idx,
                        );
                        let eval = R1CSEval::<F>::from_cycle_inputs(&row_inputs);

                        let (az_at_full_idx, bz_at_full_idx) = if !selector {
                            (
                                eval.az_at_r_first_group(&self.lagrange_evals_r0),
                                eval.bz_at_r_first_group(&self.lagrange_evals_r0),
                            )
                        } else {
                            (
                                eval.az_at_r_second_group(&self.lagrange_evals_r0),
                                eval.bz_at_r_second_group(&self.lagrange_evals_r0),
                            )
                        };

                        az_chunk[j] = az_at_full_idx;
                        bz_chunk[j] = bz_at_full_idx;
                        az_grid[j] = az_at_full_idx;
                        bz_grid[j] = bz_at_full_idx;
                    }

                    MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                        &az_grid,
                        &mut buff_a,
                        &mut tmp,
                        num_vars,
                    );
                    MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                        &bz_grid,
                        &mut buff_b,
                        &mut tmp,
                        num_vars,
                    );

                    for idx in 0..three_pow_dim {
                        local_ans[idx] = buff_a[idx] * buff_b[idx] * E_out[i];
                    }

                    local_ans
                })
                .reduce(
                    || vec![F::zero(); three_pow_dim],
                    |mut acc, local_ans| {
                        for idx in 0..three_pow_dim {
                            acc[idx] += local_ans[idx];
                        }
                        acc
                    },
                )
        } else {
            // Handle case where E_in has multiple elements
            let num_xin_bits = E_in.len().log_2();
            az.par_chunks_exact_mut(grid_size * E_in.len())
                .zip(bz.par_chunks_exact_mut(grid_size * E_in.len()))
                .enumerate()
                .map(|(x_out, (az_outer_chunk, bz_outer_chunk))| {
                    let mut local_ans = vec![F::zero(); three_pow_dim];
                    let mut az_grid = vec![F::zero(); grid_size];
                    let mut bz_grid = vec![F::zero(); grid_size];
                    let mut buff_a = vec![F::zero(); three_pow_dim];
                    let mut buff_b = vec![F::zero(); three_pow_dim];
                    let mut tmp = vec![F::zero(); three_pow_dim];

                    for x_in in 0..E_in.len() {
                        let i = (x_out << num_xin_bits) | x_in;

                        // Fill grids for this (x_out, x_in) pair
                        for j in 0..grid_size {
                            let full_idx = grid_size * i + j;
                            let time_step_idx = full_idx >> 1;
                            let selector = (full_idx & 1) == 1;

                            let row_inputs = R1CSCycleInputs::from_trace::<F>(
                                &self.bytecode_preprocessing,
                                &self.trace,
                                time_step_idx,
                            );
                            let eval = R1CSEval::<F>::from_cycle_inputs(&row_inputs);

                            let (az_at_full_idx, bz_at_full_idx) = if !selector {
                                (
                                    eval.az_at_r_first_group(&self.lagrange_evals_r0),
                                    eval.bz_at_r_first_group(&self.lagrange_evals_r0),
                                )
                            } else {
                                (
                                    eval.az_at_r_second_group(&self.lagrange_evals_r0),
                                    eval.bz_at_r_second_group(&self.lagrange_evals_r0),
                                )
                            };

                            let offset_in_chunk = x_in * grid_size + j;
                            az_outer_chunk[offset_in_chunk] = az_at_full_idx;
                            bz_outer_chunk[offset_in_chunk] = bz_at_full_idx;
                            az_grid[j] = az_at_full_idx;
                            bz_grid[j] = bz_at_full_idx;
                        }

                        MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                            &az_grid,
                            &mut buff_a,
                            &mut tmp,
                            num_vars,
                        );
                        MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                            &bz_grid,
                            &mut buff_b,
                            &mut tmp,
                            num_vars,
                        );

                        let e_product = E_out[x_out] * E_in[x_in];
                        for idx in 0..three_pow_dim {
                            local_ans[idx] += buff_a[idx] * buff_b[idx] * e_product;
                        }
                    }

                    local_ans
                })
                .reduce(
                    || vec![F::zero(); three_pow_dim],
                    |mut acc, local_ans| {
                        for idx in 0..three_pow_dim {
                            acc[idx] += local_ans[idx];
                        }
                        acc
                    },
                )
        };
        self.t_prime_poly = Some(MultiquadraticPolynomial::new(num_vars, ans));
        self.az = Some(DensePolynomial::new(az));
        self.bz = Some(DensePolynomial::new(bz));
    }

    #[tracing::instrument(
        skip_all,
        name = "OuterRemainingSumcheckProver::materialise_poly_from_trace"
    )]
    fn materialise_polynomials_from_trace(&mut self, window_size: usize) {
        let split_eq_poly = &self.split_eq_poly;
        let is_not_first_round_of_sumcheck = split_eq_poly.num_challenges() > 0;
        match (is_not_first_round_of_sumcheck, window_size == 1) {
            (true, true) => self.materialise_polynomials_from_trace_parallel_dim_one(),
            (true, false) => {
                self.fused_materialise_polynomials_general_with_multiquadratic(window_size)
            }
            (false, true) => self.materialise_polynomials_from_trace_round_zero_dim_one(),
            (false, false) => self.fused_materialise_polynomials_round_zero(window_size),
        }
        //if split_eq_poly.num_challenges() > 0 {
        //    if window_size == 1 {
        //        self.materialise_polynomials_from_trace_parallel_dim_one();
        //    } else {
        //        self.fused_materialise_polynomials_general_with_multiquadratic(window_size);
        //    }
        //} else {
        //    if window_size == 1 {
        //        self.materialise_polynomials_from_trace_round_zero_dim_one();
        //    } else {
        //        self.fused_materialise_polynomials_round_zero(window_size);
        //    }
        //}
    }

    // Compute prover message directly for window size 1
    fn _remaining_quadratic_evals(&mut self) -> (F, F) {
        let eq_poly = &self.split_eq_poly;

        let n = self.az.as_ref().expect("az should be initialized").len();
        let az = self.az.as_ref().expect("az should be initialized");
        let bz = self.bz.as_ref().expect("bz should be initialized");

        debug_assert_eq!(n, bz.len());
        if eq_poly.E_in_current_len() == 1 {
            // groups are pairs (0,1)
            let groups = n / 2;
            let (t0_unr, tinf_unr) = (0..groups)
                .into_par_iter()
                .map(|g| {
                    let az0 = az[2 * g];
                    let az1 = az[2 * g + 1];
                    let bz0 = bz[2 * g];
                    let bz1 = bz[2 * g + 1];
                    let eq = eq_poly.E_out_current()[g];
                    let p0 = az0 * bz0;
                    let slope = (az1 - az0) * (bz1 - bz0);
                    let t0_unr = eq.mul_unreduced::<9>(p0);
                    let tinf_unr = eq.mul_unreduced::<9>(slope);
                    (t0_unr, tinf_unr)
                })
                .reduce(
                    || (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero()),
                    |a, b| (a.0 + b.0, a.1 + b.1),
                );
            (
                F::from_montgomery_reduce::<9>(t0_unr),
                F::from_montgomery_reduce::<9>(tinf_unr),
            )
        } else {
            let num_x1_bits = eq_poly.E_in_current_len().log_2();
            let x1_len = eq_poly.E_in_current_len();
            let x2_len = eq_poly.E_out_current_len();
            let (sum0_unr, suminf_unr) = (0..x2_len)
                .into_par_iter()
                .map(|x2| {
                    let mut inner0_unr = F::Unreduced::<9>::zero();
                    let mut inner_inf_unr = F::Unreduced::<9>::zero();
                    for x1 in 0..x1_len {
                        let g = (x2 << num_x1_bits) | x1;
                        let az0 = az[2 * g];
                        let az1 = az[2 * g + 1];
                        let bz0 = bz[2 * g];
                        let bz1 = bz[2 * g + 1];
                        let e_in = eq_poly.E_in_current()[x1];
                        let p0 = az0 * bz0;
                        let slope = (az1 - az0) * (bz1 - bz0);
                        inner0_unr += e_in.mul_unreduced::<9>(p0);
                        inner_inf_unr += e_in.mul_unreduced::<9>(slope);
                    }
                    let e_out = eq_poly.E_out_current()[x2];
                    let inner0_red = F::from_montgomery_reduce::<9>(inner0_unr);
                    let inner_inf_red = F::from_montgomery_reduce::<9>(inner_inf_unr);
                    let t0_unr = e_out.mul_unreduced::<9>(inner0_red);
                    let tinf_unr = e_out.mul_unreduced::<9>(inner_inf_red);
                    (t0_unr, tinf_unr)
                })
                .reduce(
                    || (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero()),
                    |a, b| (a.0 + b.0, a.1 + b.1),
                );
            (
                F::from_montgomery_reduce::<9>(sum0_unr),
                F::from_montgomery_reduce::<9>(suminf_unr),
            )
        }
    }

    pub fn compute_t_evals(&self, window_size: usize) -> (F, F) {
        let t_prime_poly = self
            .t_prime_poly
            .as_ref()
            .expect("t_prime_poly should be initialized");

        // Equality weights over the active window bits (all but the first).
        let e_active = self.split_eq_poly.E_active_for_window(window_size);
        let t_prime_0 = t_prime_poly.project_to_first_variable(&e_active, 0);
        let t_prime_inf = t_prime_poly.project_to_first_variable(&e_active, INFINITY);
        (t_prime_0, t_prime_inf)
    }

    #[tracing::instrument(
        skip_all,
        name = "OuterRemainingSumcheckProver::compute_evaluation_grid_from_polynomials_serial"
    )]

    pub fn final_sumcheck_evals(&self) -> [F; 2] {
        let az = self.az.as_ref().expect("az should be initialized");
        let bz = self.bz.as_ref().expect("bz should be initialized");

        let az0 = if !az.is_empty() { az[0] } else { F::zero() };
        let bz0 = if !bz.is_empty() { bz[0] } else { F::zero() };
        [az0, bz0]
    }
}

impl<F: JoltField, T: Transcript, S: StreamingSchedule + Allocative> SumcheckInstanceProver<F, T>
    for OuterRemainingSumcheckProver<'_, F, S>
{
    fn degree(&self) -> usize {
        OUTER_REMAINING_DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.params.input_claim
    }

    #[tracing::instrument(skip_all, name = "OuterRemainingSumcheckProver::compute_message")]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        // The schedule determines how many variables we process in this window.
        // In streaming mode: exponentially growing windows (1, 2, 4, 8, ...)
        // In linear mode: single-variable windows (1, 1, 1, ...)
        let num_unbound_vars = self.schedule.num_unbound_vars(round);

        if self.schedule.is_switch_over_point(round) {
            // TRANSITION: Streaming → Linear mode
            // At this point we've done all streaming rounds. Now we:
            // 1. Materialize az/bz polynomials from the trace (O(n) memory)
            // 2. Compute the first evaluation grid from those polynomials
            // After this, we never touch the trace again.
            self.materialise_polynomials_from_trace(num_unbound_vars);
        } else if self.schedule.is_window_start(round) {
            // WINDOW START: Need to (re)compute the evaluation grid for this window.
            // The grid holds partial sums over {0,1,∞}^d needed for the sumcheck.
            if self.schedule.before_switch_over_point(round) {
                // STREAMING MODE: Compute grid directly from trace.
                // No polynomials materialized; we re-scan the trace each window.
                // Memory: O(3^d) for grid, where d = window size.
                self.compute_evaluation_grid_from_trace(num_unbound_vars);
            } else {
                // LINEAR MODE: Compute grid from materialized az/bz polynomials.
                // Faster per-round since polynomials are in memory.
                self.compute_evaluation_grid_from_polynomials_parallel(num_unbound_vars);
            }
        }
        // Else: mid-window round, just reuse the existing grid (handled by bind)

        // Extract T'(0) and T'(∞) from the grid, then compute the degree-3 univariate
        // polynomial T(X) = eq(τ, x) · [Az(x) · Bz(x) - u · Cz(x)] using Gruen's formula.
        let (t_prime_0, t_prime_inf) = self.compute_t_evals(num_unbound_vars);
        self.split_eq_poly
            .gruen_poly_deg_3(t_prime_0, t_prime_inf, previous_claim)
    }

    #[tracing::instrument(skip_all, name = "OuterRemainingSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        self.split_eq_poly.bind(r_j);

        // Always bind the multiquadratic polynomial
        let t_prime_poly = self
            .t_prime_poly
            .as_mut()
            .expect("t_prime_poly should be initialized");
        t_prime_poly.bind(r_j, BindingOrder::LowToHigh);

        if self.schedule.before_switch_over_point(round) {
            // Streaming mode: update r_grid for weighting trace evaluations
            debug_assert!(
                self.az.is_none(),
                "az should not be materialized before switch-over"
            );
            self.r_grid.update(r_j);
        } else {
            // Linear mode: bind the materialized az/bz polynomials
            debug_assert!(
                self.az.is_some(),
                "az should be materialized at or after switch-over"
            );
            // NOTE: As we are binding low-to-high in streaming
            // I need to revisit the binding algorithm and optimise
            // cache lines again
            rayon::join(
                || {
                    self.az
                        .as_mut()
                        .expect("az should be initialised")
                        .bind_parallel(r_j, BindingOrder::LowToHigh)
                },
                || {
                    self.bz
                        .as_mut()
                        .expect("bz should be initialised")
                        .bind_parallel(r_j, BindingOrder::LowToHigh)
                },
            );
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle = OuterRemainingSumcheckParams::get_inputs_opening_point(sumcheck_challenges);

        // Compute claimed witness evals and append virtual openings for all R1CS inputs
        let claimed_witness_evals =
            R1CSEval::compute_claimed_inputs(&self.bytecode_preprocessing, &self.trace, &r_cycle);

        for (i, input) in ALL_R1CS_INPUTS.iter().enumerate() {
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::from(input),
                SumcheckId::SpartanOuter,
                r_cycle.clone(),
                claimed_witness_evals[i],
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct OuterRemainingSumcheckVerifier<F: JoltField> {
    params: OuterRemainingSumcheckParams<F>,
    key: UniformSpartanKey<F>,
}

impl<F: JoltField> OuterRemainingSumcheckVerifier<F> {
    pub fn new(num_cycles_bits: usize, uni: &UniSkipState<F>, key: UniformSpartanKey<F>) -> Self {
        let params = OuterRemainingSumcheckParams::new(num_cycles_bits, uni);
        Self { params, key }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for OuterRemainingSumcheckVerifier<F>
{
    fn degree(&self) -> usize {
        OUTER_REMAINING_DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<F>) -> F {
        self.params.input_claim
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r1cs_input_evals = ALL_R1CS_INPUTS.map(|input| {
            accumulator
                .get_virtual_polynomial_opening((&input).into(), SumcheckId::SpartanOuter)
                .1
        });

        // Randomness used to bind the rows of R1CS matrices A,B.
        let rx_constr = &[sumcheck_challenges[0], self.params.r0_uniskip];
        // Compute sum_y A(rx_constr, y)*z(y) * sum_y B(rx_constr, y)*z(y).
        let inner_sum_prod = self
            .key
            .evaluate_inner_sum_product_at_point(rx_constr, r1cs_input_evals);

        let tau = &self.params.tau;
        let tau_high = &tau[tau.len() - 1];
        let tau_high_bound_r0 = LagrangePolynomial::<F>::lagrange_kernel::<
            F::Challenge,
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(tau_high, &self.params.r0_uniskip);
        let tau_low = &tau[..tau.len() - 1];
        let r_tail_reversed: Vec<F::Challenge> =
            sumcheck_challenges.iter().rev().copied().collect();
        let tau_bound_r_tail_reversed = EqPolynomial::mle(tau_low, &r_tail_reversed);
        tau_high_bound_r0 * tau_bound_r_tail_reversed * inner_sum_prod
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle = OuterRemainingSumcheckParams::get_inputs_opening_point(sumcheck_challenges);
        for input in &ALL_R1CS_INPUTS {
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::from(input),
                SumcheckId::SpartanOuter,
                r_cycle.clone(),
            );
        }
    }
}

struct OuterRemainingSumcheckParams<F: JoltField> {
    /// Number of cycle bits for splitting opening points (consistent across prover/verifier)
    /// Total number of rounds is `1 + num_cycles_bits`
    num_cycles_bits: usize,
    /// The tau vector (length `2 + num_cycles_bits`, sampled at the beginning for Lagrange + eq poly)
    tau: Vec<F::Challenge>,
    /// The univariate-skip first round challenge
    r0_uniskip: F::Challenge,
    /// Claim after the univariate-skip first round, updated every round
    input_claim: F,
}

impl<F: JoltField> OuterRemainingSumcheckParams<F> {
    fn new(num_cycles_bits: usize, uni: &UniSkipState<F>) -> Self {
        Self {
            num_cycles_bits,
            tau: uni.tau.clone(),
            r0_uniskip: uni.r0,
            input_claim: uni.claim_after_first,
        }
    }

    fn num_rounds(&self) -> usize {
        1 + self.num_cycles_bits
    }

    fn get_inputs_opening_point(
        sumcheck_challenges: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let r_cycle = sumcheck_challenges[1..].to_vec();
        OpeningPoint::<LITTLE_ENDIAN, F>::new(r_cycle).match_endianness()
    }
}
