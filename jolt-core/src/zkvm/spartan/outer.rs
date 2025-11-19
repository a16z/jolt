use std::sync::Arc;

use allocative::Allocative;
use ark_std::Zero;
use rayon::prelude::*;
use tracer::instruction::Cycle;

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

#[cfg(test)]
use crate::zkvm::r1cs::constraints::{R1CS_CONSTRAINTS_FIRST_GROUP, R1CS_CONSTRAINTS_SECOND_GROUP};
#[cfg(test)]
use crate::zkvm::r1cs::inputs::JoltR1CSInputs;

/// Degree bound of the sumcheck round polynomials for [`OuterRemainingSumcheckVerifier`].
const OUTER_REMAINING_DEGREE_BOUND: usize = 3;
const INFINITY: usize = 2; // 2 represents ∞ in base-3

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
pub struct OuterRemainingSumcheckProver<F: JoltField, S: StreamingSchedule> {
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
    lagrange_evals_r0: [F; 10],
    schedule: S,
    t_0: Option<F>,
    t_inf: Option<F>,
}

impl<F: JoltField, S: StreamingSchedule> OuterRemainingSumcheckProver<F, S> {
    #[tracing::instrument(skip_all, name = "OuterRemainingSumcheckProver::gen")]
    pub fn gen(
        trace: Arc<Vec<Cycle>>,
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

        // TODO: Double check this binding order
        let mut r_grid = ExpandingTable::new(1 << n_cycle_vars, BindingOrder::LowToHigh);
        //let mut r_grid =
        //    ExpandingTable::new_with_order(1 << n_cycle_vars, ExpansionOrder::MostSignificantBit);
        r_grid.reset(F::one());

        Self {
            split_eq_poly,
            bytecode_preprocessing,
            trace,
            az: None,
            bz: None,
            t_prime_poly: None,
            r_grid,
            params: outer_params,
            lagrange_evals_r0: lagrange_evals_r,
            schedule,
            t_0: None,
            t_inf: None,
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
    fn build_grids(
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

        // Final reductions once per j.
        for j in 0..jlen {
            let az_j = acc_az[j].barrett_reduce();
            let bz_first_j = acc_bz_first[j].barrett_reduce();
            let bz_second_j = acc_bz_second[j].barrett_reduce();
            grid_az[j] = az_j;
            grid_bz[j] = bz_first_j + bz_second_j;
        }
    }

    // returns the grid of evaluations on {0,1,inf}^window_size
    // touches each cycle of the trace exactly once and in order!
    fn get_grid_gen(&mut self, window_size: usize) {
        // Use the split-eq instance to derive the current window
        // factorisation of Eq over the unbound cycle bits. This keeps the
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
        let mut scaled_w = vec![[F::zero(); OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE]; klen];
        if klen > 1 {
            debug_assert_eq!(klen, r_grid.len());
            for k in 0..klen {
                let weight = r_grid[k];
                let row = &mut scaled_w[k];
                for t in 0..OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE {
                    row[t] = lagrange_evals_r[t] * weight;
                }
            }
        } else {
            debug_assert_eq!(klen, 1);
            scaled_w[0].copy_from_slice(lagrange_evals_r);
        }

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
                    self.build_grids(
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
                    for idx in 0..three_pow_dim {
                        let val = buff_a[idx] * buff_b[idx];
                        local_res_unr[idx] += e_in_val.mul_unreduced::<9>(val);
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
    fn compute_first_quadratic_evals_and_bound_polys(
        bytecode_preprocessing: &BytecodePreprocessing,
        trace: &[Cycle],
        lagrange_evals_r: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE],
        split_eq_poly: &GruenSplitEqPolynomial<F>,
    ) -> (F, F, DensePolynomial<F>, DensePolynomial<F>) {
        let num_x_out_vals = split_eq_poly.E_out_current_len();
        let num_x_in_vals = split_eq_poly.E_in_current_len();
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
                        let row_inputs = R1CSCycleInputs::from_trace::<F>(
                            bytecode_preprocessing,
                            trace,
                            current_step_idx,
                        );
                        let eval = R1CSEval::<F>::from_cycle_inputs(&row_inputs);
                        let az0 = eval.az_at_r_first_group(lagrange_evals_r);
                        let bz0 = eval.bz_at_r_first_group(lagrange_evals_r);
                        let az1 = eval.az_at_r_second_group(lagrange_evals_r);
                        let bz1 = eval.bz_at_r_second_group(lagrange_evals_r);
                        let p0 = az0 * bz0;
                        let slope = (az1 - az0) * (bz1 - bz0);
                        let e_in = split_eq_poly.E_in_current()[x_in_val];
                        inner_sum0 += e_in.mul_unreduced::<9>(p0);
                        inner_sum_inf += e_in.mul_unreduced::<9>(slope);
                        let off = 2 * x_in_val;
                        az_chunk[off] = az0;
                        az_chunk[off + 1] = az1;
                        bz_chunk[off] = bz0;
                        bz_chunk[off + 1] = bz1;
                    }
                    let e_out = split_eq_poly.E_out_current()[x_out_val];
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

        (
            F::from_montgomery_reduce::<9>(t0_acc_unr),
            F::from_montgomery_reduce::<9>(t_inf_acc_unr),
            DensePolynomial::new(az_bound),
            DensePolynomial::new(bz_bound),
        )
    }

    fn stream_to_linear_time_serial(&mut self) {
        let num_x_out_vals = (&self.split_eq_poly).E_out_current_len();
        let num_x_in_vals = (&self.split_eq_poly).E_in_current_len();
        let r_grid = &self.r_grid;
        let num_r_vals = r_grid.len();

        println!(
            "num_out_vals: {:?}, num_in_vals: {:?} r_grid len {:?}",
            num_x_out_vals, num_x_in_vals, num_r_vals
        );

        let groups_exact = num_x_out_vals * num_x_in_vals * num_r_vals;
        debug_assert_eq!(groups_exact, (&self.trace).len());

        // Output arrays are sized by (x_out, x_in) pairs
        let output_size = num_x_out_vals * num_x_in_vals;
        let mut az_bound: Vec<F> = unsafe_allocate_zero_vec(2 * output_size);
        let mut bz_bound: Vec<F> = unsafe_allocate_zero_vec(2 * output_size);

        let num_r_bits = num_r_vals.log_2();
        let num_x_in_bits = num_x_in_vals.log_2();

        let mut t0_acc = F::zero();
        let mut t_inf_acc = F::zero();

        // Serial iteration over all (x_out, x_in) pairs
        for x_out_val in 0..num_x_out_vals {
            for x_in_val in 0..num_x_in_vals {
                // Initialize accumulators for this (x_out, x_in) pair
                let mut az0_sum = F::zero();
                let mut az1_sum = F::zero();
                let mut bz0_sum = F::zero();
                let mut bz1_sum = F::zero();

                // Sum over all r values
                for r_idx in 0..num_r_vals {
                    let current_step_idx = (x_out_val << (num_x_in_bits + num_r_bits))
                        | (x_in_val << num_r_bits)
                        | r_idx;
                    println!("Current step idx: {:?}", current_step_idx);
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

                    let r_eval = r_grid[r_idx];

                    // Accumulate: A[x_out||x_in||r] * r_val[r] (NO eq polynomials)
                    az0_sum += az0 * r_eval;
                    az1_sum += az1 * r_eval;
                    bz0_sum += bz0 * r_eval;
                    bz1_sum += bz1 * r_eval;
                }

                // Store the summed values in Az and Bz arrays
                let pair_idx = x_out_val * num_x_in_vals + x_in_val;
                let buffer_offset = 2 * pair_idx;
                az_bound[buffer_offset] = az0_sum;
                az_bound[buffer_offset + 1] = az1_sum;
                bz_bound[buffer_offset] = bz0_sum;
                bz_bound[buffer_offset + 1] = bz1_sum;

                // For t_0 and t_inf, NOW apply eq polynomials
                let e_in = (&self.split_eq_poly).E_in_current()[x_in_val];
                let e_out = (&self.split_eq_poly).E_out_current()[x_out_val];

                let p0 = az0_sum * bz0_sum;
                let slope = (az1_sum - az0_sum) * (bz1_sum - bz0_sum);

                // Accumulate with eq polynomial weighting
                t0_acc += e_out * e_in * p0;
                t_inf_acc += e_out * e_in * slope;
            }
        }

        self.az = Some(DensePolynomial::new(az_bound));
        self.bz = Some(DensePolynomial::new(bz_bound));
        //self.t_0 = Some(t0_acc);
        //self.t_inf = Some(t_inf_acc);
    }
    #[tracing::instrument(skip_all, name = "OuterRemainingSumcheckProver::stream_to_linear_time")]
    fn stream_to_linear_time_round_zero(&mut self) {
        let num_x_out_vals = (&self.split_eq_poly).E_out_current_len();
        let num_x_in_vals = (&self.split_eq_poly).E_in_current_len();
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
                        let e_in = (&self.split_eq_poly).E_in_current()[x_in_val];
                        inner_sum0 += e_in.mul_unreduced::<9>(p0);
                        inner_sum_inf += e_in.mul_unreduced::<9>(slope);
                        let off = 2 * x_in_val;
                        az_chunk[off] = az0;
                        az_chunk[off + 1] = az1;
                        bz_chunk[off] = bz0;
                        bz_chunk[off + 1] = bz1;
                    }
                    let e_out = (&self.split_eq_poly).E_out_current()[x_out_val];
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
        self.t_0 = Some(F::from_montgomery_reduce::<9>(t0_acc_unr));
        self.t_inf = Some(F::from_montgomery_reduce::<9>(t_inf_acc_unr))
    }

    // TODO:(ari) This is 2.5x slower than it needs to be right now.
    // Currently this is binding Az and Bz -- but I can fuse this.
    #[tracing::instrument(skip_all, name = "OuterRemainingSumcheckProver::stream_to_linear_time")]
    fn stream_to_linear_time(&mut self) {
        let split_eq_poly = &self.split_eq_poly;
        // helper constants
        let jlen = 1 << (split_eq_poly.get_num_vars() - split_eq_poly.num_challenges());
        let klen = 1 << split_eq_poly.num_challenges();
        // Precompute scaled Lagrange weights for all k so the parallel
        // conversion reuses them instead of recomputing per (j, k).
        let lagrange_evals_r = &self.lagrange_evals_r0;
        let r_grid = &self.r_grid;

        let mut scaled_w = vec![[F::zero(); OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE]; klen];
        if klen > 1 {
            debug_assert_eq!(klen, r_grid.len());
            for k in 0..klen {
                let weight = r_grid[k];
                let row = &mut scaled_w[k];
                for t in 0..OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE {
                    row[t] = lagrange_evals_r[t] * weight;
                }
            }
            let mut ret_az = unsafe_allocate_zero_vec(jlen);
            let mut ret_bz = unsafe_allocate_zero_vec(jlen);
            // Parallelize over j for the linear-time conversion.
            self.build_grids(&mut ret_az, &mut ret_bz, jlen, klen, 0, true, &scaled_w);
            self.az = Some(DensePolynomial::new(ret_az));
            self.bz = Some(DensePolynomial::new(ret_bz));
        } else {
            //debug_assert_eq!(klen, 1);
            //scaled_w[0].copy_from_slice(lagrange_evals_r);
            self.stream_to_linear_time_round_zero();
        }
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
    fn remaining_quadratic_evals(&mut self) -> (F, F) {
        if self.t_0.is_some() {
            println!("Brilliant someone already did my work.");
            let t_0 = self.t_0.unwrap();
            let t_inf = self.t_inf.unwrap();
            self.t_0 = None;
            self.t_inf = None;
            return (t_0, t_inf);
        }
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

    pub fn final_sumcheck_evals(&self) -> [F; 2] {
        let az = self.az.as_ref().expect("az should be initialized");
        let bz = self.bz.as_ref().expect("bz should be initialized");

        let az0 = if !az.is_empty() { az[0] } else { F::zero() };
        let bz0 = if !bz.is_empty() { bz[0] } else { F::zero() };
        [az0, bz0]
    }
}

impl<F: JoltField, T: Transcript, S: StreamingSchedule> SumcheckInstanceProver<F, T>
    for OuterRemainingSumcheckProver<F, S>
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

    //#[tracing::instrument(skip_all, name = "OuterRemainingSumcheckProver::compute_message")]
    //fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
    //    let (t0, t_inf) = if round == 0 {
    //        self.first_round_evals
    //    } else {
    //        self.remaining_quadratic_evals()
    //    };
    //    self.split_eq_poly
    //        .gruen_poly_deg_3(t0, t_inf, previous_claim)
    //}
    #[tracing::instrument(skip_all, name = "OuterRemainingSumcheckProver::compute_message")]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let (t0, t_inf) = if self.schedule.is_streaming(round) {
            let num_unbound_vars = self.schedule.num_unbound_vars(round);

            if self.schedule.is_window_start(round) {
                // Build the multiquadratic t'(z) for this window using the
                // slice-based Eq factorisation provided by the simple
                // split-eq instance (head vs window bits).
                self.get_grid_gen(num_unbound_vars);
            }
            // Use the multiquadratic polynomial to compute the message
            let t_prime_poly = self
                .t_prime_poly
                .as_ref()
                .expect("t_prime_poly should be initialized");
            // Equality weights over the active window bits (all but the first).
            let e_active = self.split_eq_poly.E_active_for_window(num_unbound_vars);
            let t_prime_0 = t_prime_poly.project_to_first_variable(&e_active, 0);
            let t_prime_inf = t_prime_poly.project_to_first_variable(&e_active, INFINITY);

            (t_prime_0, t_prime_inf)
        } else {
            // LINEAR PHASE
            //println!("In Linear phase| Round: {:?}", round);
            if self.schedule.is_first_linear(round) {
                self.stream_to_linear_time();
            }
            // For now, just use quadratic evals
            let (t0, t_inf) = self.remaining_quadratic_evals();
            (t0, t_inf)
        };
        // Compute the Gruen cubic using the split-eq implementation.
        self.split_eq_poly
            .gruen_poly_deg_3(t0, t_inf, previous_claim)
        //vec![evals[0], evals[1], evals[2]]
    }

    #[tracing::instrument(skip_all, name = "OuterRemainingSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        self.split_eq_poly.bind(r_j);

        if self.schedule.is_streaming(round) {
            let t_prime_poly = self
                .t_prime_poly
                .as_mut()
                .expect("t_prime_poly should be initialized");
            t_prime_poly.bind(r_j, BindingOrder::LowToHigh);
            self.r_grid.update(r_j);
        } else {
            // TODO: Unless this is the last round I should also
            // manifest evals for next round : Fused bind + eval;
            // Bind the split-eq instance in lock-step with the outer sumcheck.
            // TODO: so we need a new bind_parallel algorithm
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
        let opening_point = self.params.get_opening_point(sumcheck_challenges);

        // Append Az, Bz claims and corresponding opening point
        let claims = self.final_sumcheck_evals();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SpartanAz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
            claims[0],
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SpartanBz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
            claims[1],
        );

        // Handle witness openings at r_cycle (use consistent split length)
        let (r_cycle, _rx_var) = opening_point.r.split_at(self.params.num_cycles_bits);

        // Compute claimed witness evals and append virtual openings for all R1CS inputs
        let claimed_witness_evals =
            R1CSEval::compute_claimed_inputs(&self.bytecode_preprocessing, &self.trace, r_cycle);

        #[cfg(test)]
        {
            // Recompute Az,Bz at the final opening point USING ONLY the claimed witness MLEs z(r_cycle),
            // then compare to the prover's final Az,Bz claims. This validates the consistency wiring
            // between the outer sumcheck and the witness openings.

            // Prover's final Az,Bz claims (after all bindings)
            let claims = self.final_sumcheck_evals();

            // Extract streaming-round challenge r_stream from the opening point tail (after r_cycle)
            let (_, rx_tail) = opening_point.r.split_at(self.params.num_cycles_bits);
            let r_stream = rx_tail[0];

            // Build z(r_cycle) vector extended with a trailing 1 for the constant column
            let const_col = JoltR1CSInputs::num_inputs();
            let mut z_cycle_ext = claimed_witness_evals.to_vec();
            z_cycle_ext.push(F::one());

            // Lagrange weights over the univariate-skip base domain at r0
            let w = LagrangePolynomial::<F>::evals::<F::Challenge, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE>(
                &self.params.r0_uniskip,
            );

            // Group 0 fused Az,Bz via dot product of LC with z(r_cycle)
            let mut az_g0 = F::zero();
            let mut bz_g0 = F::zero();
            for i in 0..R1CS_CONSTRAINTS_FIRST_GROUP.len() {
                let lc_a = &R1CS_CONSTRAINTS_FIRST_GROUP[i].cons.a;
                let lc_b = &R1CS_CONSTRAINTS_FIRST_GROUP[i].cons.b;
                az_g0 += w[i] * lc_a.dot_eq_ry::<F>(&z_cycle_ext, const_col);
                bz_g0 += w[i] * lc_b.dot_eq_ry::<F>(&z_cycle_ext, const_col);
            }

            // Group 1 fused Az,Bz (use same Lagrange weights order as construction)
            let mut az_g1 = F::zero();
            let mut bz_g1 = F::zero();
            let g2_len = core::cmp::min(
                R1CS_CONSTRAINTS_SECOND_GROUP.len(),
                OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
            );
            for i in 0..g2_len {
                let lc_a = &R1CS_CONSTRAINTS_SECOND_GROUP[i].cons.a;
                let lc_b = &R1CS_CONSTRAINTS_SECOND_GROUP[i].cons.b;
                az_g1 += w[i] * lc_a.dot_eq_ry::<F>(&z_cycle_ext, const_col);
                bz_g1 += w[i] * lc_b.dot_eq_ry::<F>(&z_cycle_ext, const_col);
            }

            // Bind by r_stream to match the outer streaming combination used for final Az,Bz
            let az_final = az_g0 + r_stream * (az_g1 - az_g0);
            let bz_final = bz_g0 + r_stream * (bz_g1 - bz_g0);

            assert_eq!(
                az_final, claims[0],
                "Az final eval mismatch vs claims from evaluating R1CS inputs at r_cycle: recomputed={} claimed={}",
                az_final, claims[0]
            );
            assert_eq!(
                bz_final, claims[1],
                "Bz final eval mismatch vs claims from evaluating R1CS inputs at r_cycle: recomputed={} claimed={}",
                bz_final, claims[1]
            );
        }

        for (i, input) in ALL_R1CS_INPUTS.iter().enumerate() {
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::from(input),
                SumcheckId::SpartanOuter,
                OpeningPoint::new(r_cycle.to_vec()),
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
}

impl<F: JoltField> OuterRemainingSumcheckVerifier<F> {
    pub fn new(num_cycles_bits: usize, uni: &UniSkipState<F>) -> Self {
        let params = OuterRemainingSumcheckParams::new(num_cycles_bits, uni);
        Self { params }
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
        let (_, claim_Az) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter);
        let (_, claim_Bz) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanBz, SumcheckId::SpartanOuter);

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
        tau_high_bound_r0 * tau_bound_r_tail_reversed * claim_Az * claim_Bz
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.get_opening_point(sumcheck_challenges);

        // Populate Az, Bz openings at the full outer opening point
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SpartanAz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SpartanBz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
        );

        // Append witness openings at r_cycle (no claims at verifier) for all R1CS inputs
        let (r_cycle, _rx_var) = opening_point.r.split_at(self.params.num_cycles_bits);
        ALL_R1CS_INPUTS.iter().for_each(|input| {
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::from(input),
                SumcheckId::SpartanOuter,
                OpeningPoint::new(r_cycle.to_vec()),
            );
        });
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

    fn get_opening_point(
        &self,
        sumcheck_challenges: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let r_tail = sumcheck_challenges;
        let r_full = [&[self.r0_uniskip], r_tail].concat();
        OpeningPoint::<LITTLE_ENDIAN, F>::new(r_full).match_endianness()
    }
}
