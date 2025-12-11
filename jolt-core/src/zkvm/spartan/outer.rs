use std::sync::Arc;

use allocative::Allocative;
use ark_std::Zero;
use rayon::prelude::*;
use tracer::instruction::Cycle;

use crate::field::{FMAdd, JoltField, MontgomeryReduce};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::lagrange_poly::LagrangePolynomial;
use crate::poly::multilinear_polynomial::BindingOrder;
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::subprotocols::univariate_skip::build_uniskip_first_round_poly;
use crate::transcripts::Transcript;
use crate::utils::accumulation::Acc8S;
use crate::utils::math::Math;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::r1cs::constraints::OUTER_FIRST_ROUND_POLY_DEGREE_BOUND;
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

#[derive(Allocative, Clone)]
pub struct OuterUniSkipParams<F: JoltField> {
    pub tau: Vec<F::Challenge>,
}

impl<F: JoltField> OuterUniSkipParams<F> {
    pub fn new<T: Transcript>(key: &UniformSpartanKey<F>, transcript: &mut T) -> Self {
        let num_rounds_x: usize = key.num_rows_bits();
        let tau = transcript.challenge_vector_optimized::<F>(num_rounds_x);
        Self { tau }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for OuterUniSkipParams<F> {
    fn degree(&self) -> usize {
        OUTER_FIRST_ROUND_POLY_DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        1
    }

    fn input_claim(&self, _: &dyn OpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        challenges.to_vec().into()
    }
}

/// Uni-skip instance for Spartan outer sumcheck, computing the first-round polynomial only.
#[derive(Allocative)]
pub struct OuterUniSkipProver<F: JoltField> {
    params: OuterUniSkipParams<F>,
    /// Evaluations of t1(Z) at the extended univariate-skip targets (outside base window)
    extended_evals: [F; OUTER_UNIVARIATE_SKIP_DEGREE],
    /// Verifier challenge for this univariate skip round
    r0: Option<F::Challenge>,
    /// Prover message for this univariate skip round
    uni_poly: Option<UniPoly<F>>,
}

impl<F: JoltField> OuterUniSkipProver<F> {
    #[tracing::instrument(skip_all, name = "OuterUniSkipInstanceProver::initialize")]
    pub fn initialize(
        params: OuterUniSkipParams<F>,
        trace: &[Cycle],
        bytecode_preprocessing: &BytecodePreprocessing,
    ) -> Self {
        let extended = Self::compute_univariate_skip_extended_evals(
            bytecode_preprocessing,
            trace,
            &params.tau,
        );

        let instance = Self {
            params,
            extended_evals: extended,
            r0: None,
            uni_poly: None,
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

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for OuterUniSkipProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "OuterUniSkipInstanceProver::compute_poly")]
    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UniPoly<F> {
        // Load extended univariate-skip evaluations from prover state
        let extended_evals = &self.extended_evals;

        let tau_high = self.params.tau[self.params.tau.len() - 1];

        // Compute the univariate-skip first round polynomial s1(Y) = L(τ_high, Y) · t1(Y)
        let uni_poly = build_uniskip_first_round_poly::<
            F,
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
            OUTER_UNIVARIATE_SKIP_DEGREE,
            OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE,
            OUTER_FIRST_ROUND_POLY_NUM_COEFFS,
        >(None, extended_evals, tau_high);

        self.uni_poly = Some(uni_poly.clone());
        uni_poly
    }

    fn ingest_challenge(&mut self, _: F::Challenge, _round: usize) {
        // Nothing to do
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        debug_assert_eq!(opening_point.len(), 1);
        let claim = self.uni_poly.as_ref().unwrap().evaluate(&opening_point[0]);

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::UnivariateSkip,
            SumcheckId::SpartanOuter,
            opening_point,
            claim,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct OuterUniSkipVerifier<F: JoltField> {
    pub params: OuterUniSkipParams<F>,
}

impl<F: JoltField> OuterUniSkipVerifier<F> {
    pub fn new<T: Transcript>(key: &UniformSpartanKey<F>, transcript: &mut T) -> Self {
        let params = OuterUniSkipParams::new(key, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for OuterUniSkipVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        _accumulator: &VerifierOpeningAccumulator<F>,
        _sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) -> F {
        unimplemented!("Unused for univariate skip")
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        debug_assert_eq!(opening_point.len(), 1);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::UnivariateSkip,
            SumcheckId::SpartanOuter,
            opening_point,
        );
    }
}

pub struct OuterRemainingSumcheckParams<F: JoltField> {
    /// Number of cycle bits for splitting opening points (consistent across prover/verifier)
    /// Total number of rounds is `1 + num_cycles_bits`
    pub num_cycles_bits: usize,
    /// Verifier challenge for univariate skip round
    pub r0: F::Challenge,
    /// The tau vector (length 1 + n_cycle_vars), available to prover and verifier
    pub tau: Vec<F::Challenge>,
}

impl<F: JoltField> OuterRemainingSumcheckParams<F> {
    pub fn new(
        trace_len: usize,
        uni_skip_params: OuterUniSkipParams<F>,
        opening_accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let (r_uni_skip, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnivariateSkip,
            SumcheckId::SpartanOuter,
        );
        debug_assert_eq!(r_uni_skip.len(), 1);
        let r0 = r_uni_skip[0];

        Self {
            num_cycles_bits: trace_len.log_2(),
            tau: uni_skip_params.tau,
            r0,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for OuterRemainingSumcheckParams<F> {
    fn num_rounds(&self) -> usize {
        1 + self.num_cycles_bits
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let r_cycle = challenges[1..].to_vec();
        OpeningPoint::<LITTLE_ENDIAN, F>::new(r_cycle).match_endianness()
    }

    fn degree(&self) -> usize {
        OUTER_REMAINING_DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, uni_skip_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnivariateSkip,
            SumcheckId::SpartanOuter,
        );
        uni_skip_claim
    }
}

/// SumcheckInstance for Spartan outer rounds after the univariate-skip first round.
/// Round 0 in this instance corresponds to the "streaming" round; subsequent rounds
/// use the remaining linear-time algorithm over cycle variables.
#[derive(Allocative)]
pub struct OuterRemainingSumcheckProver<F: JoltField> {
    #[allocative(skip)]
    bytecode_preprocessing: BytecodePreprocessing,
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    split_eq_poly: GruenSplitEqPolynomial<F>,
    az: DensePolynomial<F>,
    bz: DensePolynomial<F>,
    /// The first round evals (t0, t_inf) computed from a streaming pass over the trace
    first_round_evals: (F, F),
    #[allocative(skip)]
    params: OuterRemainingSumcheckParams<F>,
}

impl<F: JoltField> OuterRemainingSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "OuterRemainingSumcheckProver::initialize")]
    pub fn initialize(
        params: OuterRemainingSumcheckParams<F>,
        trace: Arc<Vec<Cycle>>,
        bytecode_preprocessing: &BytecodePreprocessing,
    ) -> Self {
        let bytecode_preprocessing = bytecode_preprocessing.clone();

        let lagrange_evals_r = LagrangePolynomial::<F>::evals::<
            F::Challenge,
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&params.r0);

        let tau_high = params.tau[params.tau.len() - 1];
        let tau_low = &params.tau[..params.tau.len() - 1];

        let lagrange_tau_r0 = LagrangePolynomial::<F>::lagrange_kernel::<
            F::Challenge,
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&params.r0, &tau_high);

        let split_eq_poly: GruenSplitEqPolynomial<F> =
            GruenSplitEqPolynomial::<F>::new_with_scaling(
                tau_low,
                BindingOrder::LowToHigh,
                Some(lagrange_tau_r0),
            );

        let (t0, t_inf, az_bound, bz_bound) = Self::compute_first_quadratic_evals_and_bound_polys(
            &bytecode_preprocessing,
            &trace,
            &lagrange_evals_r,
            &split_eq_poly,
        );

        Self {
            split_eq_poly,
            bytecode_preprocessing,
            trace,
            az: az_bound,
            bz: bz_bound,
            first_round_evals: (t0, t_inf),
            params,
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

    // No special binding path needed; az/bz hold interleaved [lo,hi] ready for binding

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
        let n = self.az.len();
        debug_assert_eq!(n, self.bz.len());
        let [t0, tinf] = self.split_eq_poly.par_fold_out_in_unreduced::<9, 2>(&|g| {
            let az0 = self.az[2 * g];
            let az1 = self.az[2 * g + 1];
            let bz0 = self.bz[2 * g];
            let bz1 = self.bz[2 * g + 1];
            let p0 = az0 * bz0;
            let slope = (az1 - az0) * (bz1 - bz0);
            [p0, slope]
        });
        (t0, tinf)
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for OuterRemainingSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "OuterRemainingSumcheckProver::compute_message")]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let (t0, t_inf) = if round == 0 {
            self.first_round_evals
        } else {
            self.remaining_quadratic_evals()
        };
        self.split_eq_poly
            .gruen_poly_deg_3(t0, t_inf, previous_claim)
    }

    #[tracing::instrument(skip_all, name = "OuterRemainingSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        rayon::join(
            || self.az.bind_parallel(r_j, BindingOrder::LowToHigh),
            || self.bz.bind_parallel(r_j, BindingOrder::LowToHigh),
        );

        // Bind eq_poly for next round
        self.split_eq_poly.bind(r_j);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle = self.params.normalize_opening_point(sumcheck_challenges);

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
    pub fn new(
        key: UniformSpartanKey<F>,
        trace_len: usize,
        uni_skip_params: OuterUniSkipParams<F>,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params =
            OuterRemainingSumcheckParams::new(trace_len, uni_skip_params, opening_accumulator);
        Self { params, key }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for OuterRemainingSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
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
        let rx_constr = &[sumcheck_challenges[0], self.params.r0];
        // Compute sum_y A(rx_constr, y)*z(y) * sum_y B(rx_constr, y)*z(y).
        let inner_sum_prod = self
            .key
            .evaluate_inner_sum_product_at_point(rx_constr, r1cs_input_evals);

        let tau = &self.params.tau;
        let tau_high = &tau[tau.len() - 1];
        let tau_high_bound_r0 = LagrangePolynomial::<F>::lagrange_kernel::<
            F::Challenge,
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(tau_high, &self.params.r0);
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
        let r_cycle = self.params.normalize_opening_point(sumcheck_challenges);
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
