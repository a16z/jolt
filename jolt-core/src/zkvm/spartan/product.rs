use std::iter::zip;
use std::sync::Arc;

use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_std::Zero;

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
use crate::zkvm::instruction::{CircuitFlags, InstructionFlags};
use crate::zkvm::r1cs::constraints::{
    NUM_PRODUCT_VIRTUAL, PRODUCT_CONSTRAINTS, PRODUCT_VIRTUAL_FIRST_ROUND_POLY_DEGREE_BOUND,
    PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS, PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE,
    PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
    PRODUCT_VIRTUAL_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE,
};
use crate::zkvm::r1cs::evaluation::ProductVirtualEval;
use crate::zkvm::r1cs::inputs::{ProductCycleInputs, PRODUCT_UNIQUE_FACTOR_VIRTUALS};
use crate::zkvm::witness::VirtualPolynomial;
use rayon::prelude::*;
use tracer::instruction::Cycle;

// Product virtualization with univariate skip
//
// We define a "combined" left and right polynomial
// Left(x, y) = \sum_i L(y, i) * Left_i(x),
// Right(x, y) = \sum_i R(y, i) * Right_i(x),
// where Left_i(x) = one of the five left polynomials, Right_i(x) = one of the five right polynomials
// Indexing is over i \in {-2, -1, 0, 1, 2}, though this gets mapped to the 0th, 1st, ..., 4th polynomial
//
// We also need to define the combined claim:
// claim(y) = \sum_i L(y, i) * claim_i,
// where claim_i is the claim of the i-th product virtualization sumcheck
//
// The product virtualization sumcheck is then:
// \sum_y L(tau_high, y) * \sum_x eq(tau_low, x) * Left(x, y) * Right(x, y)
//   = claim(tau_high)
//
// Final claim is:
// L(tau_high, r0) * Eq(tau_low, r_tail^rev) * Left(r_tail, r0) * Right(r_tail, r0)
//
// After this, we also need to check the consistency of the Left and Right evaluations with the
// claimed evaluations of the factor polynomials. This is done in the ProductVirtualInner sumcheck.
//
// TODO (Quang): this is essentially Spartan with non-zero claims. We should unify this with Spartan outer/inner.
// Only complication is to generalize the splitting strategy
// (i.e. Spartan outer currently does uni skip for half of the constraints,
// whereas here we do it for all of them)

/// Degree of the sumcheck round polynomials for [`ProductVirtualRemainderVerifier`].
const PRODUCT_VIRTUAL_REMAINDER_DEGREE: usize = 3;

#[derive(Allocative, Clone)]
pub struct ProductVirtualUniSkipParams<F: JoltField> {
    /// τ = [τ_low || τ_high]
    /// - τ_low: the cycle-point r_cycle carried from Spartan outer (length = num_cycle_vars)
    /// - τ_high: the univariate-skip binding point sampled for the size-5 domain (length = 1)
    ///   Ordering matches outer: variables are MSB→LSB with τ_high last
    pub tau: Vec<F::Challenge>,
    /// Base evaluations (claims) for the five product terms at the base domain
    /// Order: [Product, WriteLookupOutputToRD, WritePCtoRD, ShouldBranch, ShouldJump]
    base_evals: [F; NUM_PRODUCT_VIRTUAL],
}

impl<F: JoltField> ProductVirtualUniSkipParams<F> {
    pub fn new<T: Transcript>(
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Self {
        // Reuse r_cycle from Stage 1 (outer) for τ_low, and sample τ_high
        let r_cycle = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Product, SumcheckId::SpartanOuter)
            .0
            .r;
        let tau_high = transcript.challenge_scalar_optimized::<F>();
        let mut tau = r_cycle;
        tau.push(tau_high);

        let mut base_evals: [F; NUM_PRODUCT_VIRTUAL] = [F::zero(); NUM_PRODUCT_VIRTUAL];
        for (i, cons) in PRODUCT_CONSTRAINTS.iter().enumerate() {
            let (_, eval) = opening_accumulator
                .get_virtual_polynomial_opening(cons.output, SumcheckId::SpartanOuter);
            base_evals[i] = eval;
        }
        Self { tau, base_evals }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ProductVirtualUniSkipParams<F> {
    fn input_claim(&self, _: &dyn OpeningAccumulator<F>) -> F {
        // claim = \sum_i L_i(tau_high) * base_evals[i]
        let tau_high = self.tau[self.tau.len() - 1];
        let w = LagrangePolynomial::<F>::evals::<
            F::Challenge,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&tau_high);
        let mut acc = F::zero();
        for i in 0..NUM_PRODUCT_VIRTUAL {
            acc += w[i] * self.base_evals[i];
        }
        acc
    }

    fn degree(&self) -> usize {
        PRODUCT_VIRTUAL_FIRST_ROUND_POLY_DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        1
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        challenges.to_vec().into()
    }
}

/// Uni-skip instance for product virtualization, computing the first-round polynomial only.
#[derive(Allocative)]
pub struct ProductVirtualUniSkipProver<F: JoltField> {
    /// Evaluations of t1(Z) at the extended univariate-skip targets (outside base window)
    extended_evals: [F; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE],
    params: ProductVirtualUniSkipParams<F>,
    /// Verifier challenge for this univariate skip round
    r0: Option<F::Challenge>,
    /// Prover message for this univariate skip round
    uni_poly: Option<UniPoly<F>>,
}

impl<F: JoltField> ProductVirtualUniSkipProver<F> {
    /// Initialize a new prover for the univariate skip round
    /// The 5 base evaluations are the claimed evaluations of the 5 product terms from Spartan outer
    #[tracing::instrument(skip_all, name = "ProductVirtualUniSkipInstanceProver::initialize")]
    pub fn initialize(params: ProductVirtualUniSkipParams<F>, trace: &[Cycle]) -> Self {
        // Compute extended univariate-skip evals using split-eq fold-in-out (includes R^2 scaling)
        let extended_evals = Self::compute_univariate_skip_extended_evals(trace, &params.tau);
        let instance = Self {
            extended_evals,
            params,
            r0: None,
            uni_poly: None,
        };

        #[cfg(feature = "allocative")]
        print_data_structure_heap_usage("ProductVirtualUniSkipInstance", &instance);
        instance
    }

    /// Compute the extended-domain evaluations t1(z) for univariate-skip (outside base window).
    ///
    /// - For each z target, compute
    ///   t1(z) = Σ_{x_out} E_out[x_out] · Σ_{x_in} E_in[x_in] · left_z(x) · right_z(x),
    ///   where x is the concatenation of (x_out || x_in) in MSB→LSB order.
    ///
    /// Lagrange fusion per target z on (current) extended window {−4,−3,3,4}:
    /// - Compute c[0..4] = LagrangeHelper::shift_coeffs_i32(shift(z)) using the same shifted-kernel
    ///   as outer.rs (indices correspond to the 5 base points).
    /// - Define fused values at this z by linearly combining the 5 product witnesses with c:
    ///   left_z(x)  = Σ_i c[i] · Left_i(x)
    ///   right_z(x) = Σ_i c[i] · Right_i^eff(x)
    ///   with Right_4^eff(x) = 1 − NextIsNoop(x) for the ShouldJump term only.
    ///
    /// Small-value lifting rules for integer accumulation before converting to the field:
    /// - Instruction: LeftInstructionInput is u64 → lift to i128; RightInstructionInput is S64 → i128.
    /// - WriteLookupOutputToRD: IsRdNotZero is bool/u8 → i32; flag is bool/u8 → i32.
    /// - WritePCtoRD: IsRdNotZero is bool/u8 → i32; Jump flag is bool/u8 → i32.
    /// - ShouldBranch: LookupOutput is u64 → i128; Branch flag is bool/u8 → i32.
    /// - ShouldJump: Jump flag (left) is bool/u8 → i32; Right^eff = (1 − NextIsNoop) is bool/u8 → i32.
    fn compute_univariate_skip_extended_evals(
        trace: &[Cycle],
        tau: &[F::Challenge],
    ) -> [F; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE] {
        // Build split-eq over full τ; new_with_scaling drops τ_high from the split and
        // carries a global R^2 scaling via current_scalar for balanced Montgomery reduction.
        let split_eq = GruenSplitEqPolynomial::<F>::new_with_scaling(
            tau,
            BindingOrder::LowToHigh,
            Some(F::MONTGOMERY_R_SQUARE),
        );
        let outer_scale = split_eq.get_current_scalar(); // = R^2

        // Fold-out-in across (x_out, x_in) using signed Montgomery accumulators, mirroring outer.rs
        split_eq
            .par_fold_out_in(
                || [Acc8S::<F>::zero(); PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE],
                |inner, g, _x_in, e_in| {
                    // Materialize product-cycle row with raw types for this group index
                    let row = ProductCycleInputs::from_trace::<F>(trace, g);

                    // For each extended target j, compute fused left·right integer product using shared evaluator
                    for j in 0..PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE {
                        let prod_s256 =
                            ProductVirtualEval::extended_fused_product_at_j::<F>(&row, j);

                        // Fold e_in into signed 8-limb accumulator for this j
                        inner[j].fmadd(&e_in, &prod_s256);
                    }
                },
                |_x_out, e_out, inner| {
                    let mut out =
                        [F::Unreduced::<9>::zero(); PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE];
                    for j in 0..PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE {
                        let reduced = inner[j].montgomery_reduce();
                        out[j] = e_out.mul_unreduced::<9>(reduced);
                    }
                    out
                },
                |mut a, b| {
                    for j in 0..PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE {
                        a[j] += b[j];
                    }
                    a
                },
            )
            .map(|x| F::from_montgomery_reduce::<9>(x) * outer_scale)
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ProductVirtualUniSkipProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(
        skip_all,
        name = "ProductVirtualUniSkipInstanceProver::compute_message"
    )]
    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UniPoly<F> {
        // Load base evals from shared instance and extended from prover state
        let base = self.params.base_evals;
        let tau_high = self.params.tau[self.params.tau.len() - 1];

        // Compute the univariate-skip first round polynomial s1(Y) = L(τ_high, Y) · t1(Y)
        let uni_poly = build_uniskip_first_round_poly::<
            F,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE,
            PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS,
        >(Some(&base), &self.extended_evals, tau_high);

        self.uni_poly = Some(uni_poly.clone());
        uni_poly
    }

    fn ingest_challenge(&mut self, _: <F as JoltField>::Challenge, _: usize) {
        // Nothing to do
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        debug_assert_eq!(opening_point.len(), 1);
        let claim = self.uni_poly.as_ref().unwrap().evaluate(&opening_point[0]);

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::UnivariateSkip,
            SumcheckId::ProductVirtualization,
            opening_point,
            claim,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct ProductVirtualUniSkipVerifier<F: JoltField> {
    pub params: ProductVirtualUniSkipParams<F>,
}

impl<F: JoltField> ProductVirtualUniSkipVerifier<F> {
    pub fn new<T: Transcript>(
        opening_accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Self {
        let params = ProductVirtualUniSkipParams::new(opening_accumulator, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for ProductVirtualUniSkipVerifier<F>
{
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
            SumcheckId::ProductVirtualization,
            opening_point,
        );
    }
}

pub struct ProductVirtualRemainderParams<F: JoltField> {
    /// Number of cycle variables to bind in this remainder (equals log2(T))
    n_cycle_vars: usize,
    /// Verifier challenge for univariate skip round
    r0: F::Challenge,
    /// The tau vector (length 1 + n_cycle_vars), available to prover and verifier
    tau: Vec<F::Challenge>,
}

impl<F: JoltField> ProductVirtualRemainderParams<F> {
    pub fn new(
        trace_len: usize,
        uni_skip_params: ProductVirtualUniSkipParams<F>,
        opening_accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let (r_uni_skip, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnivariateSkip,
            SumcheckId::ProductVirtualization,
        );
        debug_assert_eq!(r_uni_skip.len(), 1);
        let r0 = r_uni_skip[0];

        Self {
            n_cycle_vars: trace_len.log_2(),
            tau: uni_skip_params.tau,
            r0,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ProductVirtualRemainderParams<F> {
    fn num_rounds(&self) -> usize {
        self.n_cycle_vars
    }

    fn degree(&self) -> usize {
        PRODUCT_VIRTUAL_REMAINDER_DEGREE
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, uni_skip_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnivariateSkip,
            SumcheckId::ProductVirtualization,
        );
        uni_skip_claim
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

// TODO: Update docs after merging uni skip round with this sumcheck.
/// Remaining rounds for Product Virtualization after the univariate-skip first round.
/// Mirrors the structure of `OuterRemainingSumcheck` with product-virtualization-specific wiring.
///
/// Final claim (what the prover's last claim must equal, and what the verifier computes):
///
/// Let r₀ be the univariate-skip challenge, and r_tail the remaining cycle-variable challenges
/// bound by this instance (low-to-high from the prover's perspective; the verifier uses the
/// reversed vector `r_tail^rev` when evaluating Eq_τ over τ_low).
///
/// Define Lagrange weights over the size-5 domain at r₀:
///   w_i := L_i(r₀) for i ∈ {0..4} corresponding to
///          [Instruction, WriteLookupOutputToRD, WritePCtoRD, ShouldBranch, ShouldJump].
///
/// Define fused left/right evaluations at the cycle point r_tail:
///   left_eval  := Σ_i w_i · eval(Left_i,  r_tail)
///   right_eval := Σ_i w_i · eval(Right_i, r_tail), except for ShouldJump where
///                 Right_4^eff := 1 − NextIsNoop, i.e., use (1 − eval(NextIsNoop, r_tail)).
///
/// Let
///   E_high := L(τ_high, r₀)  (Lagrange kernel over the size-5 domain)
///   E_low  := Eq_τ_low(τ_low, r_tail^rev)  (multilinear Eq kernel on the cycle variables)
///
/// Then the expected final claim is
///   expected = E_high · E_low · left_eval · right_eval.
///
/// The verifier computes this in `expected_output_claim`. The prover’s final emitted claim
/// after all rounds must match it. Note that `final_sumcheck_evals()` returns the first entries
/// of the fully-bound fused left/right polynomials (used for openings); these are not the final
/// claim themselves but are used to perform the subsequent opening checks.
#[derive(Allocative)]
pub struct ProductVirtualRemainderProver<F: JoltField> {
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    split_eq_poly: GruenSplitEqPolynomial<F>,
    left: DensePolynomial<F>,
    right: DensePolynomial<F>,
    /// The first round evals (t0, t_inf) computed from a streaming pass over the trace
    first_round_evals: (F, F),
    #[allocative(skip)]
    params: ProductVirtualRemainderParams<F>,
}

impl<F: JoltField> ProductVirtualRemainderProver<F> {
    #[tracing::instrument(skip_all, name = "ProductVirtualRemainderProver::initialize")]
    pub fn initialize(params: ProductVirtualRemainderParams<F>, trace: Arc<Vec<Cycle>>) -> Self {
        let lagrange_evals_r = LagrangePolynomial::<F>::evals::<
            F::Challenge,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&params.r0);

        let tau_high = params.tau[params.tau.len() - 1];
        let tau_low = &params.tau[..params.tau.len() - 1];

        let lagrange_tau_r0 = LagrangePolynomial::<F>::lagrange_kernel::<
            F::Challenge,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&tau_high, &params.r0);

        let split_eq_poly: GruenSplitEqPolynomial<F> =
            GruenSplitEqPolynomial::<F>::new_with_scaling(
                tau_low,
                BindingOrder::LowToHigh,
                Some(lagrange_tau_r0),
            );

        let (t0, t_inf, left_bound, right_bound) =
            Self::compute_first_quadratic_evals_and_bound_polys(
                &trace,
                &lagrange_evals_r,
                &split_eq_poly,
            );

        Self {
            split_eq_poly,
            trace,
            left: left_bound,
            right: right_bound,
            first_round_evals: (t0, t_inf),
            params,
        }
    }

    /// Compute the quadratic evaluations for the streaming round (right after univariate skip).
    ///
    /// After binding the univariate-skip variable at r0, we must
    /// compute the cubic round polynomial endpoints over the cycle variables only:
    ///   t(0)  = Σ_{x_out} E_out · Σ_{x_in} E_in · Left_0(x) · Right_0(x)
    ///   t(∞)  = Σ_{x_out} E_out · Σ_{x_in} E_in · (Left_1−Left_0) · (Right_1−Right_0)
    /// We also build per-(x_out,x_in) interleaved coefficients [lo, hi] in order to bind them by r_0
    /// once, after which remaining rounds bind linearly over the cycle variables.
    ///
    /// Product virtualization specifics:
    /// - Left/Right are fused linear combinations of five per-type witnesses with Lagrange
    ///   weights w_i = L_i(r0) over the size-5 domain.
    /// - For ShouldJump, the effective right factor is (1 − NextIsNoop).
    /// - We follow outer's delayed-reduction pattern across x_in to reduce modular reductions.
    #[inline]
    fn compute_first_quadratic_evals_and_bound_polys(
        trace: &[Cycle],
        weights_at_r0: &[F; NUM_PRODUCT_VIRTUAL],
        split_eq_poly: &GruenSplitEqPolynomial<F>,
    ) -> (F, F, DensePolynomial<F>, DensePolynomial<F>) {
        let num_x_out_vals = split_eq_poly.E_out_current_len();
        let num_x_in_vals = split_eq_poly.E_in_current_len();
        let iter_num_x_in_vars = num_x_in_vals.log_2();

        let groups_exact = num_x_out_vals
            .checked_mul(num_x_in_vals)
            .expect("overflow computing groups_exact");

        // Preallocate interleaved buffers once ([lo, hi] per entry)
        let mut left_bound: Vec<F> = unsafe_allocate_zero_vec(2 * groups_exact);
        let mut right_bound: Vec<F> = unsafe_allocate_zero_vec(2 * groups_exact);

        // Parallel over x_out groups using exact-sized mutable chunks, with per-worker fold
        let (t0_acc_unr, t_inf_acc_unr) = left_bound
            .par_chunks_exact_mut(2 * num_x_in_vals)
            .zip(right_bound.par_chunks_exact_mut(2 * num_x_in_vals))
            .enumerate()
            .fold(
                || (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero()),
                |(mut acc0, mut acci), (x_out_val, (left_chunk, right_chunk))| {
                    let mut inner_sum0 = F::Unreduced::<9>::zero();
                    let mut inner_sum_inf = F::Unreduced::<9>::zero();
                    for x_in_val in 0..num_x_in_vals {
                        let base_idx = (x_out_val << iter_num_x_in_vars) | x_in_val;
                        let idx_lo = base_idx << 1;
                        let idx_hi = idx_lo + 1;

                        // Materialize rows for lo and hi once
                        let row_lo = ProductCycleInputs::from_trace::<F>(trace, idx_lo);
                        let row_hi = ProductCycleInputs::from_trace::<F>(trace, idx_hi);

                        let (left0, right0) = ProductVirtualEval::fused_left_right_at_r::<F>(
                            &row_lo,
                            &weights_at_r0[..],
                        );
                        let (left1, right1) = ProductVirtualEval::fused_left_right_at_r::<F>(
                            &row_hi,
                            &weights_at_r0[..],
                        );

                        let p0 = left0 * right0;
                        let slope = (left1 - left0) * (right1 - right0);
                        let e_in = split_eq_poly.E_in_current()[x_in_val];
                        inner_sum0 += e_in.mul_unreduced::<9>(p0);
                        inner_sum_inf += e_in.mul_unreduced::<9>(slope);
                        let off = 2 * x_in_val;
                        left_chunk[off] = left0;
                        left_chunk[off + 1] = left1;
                        right_chunk[off] = right0;
                        right_chunk[off + 1] = right1;
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
            DensePolynomial::new(left_bound),
            DensePolynomial::new(right_bound),
        )
    }

    /// Compute the quadratic endpoints for remaining rounds.
    fn remaining_quadratic_evals(&self) -> (F, F) {
        let n = self.left.len();
        debug_assert_eq!(n, self.right.len());
        let [t0, tinf] = self.split_eq_poly.par_fold_out_in_unreduced::<9, 2>(&|g| {
            let l0 = self.left[2 * g];
            let l1 = self.left[2 * g + 1];
            let r0 = self.right[2 * g];
            let r1 = self.right[2 * g + 1];
            let p0 = l0 * r0;
            let slope = (l1 - l0) * (r1 - r0);
            [p0, slope]
        });
        (t0, tinf)
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for ProductVirtualRemainderProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "ProductVirtualRemainderProver::compute_message")]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let (t0, t_inf) = if round == 0 {
            self.first_round_evals
        } else {
            self.remaining_quadratic_evals()
        };
        self.split_eq_poly
            .gruen_poly_deg_3(t0, t_inf, previous_claim)
    }

    #[tracing::instrument(skip_all, name = "ProductVirtualRemainderProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        rayon::join(
            || self.left.bind_parallel(r_j, BindingOrder::LowToHigh),
            || self.right.bind_parallel(r_j, BindingOrder::LowToHigh),
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
        let claims = ProductVirtualEval::compute_claimed_factors::<F>(&self.trace, &r_cycle);
        for (poly, claim) in zip(PRODUCT_UNIQUE_FACTOR_VIRTUALS, claims) {
            accumulator.append_virtual(
                transcript,
                poly,
                SumcheckId::ProductVirtualization,
                r_cycle.clone(),
                claim,
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct ProductVirtualRemainderVerifier<F: JoltField> {
    params: ProductVirtualRemainderParams<F>,
}

impl<F: JoltField> ProductVirtualRemainderVerifier<F> {
    pub fn new(
        trace_len: usize,
        uni_skip_params: ProductVirtualUniSkipParams<F>,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params =
            ProductVirtualRemainderParams::new(trace_len, uni_skip_params, opening_accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for ProductVirtualRemainderVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        // Lagrange weights at r0
        let w = LagrangePolynomial::<F>::evals::<
            F::Challenge,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&self.params.r0);

        // Fetch factor claims
        let l_inst = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LeftInstructionInput,
                SumcheckId::ProductVirtualization,
            )
            .1;
        let r_inst = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RightInstructionInput,
                SumcheckId::ProductVirtualization,
            )
            .1;
        let is_rd_not_zero = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::InstructionFlags(InstructionFlags::IsRdNotZero),
                SumcheckId::ProductVirtualization,
            )
            .1;
        let wl_flag = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::OpFlags(CircuitFlags::WriteLookupOutputToRD),
                SumcheckId::ProductVirtualization,
            )
            .1;
        let j_flag = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::OpFlags(CircuitFlags::Jump),
                SumcheckId::ProductVirtualization,
            )
            .1;
        let lookup_out = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::ProductVirtualization,
            )
            .1;
        let branch_flag = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::InstructionFlags(InstructionFlags::Branch),
                SumcheckId::ProductVirtualization,
            )
            .1;
        let next_is_noop = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NextIsNoop,
                SumcheckId::ProductVirtualization,
            )
            .1;

        let fused_left = w[0] * l_inst
            + w[1] * is_rd_not_zero
            + w[2] * is_rd_not_zero
            + w[3] * lookup_out
            + w[4] * j_flag;
        let fused_right = w[0] * r_inst
            + w[1] * wl_flag
            + w[2] * j_flag
            + w[3] * branch_flag
            + w[4] * (F::one() - next_is_noop);

        // Multiply by L(τ_high, r0) and Eq(τ_low, r_tail^rev)
        let tau_high = &self.params.tau[self.params.tau.len() - 1];
        let tau_high_bound_r0 = LagrangePolynomial::<F>::lagrange_kernel::<
            F::Challenge,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(tau_high, &self.params.r0);
        let tau_low = &self.params.tau[..self.params.tau.len() - 1];
        let r_tail_reversed: Vec<F::Challenge> =
            sumcheck_challenges.iter().rev().copied().collect();
        let tau_bound_r_tail_reversed = EqPolynomial::mle(tau_low, &r_tail_reversed);

        tau_high_bound_r0 * tau_bound_r_tail_reversed * fused_left * fused_right
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        for vp in PRODUCT_UNIQUE_FACTOR_VIRTUALS.iter() {
            accumulator.append_virtual(
                transcript,
                *vp,
                SumcheckId::ProductVirtualization,
                opening_point.clone(),
            );
        }
    }
}
