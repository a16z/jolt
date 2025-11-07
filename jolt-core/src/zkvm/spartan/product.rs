use allocative::Allocative;
use ark_std::Zero;

use crate::field::{FMAdd, JoltField, MontgomeryReduce};
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::lagrange_poly::{LagrangeHelper, LagrangePolynomial};
use crate::poly::multilinear_polynomial::BindingOrder;
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_prover::{
    SumcheckInstanceProver, UniSkipFirstRoundInstanceProver,
};
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::subprotocols::univariate_skip::{
    build_uniskip_first_round_poly, uniskip_targets, UniSkipState,
};
use crate::transcripts::Transcript;
use crate::utils::accumulation::Acc8S;
use crate::utils::math::Math;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::instruction::{CircuitFlags, InstructionFlags};
use crate::zkvm::r1cs::evaluation::ProductVirtualEval;
use crate::zkvm::r1cs::inputs::{ProductCycleInputs, PRODUCT_UNIQUE_FACTOR_VIRTUALS};
use crate::zkvm::witness::VirtualPolynomial;
use ark_ff::biginteger::S128;
use rayon::prelude::*;
use std::sync::Arc;
use tracer::instruction::Cycle;

/// Product virtualization with univariate skip
///
/// We define a "combined" left and right polynomial
/// Left(x, y) = \sum_i L(y, i) * Left_i(x),
/// Right(x, y) = \sum_i R(y, i) * Right_i(x),
/// where Left_i(x) = one of the five left polynomials, Right_i(x) = one of the five right polynomials
/// Indexing is over i \in {-2, -1, 0, 1, 2}, though this gets mapped to the 0th, 1st, ..., 4th polynomial
///
/// We also need to define the combined claim:
/// claim(y) = \sum_i L(y, i) * claim_i,
/// where claim_i is the claim of the i-th product virtualization sumcheck
///
/// The product virtualization sumcheck is then:
/// \sum_y L(tau_high, y) * \sum_x eq(tau_low, x) * Left(x, y) * Right(x, y)
///   = claim(tau_high)
///
/// Final claim is:
/// L(tau_high, r0) * Eq(tau_low, r_tail^rev) * Left(r_tail, r0) * Right(r_tail, r0)
///
/// After this, we also need to check the consistency of the Left and Right evaluations with the
/// claimed evaluations of the factor polynomials. This is done in the ProductVirtualInner sumcheck.
///
/// TODO (Quang): this is essentially Spartan with non-zero claims. We should unify this with Spartan outer/inner.
/// Only complication is to generalize the splitting strategy
/// (i.e. Spartan outer currently does uni skip for half of the constraints,
/// whereas here we do it for all of them)
/// Fixed list of product virtual polynomials, in canonical order
pub const PRODUCT_VIRTUAL_TERMS: [VirtualPolynomial; NUM_PRODUCT_VIRTUAL] = [
    VirtualPolynomial::Product,               // Instruction
    VirtualPolynomial::WriteLookupOutputToRD, // WriteLookupOutputToRD
    VirtualPolynomial::WritePCtoRD,           // WritePCtoRD
    VirtualPolynomial::ShouldBranch,          // ShouldBranch
    VirtualPolynomial::ShouldJump,            // ShouldJump
];

pub const NUM_PRODUCT_VIRTUAL: usize = 5;
pub const PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE: usize = NUM_PRODUCT_VIRTUAL;
pub const PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE: usize = NUM_PRODUCT_VIRTUAL - 1;
pub const PRODUCT_VIRTUAL_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE: usize =
    2 * PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE + 1;
pub const PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS: usize =
    3 * PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE + 1;

/// Degree of the sumcheck round polynomials for [`ProductVirtualRemainderVerifier`].
const PRODUCT_VIRTUAL_REMAINDER_DEGREE: usize = 3;

/// Uni-skip instance for product virtualization, computing the first-round polynomial only.
#[derive(Allocative)]
pub struct ProductVirtualUniSkipInstanceProver<F: JoltField> {
    /// Evaluations of t1(Z) at the extended univariate-skip targets (outside base window)
    extended_evals: [F; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE],
    #[allocative(skip)]
    params: ProductVirtualUniSkipInstanceParams<F>,
}

impl<F: JoltField> ProductVirtualUniSkipInstanceProver<F> {
    /// Initialize a new prover for the univariate skip round
    /// The 5 base evaluations are the claimed evaluations of the 5 product terms from Spartan outer
    #[tracing::instrument(skip_all, name = "ProductVirtualUniSkipInstanceProver::gen")]
    pub fn gen<PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, PCS>,
        opening_accumulator: &ProverOpeningAccumulator<F>,
        tau: &[F::Challenge],
    ) -> Self {
        let params = ProductVirtualUniSkipInstanceParams::new(opening_accumulator, tau);

        let (_, _, trace, _, _) = state_manager.get_prover_data();

        let tau_low = &tau[..tau.len() - 1];
        let extended_evals = Self::compute_univariate_skip_extended_evals(trace, tau_low);

        let instance = Self {
            extended_evals,
            params,
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
        tau_low: &[F::Challenge],
    ) -> [F; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE] {
        // Split-Eq over cycle variables
        let m = tau_low.len() / 2;
        let (tau_out, tau_in) = tau_low.split_at(m);
        // Compute the split eq polynomial, one scaled by R^2 in order to balance against
        // Montgomery (not Barrett) reduction later on in 8-limb signed accumulation
        // of e_in * (left * right)
        let (E_out, E_in) = rayon::join(
            || EqPolynomial::evals_with_scaling(tau_out, Some(F::MONTGOMERY_R_SQUARE)),
            || EqPolynomial::evals(tau_in),
        );

        let num_x_out_vals = E_out.len();
        let num_x_in_vals = E_in.len();
        let iter_num_x_in_vars = num_x_in_vals.log_2();

        // Precompute per-target Lagrange integer coefficient vectors for extended targets
        let base_left: i64 = -((PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE as i64 - 1) / 2);
        let targets = uniskip_targets::<
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE,
        >();
        let coeffs_per_j: [[i32; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE];
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE] = core::array::from_fn(|j| {
            let shift = targets[j] - base_left;
            LagrangeHelper::shift_coeffs_i32::<PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE>(shift)
        });

        // Parallelize across x_out chunks
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

        let extended_evals: [F; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE] = (0..num_parallel_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let x_out_start = chunk_idx * x_out_chunk_size;
                let x_out_end = core::cmp::min((chunk_idx + 1) * x_out_chunk_size, num_x_out_vals);
                let mut local_acc_unr: [F::Unreduced<9>; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE] =
                    [F::Unreduced::<9>::zero(); PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE];

                for x_out_val in x_out_start..x_out_end {
                    let e_out = E_out[x_out_val];
                    // Accumulate across x_in using 8-limb signed accumulators per j
                    let mut inner_acc: [Acc8S<F>; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE] =
                        [Acc8S::<F>::zero(); PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE];
                    for x_in_val in 0..num_x_in_vals {
                        let e_in = if num_x_in_vals == 1 {
                            E_in[0]
                        } else {
                            E_in[x_in_val]
                        };
                        let idx = (x_out_val << iter_num_x_in_vars) | x_in_val;

                        // Materialize product-cycle row with raw types
                        let row = ProductCycleInputs::from_trace::<F>(trace, idx);

                        // For each extended target j, compute per-product weighted left/right via integer lifting
                        for j in 0..PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE {
                            let c = &coeffs_per_j[j];

                            // Declare per-product weighted components upfront
                            let mut left_w: [i128; NUM_PRODUCT_VIRTUAL] = [0; NUM_PRODUCT_VIRTUAL];
                            let mut right_w: [i128; NUM_PRODUCT_VIRTUAL] = [0; NUM_PRODUCT_VIRTUAL];

                            // Instruction: LeftInstructionInput × RightInstructionInput
                            // left: u64 -> i128; right: S64 -> i128
                            left_w[0] = (c[0] as i128) * (row.instruction_left_input as i128);
                            right_w[0] = (c[0] as i128) * row.instruction_right_input;

                            // WriteLookupOutputToRD: is_rd_zero × WriteLookupOutputToRD_flag
                            // left: bool/u8 -> i32 -> i128; right: bool/u8 -> i32 -> i128
                            left_w[1] = (c[1] as i128)
                                * (if row.is_rd_not_zero { 1i32 } else { 0i32 } as i128);
                            right_w[1] = (c[1] as i128)
                                * (if row.write_lookup_output_to_rd_flag {
                                    1i32
                                } else {
                                    0i32
                                } as i128);

                            // WritePCtoRD: is_rd_zero × Jump_flag
                            // left: bool/u8 -> i32 -> i128; right: bool/u8 -> i32 -> i128
                            left_w[2] = (c[2] as i128) * (row.is_rd_not_zero as i32 as i128);
                            right_w[2] =
                                (c[2] as i128) * (if row.jump_flag { 1i32 } else { 0i32 } as i128);

                            // ShouldBranch: lookup_output × Branch_flag
                            // left: u64 -> i128; right: bool/u8 -> i32 -> i128
                            left_w[3] = (c[3] as i128) * (row.should_branch_lookup_output as i128);
                            right_w[3] = (c[3] as i128)
                                * (if row.should_branch_flag { 1i32 } else { 0i32 } as i128);

                            // ShouldJump: Jump_flag × (1 − NextIsNoop)
                            // left: bool/u8 -> i32 -> i128; right: bool/u8 -> i32 -> i128
                            left_w[4] =
                                (c[4] as i128) * (if row.jump_flag { 1i32 } else { 0i32 } as i128);
                            right_w[4] = (c[4] as i128)
                                * (if row.not_next_noop { 1i32 } else { 0i32 } as i128);

                            // Fuse by summing over i in i128 and multiply in bigints first
                            let mut left_sum_i128: i128 = 0;
                            let mut right_sum_i128: i128 = 0;
                            for i in 0..NUM_PRODUCT_VIRTUAL {
                                left_sum_i128 += left_w[i];
                                right_sum_i128 += right_w[i];
                            }
                            // Compute S256 = S128 × S128
                            let left_s128 = S128::from_i128(left_sum_i128);
                            let right_s128 = S128::from_i128(right_sum_i128);
                            let prod_s256 = left_s128.mul_trunc::<2, 4>(&right_s128);

                            // Fold e_in into signed 8-limb accumulator for this j
                            inner_acc[j].fmadd(&e_in, &prod_s256);
                        }
                    }
                    // Reduce inner accumulators (pos-neg Montgomery) and multiply by E_out
                    // NOTE: needs a R^2 correction factor, applied when initializing E_out
                    for j in 0..PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE {
                        let reduced = inner_acc[j].montgomery_reduce();
                        local_acc_unr[j] += e_out.mul_unreduced::<9>(reduced);
                    }
                }

                // Reduce once per target for this chunk
                let mut local_acc: [F; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE] =
                    [F::zero(); PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE];
                for j in 0..PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE {
                    local_acc[j] = F::from_montgomery_reduce::<9>(local_acc_unr[j]);
                }
                local_acc
            })
            .reduce(
                || [F::zero(); PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE],
                |mut a, b| {
                    for j in 0..PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE {
                        a[j] += b[j];
                    }
                    a
                },
            );

        extended_evals
    }
}

impl<F: JoltField, T: Transcript> UniSkipFirstRoundInstanceProver<F, T>
    for ProductVirtualUniSkipInstanceProver<F>
{
    fn input_claim(&self) -> F {
        self.params.input_claim()
    }

    #[tracing::instrument(skip_all, name = "ProductVirtualUniSkipInstanceProver::compute_poly")]
    fn compute_poly(&mut self) -> UniPoly<F> {
        // Load base evals from shared instance and extended from prover state
        let base = self.params.base_evals;
        let tau_high = self.params.tau[self.params.tau.len() - 1];

        // Compute the univariate-skip first round polynomial s1(Y) = L(τ_high, Y) · t1(Y)
        build_uniskip_first_round_poly::<
            F,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE,
            PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS,
        >(Some(&base), &self.extended_evals, tau_high)
    }
}

pub struct ProductVirtualUniSkipInstanceParams<F: JoltField> {
    /// τ = [τ_low || τ_high]
    /// - τ_low: the cycle-point r_cycle carried from Spartan outer (length = num_cycle_vars)
    /// - τ_high: the univariate-skip binding point sampled for the size-5 domain (length = 1)
    ///   Ordering matches outer: variables are MSB→LSB with τ_high last
    tau: Vec<F::Challenge>,
    /// Base evaluations (claims) for the five product terms at the base domain
    /// Order: [Product, WriteLookupOutputToRD, WritePCtoRD, ShouldBranch, ShouldJump]
    base_evals: [F; NUM_PRODUCT_VIRTUAL],
}

impl<F: JoltField> ProductVirtualUniSkipInstanceParams<F> {
    pub fn new(opening_accumulator: &dyn OpeningAccumulator<F>, tau: &[F::Challenge]) -> Self {
        let mut base_evals: [F; NUM_PRODUCT_VIRTUAL] = [F::zero(); NUM_PRODUCT_VIRTUAL];
        for (i, vp) in PRODUCT_VIRTUAL_TERMS.iter().enumerate() {
            let (_, eval) =
                opening_accumulator.get_virtual_polynomial_opening(*vp, SumcheckId::SpartanOuter);
            base_evals[i] = eval;
        }
        Self {
            tau: tau.to_vec(),
            base_evals,
        }
    }

    pub fn input_claim(&self) -> F {
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
}

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
    #[tracing::instrument(skip_all, name = "ProductVirtualRemainderProver::gen")]
    pub fn gen<PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, PCS>,
        num_cycle_vars: usize,
        uni: &UniSkipState<F>,
    ) -> Self {
        let (_, _, trace, _program_io, _final_mem) = state_manager.get_prover_data();

        let lagrange_evals_r = LagrangePolynomial::<F>::evals::<
            F::Challenge,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&uni.r0);

        let tau_high = uni.tau[uni.tau.len() - 1];
        let tau_low = &uni.tau[..uni.tau.len() - 1];

        let lagrange_tau_r0 = LagrangePolynomial::<F>::lagrange_kernel::<
            F::Challenge,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&tau_high, &uni.r0);

        let split_eq_poly: GruenSplitEqPolynomial<F> =
            GruenSplitEqPolynomial::<F>::new_with_scaling(
                tau_low,
                BindingOrder::LowToHigh,
                Some(lagrange_tau_r0),
            );

        let (t0, t_inf, left_bound, right_bound) =
            Self::compute_first_quadratic_evals_and_bound_polys(
                trace,
                &lagrange_evals_r,
                &split_eq_poly,
            );

        Self {
            split_eq_poly,
            trace: state_manager.get_trace_arc(),
            left: left_bound,
            right: right_bound,
            first_round_evals: (t0, t_inf),
            params: ProductVirtualRemainderParams::new(num_cycle_vars, uni),
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
        let eq_poly = &self.split_eq_poly;

        let n = self.left.len();
        debug_assert_eq!(n, self.right.len());
        if eq_poly.E_in_current_len() == 1 {
            let groups = n / 2;
            let (t0_unr, tinf_unr) = (0..groups)
                .into_par_iter()
                .map(|g| {
                    let l0 = self.left[2 * g];
                    let l1 = self.left[2 * g + 1];
                    let r0 = self.right[2 * g];
                    let r1 = self.right[2 * g + 1];
                    let eq = eq_poly.E_out_current()[g];
                    let p0 = l0 * r0;
                    let slope = (l1 - l0) * (r1 - r0);
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
                        let l0 = self.left[2 * g];
                        let l1 = self.left[2 * g + 1];
                        let r0 = self.right[2 * g];
                        let r1 = self.right[2 * g + 1];
                        let e_in = eq_poly.E_in_current()[x1];
                        let p0 = l0 * r0;
                        let slope = (l1 - l0) * (r1 - r0);
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

    /// Returns final per-virtual-polynomial evaluations needed for openings.
    pub fn final_sumcheck_evals(&self) -> [F; 2] {
        let l0 = if !self.left.is_empty() {
            self.left[0]
        } else {
            F::zero()
        };
        let r0 = if !self.right.is_empty() {
            self.right[0]
        } else {
            F::zero()
        };
        [l0, r0]
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for ProductVirtualRemainderProver<F>
{
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.params.input_claim
    }

    #[tracing::instrument(
        skip_all,
        name = "ProductVirtualRemainderProver::compute_prover_message"
    )]
    fn compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F> {
        let (t0, t_inf) = if round == 0 {
            self.first_round_evals
        } else {
            self.remaining_quadratic_evals()
        };
        let evals = self
            .split_eq_poly
            .gruen_evals_deg_3(t0, t_inf, previous_claim);
        vec![evals[0], evals[1], evals[2]]
    }

    #[tracing::instrument(skip_all, name = "ProductVirtualRemainderProver::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
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
        let opening_point = self.params.get_opening_point(sumcheck_challenges);
        // Append fused left/right product claims and factor claims under fused ID at opening_point
        debug_assert_eq!(opening_point.len(), self.params.n_cycle_vars + 1);
        // Split opening_point.r into (r_cycle, r0) after reversal: first num_cycle_vars are r_cycle
        let (r_cycle, _r0_slice) = opening_point.r.split_at(self.params.n_cycle_vars);

        // Compute claimed unique factor evaluations at r_cycle in one pass
        let claims = ProductVirtualEval::compute_claimed_factors::<F>(&self.trace, r_cycle);

        // Append fused left/right product openings akin to outer (SpartanAz/Bz)
        let lr = self.final_sumcheck_evals();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::FusedProductLeft,
            SumcheckId::ProductVirtualization,
            opening_point.clone(),
            lr[0],
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::FusedProductRight,
            SumcheckId::ProductVirtualization,
            opening_point.clone(),
            lr[1],
        );

        // Append the 8 unique factor openings in canonical order
        for (i, vp) in PRODUCT_UNIQUE_FACTOR_VIRTUALS.iter().enumerate() {
            accumulator.append_virtual(
                transcript,
                *vp,
                SumcheckId::ProductVirtualization,
                OpeningPoint::new(r_cycle.to_vec()),
                claims[i],
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
    pub fn new(n_cycle_vars: usize, uni: &UniSkipState<F>) -> Self {
        let params = ProductVirtualRemainderParams::new(n_cycle_vars, uni);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for ProductVirtualRemainderVerifier<F>
{
    fn degree(&self) -> usize {
        PRODUCT_VIRTUAL_REMAINDER_DEGREE
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
        // Retrieve fused left/right product evaluations under ProductVirtualization
        let (_, fused_left) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::FusedProductLeft,
            SumcheckId::ProductVirtualization,
        );
        let (_, fused_right) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::FusedProductRight,
            SumcheckId::ProductVirtualization,
        );

        // Multiply by L(τ_high, r0) and Eq(τ_low, r_tail^rev)
        let tau_high = &self.params.tau[self.params.tau.len() - 1];
        let tau_high_bound_r0 = LagrangePolynomial::<F>::lagrange_kernel::<
            F::Challenge,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(tau_high, &self.params.r0_uniskip);
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
        let opening_point = self.params.get_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::FusedProductLeft,
            SumcheckId::ProductVirtualization,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::FusedProductRight,
            SumcheckId::ProductVirtualization,
            opening_point.clone(),
        );
        // Factors are opened at r_cycle (slice out r0), mirroring outer.rs
        let (r_cycle, _r0_slice) = opening_point.r.split_at(self.params.n_cycle_vars);
        let cycle_only_point = OpeningPoint::new(r_cycle.to_vec());
        for vp in PRODUCT_UNIQUE_FACTOR_VIRTUALS.iter() {
            accumulator.append_virtual(
                transcript,
                *vp,
                SumcheckId::ProductVirtualization,
                cycle_only_point.clone(),
            );
        }
    }
}

struct ProductVirtualRemainderParams<F: JoltField> {
    /// Number of cycle variables to bind in this remainder (equals log2(T))
    n_cycle_vars: usize,
    /// The univariate-skip first round challenge r0
    r0_uniskip: F::Challenge,
    /// Claim after the univariate-skip first round, updated every round
    input_claim: F,
    /// The tau vector (length 1 + n_cycle_vars), available to prover and verifier
    tau: Vec<F::Challenge>,
}

impl<F: JoltField> ProductVirtualRemainderParams<F> {
    fn new(n_cycle_vars: usize, uni: &UniSkipState<F>) -> Self {
        Self {
            n_cycle_vars,
            r0_uniskip: uni.r0,
            input_claim: uni.claim_after_first,
            tau: uni.tau.clone(),
        }
    }

    fn num_rounds(&self) -> usize {
        self.n_cycle_vars
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

/// Zero-round check to check that the final claimed evals of the fused left & right product terms
/// are the same as the linear combination of the individual claimed evals of the unique factors.
///
/// This is essentially Spartan inner sumcheck, but we check the claim without any sumcheck
/// because we can.
///
/// To be precise, the relation to be checked is:
/// fused_left_claim = sum_i lagrange_coeffs[i] * left_factor_claim[i], AND
/// fused_right_claim = sum_i lagrange_coeffs[i] * right_factor_claim[i]
///
/// (batched together by a gamma challenge)
///
/// where the factor are determined according to the virtual polynomials
/// (i.e. a single ShouldJump claimed eval participates in two (right) factor claim)
#[derive(Allocative)]
pub struct ProductVirtualInnerProver<F: JoltField> {
    #[allocative(skip)]
    params: ProductVirtualInnerParams<F>,
}

impl<F: JoltField> ProductVirtualInnerProver<F> {
    #[tracing::instrument(skip_all, name = "ProductVirtualInnerProver::new")]
    pub fn new(
        opening_accumulator: &ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = ProductVirtualInnerParams::new(opening_accumulator, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ProductVirtualInnerProver<F> {
    // Trivial sum-check, no round, no computation of messages or binding, only verification
    fn degree(&self) -> usize {
        0
    }

    fn num_rounds(&self) -> usize {
        0
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        Vec::new()
    }

    fn bind(&mut self, _r_j: F::Challenge, _round: usize) {}

    fn cache_openings(
        &self,
        _accumulator: &mut ProverOpeningAccumulator<F>,
        _transcript: &mut T,
        _sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct ProductVirtualInnerVerifier<F: JoltField> {
    params: ProductVirtualInnerParams<F>,
}

impl<F: JoltField> ProductVirtualInnerVerifier<F> {
    pub fn new(
        opening_accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = ProductVirtualInnerParams::new(opening_accumulator, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for ProductVirtualInnerVerifier<F>
{
    fn degree(&self) -> usize {
        0
    }

    fn num_rounds(&self) -> usize {
        0
    }

    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        _sumcheck_challenges: &[F::Challenge],
    ) -> F {
        // Lagrange weights at r0
        let w = LagrangePolynomial::<F>::evals::<
            F::Challenge,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&self.params.r0_uniskip);

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

        // TODO: make this logic less brittle
        let left_sum = w[0] * l_inst
            + w[1] * is_rd_not_zero
            + w[2] * is_rd_not_zero
            + w[3] * lookup_out
            + w[4] * j_flag;
        let right_sum = w[0] * r_inst
            + w[1] * wl_flag
            + w[2] * j_flag
            + w[3] * branch_flag
            + w[4] * (F::one() - next_is_noop);

        left_sum + self.params.gamma * right_sum
    }

    fn cache_openings(
        &self,
        _accumulator: &mut VerifierOpeningAccumulator<F>,
        _transcript: &mut T,
        _sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
    }
}

struct ProductVirtualInnerParams<F: JoltField> {
    r0_uniskip: F::Challenge,
    gamma: F::Challenge,
}

impl<F: JoltField> ProductVirtualInnerParams<F> {
    fn new(
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let gamma = transcript.challenge_scalar_optimized::<F>();
        let (pt_left, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::FusedProductLeft,
            SumcheckId::ProductVirtualization,
        );
        let r0 = *pt_left
            .r
            .last()
            .expect("ProductVirtualInner requires r0 in opening point");
        Self {
            r0_uniskip: r0,
            gamma,
        }
    }

    pub fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, fused_left) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::FusedProductLeft,
            SumcheckId::ProductVirtualization,
        );
        let (_, fused_right) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::FusedProductRight,
            SumcheckId::ProductVirtualization,
        );
        fused_left + self.gamma * fused_right
    }
}
