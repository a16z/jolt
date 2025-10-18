use allocative::Allocative;
use ark_std::Zero;
use std::cell::RefCell;
use std::rc::Rc;

use crate::field::AccumulateInPlace;
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
use crate::subprotocols::univariate_skip::{
    build_uniskip_first_round_poly, uniskip_targets, UniSkipState,
};
use crate::transcripts::Transcript;
use crate::utils::accumulation::{acc8s_fmadd_s256, Acc6S, Acc6U, Acc8Signed};
use crate::utils::math::Math;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::instruction::{CircuitFlags, InstructionFlags};
use crate::zkvm::r1cs::inputs::{
    compute_claimed_product_factor_evals, ProductCycleInputs, PRODUCT_UNIQUE_FACTOR_VIRTUALS,
};
use crate::zkvm::witness::VirtualPolynomial;
use crate::zkvm::JoltSharedPreprocessing;
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

/// Uni-skip instance for product virtualization, computing the first-round polynomial only.
#[derive(Allocative)]
pub struct ProductVirtualUniSkipInstance<F: JoltField> {
    /// τ = [τ_low || τ_high]
    /// - τ_low: the cycle-point r_cycle carried from Spartan outer (length = num_cycle_vars)
    /// - τ_high: the univariate-skip binding point sampled for the size-5 domain (length = 1)
    ///   Ordering matches outer: variables are MSB→LSB with τ_high last
    tau: Vec<F::Challenge>,
    /// Base evaluations (claims) for the five product terms at the base domain
    /// Order: [Product, WriteLookupOutputToRD, WritePCtoRD, ShouldBranch, ShouldJump]
    base_evals: [F; NUM_PRODUCT_VIRTUAL],
    /// Prover-only state (None on verifier)
    prover_state: Option<ProductVirtualUniSkipProverState<F>>,
}

#[derive(Allocative)]
struct ProductVirtualUniSkipProverState<F: JoltField> {
    /// Evaluations of t1(Z) at the extended univariate-skip targets (outside base window)
    extended_evals: [F; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE],
}

impl<F: JoltField> ProductVirtualUniSkipInstance<F> {
    /// Initialize a new prover for the univariate skip round
    /// The 5 base evaluations are the claimed evaluations of the 5 product terms from Spartan outer
    #[tracing::instrument(skip_all, name = "ProductVirtualUniSkipInstance::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        tau: &[F::Challenge],
    ) -> Self {
        let (_preprocessing, trace, _program_io, _final_mem) = state_manager.get_prover_data();

        // Get base evaluations from outer sumcheck claims
        let acc = state_manager.get_prover_accumulator();
        let mut base_evals: [F; NUM_PRODUCT_VIRTUAL] = [F::zero(); NUM_PRODUCT_VIRTUAL];
        for (i, vp) in PRODUCT_VIRTUAL_TERMS.iter().enumerate() {
            let (_, eval) = acc
                .borrow()
                .get_virtual_polynomial_opening(*vp, SumcheckId::SpartanOuter);
            base_evals[i] = eval;
        }

        let tau_low = &tau[..tau.len() - 1];
        let extended_evals = Self::compute_univariate_skip_extended_evals(trace, tau_low);

        let instance = Self {
            tau: tau.to_vec(),
            base_evals,
            prover_state: Some(ProductVirtualUniSkipProverState { extended_evals }),
        };

        #[cfg(feature = "allocative")]
        print_data_structure_heap_usage("ProductVirtualUniSkipInstance", &instance);
        instance
    }

    pub fn new_verifier(
        state_manager: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
        tau: &[F::Challenge],
    ) -> Self {
        let acc = state_manager.get_verifier_accumulator();
        let mut base_evals: [F; NUM_PRODUCT_VIRTUAL] = [F::zero(); NUM_PRODUCT_VIRTUAL];
        for (i, vp) in PRODUCT_VIRTUAL_TERMS.iter().enumerate() {
            let (_, eval) = acc
                .borrow()
                .get_virtual_polynomial_opening(*vp, SumcheckId::SpartanOuter);
            base_evals[i] = eval;
        }
        Self {
            tau: tau.to_vec(),
            base_evals,
            prover_state: None,
        }
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
    /// - WriteLookupOutputToRD: RdWa is u8 → i32; flag is bool/u8 → i32.
    /// - WritePCtoRD: RdWa is u8 → i32; Jump flag is bool/u8 → i32.
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
        // Montgomery (not Barrett) reduction later on in signed accumulation of e_in * (left * right)
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
                    let mut inner_acc: [Acc8Signed<F>; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE] =
                        [Acc8Signed::<F>::new(); PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE];
                    for x_in_val in 0..num_x_in_vals {
                        let e_in = if num_x_in_vals == 0 {
                            F::one()
                        } else if num_x_in_vals == 1 {
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

                            // WriteLookupOutputToRD: rd_addr × WriteLookupOutputToRD_flag
                            // left: u8 -> i32 -> i128; right: bool/u8 -> i32 -> i128
                            left_w[1] = (c[1] as i128) * (row.rd_addr as i32 as i128);
                            right_w[1] = (c[1] as i128)
                                * (if row.write_lookup_output_to_rd_flag {
                                    1i32
                                } else {
                                    0i32
                                } as i128);

                            // WritePCtoRD: rd_addr × Jump_flag
                            // left: u8 -> i32 -> i128; right: bool/u8 -> i32 -> i128
                            left_w[2] = (c[2] as i128) * (row.rd_addr as i32 as i128);
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
                            acc8s_fmadd_s256(&mut inner_acc[j], &e_in, prod_s256);
                        }
                    }
                    // Reduce inner accumulators (pos-neg Montgomery) and apply E_out
                    for j in 0..PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE {
                        let reduced = inner_acc[j].reduce_to_field();
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

impl<F: JoltField, T: Transcript> UniSkipFirstRoundInstance<F, T>
    for ProductVirtualUniSkipInstance<F>
{
    const DEGREE_BOUND: usize = PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS - 1;
    const DOMAIN_SIZE: usize = NUM_PRODUCT_VIRTUAL;

    fn input_claim(&self) -> F {
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

    fn compute_poly(&mut self) -> UniPoly<F> {
        // Load base evals from shared instance and extended from prover state
        let base = self.base_evals;
        let extended = if let Some(ps) = &self.prover_state {
            ps.extended_evals
        } else {
            [F::zero(); PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE]
        };

        let tau_high = self.tau[self.tau.len() - 1];

        // Compute the univariate-skip first round polynomial s1(Y) = L(τ_high, Y) · t1(Y)
        build_uniskip_first_round_poly::<
            F,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE,
            PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS,
            false, // base evals are non-zero (product claims from outer sumcheck)
        >(&base, &extended, tau_high)
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
pub struct ProductVirtualRemainder<F: JoltField> {
    /// Number of cycle variables to bind in this remainder (equals log2(T))
    pub num_cycle_vars: usize,
    /// The univariate-skip first round challenge r0
    pub r0_uniskip: F::Challenge,
    /// Claim after the univariate-skip first round, updated every round
    pub input_claim: F,
    /// The tau vector (length 1 + num_cycle_vars), available to prover and verifier
    pub tau: Vec<F::Challenge>,
    /// Prover-only state (None on verifier)
    pub prover_state: Option<ProductVirtualProverState<F>>,
}

#[derive(Allocative)]
pub struct ProductVirtualStreamingCache<F: JoltField> {
    pub t0: F,
    pub t_inf: F,
    pub left_lo: Vec<F>,
    pub left_hi: Vec<F>,
    pub right_lo: Vec<F>,
    pub right_hi: Vec<F>,
}

#[derive(Allocative)]
pub struct ProductVirtualProverState<F: JoltField> {
    #[allocative(skip)]
    pub preprocess: Arc<JoltSharedPreprocessing>,
    #[allocative(skip)]
    pub trace: Arc<Vec<Cycle>>,
    pub split_eq_poly: GruenSplitEqPolynomial<F>,
    pub left: DensePolynomial<F>,
    pub right: DensePolynomial<F>,
    pub streaming_cache: Option<ProductVirtualStreamingCache<F>>,
}

impl<F: JoltField> ProductVirtualRemainder<F> {
    #[tracing::instrument(skip_all, name = "ProductVirtualRemainder::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        num_cycle_vars: usize,
        uni: &UniSkipState<F>,
    ) -> Self {
        let (preprocessing, trace, _program_io, _final_mem) = state_manager.get_prover_data();

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

        let streaming_cache =
            Self::compute_streaming_round_cache(trace, &lagrange_evals_r, &split_eq_poly);

        Self {
            num_cycle_vars,
            r0_uniskip: uni.r0,
            input_claim: uni.claim_after_first,
            tau: uni.tau.clone(),
            prover_state: Some(ProductVirtualProverState {
                split_eq_poly,
                preprocess: Arc::new(preprocessing.shared.clone()),
                trace: Arc::new(trace.to_vec()),
                left: DensePolynomial::default(),
                right: DensePolynomial::default(),
                streaming_cache: Some(streaming_cache),
            }),
        }
    }

    pub fn new_verifier(num_cycle_vars: usize, uni: UniSkipState<F>) -> Self {
        Self {
            num_cycle_vars,
            r0_uniskip: uni.r0,
            input_claim: uni.claim_after_first,
            tau: uni.tau,
            prover_state: None,
        }
    }

    /// Compute the quadratic evaluations for the streaming round (right after univariate skip).
    ///
    /// After binding the univariate-skip variable at r0, we must
    /// compute the cubic round polynomial endpoints over the cycle variables only:
    ///   t(0)  = Σ_{x_out} E_out · Σ_{x_in} E_in · Left_0(x) · Right_0(x)
    ///   t(∞)  = Σ_{x_out} E_out · Σ_{x_in} E_in · (Left_1−Left_0) · (Right_1−Right_0)
    /// We also cache per-(x_out,x_in) coefficients at y∈{0,1} in order to bind them by r_0
    /// once, after which remaining rounds bind linearly over the cycle variables.
    ///
    /// Product virtualization specifics:
    /// - Left/Right are fused linear combinations of five per-type witnesses with Lagrange
    ///   weights w_i = L_i(r0) over the size-5 domain.
    /// - For ShouldJump, the effective right factor is (1 − NextIsNoop).
    /// - We follow outer’s delayed-reduction pattern across x_in to reduce modular reductions.
    fn compute_streaming_round_cache(
        trace: &[Cycle],
        weights_at_r0: &[F; NUM_PRODUCT_VIRTUAL],
        split_eq_poly: &GruenSplitEqPolynomial<F>,
    ) -> ProductVirtualStreamingCache<F> {
        let num_x_out_vals = split_eq_poly.E_out_current_len();
        let num_x_in_vals = split_eq_poly.E_in_current_len();
        let iter_num_x_in_vars = num_x_in_vals.log_2();
        let groups_exact = num_x_out_vals
            .checked_mul(num_x_in_vals)
            .expect("overflow computing groups_exact");

        // Preallocate global buffers once
        let mut left_lo: Vec<F> = unsafe_allocate_zero_vec(groups_exact);
        let mut left_hi: Vec<F> = unsafe_allocate_zero_vec(groups_exact);
        let mut right_lo: Vec<F> = unsafe_allocate_zero_vec(groups_exact);
        let mut right_hi: Vec<F> = unsafe_allocate_zero_vec(groups_exact);

        // Parallel over x_out by chunking in lockstep; each chunk has size num_x_in_vals
        let (t0_acc_unr, t_inf_acc_unr) = left_lo
            .par_chunks_mut(num_x_in_vals)
            .zip(left_hi.par_chunks_mut(num_x_in_vals))
            .zip(right_lo.par_chunks_mut(num_x_in_vals))
            .zip(right_hi.par_chunks_mut(num_x_in_vals))
            .enumerate()
            .map(
                |(
                    x_out_val,
                    (((left_lo_chunk, left_hi_chunk), right_lo_chunk), right_hi_chunk),
                )| {
                    let mut inner0 = F::Unreduced::<9>::zero();
                    let mut inner_inf = F::Unreduced::<9>::zero();
                    for x_in_val in 0..num_x_in_vals {
                        let base_idx = (x_out_val << iter_num_x_in_vars) | x_in_val;
                        let idx_lo = base_idx << 1;
                        let idx_hi = idx_lo + 1;

                        // Materialize rows for lo and hi once
                        let row_lo = ProductCycleInputs::from_trace::<F>(trace, idx_lo);
                        let row_hi = ProductCycleInputs::from_trace::<F>(trace, idx_hi);

                        // Fused left/right using small-value accumulators
                        let mut left0_acc: Acc6U<F> = Acc6U::new();
                        let mut right0_acc: Acc6S<F> = Acc6S::new();
                        left0_acc.fmadd(&weights_at_r0[0], &row_lo.instruction_left_input);
                        left0_acc.fmadd(&weights_at_r0[1], &(row_lo.rd_addr as u64));
                        left0_acc.fmadd(&weights_at_r0[2], &(row_lo.rd_addr as u64));
                        left0_acc.fmadd(&weights_at_r0[3], &row_lo.should_branch_lookup_output);
                        if row_lo.jump_flag {
                            left0_acc.fmadd(&weights_at_r0[4], &1u64);
                        }
                        right0_acc.fmadd(&weights_at_r0[0], &row_lo.instruction_right_input);
                        if row_lo.write_lookup_output_to_rd_flag {
                            right0_acc.fmadd(&weights_at_r0[1], &1i128);
                        }
                        if row_lo.jump_flag {
                            right0_acc.fmadd(&weights_at_r0[2], &1i128);
                        }
                        if row_lo.should_branch_flag {
                            right0_acc.fmadd(&weights_at_r0[3], &1i128);
                        }
                        if row_lo.not_next_noop {
                            right0_acc.fmadd(&weights_at_r0[4], &1i128);
                        }
                        let left0 = left0_acc.reduce();
                        let right0 = right0_acc.reduce();

                        let mut left1_acc: Acc6U<F> = Acc6U::new();
                        let mut right1_acc: Acc6S<F> = Acc6S::new();
                        left1_acc.fmadd(&weights_at_r0[0], &row_hi.instruction_left_input);
                        left1_acc.fmadd(&weights_at_r0[1], &(row_hi.rd_addr as u64));
                        left1_acc.fmadd(&weights_at_r0[2], &(row_hi.rd_addr as u64));
                        left1_acc.fmadd(&weights_at_r0[3], &row_hi.should_branch_lookup_output);
                        if row_hi.jump_flag {
                            left1_acc.fmadd(&weights_at_r0[4], &1u64);
                        }
                        right1_acc.fmadd(&weights_at_r0[0], &row_hi.instruction_right_input);
                        if row_hi.write_lookup_output_to_rd_flag {
                            right1_acc.fmadd(&weights_at_r0[1], &1i128);
                        }
                        if row_hi.jump_flag {
                            right1_acc.fmadd(&weights_at_r0[2], &1i128);
                        }
                        if row_hi.should_branch_flag {
                            right1_acc.fmadd(&weights_at_r0[3], &1i128);
                        }
                        if row_hi.not_next_noop {
                            right1_acc.fmadd(&weights_at_r0[4], &1i128);
                        }
                        let left1 = left1_acc.reduce();
                        let right1 = right1_acc.reduce();

                        let e_in = if num_x_in_vals == 1 {
                            split_eq_poly.E_in_current()[0]
                        } else {
                            split_eq_poly.E_in_current()[x_in_val]
                        };
                        let p0 = left0 * right0;
                        let slope = (left1 - left0) * (right1 - right0);
                        inner0 += e_in.mul_unreduced::<9>(p0);
                        inner_inf += e_in.mul_unreduced::<9>(slope);
                        left_lo_chunk[x_in_val] = left0;
                        right_lo_chunk[x_in_val] = right0;
                        left_hi_chunk[x_in_val] = left1;
                        right_hi_chunk[x_in_val] = right1;
                    }
                    let e_out = split_eq_poly.E_out_current()[x_out_val];
                    let reduced0 = F::from_montgomery_reduce::<9>(inner0);
                    let reduced_inf = F::from_montgomery_reduce::<9>(inner_inf);
                    (
                        e_out.mul_unreduced::<9>(reduced0),
                        e_out.mul_unreduced::<9>(reduced_inf),
                    )
                },
            )
            .reduce(
                || (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero()),
                |a, b| (a.0 + b.0, a.1 + b.1),
            );

        ProductVirtualStreamingCache {
            t0: F::from_montgomery_reduce::<9>(t0_acc_unr),
            t_inf: F::from_montgomery_reduce::<9>(t_inf_acc_unr),
            left_lo,
            left_hi,
            right_lo,
            right_hi,
        }
    }

    fn bind_streaming_round(&mut self, r_0: F::Challenge) {
        if let Some(ps) = self.prover_state.as_mut() {
            if let Some(cache) = ps.streaming_cache.take() {
                let groups = cache.left_lo.len();
                let mut left_bound: Vec<F> = unsafe_allocate_zero_vec(groups);
                let mut right_bound: Vec<F> = unsafe_allocate_zero_vec(groups);
                let num_x_in_vals = ps.split_eq_poly.E_in_current_len();

                // Parallelize over x_out by chunking destination slices
                left_bound
                    .par_chunks_mut(num_x_in_vals)
                    .zip(right_bound.par_chunks_mut(num_x_in_vals))
                    .enumerate()
                    .for_each(|(xo, (l_chunk, r_chunk))| {
                        for xi in 0..num_x_in_vals {
                            let idx = xo * num_x_in_vals + xi;
                            let l0 = cache.left_lo[idx];
                            let l1 = cache.left_hi[idx];
                            let r0 = cache.right_lo[idx];
                            let r1 = cache.right_hi[idx];
                            l_chunk[xi] = l0 + r_0 * (l1 - l0);
                            r_chunk[xi] = r0 + r_0 * (r1 - r0);
                        }
                    });

                ps.left = DensePolynomial::new(left_bound);
                ps.right = DensePolynomial::new(right_bound);
                return;
            }
        }
        panic!("Streaming cache missing; cannot bind first round");
    }

    /// Compute the quadratic endpoints for remaining rounds.
    fn remaining_quadratic_evals(&self) -> (F, F) {
        let ps = self.prover_state.as_ref().expect("prover state missing");
        let eq_poly = &ps.split_eq_poly;

        let n = ps.left.len();
        debug_assert_eq!(n, ps.right.len());
        if eq_poly.E_in_current_len() == 1 {
            let groups = n / 2;
            let (t0_unr, tinf_unr) = (0..groups)
                .into_par_iter()
                .map(|g| {
                    let l0 = ps.left[2 * g];
                    let l1 = ps.left[2 * g + 1];
                    let r0 = ps.right[2 * g];
                    let r1 = ps.right[2 * g + 1];
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
                        let l0 = ps.left[2 * g];
                        let l1 = ps.left[2 * g + 1];
                        let r0 = ps.right[2 * g];
                        let r1 = ps.right[2 * g + 1];
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
        let ps_opt = self.prover_state.as_ref();
        let l0 = ps_opt
            .and_then(|s| {
                if !s.left.is_empty() {
                    Some(s.left[0])
                } else {
                    None
                }
            })
            .unwrap_or(F::zero());
        let r0 = ps_opt
            .and_then(|s| {
                if !s.right.is_empty() {
                    Some(s.right[0])
                } else {
                    None
                }
            })
            .unwrap_or(F::zero());
        [l0, r0]
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for ProductVirtualRemainder<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.num_cycle_vars
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
            ps.left.bind_parallel(r_j, BindingOrder::LowToHigh);
            ps.right.bind_parallel(r_j, BindingOrder::LowToHigh);
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
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r_tail: &[F::Challenge],
    ) -> F {
        let acc = accumulator.as_ref().expect("accumulator required").borrow();

        // Retrieve fused left/right product evaluations under ProductVirtualization
        let (_, fused_left) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::FusedProductLeft,
            SumcheckId::ProductVirtualization,
        );
        let (_, fused_right) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::FusedProductRight,
            SumcheckId::ProductVirtualization,
        );

        // Multiply by L(τ_high, r0) and Eq(τ_low, r_tail^rev)
        let tau_high = &self.tau[self.tau.len() - 1];
        let tau_high_bound_r0 = LagrangePolynomial::<F>::lagrange_kernel::<
            F::Challenge,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(tau_high, &self.r0_uniskip);
        let tau_low = &self.tau[..self.tau.len() - 1];
        let r_tail_reversed: Vec<F::Challenge> = r_tail.iter().rev().copied().collect();
        let tau_bound_r_tail_reversed = EqPolynomial::mle(tau_low, &r_tail_reversed);

        tau_high_bound_r0 * tau_bound_r_tail_reversed * fused_left * fused_right
    }

    fn normalize_opening_point(&self, r_tail: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        // Construct full r = [r0 || r_tail] and reverse to match outer convention
        let mut r_full: Vec<F::Challenge> = Vec::with_capacity(1 + r_tail.len());
        r_full.push(self.r0_uniskip);
        r_full.extend_from_slice(r_tail);
        let r_reversed: Vec<F::Challenge> = r_full.into_iter().rev().collect();
        OpeningPoint::new(r_reversed)
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        // Append fused left/right product claims and factor claims under fused ID at opening_point
        debug_assert_eq!(opening_point.r.len(), self.num_cycle_vars + 1);

        // Split opening_point.r into (r_cycle, r0) after reversal: first num_cycle_vars are r_cycle
        let (r_cycle, _r0_slice) = opening_point.r.split_at(self.num_cycle_vars);

        // Compute claimed unique factor evaluations at r_cycle in one pass
        let claims = {
            let ps = self.prover_state.as_ref().expect("prover state missing");
            compute_claimed_product_factor_evals::<F>(&ps.trace, r_cycle)
        };

        // Append fused left/right product openings akin to outer (SpartanAz/Bz)
        let lr = self.final_sumcheck_evals();
        let mut acc = accumulator.borrow_mut();
        acc.append_virtual(
            transcript,
            VirtualPolynomial::FusedProductLeft,
            SumcheckId::ProductVirtualization,
            opening_point.clone(),
            lr[0],
        );
        acc.append_virtual(
            transcript,
            VirtualPolynomial::FusedProductRight,
            SumcheckId::ProductVirtualization,
            opening_point.clone(),
            lr[1],
        );

        // Append the 8 unique factor openings in canonical order
        for (i, vp) in PRODUCT_UNIQUE_FACTOR_VIRTUALS.iter().enumerate() {
            acc.append_virtual(
                transcript,
                *vp,
                SumcheckId::ProductVirtualization,
                OpeningPoint::new(r_cycle.to_vec()),
                claims[i],
            );
        }
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        // Append fused left/right product openings and the 8 factor openings (no claims)
        let mut acc = accumulator.borrow_mut();
        acc.append_virtual(
            transcript,
            VirtualPolynomial::FusedProductLeft,
            SumcheckId::ProductVirtualization,
            opening_point.clone(),
        );
        acc.append_virtual(
            transcript,
            VirtualPolynomial::FusedProductRight,
            SumcheckId::ProductVirtualization,
            opening_point.clone(),
        );
        // Factors are opened at r_cycle (slice out r0), mirroring outer.rs
        let (r_cycle, _r0_slice) = opening_point.r.split_at(self.num_cycle_vars);
        let cycle_only_point = OpeningPoint::new(r_cycle.to_vec());
        for vp in PRODUCT_UNIQUE_FACTOR_VIRTUALS.iter() {
            acc.append_virtual(
                transcript,
                *vp,
                SumcheckId::ProductVirtualization,
                cycle_only_point.clone(),
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
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
pub struct ProductVirtualInner<F: JoltField> {
    claim: F,
    r0_uniskip: F::Challenge,
    gamma: F::Challenge,
}

impl<F: JoltField> ProductVirtualInner<F> {
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        // Sample gamma like inner.rs
        let gamma: F::Challenge = state_manager
            .transcript
            .borrow_mut()
            .challenge_scalar_optimized::<F>();
        let acc = state_manager.get_prover_accumulator();
        let acc_ref = acc.borrow();
        // Fused product claims (their opening point includes r0)
        let (pt_left, fused_left) = acc_ref.get_virtual_polynomial_opening(
            VirtualPolynomial::FusedProductLeft,
            SumcheckId::ProductVirtualization,
        );
        let (_, fused_right) = acc_ref.get_virtual_polynomial_opening(
            VirtualPolynomial::FusedProductRight,
            SumcheckId::ProductVirtualization,
        );
        let r0 = *pt_left
            .r
            .last()
            .expect("ProductVirtualInner requires r0 in opening point");
        let claim = fused_left + gamma * fused_right;
        Self {
            claim,
            r0_uniskip: r0,
            gamma,
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let gamma: F::Challenge = state_manager
            .transcript
            .borrow_mut()
            .challenge_scalar_optimized::<F>();
        let acc = state_manager.get_verifier_accumulator();
        let acc_ref = acc.borrow();
        let (pt_left, fused_left) = acc_ref.get_virtual_polynomial_opening(
            VirtualPolynomial::FusedProductLeft,
            SumcheckId::ProductVirtualization,
        );
        let (_, fused_right) = acc_ref.get_virtual_polynomial_opening(
            VirtualPolynomial::FusedProductRight,
            SumcheckId::ProductVirtualization,
        );
        let r0 = *pt_left
            .r
            .last()
            .expect("ProductVirtualInner requires r0 in opening point");
        let claim = fused_left + gamma * fused_right;
        Self {
            claim,
            r0_uniskip: r0,
            gamma,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for ProductVirtualInner<F> {
    // Trivial sum-check, no round, no computation of messages or binding, only verification
    fn degree(&self) -> usize {
        0
    }
    fn num_rounds(&self) -> usize {
        0
    }
    fn input_claim(&self) -> F {
        self.claim
    }
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        Vec::new()
    }
    fn bind(&mut self, _r_j: F::Challenge, _round: usize) {}
    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        _r: &[F::Challenge],
    ) -> F {
        let acc = accumulator.as_ref().expect("accumulator required").borrow();
        // Lagrange weights at r0
        let w = LagrangePolynomial::<F>::evals::<
            F::Challenge,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&self.r0_uniskip);

        // Fetch factor claims
        let l_inst = acc
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LeftInstructionInput,
                SumcheckId::ProductVirtualization,
            )
            .1;
        let r_inst = acc
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RightInstructionInput,
                SumcheckId::ProductVirtualization,
            )
            .1;
        let rd_wa = acc
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RdWa,
                SumcheckId::ProductVirtualization,
            )
            .1;
        let wl_flag = acc
            .get_virtual_polynomial_opening(
                VirtualPolynomial::OpFlags(CircuitFlags::WriteLookupOutputToRD),
                SumcheckId::ProductVirtualization,
            )
            .1;
        let j_flag = acc
            .get_virtual_polynomial_opening(
                VirtualPolynomial::OpFlags(CircuitFlags::Jump),
                SumcheckId::ProductVirtualization,
            )
            .1;
        let lookup_out = acc
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::ProductVirtualization,
            )
            .1;
        let branch_flag = acc
            .get_virtual_polynomial_opening(
                VirtualPolynomial::InstructionFlags(InstructionFlags::Branch),
                SumcheckId::ProductVirtualization,
            )
            .1;
        let next_is_noop = acc
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NextIsNoop,
                SumcheckId::ProductVirtualization,
            )
            .1;

        // TODO: make this logic less brittle
        let left_sum =
            w[0] * l_inst + w[1] * rd_wa + w[2] * rd_wa + w[3] * lookup_out + w[4] * j_flag;
        let right_sum = w[0] * r_inst
            + w[1] * wl_flag
            + w[2] * j_flag
            + w[3] * branch_flag
            + w[4] * (F::one() - next_is_noop);

        left_sum + self.gamma * right_sum
    }
    // No opening point or new openings
    fn normalize_opening_point(
        &self,
        _opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(Vec::new())
    }
    fn cache_openings_prover(
        &self,
        _accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        _transcript: &mut T,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
    }
    fn cache_openings_verifier(
        &self,
        _accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        _transcript: &mut T,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
    }
    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}
