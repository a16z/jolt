use allocative::Allocative;
use ark_std::Zero;
use std::cell::RefCell;
use std::rc::Rc;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::lagrange_poly::{LagrangeHelper, LagrangePolynomial};
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial};
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
use crate::utils::math::Math;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::instruction::{CircuitFlags, InstructionFlags};
use crate::zkvm::r1cs::inputs::{
    compute_claimed_product_virtual_evals, generate_virtual_product_witnesses, ProductCycleInputs,
};
use crate::zkvm::witness::VirtualPolynomial;
use crate::zkvm::JoltSharedPreprocessing;
use rayon::prelude::*;
use std::sync::Arc;
use tracer::instruction::Cycle;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Allocative, EnumIter)]
pub enum VirtualProductType {
    /// LeftInstructionInput × RightInstructionInput
    Instruction,
    /// rd_addr × WriteLookupOutputToRD_flag
    WriteLookupOutputToRD,
    /// rd_addr × Jump_flag
    WritePCtoRD,
    /// lookup_output × Branch_flag
    ShouldBranch,
    /// Jump_flag × (1 - NextIsNoop)
    ShouldJump,
}

#[derive(Debug, Clone, Copy)]
pub struct FactorPolynomials(VirtualPolynomial, VirtualPolynomial);

impl VirtualProductType {
    pub fn get_virtual_polynomial(&self) -> VirtualPolynomial {
        match self {
            VirtualProductType::Instruction => VirtualPolynomial::Product,
            VirtualProductType::WriteLookupOutputToRD => VirtualPolynomial::WriteLookupOutputToRD,
            VirtualProductType::WritePCtoRD => VirtualPolynomial::WritePCtoRD,
            VirtualProductType::ShouldBranch => VirtualPolynomial::ShouldBranch,
            VirtualProductType::ShouldJump => VirtualPolynomial::ShouldJump,
        }
    }
    pub fn get_factor_polynomials(&self) -> FactorPolynomials {
        match self {
            VirtualProductType::Instruction => FactorPolynomials(
                VirtualPolynomial::LeftInstructionInput,
                VirtualPolynomial::RightInstructionInput,
            ),
            VirtualProductType::WriteLookupOutputToRD => FactorPolynomials(
                VirtualPolynomial::RdWa,
                VirtualPolynomial::OpFlags(CircuitFlags::WriteLookupOutputToRD),
            ),
            VirtualProductType::WritePCtoRD => FactorPolynomials(
                VirtualPolynomial::RdWa,
                VirtualPolynomial::OpFlags(CircuitFlags::Jump),
            ),
            VirtualProductType::ShouldBranch => FactorPolynomials(
                VirtualPolynomial::LookupOutput,
                VirtualPolynomial::InstructionFlags(InstructionFlags::Branch),
            ),
            VirtualProductType::ShouldJump => FactorPolynomials(
                VirtualPolynomial::OpFlags(CircuitFlags::Jump),
                VirtualPolynomial::NextIsNoop,
            ),
        }
    }
}

// Product virtualization with univariate skip, fusing 5 product claims into one and reducing memory usage
// Eventually will supersede the above (old) implementation.
// For now limit the addition logic to this file (and maybe helper files like univariate_skip.rs and inputs.rs).

// Idea: we define a "combined" left and right polynomial
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
    /// Ordering matches outer: variables are MSB→LSB with τ_high last
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
        for (i, pt) in VirtualProductType::iter().enumerate() {
            let vp = pt.get_virtual_polynomial();
            let (_, eval) = acc
                .borrow()
                .get_virtual_polynomial_opening(vp, SumcheckId::SpartanOuter);
            base_evals[i] = eval;
        }

        println!("base_evals: {:?}", base_evals);

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
        for (i, pt) in VirtualProductType::iter().enumerate() {
            let vp = pt.get_virtual_polynomial();
            let (_, eval) = acc
                .borrow()
                .get_virtual_polynomial_opening(vp, SumcheckId::SpartanOuter);
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
    /// Split-Eq logic (same as outer.rs):
    /// - Split τ_low into (τ_out, τ_in) with m = |τ_low|/2. Precompute
    ///   E_out = EqPolynomial::evals(τ_out), E_in = EqPolynomial::evals(τ_in).
    /// - For each z target, compute
    ///     t1(z) = Σ_{x_out} E_out[x_out] · Σ_{x_in} E_in[x_in] · left_z(x) · right_z(x),
    ///   where x is the concatenation of (x_out || x_in) in MSB→LSB order.
    ///
    /// Lagrange fusion per target z on extended window {−4,−3,3,4}:
    /// - Compute c[0..4] = LagrangeHelper::shift_coeffs_i32(shift(z)) using the same shifted-kernel
    ///   as outer.rs (indices correspond to the 5 base points).
    /// - Define fused values at this z by linearly combining the 5 product witnesses with c:
    ///     left_z(x)  = Σ_i c[i] · Left_i(x)
    ///     right_z(x) = Σ_i c[i] · Right_i^eff(x)
    ///   with Right_4^eff(x) = 1 − NextIsNoop(x) for the ShouldJump term only.
    ///
    /// Small-value lifting rules for integer accumulation before converting to the field:
    /// - Instruction: LeftInstructionInput is u64 → lift to i128; RightInstructionInput is S64 → i128.
    /// - WriteLookupOutputToRD: RdWa is u8 → i32; flag is bool/u8 → i32.
    /// - WritePCtoRD: RdWa is u8 → i32; Jump flag is bool/u8 → i32.
    /// - ShouldBranch: LookupOutput is u64 → i128; Branch flag is bool/u8 → i32.
    /// - ShouldJump: Jump flag (left) is bool/u8 → i32; Right^eff = (1 − NextIsNoop) is bool/u8 → i32.
    ///
    /// Implementation guidance:
    /// - For each z, form left_z and right_z by first summing with integer accumulators of the
    ///   indicated width (i32 for {bool,u8}, i128 for {u64,S64,i128}).
    /// - Convert these fused sums to field elements and then multiply left_z · right_z in the field.
    /// - Apply the split-Eq weights E_out and E_in exactly as in outer.rs and sum over x.
    /// - Return t1(z) for all extended targets in the order of uniskip_targets.
    fn compute_univariate_skip_extended_evals(
        trace: &[Cycle],
        tau_low: &[F::Challenge],
    ) -> [F; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE] {
        // Split-Eq over cycle variables
        let m = tau_low.len() / 2;
        let (tau_out, tau_in) = tau_low.split_at(m);
        let (E_out, E_in) = rayon::join(
            || EqPolynomial::evals(tau_out),
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
                let mut local_acc: [F; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE] =
                    [F::zero(); PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE];

                for x_out_val in x_out_start..x_out_end {
                    let e_out = E_out[x_out_val];
                    // Delayed reduction accumulators over x_in per target j
                    let mut inner_acc: [F::Unreduced<9>; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE] =
                        [F::Unreduced::<9>::zero(); PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE];
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

                            // Weighted per-product left components (i128)
                            let mut left_w: [i128; NUM_PRODUCT_VIRTUAL] = [0; NUM_PRODUCT_VIRTUAL];
                            left_w[0] = (c[0] as i128) * (row.instruction_left_input as i128); // u64 -> i128
                            left_w[1] = (c[1] as i128)
                                * (row.write_lookup_output_to_rd_rd_addr as i32 as i128); // u8 -> i32 -> i128
                            left_w[2] =
                                (c[2] as i128) * (row.write_pc_to_rd_rd_addr as i32 as i128); // u8 -> i32 -> i128
                            left_w[3] = (c[3] as i128) * (row.should_branch_lookup_output as i128); // u64 -> i128
                            left_w[4] = (c[4] as i128)
                                * (if row.should_jump_flag { 1i32 } else { 0i32 } as i128); // bool/u8 -> i32 -> i128

                            // Weighted per-product right components with Right_4^eff (i128)
                            let mut right_w: [i128; NUM_PRODUCT_VIRTUAL] = [0; NUM_PRODUCT_VIRTUAL];
                            right_w[0] = (c[0] as i128) * row.instruction_right_input; // S64 -> i128
                            right_w[1] = (c[1] as i128)
                                * (if row.write_lookup_output_to_rd_flag {
                                    1i32
                                } else {
                                    0i32
                                } as i128); // bool/u8 -> i32 -> i128
                            right_w[2] = (c[2] as i128)
                                * (if row.write_pc_to_rd_flag { 1i32 } else { 0i32 } as i128); // bool/u8 -> i32 -> i128
                            right_w[3] = (c[3] as i128)
                                * (if row.should_branch_flag { 1i32 } else { 0i32 } as i128); // bool/u8 -> i32 -> i128
                            right_w[4] = (c[4] as i128)
                                * (if row.not_next_noop { 1i32 } else { 0i32 } as i128); // (1-NextIsNoop) -> i32 -> i128

                            // Convert per-product to field once and fuse by summing over i
                            let mut left_sum = F::zero();
                            let mut right_sum = F::zero();
                            for i in 0..NUM_PRODUCT_VIRTUAL {
                                left_sum += F::from_i128(left_w[i]);
                                right_sum += F::from_i128(right_w[i]);
                            }
                            // Delayed reduction: multiply by E_in unreduced, reduce later, then apply E_out
                            let prod = left_sum * right_sum;
                            inner_acc[j] += e_in.mul_unreduced::<9>(prod);
                        }
                    }
                    // Reduce inner accumulators and apply E_out
                    for j in 0..PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE {
                        let reduced = F::from_montgomery_reduce::<9>(inner_acc[j]);
                        local_acc[j] += e_out * reduced;
                    }
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

        #[cfg(debug_assertions)]
        {
            let w_dbg = LagrangePolynomial::<F>::evals::<
                F::Challenge,
                PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
            >(&tau_high);
            eprintln!(
                "[pv-uniskip] base_evals={:?} w_at_tau_high={:?}",
                base, w_dbg
            );
        }

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
        >(&uni.r0, &tau_high);

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
    /// Outer strategy (mirrored here): after binding the univariate-skip variable at r0, we must
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
        // Build witness polynomials for the five product types
        let product_types: Vec<VirtualProductType> = VirtualProductType::iter().collect();
        let witnesses: Vec<(MultilinearPolynomial<F>, MultilinearPolynomial<F>)> = product_types
            .iter()
            .map(|pt| generate_virtual_product_witnesses::<F>(*pt, trace))
            .collect();

        let num_x_out_vals = split_eq_poly.E_out_current_len();
        let num_x_in_vals = split_eq_poly.E_in_current_len();
        let iter_num_x_in_vars = num_x_in_vals.log_2();

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
            left_lo: Vec<F>,
            left_hi: Vec<F>,
            right_lo: Vec<F>,
            right_hi: Vec<F>,
            sum0_unr: F::Unreduced<9>,
            sum_inf_unr: F::Unreduced<9>,
        }

        let chunk_results: Vec<ChunkOut<F>> = (0..num_parallel_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let x_out_start = chunk_idx * x_out_chunk_size;
                let x_out_end = core::cmp::min((chunk_idx + 1) * x_out_chunk_size, num_x_out_vals);
                let chunk_width = x_out_end.saturating_sub(x_out_start);
                let local_len = chunk_width.saturating_mul(num_x_in_vals);
                let mut local_left_lo: Vec<F> = unsafe_allocate_zero_vec(local_len);
                let mut local_left_hi: Vec<F> = unsafe_allocate_zero_vec(local_len);
                let mut local_right_lo: Vec<F> = unsafe_allocate_zero_vec(local_len);
                let mut local_right_hi: Vec<F> = unsafe_allocate_zero_vec(local_len);

                let mut task_sum0 = F::Unreduced::<9>::zero();
                let mut task_sum_inf = F::Unreduced::<9>::zero();
                let mut local_idx = 0usize;
                for x_out_val in x_out_start..x_out_end {
                    let mut inner0 = F::Unreduced::<9>::zero();
                    let mut inner_inf = F::Unreduced::<9>::zero();
                    for x_in_val in 0..num_x_in_vals {
                        let current_step_idx = (x_out_val << iter_num_x_in_vars) | x_in_val;

                        // Compute fused left/right at 0 and 1 for this group
                        let mut left0 = F::zero();
                        let mut left1 = F::zero();
                        let mut right0 = F::zero();
                        let mut right1 = F::zero();
                        for (i, (l, r)) in witnesses.iter().enumerate() {
                            let l0_i = l.get_bound_coeff(2 * current_step_idx);
                            let l1_i = l.get_bound_coeff(2 * current_step_idx + 1);
                            let mut r0_i = r.get_bound_coeff(2 * current_step_idx);
                            let mut r1_i = r.get_bound_coeff(2 * current_step_idx + 1);
                            // ShouldJump: use (1 - NextIsNoop)
                            if product_types[i] == VirtualProductType::ShouldJump {
                                r0_i = F::one() - r0_i;
                                r1_i = F::one() - r1_i;
                            }
                            let w = weights_at_r0[i];
                            left0 += w * l0_i;
                            left1 += w * l1_i;
                            right0 += w * r0_i;
                            right1 += w * r1_i;
                        }

                        let e_in = if num_x_in_vals == 0 {
                            F::one()
                        } else if num_x_in_vals == 1 {
                            split_eq_poly.E_in_current()[0]
                        } else {
                            split_eq_poly.E_in_current()[x_in_val]
                        };
                        let p0 = left0 * right0;
                        let slope = (left1 - left0) * (right1 - right0);
                        inner0 += e_in.mul_unreduced::<9>(p0);
                        inner_inf += e_in.mul_unreduced::<9>(slope);
                        local_left_lo[local_idx] = left0;
                        local_right_lo[local_idx] = right0;
                        local_left_hi[local_idx] = left1;
                        local_right_hi[local_idx] = right1;
                        local_idx += 1;
                    }
                    let e_out = if num_x_out_vals > 0 {
                        split_eq_poly.E_out_current()[x_out_val]
                    } else {
                        F::zero()
                    };
                    let reduced0 = F::from_montgomery_reduce::<9>(inner0);
                    let reduced_inf = F::from_montgomery_reduce::<9>(inner_inf);
                    task_sum0 += e_out.mul_unreduced::<9>(reduced0);
                    task_sum_inf += e_out.mul_unreduced::<9>(reduced_inf);
                }
                ChunkOut {
                    base: x_out_start.saturating_mul(num_x_in_vals),
                    len: local_left_lo.len(),
                    left_lo: local_left_lo,
                    left_hi: local_left_hi,
                    right_lo: local_right_lo,
                    right_hi: local_right_hi,
                    sum0_unr: task_sum0,
                    sum_inf_unr: task_sum_inf,
                }
            })
            .collect();

        let groups_exact = num_x_out_vals.saturating_mul(num_x_in_vals);
        let mut left_lo: Vec<F> = unsafe_allocate_zero_vec(groups_exact);
        let mut left_hi: Vec<F> = unsafe_allocate_zero_vec(groups_exact);
        let mut right_lo: Vec<F> = unsafe_allocate_zero_vec(groups_exact);
        let mut right_hi: Vec<F> = unsafe_allocate_zero_vec(groups_exact);
        let mut t0_acc_unr = F::Unreduced::<9>::zero();
        let mut t_inf_acc_unr = F::Unreduced::<9>::zero();
        for chunk in &chunk_results {
            t0_acc_unr += chunk.sum0_unr;
            t_inf_acc_unr += chunk.sum_inf_unr;
            let dst_base = chunk.base;
            let dst_end = dst_base + chunk.len;
            left_lo[dst_base..dst_end].copy_from_slice(&chunk.left_lo);
            left_hi[dst_base..dst_end].copy_from_slice(&chunk.left_hi);
            right_lo[dst_base..dst_end].copy_from_slice(&chunk.right_lo);
            right_hi[dst_base..dst_end].copy_from_slice(&chunk.right_hi);
        }

        ProductVirtualStreamingCache {
            t0: F::from_montgomery_reduce::<9>(t0_acc_unr),
            t_inf: F::from_montgomery_reduce::<9>(t_inf_acc_unr),
            left_lo,
            left_hi,
            right_lo,
            right_hi,
        }
    }

    /// Optional helper to bind the first remaining round using cached values.
    fn bind_streaming_round(&mut self, r_0: F::Challenge) {
        if let Some(ps) = self.prover_state.as_mut() {
            if let Some(cache) = ps.streaming_cache.take() {
                let groups = cache.left_lo.len();
                let mut left_bound: Vec<F> = unsafe_allocate_zero_vec(groups);
                let mut right_bound: Vec<F> = unsafe_allocate_zero_vec(groups);

                // Parallelize by x_out chunks to mirror outer binding performance
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
                    left: Vec<F>,
                    right: Vec<F>,
                }

                let chunks: Vec<BoundChunk<F>> = (0..chunk_threads)
                    .into_par_iter()
                    .map(|chunk_idx| {
                        let x_out_start = chunk_idx * chunk_size;
                        if x_out_start >= num_x_out_vals {
                            return BoundChunk {
                                base: 0,
                                left: Vec::new(),
                                right: Vec::new(),
                            };
                        }
                        let x_out_end = core::cmp::min(x_out_start + chunk_size, num_x_out_vals);
                        let local_len = (x_out_end - x_out_start).saturating_mul(num_x_in_vals);
                        let mut local_left: Vec<F> = Vec::with_capacity(local_len);
                        let mut local_right: Vec<F> = Vec::with_capacity(local_len);
                        for xo in x_out_start..x_out_end {
                            for xi in 0..num_x_in_vals {
                                let idx = xo * num_x_in_vals + xi;
                                let l0 = cache.left_lo[idx];
                                let l1 = cache.left_hi[idx];
                                let r0 = cache.right_lo[idx];
                                let r1 = cache.right_hi[idx];
                                local_left.push(l0 + r_0 * (l1 - l0));
                                local_right.push(r0 + r_0 * (r1 - r0));
                            }
                        }
                        BoundChunk {
                            base: x_out_start * num_x_in_vals,
                            left: local_left,
                            right: local_right,
                        }
                    })
                    .collect();

                for c in &chunks {
                    let end = c.base + c.left.len();
                    left_bound[c.base..end].copy_from_slice(&c.left);
                    right_bound[c.base..end].copy_from_slice(&c.right);
                }

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

        // Lagrange weights at r0
        let w = LagrangePolynomial::<F>::evals::<
            F::Challenge,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&self.r0_uniskip);

        // Per-type evals at fused SumcheckId
        let mut left_eval = F::zero();
        let mut right_eval = F::zero();
        for (i, pt) in VirtualProductType::iter().enumerate() {
            let FactorPolynomials(lp, rp) = pt.get_factor_polynomials();
            let (_, le) = acc.get_virtual_polynomial_opening(lp, SumcheckId::ProductVirtualization);
            let (_, mut re) =
                acc.get_virtual_polynomial_opening(rp, SumcheckId::ProductVirtualization);
            if matches!(pt, VirtualProductType::ShouldJump) {
                re = F::one() - re;
            }
            left_eval += w[i] * le;
            right_eval += w[i] * re;
        }

        // Multiply by L(τ_high, r0) and Eq(τ_low, r_tail^rev)
        let tau_high = &self.tau[self.tau.len() - 1];
        let tau_high_bound_r0 = LagrangePolynomial::<F>::lagrange_kernel::<
            F::Challenge,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(tau_high, &self.r0_uniskip);
        let tau_low = &self.tau[..self.tau.len() - 1];
        let r_tail_reversed: Vec<F::Challenge> = r_tail.iter().rev().copied().collect();
        let tau_bound_r_tail_reversed = EqPolynomial::mle(tau_low, &r_tail_reversed);

        tau_high_bound_r0 * tau_bound_r_tail_reversed * left_eval * right_eval
    }

    fn normalize_opening_point(&self, r_tail: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        // Reverse the challenge (since we bind in small endian), then return
        // Note: we do not need to append the univariate skip challenge
        let r_reversed: Vec<F::Challenge> = r_tail.iter().rev().copied().collect();
        OpeningPoint::new(r_reversed)
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        // Append 10 virtual openings (5 left, 5 right) under fused ID at r_cycle
        let (r_cycle, _) = opening_point.r.split_at(self.num_cycle_vars);
        let r_cycle_op = OpeningPoint::new(r_cycle.to_vec());

        // Compute claimed product-virtual evaluations at r_cycle in one pass
        let claims = {
            let ps = self.prover_state.as_ref().expect("prover state missing");
            compute_claimed_product_virtual_evals::<F>(&ps.trace, r_cycle)
        };

        // Map claims to virtual polynomials in the enum order
        for (i, pt) in VirtualProductType::iter().enumerate() {
            let FactorPolynomials(lp, rp) = pt.get_factor_polynomials();
            let left_claim = claims[2 * i];
            let right_claim = claims[2 * i + 1];
            let mut acc = accumulator.borrow_mut();
            acc.append_virtual(
                transcript,
                lp,
                SumcheckId::ProductVirtualization,
                r_cycle_op.clone(),
                left_claim,
            );
            acc.append_virtual(
                transcript,
                rp,
                SumcheckId::ProductVirtualization,
                r_cycle_op.clone(),
                right_claim,
            );
        }
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        // Append the same 10 virtual openings (no claims) under fused ID at opening point
        let mut acc = accumulator.borrow_mut();
        for pt in VirtualProductType::iter() {
            let FactorPolynomials(lp, rp) = pt.get_factor_polynomials();
            acc.append_virtual(
                transcript,
                lp,
                SumcheckId::ProductVirtualization,
                opening_point.clone(),
            );
            acc.append_virtual(
                transcript,
                rp,
                SumcheckId::ProductVirtualization,
                opening_point.clone(),
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}
