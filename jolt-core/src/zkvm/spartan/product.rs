// use allocative::Allocative;
use std::cell::RefCell;
use std::rc::Rc;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::lagrange_poly::{LagrangeHelper, LagrangePolynomial};
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialEvaluation};
use crate::poly::opening_proof::{
    OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
};
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck::{SumcheckInstance, UniSkipFirstRoundInstance};
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::subprotocols::univariate_skip::{build_uniskip_first_round_poly, uniskip_targets, UniSkipState};
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::instruction::{CircuitFlags, InstructionFlags};
use crate::zkvm::r1cs::inputs::generate_virtual_product_witnesses;
use crate::zkvm::witness::VirtualPolynomial;
use crate::zkvm::JoltSharedPreprocessing;
use rayon::prelude::*;
use std::sync::Arc;
use tracer::instruction::Cycle;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "allocative", derive(Allocative))]
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

// First, we define our constants

pub const NUM_PRODUCT_VIRTUAL: usize = 5;
pub const PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE: usize = NUM_PRODUCT_VIRTUAL;
pub const PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE: usize = NUM_PRODUCT_VIRTUAL - 1;
pub const PRODUCT_VIRTUAL_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE: usize =
    2 * PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE + 1;
pub const PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS: usize =
    3 * PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE + 1;

/// Uni-skip instance for product virtualization, computing the first-round polynomial only.
pub struct ProductVirtualUniSkipInstance<F: JoltField> {
    tau: Vec<F::Challenge>,
    /// Prover-only state (None on verifier)
    prover_state: Option<ProductVirtualUniSkipProverState<F>>,
}

#[derive(Clone, Debug)]
struct ProductVirtualUniSkipProverState<F: JoltField> {
    /// Evaluations of t1(Z) at the extended univariate-skip targets (outside base window)
    extended_evals: [F; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE],
}

impl<F: JoltField> ProductVirtualUniSkipInstance<F> {
    #[tracing::instrument(skip_all, name = "ProductVirtualUniSkipInstance::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        tau: &[F::Challenge],
    ) -> Self {
        let (_preprocessing, trace, _program_io, _final_mem) = state_manager.get_prover_data();

        let tau_low = &tau[..tau.len() - 1];
        let extended = Self::compute_univariate_skip_extended_evals(trace, tau_low);

        Self {
            tau: tau.to_vec(),
            prover_state: Some(ProductVirtualUniSkipProverState {
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

    fn compute_univariate_skip_extended_evals(
        trace: &[tracer::instruction::Cycle],
        tau_low: &[F::Challenge],
    ) -> [F; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE] {
        // Build five (left, right) polynomials in the specified order
        let product_types = [
            VirtualProductType::Instruction,
            VirtualProductType::WriteLookupOutputToRD,
            VirtualProductType::WritePCtoRD,
            VirtualProductType::ShouldBranch,
            VirtualProductType::ShouldJump,
        ];

        let witnesses: Vec<(MultilinearPolynomial<F>, MultilinearPolynomial<F>)> = product_types
            .iter()
            .map(|pt| generate_virtual_product_witnesses::<F>(*pt, trace))
            .collect();

        // Eq table over cycle variables
        let eq = EqPolynomial::evals(tau_low);
        let t_len = eq.len();

        // Precompute M[i][j] = Σ_x eq[x] * left_i[x] * right_j[x]
        let mut M: [[F; NUM_PRODUCT_VIRTUAL]; NUM_PRODUCT_VIRTUAL] =
            [[F::zero(); NUM_PRODUCT_VIRTUAL]; NUM_PRODUCT_VIRTUAL];

        for x in 0..t_len {
            let e = eq[x];
            // Cache left/right values at x for all i
            let mut left_vals: [F; NUM_PRODUCT_VIRTUAL] = [F::zero(); NUM_PRODUCT_VIRTUAL];
            let mut right_vals: [F; NUM_PRODUCT_VIRTUAL] = [F::zero(); NUM_PRODUCT_VIRTUAL];
            for (i, (l, r)) in witnesses.iter().enumerate() {
                left_vals[i] = l.get_coeff(x);
                right_vals[i] = r.get_coeff(x);
            }
            for i in 0..NUM_PRODUCT_VIRTUAL {
                let li = left_vals[i];
                for j in 0..NUM_PRODUCT_VIRTUAL {
                    let rj = right_vals[j];
                    M[i][j] += e * li * rj;
                }
            }
        }

        // Compute t1(z) for extended domain targets outside base window using c^T M c
        let base_left: i64 = -((PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE as i64 - 1) / 2);
        let targets = uniskip_targets::<
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE,
        >();
        let mut out: [F; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE] =
            [F::zero(); PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE];

        for (idx, &z) in targets.iter().enumerate() {
            let shift = z - base_left;
            let coeffs_i32 = LagrangeHelper::shift_coeffs_i32::<
                PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
            >(shift);
            let mut c: [F; NUM_PRODUCT_VIRTUAL] = [F::zero(); NUM_PRODUCT_VIRTUAL];
            for i in 0..NUM_PRODUCT_VIRTUAL {
                c[i] = F::from_i64(coeffs_i32[i] as i64);
            }

            // Compute c^T M c
            let mut acc = F::zero();
            for i in 0..NUM_PRODUCT_VIRTUAL {
                let ci = c[i];
                let mut row_acc = F::zero();
                for j in 0..NUM_PRODUCT_VIRTUAL {
                    row_acc += M[i][j] * c[j];
                }
                acc += ci * row_acc;
            }
            out[idx] = acc;
        }

        out
    }
}

impl<F: JoltField, T: Transcript> UniSkipFirstRoundInstance<F, T>
    for ProductVirtualUniSkipInstance<F>
{
    const DEGREE_BOUND: usize = PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS - 1;
    const DOMAIN_SIZE: usize = NUM_PRODUCT_VIRTUAL;

    fn input_claim(&self) -> F {
        F::zero()
    }

    fn compute_poly(&mut self) -> UniPoly<F> {
        // Load extended univariate-skip evaluations from prover state (prover only)
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
        >(&extended, tau_high)
    }
}

/// Remaining rounds for Product Virtualization after the univariate-skip first round.
/// Mirrors the structure of `OuterRemainingSumcheck` with product-virtualization-specific wiring.
pub struct ProductVirtualRemainder<F: JoltField> {
    /// Number of cycle bits to bind in this remainder (equals log2(T))
    pub num_cycles_bits: usize,
    /// The univariate-skip first round challenge r0
    pub r0_uniskip: F::Challenge,
    /// Claim after the univariate-skip first round, updated every round
    pub input_claim: F,
    /// The tau vector (length 1 + num_cycles_bits), available to prover and verifier
    pub tau: Vec<F::Challenge>,
    /// Prover-only state (None on verifier)
    pub prover_state: Option<ProductVirtualProverState<F>>,
}

#[derive(Clone, Debug)]
pub struct ProductVirtualStreamingCache<F: JoltField> {
    pub t0: F,
    pub t_inf: F,
    pub left_lo: Vec<F>,
    pub left_hi: Vec<F>,
    pub right_lo: Vec<F>,
    pub right_hi: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct ProductVirtualProverState<F: JoltField> {
    pub split_eq_poly: GruenSplitEqPolynomial<F>,
    pub preprocess: Arc<JoltSharedPreprocessing>,
    pub trace: Arc<Vec<Cycle>>,
    pub left: DensePolynomial<F>,
    pub right: DensePolynomial<F>,
    pub streaming_cache: Option<ProductVirtualStreamingCache<F>>,
}

impl<F: JoltField> ProductVirtualRemainder<F> {
    #[tracing::instrument(skip_all, name = "ProductVirtualRemainder::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        uni: &UniSkipState<F>,
    ) -> Self {
        let (preprocessing, trace, _program_io, _final_mem) = state_manager.get_prover_data();
        let num_cycles_bits = trace.len().log_2();

        // Build split-eq over cycle variables (no extra scaling needed here)
        let r_cycle = {
            let acc = state_manager.get_prover_accumulator();
            // Retrieve any outer-opened VP; use Product for consistency
            let (outer_opening, _) = acc.borrow().get_virtual_polynomial_opening(
                VirtualPolynomial::Product,
                SumcheckId::SpartanOuter,
            );
            let (r_cycle, _rx) = outer_opening.r.split_at(num_cycles_bits);
            r_cycle.to_vec()
        };
        let split_eq_poly = GruenSplitEqPolynomial::new(&r_cycle, BindingOrder::LowToHigh);

        // Compute Lagrange weights at r0 for fusing the 5 product types
        let weights = LagrangePolynomial::<F>::evals::<
            F::Challenge,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&uni.r0);

        let streaming_cache = Self::compute_streaming_round_cache(trace, &weights, &split_eq_poly);

        Self {
            num_cycles_bits,
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

    pub fn new_verifier(
        num_cycles_bits: usize,
        uni: UniSkipState<F>,
    ) -> Self {
        Self {
            num_cycles_bits,
            r0_uniskip: uni.r0,
            input_claim: uni.claim_after_first,
            tau: uni.tau,
            prover_state: None,
        }
    }

    /// Optional helper to compute any per-round cached values immediately after uni-skip.
    fn compute_streaming_round_cache(
        trace: &[Cycle],
        weights_at_r0: &[F; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE],
        split_eq_poly: &GruenSplitEqPolynomial<F>,
    ) -> ProductVirtualStreamingCache<F> {
        // Build witness polynomials for the five product types
        let product_types = [
            VirtualProductType::Instruction,
            VirtualProductType::WriteLookupOutputToRD,
            VirtualProductType::WritePCtoRD,
            VirtualProductType::ShouldBranch,
            VirtualProductType::ShouldJump,
        ];
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
                let mut local_left_lo: Vec<F> = Vec::with_capacity(local_len);
                let mut local_left_hi: Vec<F> = Vec::with_capacity(local_len);
                let mut local_right_lo: Vec<F> = Vec::with_capacity(local_len);
                let mut local_right_hi: Vec<F> = Vec::with_capacity(local_len);

                let mut task_sum0 = F::zero();
                let mut task_sum_inf = F::zero();
                for x_out_val in x_out_start..x_out_end {
                    let mut inner0 = F::zero();
                    let mut inner_inf = F::zero();
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
                        inner0 += e_in * left0 * right0;
                        inner_inf += e_in * (left1 - left0) * (right1 - right0);
                        local_left_lo.push(left0);
                        local_right_lo.push(right0);
                        local_left_hi.push(left1);
                        local_right_hi.push(right1);
                    }
                    let e_out = if num_x_out_vals > 0 {
                        split_eq_poly.E_out_current()[x_out_val]
                    } else {
                        F::zero()
                    };
                    task_sum0 += e_out * inner0;
                    task_sum_inf += e_out * inner_inf;
                }
                ChunkOut {
                    base: x_out_start.saturating_mul(num_x_in_vals),
                    len: local_left_lo.len(),
                    left_lo: local_left_lo,
                    left_hi: local_left_hi,
                    right_lo: local_right_lo,
                    right_hi: local_right_hi,
                    sum0: task_sum0,
                    sum_inf: task_sum_inf,
                }
            })
            .collect();

        let groups_exact = num_x_out_vals.saturating_mul(num_x_in_vals);
        let mut left_lo: Vec<F> = Vec::with_capacity(groups_exact);
        let mut left_hi: Vec<F> = Vec::with_capacity(groups_exact);
        let mut right_lo: Vec<F> = Vec::with_capacity(groups_exact);
        let mut right_hi: Vec<F> = Vec::with_capacity(groups_exact);
        left_lo.resize(groups_exact, F::zero());
        left_hi.resize(groups_exact, F::zero());
        right_lo.resize(groups_exact, F::zero());
        right_hi.resize(groups_exact, F::zero());
        let mut t0_acc = F::zero();
        let mut t_inf_acc = F::zero();
        for chunk in &chunk_results {
            t0_acc += chunk.sum0;
            t_inf_acc += chunk.sum_inf;
            let dst_base = chunk.base;
            let dst_end = dst_base + chunk.len;
            left_lo[dst_base..dst_end].copy_from_slice(&chunk.left_lo);
            left_hi[dst_base..dst_end].copy_from_slice(&chunk.left_hi);
            right_lo[dst_base..dst_end].copy_from_slice(&chunk.right_lo);
            right_hi[dst_base..dst_end].copy_from_slice(&chunk.right_hi);
        }

        ProductVirtualStreamingCache {
            t0: t0_acc,
            t_inf: t_inf_acc,
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
                let mut left_bound: Vec<F> = Vec::with_capacity(groups);
                let mut right_bound: Vec<F> = Vec::with_capacity(groups);
                for idx in 0..groups {
                    let l0 = cache.left_lo[idx];
                    let l1 = cache.left_hi[idx];
                    let r0 = cache.right_lo[idx];
                    let r1 = cache.right_hi[idx];
                    left_bound.push(l0 + r_0 * (l1 - l0));
                    right_bound.push(r0 + r_0 * (r1 - r0));
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
            (0..groups)
                .into_par_iter()
                .map(|g| {
                    let l0 = ps.left[2 * g];
                    let l1 = ps.left[2 * g + 1];
                    let r0 = ps.right[2 * g];
                    let r1 = ps.right[2 * g + 1];
                    let eq = eq_poly.E_out_current()[g];
                    let t0 = eq * l0 * r0;
                    let tinf = eq * (l1 - l0) * (r1 - r0);
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
                        let l0 = ps.left[2 * g];
                        let l1 = ps.left[2 * g + 1];
                        let r0 = ps.right[2 * g];
                        let r1 = ps.right[2 * g + 1];
                        let e_in = eq_poly.E_in_current()[x1];
                        inner0 += e_in * l0 * r0;
                        inner_inf += e_in * (l1 - l0) * (r1 - r0);
                    }
                    let e_out = eq_poly.E_out_current()[x2];
                    (e_out * inner0, e_out * inner_inf)
                })
                .reduce(|| (F::zero(), F::zero()), |a, b| (a.0 + b.0, a.1 + b.1))
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
        self.num_cycles_bits
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
        let product_types = [
            VirtualProductType::Instruction,
            VirtualProductType::WriteLookupOutputToRD,
            VirtualProductType::WritePCtoRD,
            VirtualProductType::ShouldBranch,
            VirtualProductType::ShouldJump,
        ];
        let mut left_eval = F::zero();
        let mut right_eval = F::zero();
        for (i, pt) in product_types.iter().enumerate() {
            let FactorPolynomials(lp, rp) = pt.get_factor_polynomials();
            let (_, le) =
                acc.get_virtual_polynomial_opening(lp, SumcheckId::ProductVirtualization);
            let (_, mut re) =
                acc.get_virtual_polynomial_opening(rp, SumcheckId::ProductVirtualization);
            if matches!(pt, VirtualProductType::ShouldJump) {
                re = F::one() - re;
            }
            left_eval += w[i] * le;
            right_eval += w[i] * re;
        }

        // Multiply by L(τ_high, r0) and Eq(τ_low, r_tail^rev), matching outer
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
        // Append 10 virtual openings (5 left, 5 right) under fused ID at r_cycle
        let (r_cycle, _) = opening_point.r.split_at(self.num_cycles_bits);
        let r_cycle_op = OpeningPoint::new(r_cycle.to_vec());

        // Rebuild per-type witness polynomials ephemerally and evaluate at r_cycle
        let trace = {
            let ps = self.prover_state.as_ref().expect("prover state missing");
            ps.trace.clone()
        };
        let product_types = [
            VirtualProductType::Instruction,
            VirtualProductType::WriteLookupOutputToRD,
            VirtualProductType::WritePCtoRD,
            VirtualProductType::ShouldBranch,
            VirtualProductType::ShouldJump,
        ];
        for pt in product_types.iter() {
            let (l, r) = generate_virtual_product_witnesses::<F>(*pt, &trace);
            let FactorPolynomials(lp, rp) = pt.get_factor_polynomials();

            // Evaluate bound polynomials at r_cycle and append claims
            let le = MultilinearPolynomial::<F>::evaluate(&l, &r_cycle);
            let mut re = MultilinearPolynomial::<F>::evaluate(&r, &r_cycle);
            if matches!(pt, VirtualProductType::ShouldJump) {
                re = F::one() - re;
            }
            let mut acc = accumulator.borrow_mut();
            acc.append_virtual(
                transcript,
                lp,
                SumcheckId::ProductVirtualization,
                r_cycle_op.clone(),
                le,
            );
            acc.append_virtual(
                transcript,
                rp,
                SumcheckId::ProductVirtualization,
                r_cycle_op.clone(),
                re,
            );
        }
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        // Append the same 10 virtual openings (no claims) under fused ID at r_cycle
        let (r_cycle, _) = opening_point.r.split_at(self.num_cycles_bits);
        let r_cycle_op = OpeningPoint::new(r_cycle.to_vec());
        let product_types = [
            VirtualProductType::Instruction,
            VirtualProductType::WriteLookupOutputToRD,
            VirtualProductType::WritePCtoRD,
            VirtualProductType::ShouldBranch,
            VirtualProductType::ShouldJump,
        ];
        let mut acc = accumulator.borrow_mut();
        for pt in product_types.iter() {
            let FactorPolynomials(lp, rp) = pt.get_factor_polynomials();
            acc.append_virtual(
                transcript,
                lp,
                SumcheckId::ProductVirtualization,
                r_cycle_op.clone(),
            );
            acc.append_virtual(
                transcript,
                rp,
                SumcheckId::ProductVirtualization,
                r_cycle_op.clone(),
            );
        }
    }
}
