use allocative::Allocative;
use std::cell::RefCell;
use std::rc::Rc;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::lagrange_poly::LagrangeHelper;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{
    OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
};
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck::{SumcheckInstance, UniSkipFirstRoundInstance};
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::utils::univariate_skip::{build_uniskip_first_round_poly, uniskip_targets};
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::instruction::{CircuitFlags, InstructionFlags};
use crate::zkvm::r1cs::inputs::generate_virtual_product_witnesses;
use crate::zkvm::witness::VirtualPolynomial;
use rayon::prelude::*;

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

    pub fn get_sumcheck_id(&self) -> SumcheckId {
        match self {
            VirtualProductType::Instruction => SumcheckId::ProductVirtualization,
            VirtualProductType::WriteLookupOutputToRD => {
                SumcheckId::WriteLookupOutputToRDVirtualization
            }
            VirtualProductType::WritePCtoRD => SumcheckId::WritePCtoRDVirtualization,
            VirtualProductType::ShouldBranch => SumcheckId::ShouldBranchVirtualization,
            VirtualProductType::ShouldJump => SumcheckId::ShouldJumpVirtualization,
        }
    }
}

#[derive(Allocative)]
struct ProductVirtualizationSumcheckProverState<F: JoltField> {
    left_input_poly: MultilinearPolynomial<F>,
    right_input_poly: MultilinearPolynomial<F>,
    eq_r_cycle: GruenSplitEqPolynomial<F>,
}

#[derive(Allocative)]
pub struct ProductVirtualizationSumcheck<F: JoltField> {
    #[allocative(skip)]
    product_type: VirtualProductType,
    input_claim: F,
    log_T: usize,
    prover_state: Option<ProductVirtualizationSumcheckProverState<F>>,
}

impl<F: JoltField> ProductVirtualizationSumcheck<F> {
    #[tracing::instrument(skip_all, name = "ProductVirtualizationSumcheck::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        product_type: VirtualProductType,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (_, trace, _, _) = state_manager.get_prover_data();

        let (left_input_poly, right_input_poly) =
            generate_virtual_product_witnesses(product_type, trace);

        let accumulator = state_manager.get_prover_accumulator();
        let virtual_poly = product_type.get_virtual_polynomial();
        let (outer_sumcheck_r, input_claim) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(virtual_poly, SumcheckId::SpartanOuter);

        let (r_cycle, _rx_var) = outer_sumcheck_r.split_at(trace.len().log_2());

        Self {
            product_type,
            input_claim,
            log_T: r_cycle.len(),
            prover_state: Some(ProductVirtualizationSumcheckProverState {
                left_input_poly,
                right_input_poly,
                eq_r_cycle: GruenSplitEqPolynomial::new(&r_cycle.r, BindingOrder::LowToHigh),
            }),
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        product_type: VirtualProductType,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let accumulator = state_manager.get_verifier_accumulator();
        let virtual_poly = product_type.get_virtual_polynomial();
        let (_, input_claim) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(virtual_poly, SumcheckId::SpartanOuter);
        let (_, _, T) = state_manager.get_verifier_data();

        Self {
            product_type,
            input_claim,
            prover_state: None,
            log_T: T.log_2(),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for ProductVirtualizationSumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.log_T
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    #[tracing::instrument(
        skip_all,
        name = "ProductVirtualizationSumcheck::compute_prover_message"
    )]
    fn compute_prover_message(&mut self, _round: usize, previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        const DEGREE: usize = 3;

        let eq = &prover_state.eq_r_cycle;
        let left_input_poly = &prover_state.left_input_poly;
        let right_input_poly = &prover_state.right_input_poly;

        let quadratic_coeffs: [F; DEGREE - 1] = if eq.E_in_current_len() == 1 {
            // E_in is fully bound
            (0..eq.len() / 2)
                .into_par_iter()
                .map(|j| {
                    let eq_eval = eq.E_out_current()[j];
                    let left_evals = [
                        left_input_poly.get_bound_coeff(2 * j),
                        left_input_poly.get_bound_coeff(2 * j + 1),
                    ];
                    let right_evals = [
                        right_input_poly.get_bound_coeff(2 * j),
                        right_input_poly.get_bound_coeff(2 * j + 1),
                    ];

                    [
                        eq_eval * left_evals[0] * right_evals[0],
                        eq_eval
                            * (left_evals[1] - left_evals[0])
                            * (right_evals[1] - right_evals[0]),
                    ]
                })
                .reduce(
                    || [F::zero(); DEGREE - 1],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
        } else {
            // E_in has not been fully bound - use nested structure
            let num_x_in_bits = eq.E_in_current_len().log_2();
            let x_bitmask = (1 << num_x_in_bits) - 1;
            let chunk_size = 1 << num_x_in_bits;

            (0..eq.len() / 2)
                .collect::<Vec<_>>()
                .par_chunks(chunk_size)
                .enumerate()
                .map(|(x_out, chunk)| {
                    let E_out_eval = eq.E_out_current()[x_out];

                    let chunk_evals = chunk
                        .par_iter()
                        .map(|j| {
                            let x_in = j & x_bitmask;
                            let E_in_eval = eq.E_in_current()[x_in];
                            let left_evals = [
                                left_input_poly.get_bound_coeff(2 * j),
                                left_input_poly.get_bound_coeff(2 * j + 1),
                            ];
                            let right_evals = [
                                right_input_poly.get_bound_coeff(2 * j),
                                right_input_poly.get_bound_coeff(2 * j + 1),
                            ];

                            // Inner eq contribution
                            [
                                E_in_eval * left_evals[0] * right_evals[0],
                                E_in_eval
                                    * (left_evals[1] - left_evals[0])
                                    * (right_evals[1] - right_evals[0]),
                            ]
                        })
                        .reduce(
                            || [F::zero(); DEGREE - 1],
                            |running, new| [running[0] + new[0], running[1] + new[1]],
                        );

                    // Outer eq contribution
                    [E_out_eval * chunk_evals[0], E_out_eval * chunk_evals[1]]
                })
                .reduce(
                    || [F::zero(); DEGREE - 1],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
        };

        eq.gruen_evals_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim)
            .to_vec()
    }

    #[tracing::instrument(skip_all, name = "ProductVirtualizationSumcheck::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        prover_state.eq_r_cycle.bind(r_j);
        rayon::join(
            || {
                prover_state
                    .left_input_poly
                    .bind_parallel(r_j, BindingOrder::LowToHigh)
            },
            || {
                prover_state
                    .right_input_poly
                    .bind_parallel(r_j, BindingOrder::LowToHigh)
            },
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap().borrow();

        // Get r_cycle from the SpartanOuter sumcheck opening point
        let virtual_poly = self.product_type.get_virtual_polynomial();
        let (outer_sumcheck_opening, _) =
            accumulator.get_virtual_polynomial_opening(virtual_poly, SumcheckId::SpartanOuter);
        let outer_sumcheck_r = &outer_sumcheck_opening.r;
        let (r_cycle, _) = outer_sumcheck_r.split_at(self.log_T);

        let sumcheck_id = self.product_type.get_sumcheck_id();
        let FactorPolynomials(left_poly, right_poly) = self.product_type.get_factor_polynomials();
        let (_, left_eval) = accumulator.get_virtual_polynomial_opening(left_poly, sumcheck_id);
        let (_, mut right_eval) =
            accumulator.get_virtual_polynomial_opening(right_poly, sumcheck_id);
        // Special case (1 - NextIsNoop) for ShouldJump
        if let (VirtualProductType::ShouldJump, VirtualPolynomial::NextIsNoop) =
            (self.product_type, right_poly)
        {
            right_eval = F::one() - right_eval;
        }

        let eq_eval = EqPolynomial::mle(&r.iter().rev().copied().collect::<Vec<_>>(), r_cycle);
        eq_eval * left_eval * right_eval
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().rev().copied().collect())
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let left_input_eval = prover_state.left_input_poly.final_sumcheck_claim();
        let right_input_eval = prover_state.right_input_poly.final_sumcheck_claim();

        let sumcheck_id = self.product_type.get_sumcheck_id();
        let FactorPolynomials(left_poly, right_poly) = self.product_type.get_factor_polynomials();
        accumulator.borrow_mut().append_virtual(
            transcript,
            left_poly,
            sumcheck_id,
            opening_point.clone(),
            left_input_eval,
        );
        // Special case (1 - NextIsNoop) for ShouldJump
        let right_eval = if let (VirtualProductType::ShouldJump, VirtualPolynomial::NextIsNoop) =
            (self.product_type, right_poly)
        {
            F::one() - right_input_eval
        } else {
            right_input_eval
        };
        accumulator.borrow_mut().append_virtual(
            transcript,
            right_poly,
            sumcheck_id,
            opening_point,
            right_eval,
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let sumcheck_id = self.product_type.get_sumcheck_id();
        let FactorPolynomials(left_poly, right_poly) = self.product_type.get_factor_polynomials();
        accumulator.borrow_mut().append_virtual(
            transcript,
            left_poly,
            sumcheck_id,
            opening_point.clone(),
        );
        accumulator
            .borrow_mut()
            .append_virtual(transcript, right_poly, sumcheck_id, opening_point);
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

// NEW: univariate skip for product virtualization, fusing 5 sumchecks into one and reducing memory usage
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

const NUM_PRODUCT_VIRTUAL: usize = 5;
const PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE: usize = NUM_PRODUCT_VIRTUAL;
const PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE: usize = NUM_PRODUCT_VIRTUAL - 1;
const PRODUCT_VIRTUAL_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE: usize =
    2 * PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE + 1;
const PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS: usize =
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
    pub input_claim: F,
    pub r0_uniskip: F::Challenge,
    pub total_rounds: usize,
    /// Optional verifier-only state (e.g., tau vector when needed on verifier)
    pub tau: Option<Vec<F::Challenge>>,
    /// Prover-only state (None on verifier)
    pub prover_state: Option<ProductVirtualRemainderProverState<F>>,
}

#[derive(Clone, Debug)]
pub struct ProductVirtualRemainderProverState<F: JoltField> {
    left: DensePolynomial<F>,
    right: DensePolynomial<F>,
}

impl<F: JoltField> ProductVirtualRemainder<F> {
    #[tracing::instrument(skip_all, name = "ProductVirtualRemainder::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _input_claim: F,
        _r0_uniskip: F::Challenge,
        _total_rounds: usize,
    ) -> Self {
        todo!()
    }

    pub fn new_verifier(
        _input_claim: F,
        _r0_uniskip: F::Challenge,
        _total_rounds: usize,
        _tau: Vec<F::Challenge>,
    ) -> Self {
        todo!()
    }

    /// Optional helper to compute any per-round cached values immediately after uni-skip.
    fn compute_streaming_round_cache(&self) {
        todo!()
    }

    /// Optional helper to bind the first remaining round using cached values.
    fn bind_streaming_round(&mut self, _r_0: F::Challenge) {
        todo!()
    }

    /// Compute the quadratic endpoints for remaining rounds.
    fn remaining_quadratic_evals(&self) -> (F, F) {
        todo!()
    }

    /// Returns final per-virtual-polynomial evaluations needed for openings.
    pub fn final_sumcheck_evals(&self) -> [F; 2] {
        todo!()
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for ProductVirtualRemainder<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.total_rounds
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        todo!()
    }

    fn bind(&mut self, _r_j: F::Challenge, _round: usize) {
        todo!()
    }

    fn expected_output_claim(
        &self,
        _accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        _r_tail: &[F::Challenge],
    ) -> F {
        todo!()
    }

    fn normalize_opening_point(&self, _r_tail: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        todo!()
    }

    fn cache_openings_prover(
        &self,
        _accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        _transcript: &mut T,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        todo!()
    }

    fn cache_openings_verifier(
        &self,
        _accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        _transcript: &mut T,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        todo!()
    }
}
