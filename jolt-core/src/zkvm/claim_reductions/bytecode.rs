//! Two-phase bytecode claim reduction (Stage 6b cycle -> Stage 7 address).

use std::cell::RefCell;

use allocative::Allocative;
use rayon::prelude::*;

use crate::field::JoltField;
use crate::poly::commitment::dory::{DoryGlobals, DoryLayout};
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
#[cfg(feature = "zk")]
use crate::poly::opening_proof::OpeningId;
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::unipoly::UniPoly;
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::{InputClaimConstraint, OutputClaimConstraint, ValueSource};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::bytecode::chunks::committed_lanes;
use crate::zkvm::bytecode::read_raf_checking::BytecodeReadRafSumcheckParams;
use crate::zkvm::claim_reductions::{
    permute_precommitted_polys, precommitted_skip_round_scale, PrecommittedClaimReduction,
    PrecommittedPhase, PrecommittedSchedulingReference, TWO_PHASE_DEGREE_BOUND,
};
use crate::zkvm::instruction::{
    CircuitFlags, InstructionFlags, NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS,
};
use crate::zkvm::lookup_table::LookupTables;
use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use common::constants::{REGISTER_COUNT, XLEN};
use strum::EnumCount;

const NUM_VAL_STAGES: usize = 5;

fn debug_bytecode_reduction_enabled() -> bool {
    std::env::var("JOLT_DEBUG_BYTECODE_REDUCTION")
        .map(|v| {
            let value = v.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "" | "0" | "false" | "off")
        })
        .unwrap_or(false)
}

#[derive(Clone, Allocative)]
pub struct BytecodeClaimReductionParams<F: JoltField> {
    pub phase: PrecommittedPhase,
    pub precommitted: PrecommittedClaimReduction<F>,
    pub eta: F,
    pub eta_powers: [F; NUM_VAL_STAGES],
    /// Eq weights over high bytecode address bits (one per committed chunk).
    pub chunk_rbc_weights: Vec<F>,
    pub bytecode_T: usize,
    pub log_t: usize,
    /// Number of initial cycle rounds that must follow IncClaimReduction ordering.
    pub dense_cycle_prefix_vars: usize,
    pub bytecode_chunk_count: usize,
    pub bytecode_col_vars: usize,
    pub bytecode_row_vars: usize,
    pub r_bc: OpeningPoint<BIG_ENDIAN, F>,
    pub lane_weights: Vec<F>,
}

impl<F: JoltField> BytecodeClaimReductionParams<F> {
    pub fn new(
        bytecode_read_raf_params: &BytecodeReadRafSumcheckParams<F>,
        full_bytecode_len: usize,
        bytecode_chunk_count: usize,
        scheduling_reference: PrecommittedSchedulingReference,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let log_t = DoryGlobals::main_t().log_2();
        assert!(
            full_bytecode_len.is_multiple_of(bytecode_chunk_count),
            "bytecode chunk count ({bytecode_chunk_count}) must divide bytecode_len ({full_bytecode_len})"
        );
        let bytecode_t = (full_bytecode_len / bytecode_chunk_count).log_2();
        let bytecode_t_full = full_bytecode_len.log_2();

        let eta: F = transcript.challenge_scalar();
        let mut eta_powers = [F::one(); NUM_VAL_STAGES];
        for i in 1..NUM_VAL_STAGES {
            eta_powers[i] = eta_powers[i - 1] * eta;
        }

        let (r_bc_full, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::BytecodeReadRafAddrClaim,
            SumcheckId::BytecodeReadRafAddressPhase,
        );
        debug_assert_eq!(r_bc_full.r.len(), bytecode_t_full);
        let dropped_bits = bytecode_t_full - bytecode_t;
        let chunk_rbc_weights = if dropped_bits == 0 {
            vec![F::one()]
        } else {
            EqPolynomial::<F>::evals(&r_bc_full.r[..dropped_bits])
        };
        debug_assert_eq!(chunk_rbc_weights.len(), bytecode_chunk_count);
        let r_bc = OpeningPoint::new(r_bc_full.r[dropped_bits..].to_vec());

        let lane_weights = compute_lane_weights(bytecode_read_raf_params, accumulator, &eta_powers);

        // bytecode_K is the committed lane capacity (already next-power-of-two padded).
        let bytecode_k = committed_lanes();
        let total_vars = bytecode_k.log_2() + bytecode_t;
        // Bytecode uses its own balanced dimensions (independent from Main).
        // In Stage 8 it is embedded as a top-left block in Joint.
        let (bytecode_col_vars, bytecode_row_vars) = DoryGlobals::balanced_sigma_nu(total_vars);
        let precommitted = PrecommittedClaimReduction::new(
            total_vars,
            bytecode_row_vars,
            bytecode_col_vars,
            scheduling_reference,
        );
        // Align all precommitted scheduling/permutation to the shared reference domain.

        Self {
            phase: PrecommittedPhase::CycleVariables,
            precommitted,
            eta,
            eta_powers,
            chunk_rbc_weights,
            bytecode_T: bytecode_t,
            log_t,
            dense_cycle_prefix_vars: log_t,
            bytecode_chunk_count,
            bytecode_col_vars,
            bytecode_row_vars,
            r_bc,
            lane_weights,
        }
    }

    pub fn num_address_phase_rounds(&self) -> usize {
        self.precommitted.num_address_phase_rounds()
    }
}

impl<F: JoltField> BytecodeClaimReductionParams<F> {
    fn is_cycle_phase(&self) -> bool {
        self.phase == PrecommittedPhase::CycleVariables
    }

    fn is_cycle_phase_round(&self, round: usize) -> bool {
        self.precommitted.is_cycle_phase_round(round)
    }

    fn is_address_phase_round(&self, round: usize) -> bool {
        self.precommitted.is_address_phase_round(round)
    }

    fn cycle_alignment_rounds(&self) -> usize {
        self.precommitted.cycle_alignment_rounds()
    }

    fn address_alignment_rounds(&self) -> usize {
        self.precommitted.address_alignment_rounds()
    }

    pub fn transition_to_address_phase(&mut self) {
        self.phase = PrecommittedPhase::AddressVariables;
    }

    fn num_rounds_for_current_phase(&self) -> usize {
        self.precommitted
            .num_rounds_for_phase(self.is_cycle_phase())
    }

    pub fn round_offset(&self, max_num_rounds: usize) -> usize {
        self.precommitted
            .round_offset(self.is_cycle_phase(), max_num_rounds)
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for BytecodeClaimReductionParams<F> {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        match self.phase {
            PrecommittedPhase::CycleVariables => (0..NUM_VAL_STAGES)
                .map(|stage| {
                    let (_, val_claim) = accumulator.get_virtual_polynomial_opening(
                        VirtualPolynomial::BytecodeValStage(stage),
                        SumcheckId::BytecodeReadRafAddressPhase,
                    );
                    self.eta_powers[stage] * val_claim
                })
                .sum(),
            PrecommittedPhase::AddressVariables => {
                accumulator
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::BytecodeClaimReductionIntermediate,
                        SumcheckId::BytecodeClaimReductionCyclePhase,
                    )
                    .1
            }
        }
    }

    fn degree(&self) -> usize {
        TWO_PHASE_DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds_for_current_phase()
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        self.precommitted.normalize_opening_point(
            self.is_cycle_phase(),
            challenges,
            self.dense_cycle_prefix_vars,
        )
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        match self.phase {
            PrecommittedPhase::CycleVariables => {
                let openings: Vec<OpeningId> = (0..NUM_VAL_STAGES)
                    .map(|stage| {
                        OpeningId::virt(
                            VirtualPolynomial::BytecodeValStage(stage),
                            SumcheckId::BytecodeReadRafAddressPhase,
                        )
                    })
                    .collect();
                InputClaimConstraint::all_weighted_openings(&openings)
            }
            PrecommittedPhase::AddressVariables => InputClaimConstraint::direct(OpeningId::virt(
                VirtualPolynomial::BytecodeClaimReductionIntermediate,
                SumcheckId::BytecodeClaimReductionCyclePhase,
            )),
        }
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(&self, _: &dyn OpeningAccumulator<F>) -> Vec<F> {
        match self.phase {
            PrecommittedPhase::CycleVariables => self.eta_powers.to_vec(),
            PrecommittedPhase::AddressVariables => Vec::new(),
        }
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        match self.phase {
            PrecommittedPhase::CycleVariables => {
                Some(OutputClaimConstraint::direct(OpeningId::virt(
                    VirtualPolynomial::BytecodeClaimReductionIntermediate,
                    SumcheckId::BytecodeClaimReductionCyclePhase,
                )))
            }
            PrecommittedPhase::AddressVariables => {
                let terms = (0..self.bytecode_chunk_count)
                    .map(|chunk_idx| {
                        let opening = OpeningId::committed(
                            CommittedPolynomial::BytecodeChunk(chunk_idx),
                            SumcheckId::BytecodeClaimReduction,
                        );
                        (
                            ValueSource::Challenge(chunk_idx),
                            ValueSource::Opening(opening),
                        )
                    })
                    .collect();
                Some(OutputClaimConstraint::linear(terms))
            }
        }
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        match self.phase {
            PrecommittedPhase::CycleVariables => vec![],
            PrecommittedPhase::AddressVariables => {
                let eq_combined = evaluate_bytecode_eq_combined(self, sumcheck_challenges);
                let scale: F = precommitted_skip_round_scale(&self.precommitted);
                self.chunk_rbc_weights
                    .iter()
                    .map(|w| *w * eq_combined * scale)
                    .collect()
            }
        }
    }
}

#[derive(Allocative)]
pub struct BytecodeClaimReductionProver<F: JoltField> {
    params: BytecodeClaimReductionParams<F>,
    value_poly: MultilinearPolynomial<F>,
    eq_poly: MultilinearPolynomial<F>,
    scale: F,
    chunk_value_polys: Vec<MultilinearPolynomial<F>>,
    pending_round_poly: Option<UniPoly<F>>,
    running_claim: Option<F>,
}

impl<F: JoltField> BytecodeClaimReductionProver<F> {
    pub fn params(&self) -> &BytecodeClaimReductionParams<F> {
        &self.params
    }

    pub fn transition_to_address_phase(&mut self) {
        self.params.transition_to_address_phase();
    }

    pub fn initialize(
        params: BytecodeClaimReductionParams<F>,
        raw_chunk_polys: &[MultilinearPolynomial<F>],
    ) -> Self {
        let eq_cycle = EqPolynomial::<F>::evals(&params.r_bc.r);
        let eq_coeffs_template: Vec<F> = (0..raw_chunk_polys[0].len())
            .map(|idx| {
                let (lane, cycle) = native_index_to_lane_cycle(&params, idx);
                params.lane_weights[lane] * eq_cycle[cycle]
            })
            .collect();

        let raw_value_coeffs: Vec<F> = (0..raw_chunk_polys[0].len())
            .into_par_iter()
            .map(|idx| {
                raw_chunk_polys
                    .iter()
                    .zip(params.chunk_rbc_weights.iter())
                    .map(|(poly, weight)| poly.get_coeff(idx) * *weight)
                    .sum::<F>()
            })
            .collect();
        let mut coeffs_by_poly = Vec::with_capacity(2 + raw_chunk_polys.len());
        coeffs_by_poly.push(raw_value_coeffs);
        coeffs_by_poly.push(eq_coeffs_template);
        for raw_chunk_poly in raw_chunk_polys.iter() {
            let raw_chunk_coeffs: Vec<F> = (0..raw_chunk_poly.len())
                .map(|idx| raw_chunk_poly.get_coeff(idx))
                .collect();
            coeffs_by_poly.push(raw_chunk_coeffs);
        }
        let mut permuted_polys =
            permute_precommitted_polys(coeffs_by_poly, &params.precommitted).into_iter();
        let value_poly = permuted_polys
            .next()
            .expect("expected permuted bytecode value polynomial");
        let eq_poly = permuted_polys
            .next()
            .expect("expected permuted bytecode eq polynomial");
        let chunk_value_polys: Vec<MultilinearPolynomial<F>> = permuted_polys.collect();

        if debug_bytecode_reduction_enabled() {
            let initial_true_claim: F = (0..value_poly.len())
                .map(|i| value_poly.get_bound_coeff(i) * eq_poly.get_bound_coeff(i))
                .sum();
            tracing::info!(
                "BytecodeClaimReduction initialize value_len={} eq_len={} initial_true_claim={}",
                value_poly.len(),
                eq_poly.len(),
                initial_true_claim
            );
        }

        Self {
            params,
            value_poly,
            eq_poly,
            scale: F::one(),
            chunk_value_polys,
            pending_round_poly: None,
            running_claim: None,
        }
    }

    fn bind_aux_polys(&mut self, r_j: F::Challenge) {
        for poly in self.chunk_value_polys.iter_mut() {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
    }

    fn compute_message_unscaled(&self, previous_claim_unscaled: F) -> UniPoly<F> {
        let half = self.value_poly.len() / 2;
        let evals: [F; TWO_PHASE_DEGREE_BOUND] = (0..half)
            .into_par_iter()
            .map(|j| {
                let value_evals = self
                    .value_poly
                    .sumcheck_evals_array::<TWO_PHASE_DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let eq_evals = self
                    .eq_poly
                    .sumcheck_evals_array::<TWO_PHASE_DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let mut out = [F::zero(); TWO_PHASE_DEGREE_BOUND];
                for i in 0..TWO_PHASE_DEGREE_BOUND {
                    out[i] = value_evals[i] * eq_evals[i];
                }
                out
            })
            .reduce(
                || [F::zero(); TWO_PHASE_DEGREE_BOUND],
                |mut acc, arr| {
                    acc.iter_mut().zip(arr.iter()).for_each(|(a, b)| *a += *b);
                    acc
                },
            );
        UniPoly::from_evals_and_hint(previous_claim_unscaled, &evals)
    }

    fn cycle_intermediate_claim(&self) -> F {
        let len = self.value_poly.len();
        debug_assert_eq!(len, self.eq_poly.len());
        let mut sum = F::zero();
        for i in 0..len {
            sum += self.value_poly.get_bound_coeff(i) * self.eq_poly.get_bound_coeff(i);
        }
        sum * self.scale
    }

    fn final_claim_if_ready(&self) -> Option<F> {
        if self.value_poly.len() == 1 {
            Some(self.value_poly.get_bound_coeff(0))
        } else {
            None
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for BytecodeClaimReductionProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn round_offset(&self, max_num_rounds: usize) -> usize {
        self.params.round_offset(max_num_rounds)
    }

    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let is_active_round = if self.params.is_cycle_phase() {
            self.params.is_cycle_phase_round(round)
        } else {
            self.params.is_address_phase_round(round)
        };
        if !is_active_round {
            let round_poly =
                UniPoly::from_coeff(vec![previous_claim * F::from_u64(2).inverse().unwrap()]);
            self.pending_round_poly = Some(round_poly.clone());
            return round_poly;
        }

        let trailing_cap = if self.params.is_cycle_phase() {
            self.params.cycle_alignment_rounds()
        } else {
            self.params.address_alignment_rounds()
        };
        let num_trailing_variables =
            trailing_cap.saturating_sub(self.params.num_rounds_for_current_phase());
        let scaling_factor = self.scale * F::one().mul_pow_2(num_trailing_variables);
        let prev_unscaled = previous_claim * scaling_factor.inverse().unwrap();
        let poly_unscaled = self.compute_message_unscaled(prev_unscaled);
        let round_poly = poly_unscaled * scaling_factor;
        self.pending_round_poly = Some(round_poly.clone());
        round_poly
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        if let Some(round_poly) = self.pending_round_poly.take() {
            self.running_claim = Some(round_poly.evaluate(&r_j));
        }
        let is_active_round = if self.params.is_cycle_phase() {
            self.params.is_cycle_phase_round(round)
        } else {
            self.params.is_address_phase_round(round)
        };
        if !is_active_round {
            self.scale *= F::from_u64(2).inverse().unwrap();
            return;
        }

        self.value_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.bind_aux_polys(r_j);
        if self.params.is_cycle_phase() {
            self.params.precommitted.record_cycle_challenge(r_j);
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let params = &self.params;
        let opening_point = params.normalize_opening_point(sumcheck_challenges);

        if params.phase == PrecommittedPhase::CycleVariables {
            let c_mid = self.cycle_intermediate_claim();
            let synced_cycle_claim = self.running_claim.unwrap_or(c_mid);
            if debug_bytecode_reduction_enabled() {
                tracing::info!(
                    "BytecodeClaimReduction cache cycle len={} bound_value={} bound_eq={} scale={} cycle_claim={} synced_cycle_claim={}",
                    self.value_poly.len(),
                    self.value_poly.get_bound_coeff(0),
                    self.eq_poly.get_bound_coeff(0),
                    self.scale,
                    c_mid,
                    synced_cycle_claim,
                );
            }
            accumulator.append_virtual(
                VirtualPolynomial::BytecodeClaimReductionIntermediate,
                SumcheckId::BytecodeClaimReductionCyclePhase,
                opening_point.clone(),
                synced_cycle_claim,
            );
        }

        if let Some(bytecode_claim) = self.final_claim_if_ready() {
            let chunk_claims: Vec<F> = self
                .chunk_value_polys
                .iter()
                .map(|poly| poly.final_sumcheck_claim())
                .collect();
            let weighted_chunk_sum = chunk_claims
                .iter()
                .zip(params.chunk_rbc_weights.iter())
                .map(|(claim, weight)| *claim * *weight)
                .sum::<F>();
            debug_assert_eq!(weighted_chunk_sum, bytecode_claim);
            for (chunk_idx, claim) in chunk_claims.into_iter().enumerate() {
                accumulator.append_dense(
                    CommittedPolynomial::BytecodeChunk(chunk_idx),
                    SumcheckId::BytecodeClaimReduction,
                    opening_point.r.clone(),
                    claim,
                );
            }
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct BytecodeClaimReductionVerifier<F: JoltField> {
    pub params: RefCell<BytecodeClaimReductionParams<F>>,
}

impl<F: JoltField> BytecodeClaimReductionVerifier<F> {
    pub fn new(params: BytecodeClaimReductionParams<F>) -> Self {
        Self {
            params: RefCell::new(params),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for BytecodeClaimReductionVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        unsafe { &*self.params.as_ptr() }
    }

    fn round_offset(&self, max_num_rounds: usize) -> usize {
        let params = self.params.borrow();
        params.round_offset(max_num_rounds)
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let params = self.params.borrow();
        match params.phase {
            PrecommittedPhase::CycleVariables => {
                accumulator
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::BytecodeClaimReductionIntermediate,
                        SumcheckId::BytecodeClaimReductionCyclePhase,
                    )
                    .1
            }
            PrecommittedPhase::AddressVariables => {
                let bytecode_opening: F = (0..params.bytecode_chunk_count)
                    .map(|chunk_idx| {
                        params.chunk_rbc_weights[chunk_idx]
                            * accumulator
                                .get_committed_polynomial_opening(
                                    CommittedPolynomial::BytecodeChunk(chunk_idx),
                                    SumcheckId::BytecodeClaimReduction,
                                )
                                .1
                    })
                    .sum();
                let eq_combined = evaluate_bytecode_eq_combined(&params, sumcheck_challenges);
                let scale: F = precommitted_skip_round_scale(&params.precommitted);

                bytecode_opening * eq_combined * scale
            }
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let mut params = self.params.borrow_mut();
        if params.phase == PrecommittedPhase::CycleVariables {
            let opening_point = params.normalize_opening_point(sumcheck_challenges);
            accumulator.append_virtual(
                VirtualPolynomial::BytecodeClaimReductionIntermediate,
                SumcheckId::BytecodeClaimReductionCyclePhase,
                opening_point.clone(),
            );
            let opening_point_le: OpeningPoint<LITTLE_ENDIAN, F> = opening_point.match_endianness();
            params
                .precommitted
                .set_cycle_var_challenges(opening_point_le.r);
        }

        if params.num_address_phase_rounds() == 0
            || params.phase == PrecommittedPhase::AddressVariables
        {
            let opening_point = params.normalize_opening_point(sumcheck_challenges);
            for chunk_idx in 0..params.bytecode_chunk_count {
                accumulator.append_dense(
                    CommittedPolynomial::BytecodeChunk(chunk_idx),
                    SumcheckId::BytecodeClaimReduction,
                    opening_point.r.clone(),
                );
            }
        }
    }
}

fn evaluate_bytecode_eq_combined<F: JoltField>(
    params: &BytecodeClaimReductionParams<F>,
    sumcheck_challenges: &[F::Challenge],
) -> F {
    let opening_point = params.normalize_opening_point(sumcheck_challenges);
    let lane_var_count = committed_lanes().log_2();

    let (lane_challenges, cycle_challenges) = match DoryGlobals::get_layout() {
        DoryLayout::CycleMajor => {
            let (lane, cycle) = opening_point.r.split_at(lane_var_count);
            (lane, cycle)
        }
        DoryLayout::AddressMajor => {
            let (cycle, lane) = opening_point.r.split_at(params.bytecode_T);
            (lane, cycle)
        }
    };

    debug_assert_eq!(lane_challenges.len(), lane_var_count);
    debug_assert_eq!(cycle_challenges.len(), params.r_bc.r.len());

    let eq_cycle = EqPolynomial::mle(cycle_challenges, &params.r_bc.r);
    let eq_lane = EqPolynomial::<F>::evals(lane_challenges);
    let lane_weight_eval: F = params
        .lane_weights
        .iter()
        .zip(eq_lane.iter())
        .map(|(w, eq)| *w * *eq)
        .sum();

    lane_weight_eval * eq_cycle
}

#[inline(always)]
fn native_index_to_lane_cycle<F: JoltField>(
    params: &BytecodeClaimReductionParams<F>,
    index: usize,
) -> (usize, usize) {
    let bytecode_len = 1usize << params.bytecode_T;
    match DoryGlobals::get_layout() {
        DoryLayout::CycleMajor => (index / bytecode_len, index % bytecode_len),
        DoryLayout::AddressMajor => (index % committed_lanes(), index / committed_lanes()),
    }
}

fn compute_lane_weights<F: JoltField>(
    bytecode_read_raf_params: &BytecodeReadRafSumcheckParams<F>,
    accumulator: &dyn OpeningAccumulator<F>,
    eta_powers: &[F; NUM_VAL_STAGES],
) -> Vec<F> {
    let reg_count = REGISTER_COUNT as usize;
    let total = crate::zkvm::bytecode::chunks::total_lanes();

    let rs1_start = 0usize;
    let rs2_start = rs1_start + reg_count;
    let rd_start = rs2_start + reg_count;
    let unexp_pc_idx = rd_start + reg_count;
    let imm_idx = unexp_pc_idx + 1;
    let circuit_start = imm_idx + 1;
    let instr_start = circuit_start + NUM_CIRCUIT_FLAGS;
    let lookup_start = instr_start + NUM_INSTRUCTION_FLAGS;
    let raf_flag_idx = lookup_start + LookupTables::<XLEN>::COUNT;
    debug_assert_eq!(raf_flag_idx + 1, total);

    let log_reg = reg_count.log_2();
    let r_register_4 = accumulator
        .get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersReadWriteChecking,
        )
        .0
        .r;
    let eq_r_register_4 = EqPolynomial::<F>::evals(&r_register_4[..log_reg]);

    let r_register_5 = accumulator
        .get_virtual_polynomial_opening(VirtualPolynomial::RdWa, SumcheckId::RegistersValEvaluation)
        .0
        .r;
    let eq_r_register_5 = EqPolynomial::<F>::evals(&r_register_5[..log_reg]);

    let mut weights = vec![F::zero(); committed_lanes()];

    {
        let coeff = eta_powers[0];
        let g = &bytecode_read_raf_params.stage1_gammas;
        weights[unexp_pc_idx] += coeff * g[0];
        weights[imm_idx] += coeff * g[1];
        for i in 0..NUM_CIRCUIT_FLAGS {
            weights[circuit_start + i] += coeff * g[2 + i];
        }
    }
    {
        let coeff = eta_powers[1];
        let g = &bytecode_read_raf_params.stage2_gammas;
        weights[circuit_start + (CircuitFlags::Jump as usize)] += coeff * g[0];
        weights[instr_start + (InstructionFlags::Branch as usize)] += coeff * g[1];
        weights[circuit_start + (CircuitFlags::WriteLookupOutputToRD as usize)] += coeff * g[2];
        weights[circuit_start + (CircuitFlags::VirtualInstruction as usize)] += coeff * g[3];
    }
    {
        let coeff = eta_powers[2];
        let g = &bytecode_read_raf_params.stage3_gammas;
        weights[imm_idx] += coeff * g[0];
        weights[unexp_pc_idx] += coeff * g[1];
        weights[instr_start + (InstructionFlags::LeftOperandIsRs1Value as usize)] += coeff * g[2];
        weights[instr_start + (InstructionFlags::LeftOperandIsPC as usize)] += coeff * g[3];
        weights[instr_start + (InstructionFlags::RightOperandIsRs2Value as usize)] += coeff * g[4];
        weights[instr_start + (InstructionFlags::RightOperandIsImm as usize)] += coeff * g[5];
        weights[instr_start + (InstructionFlags::IsNoop as usize)] += coeff * g[6];
        weights[circuit_start + (CircuitFlags::VirtualInstruction as usize)] += coeff * g[7];
        weights[circuit_start + (CircuitFlags::IsFirstInSequence as usize)] += coeff * g[8];
    }
    {
        let coeff = eta_powers[3];
        let g = &bytecode_read_raf_params.stage4_gammas;
        for r in 0..reg_count {
            weights[rd_start + r] += coeff * g[0] * eq_r_register_4[r];
            weights[rs1_start + r] += coeff * g[1] * eq_r_register_4[r];
            weights[rs2_start + r] += coeff * g[2] * eq_r_register_4[r];
        }
    }
    {
        let coeff = eta_powers[4];
        let g = &bytecode_read_raf_params.stage5_gammas;
        for r in 0..reg_count {
            weights[rd_start + r] += coeff * g[0] * eq_r_register_5[r];
        }
        weights[raf_flag_idx] += coeff * g[1];
        for i in 0..LookupTables::<XLEN>::COUNT {
            weights[lookup_start + i] += coeff * g[2 + i];
        }
    }

    weights
}
