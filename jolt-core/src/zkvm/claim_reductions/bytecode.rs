//! Two-phase bytecode claim reduction (Stage 6b cycle -> Stage 7 address).

use allocative::Allocative;
use rayon::prelude::*;

use crate::field::JoltField;
use crate::poly::commitment::dory::{DoryGlobals, DoryLayout};
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialBinding};
#[cfg(feature = "zk")]
use crate::poly::opening_proof::OpeningId;
use crate::poly::opening_proof::{
    AbstractVerifierOpeningAccumulator, OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator,
    SumcheckId, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::unipoly::UniPoly;
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::{InputClaimConstraint, OutputClaimConstraint, ValueSource};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::bytecode::chunks::{committed_lanes, total_lanes, BYTECODE_LANE_LAYOUT};
use crate::zkvm::claim_reductions::{
    permute_precommitted_polys, precommitted_skip_round_scale, PrecommittedClaimReduction,
    PrecommittedPhase, PrecommittedSchedulingReference, TWO_PHASE_DEGREE_BOUND,
};
use crate::zkvm::instruction::{CircuitFlags, InstructionFlags, NUM_CIRCUIT_FLAGS};
use crate::zkvm::lookup_table::LookupTables;
use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use common::constants::{REGISTER_COUNT, XLEN};
use strum::EnumCount;

use super::precommitted::{PrecommittedParams, PrecommittedProver};

const NUM_VAL_STAGES: usize = 5;

#[derive(Clone, Allocative)]
pub struct BytecodeClaimReductionParams<F: JoltField> {
    pub precommitted: PrecommittedClaimReduction<F>,
    pub eta: F,
    pub eta_powers: [F; NUM_VAL_STAGES],
    /// Eq weights over high bytecode address bits (one per committed chunk).
    pub chunk_rbc_weights: Vec<F>,
    pub log_bytecode_chunk_size: usize,
    pub bytecode_chunk_count: usize,
    pub bytecode_col_vars: usize,
    pub bytecode_row_vars: usize,
    pub r_bc: OpeningPoint<BIG_ENDIAN, F>,
    pub lane_weights: Vec<F>,
}

impl<F: JoltField> BytecodeClaimReductionParams<F> {
    pub fn new(
        bytecode_read_raf_gammas: [&[F]; NUM_VAL_STAGES],
        bytecode_len: usize,
        bytecode_chunk_count: usize,
        scheduling_reference: PrecommittedSchedulingReference,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        assert!(
            bytecode_len.is_multiple_of(bytecode_chunk_count),
            "bytecode chunk count ({bytecode_chunk_count}) must divide bytecode_len ({bytecode_len})"
        );
        let log_bytecode_chunk_size = (bytecode_len / bytecode_chunk_count).log_2();
        let log_bytecode_len = bytecode_len.log_2();

        let eta: F = transcript.challenge_scalar();
        let mut eta_powers = [F::one(); NUM_VAL_STAGES];
        for i in 1..NUM_VAL_STAGES {
            eta_powers[i] = eta_powers[i - 1] * eta;
        }

        let (r_bc_full, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::BytecodeReadRafAddrClaim,
            SumcheckId::BytecodeReadRafAddressPhase,
        );
        debug_assert_eq!(r_bc_full.r.len(), log_bytecode_len);
        let dropped_bits = log_bytecode_len - log_bytecode_chunk_size;
        let chunk_rbc_weights = if dropped_bits == 0 {
            vec![F::one()]
        } else {
            EqPolynomial::<F>::evals(&r_bc_full.r[..dropped_bits])
        };
        debug_assert_eq!(chunk_rbc_weights.len(), bytecode_chunk_count);
        let r_bc = OpeningPoint::new(r_bc_full.r[dropped_bits..].to_vec());

        let lane_weights = compute_lane_weights(bytecode_read_raf_gammas, accumulator, &eta_powers);

        let log_committed_lane_count = committed_lanes().log_2();
        let total_vars = log_committed_lane_count + log_bytecode_chunk_size;
        // Bytecode uses its own balanced dimensions (independent from Main).
        // In Stage 8 it is embedded as a top-left block in Joint.
        let (bytecode_col_vars, bytecode_row_vars) = DoryGlobals::balanced_sigma_nu(total_vars);
        let precommitted = PrecommittedClaimReduction::new(
            bytecode_row_vars,
            bytecode_col_vars,
            scheduling_reference,
        );
        // Align all precommitted scheduling/permutation to the shared reference domain.

        Self {
            precommitted,
            eta,
            eta_powers,
            chunk_rbc_weights,
            log_bytecode_chunk_size,
            bytecode_chunk_count,
            bytecode_col_vars,
            bytecode_row_vars,
            r_bc,
            lane_weights,
        }
    }
}

impl<F: JoltField> BytecodeClaimReductionParams<F> {
    fn num_rounds_for_current_phase(&self) -> usize {
        self.precommitted.num_rounds_for_current_phase()
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for BytecodeClaimReductionParams<F> {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        match self.precommitted.phase {
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
        self.precommitted.normalize_opening_point(challenges)
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        match self.precommitted.phase {
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
        match self.precommitted.phase {
            PrecommittedPhase::CycleVariables => self.eta_powers.to_vec(),
            PrecommittedPhase::AddressVariables => Vec::new(),
        }
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        match self.precommitted.phase {
            PrecommittedPhase::CycleVariables => {
                if self.precommitted.num_address_phase_rounds() > 0 {
                    return Some(OutputClaimConstraint::direct(OpeningId::virt(
                        VirtualPolynomial::BytecodeClaimReductionIntermediate,
                        SumcheckId::BytecodeClaimReductionCyclePhase,
                    )));
                }
                self.final_bytecode_output_claim_constraint()
            }
            PrecommittedPhase::AddressVariables => self.final_bytecode_output_claim_constraint(),
        }
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        match self.precommitted.phase {
            PrecommittedPhase::CycleVariables
                if self.precommitted.num_address_phase_rounds() > 0 =>
            {
                vec![]
            }
            PrecommittedPhase::CycleVariables | PrecommittedPhase::AddressVariables => {
                self.final_bytecode_output_weights(sumcheck_challenges)
            }
        }
    }
}

impl<F: JoltField> BytecodeClaimReductionParams<F> {
    #[cfg(feature = "zk")]
    fn final_bytecode_output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
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

    fn final_bytecode_output_scale(&self, sumcheck_challenges: &[F::Challenge]) -> F {
        let opening_point = self.normalize_opening_point(sumcheck_challenges);
        let eq_combined = evaluate_bytecode_eq_combined(self, &opening_point);
        let scale = match self.precommitted.phase {
            PrecommittedPhase::CycleVariables => self.precommitted.cycle_phase_skip_scale(),
            PrecommittedPhase::AddressVariables => {
                precommitted_skip_round_scale(&self.precommitted)
            }
        };
        eq_combined * scale
    }

    #[cfg(feature = "zk")]
    fn final_bytecode_output_weights(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        let output_scale = self.final_bytecode_output_scale(sumcheck_challenges);
        self.chunk_rbc_weights
            .iter()
            .map(|w| *w * output_scale)
            .collect()
    }
}

impl<F: JoltField> PrecommittedParams<F> for BytecodeClaimReductionParams<F> {
    fn precommitted(&self) -> &PrecommittedClaimReduction<F> {
        &self.precommitted
    }

    fn precommitted_mut(&mut self) -> &mut PrecommittedClaimReduction<F> {
        &mut self.precommitted
    }

    fn get_cycle_challenges<A: AbstractVerifierOpeningAccumulator<F>>(
        &self,
        accumulator: &A,
    ) -> Vec<F::Challenge> {
        let (cycle_opening_point, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::BytecodeClaimReductionIntermediate,
            SumcheckId::BytecodeClaimReductionCyclePhase,
        );
        let opening_point_le: OpeningPoint<LITTLE_ENDIAN, F> =
            cycle_opening_point.match_endianness();
        opening_point_le.r
    }
}

#[derive(Allocative)]
pub struct BytecodeClaimReductionProver<F: JoltField> {
    core: PrecommittedProver<F, BytecodeClaimReductionParams<F>>,
}

impl<F: JoltField> BytecodeClaimReductionProver<F> {
    pub fn params(&self) -> &BytecodeClaimReductionParams<F> {
        self.core.params()
    }

    pub fn transition_to_address_phase(&mut self) {
        self.core.transition_to_address_phase();
    }

    pub fn initialize(
        params: BytecodeClaimReductionParams<F>,
        raw_chunk_coeffs: &[Vec<F>],
    ) -> Self {
        let eq_cycle = EqPolynomial::<F>::evals(&params.r_bc.r);
        let eq_coeffs_template: Vec<F> = (0..raw_chunk_coeffs[0].len())
            .map(|idx| {
                let (lane, cycle) = native_index_to_lane_cycle(&params, idx);
                params.lane_weights[lane] * eq_cycle[cycle]
            })
            .collect();

        let raw_value_coeffs: Vec<F> = (0..raw_chunk_coeffs[0].len())
            .into_par_iter()
            .map(|idx| {
                raw_chunk_coeffs
                    .iter()
                    .zip(params.chunk_rbc_weights.iter())
                    .map(|(coeffs, weight)| coeffs[idx] * *weight)
                    .sum::<F>()
            })
            .collect();
        let mut coeffs_by_poly = Vec::with_capacity(2 + raw_chunk_coeffs.len());
        coeffs_by_poly.push(raw_value_coeffs);
        coeffs_by_poly.push(eq_coeffs_template);
        for coeffs in raw_chunk_coeffs.iter() {
            coeffs_by_poly.push(coeffs.clone());
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

        Self {
            core: PrecommittedProver::new(params, value_poly, eq_poly, Some(chunk_value_polys)),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for BytecodeClaimReductionProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        self.core.params()
    }

    fn round_offset(&self, _max_num_rounds: usize) -> usize {
        0
    }

    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        self.core.compute_message(round, previous_claim)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        self.core.ingest_challenge(r_j, round);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let params = self.core.params();
        let opening_point = params.normalize_opening_point(sumcheck_challenges);

        if params.is_cycle_phase() && params.precommitted.num_address_phase_rounds() > 0 {
            accumulator.append_virtual(
                VirtualPolynomial::BytecodeClaimReductionIntermediate,
                SumcheckId::BytecodeClaimReductionCyclePhase,
                opening_point.clone(),
                self.core.cycle_intermediate_claim(),
            );
        }

        if let Some(bytecode_claim) = self.core.final_claim_if_ready() {
            let chunk_claims: Vec<F> = self
                .core
                .aux_polys()
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
    pub params: BytecodeClaimReductionParams<F>,
}

impl<F: JoltField> BytecodeClaimReductionVerifier<F> {
    pub fn new(params: BytecodeClaimReductionParams<F>) -> Self {
        Self { params }
    }
}

impl<F: JoltField, T: Transcript, A: AbstractVerifierOpeningAccumulator<F>>
    SumcheckInstanceVerifier<F, T, A> for BytecodeClaimReductionVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn round_offset(&self, _max_num_rounds: usize) -> usize {
        0
    }

    fn expected_output_claim(&self, accumulator: &A, sumcheck_challenges: &[F::Challenge]) -> F {
        let params = &self.params;
        match params.precommitted.phase {
            PrecommittedPhase::CycleVariables
                if params.precommitted.num_address_phase_rounds() > 0 =>
            {
                accumulator
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::BytecodeClaimReductionIntermediate,
                        SumcheckId::BytecodeClaimReductionCyclePhase,
                    )
                    .1
            }
            PrecommittedPhase::CycleVariables | PrecommittedPhase::AddressVariables => {
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
                    .sum::<F>();

                bytecode_opening * params.final_bytecode_output_scale(sumcheck_challenges)
            }
        }
    }

    fn cache_openings(&self, accumulator: &mut A, sumcheck_challenges: &[F::Challenge]) {
        let params = &self.params;
        if params.is_cycle_phase() && params.precommitted.num_address_phase_rounds() > 0 {
            let opening_point = params.normalize_opening_point(sumcheck_challenges);
            accumulator.append_virtual(
                VirtualPolynomial::BytecodeClaimReductionIntermediate,
                SumcheckId::BytecodeClaimReductionCyclePhase,
                opening_point,
            );
        }

        if params.precommitted.num_address_phase_rounds() == 0 || !params.is_cycle_phase() {
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
    opening_point: &OpeningPoint<BIG_ENDIAN, F>,
) -> F {
    let lane_var_count = committed_lanes().log_2();

    let (lane_challenges, cycle_challenges) = match DoryGlobals::get_layout() {
        DoryLayout::CycleMajor => {
            let (lane, cycle) = opening_point.r.split_at(lane_var_count);
            (lane, cycle)
        }
        DoryLayout::AddressMajor => {
            let (cycle, lane) = opening_point.r.split_at(params.log_bytecode_chunk_size);
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
    let bytecode_len = 1usize << params.log_bytecode_chunk_size;
    match DoryGlobals::get_layout() {
        DoryLayout::CycleMajor => (index / bytecode_len, index % bytecode_len),
        DoryLayout::AddressMajor => (index % committed_lanes(), index / committed_lanes()),
    }
}

fn compute_lane_weights<F: JoltField>(
    bytecode_read_raf_gammas: [&[F]; NUM_VAL_STAGES],
    accumulator: &dyn OpeningAccumulator<F>,
    eta_powers: &[F; NUM_VAL_STAGES],
) -> Vec<F> {
    let reg_count = REGISTER_COUNT as usize;
    let layout = BYTECODE_LANE_LAYOUT;
    debug_assert_eq!(layout.raf_flag_idx + 1, total_lanes());

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
        let g = bytecode_read_raf_gammas[0];
        weights[layout.unexp_pc_idx] += coeff * g[0];
        weights[layout.imm_idx] += coeff * g[1];
        for i in 0..NUM_CIRCUIT_FLAGS {
            weights[layout.circuit_start + i] += coeff * g[2 + i];
        }
    }
    {
        let coeff = eta_powers[1];
        let g = bytecode_read_raf_gammas[1];
        weights[layout.circuit_start + (CircuitFlags::Jump as usize)] += coeff * g[0];
        weights[layout.instr_start + (InstructionFlags::Branch as usize)] += coeff * g[1];
        weights[layout.circuit_start + (CircuitFlags::WriteLookupOutputToRD as usize)] +=
            coeff * g[2];
        weights[layout.circuit_start + (CircuitFlags::VirtualInstruction as usize)] += coeff * g[3];
    }
    {
        let coeff = eta_powers[2];
        let g = bytecode_read_raf_gammas[2];
        weights[layout.imm_idx] += coeff * g[0];
        weights[layout.unexp_pc_idx] += coeff * g[1];
        weights[layout.instr_start + (InstructionFlags::LeftOperandIsRs1Value as usize)] +=
            coeff * g[2];
        weights[layout.instr_start + (InstructionFlags::LeftOperandIsPC as usize)] += coeff * g[3];
        weights[layout.instr_start + (InstructionFlags::RightOperandIsRs2Value as usize)] +=
            coeff * g[4];
        weights[layout.instr_start + (InstructionFlags::RightOperandIsImm as usize)] +=
            coeff * g[5];
        weights[layout.instr_start + (InstructionFlags::IsNoop as usize)] += coeff * g[6];
        weights[layout.circuit_start + (CircuitFlags::VirtualInstruction as usize)] += coeff * g[7];
        weights[layout.circuit_start + (CircuitFlags::IsFirstInSequence as usize)] += coeff * g[8];
    }
    {
        let coeff = eta_powers[3];
        let g = bytecode_read_raf_gammas[3];
        for r in 0..reg_count {
            weights[layout.rd_start + r] += coeff * g[0] * eq_r_register_4[r];
            weights[layout.rs1_start + r] += coeff * g[1] * eq_r_register_4[r];
            weights[layout.rs2_start + r] += coeff * g[2] * eq_r_register_4[r];
        }
    }
    {
        let coeff = eta_powers[4];
        let g = bytecode_read_raf_gammas[4];
        for r in 0..reg_count {
            weights[layout.rd_start + r] += coeff * g[0] * eq_r_register_5[r];
        }
        weights[layout.raf_flag_idx] += coeff * g[1];
        for i in 0..LookupTables::<XLEN>::COUNT {
            weights[layout.lookup_start + i] += coeff * g[2 + i];
        }
    }

    weights
}
