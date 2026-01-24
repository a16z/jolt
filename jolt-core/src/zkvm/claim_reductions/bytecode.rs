//! Two-phase Bytecode claim reduction (Stage 6b cycle → Stage 7 lane/address).
//!
//! This reduction batches the 5 bytecode Val-stage claims emitted at the Stage 6a boundary:
//! `Val_s(r_bc)` for `s = 0..5` (val-only; RAF terms excluded).
//!
//! High level:
//! - Sample `η` and form `C_in = Σ_s η^s · Val_s(r_bc)`.
//! - Define a canonical set of bytecode "lanes" (448 total) and a lane weight function
//!   `W_η(lane) = Σ_s η^s · w_s(lane)` derived from the same stage-specific gammas used to
//!   define `Val_s`.
//! - Prove, via a two-phase sumcheck, that `C_in` equals a single linear functional of the
//!   (eventual) committed bytecode chunk polynomials.
//!
//! NOTE: This module wires the reduction logic and emits openings for bytecode chunk polynomials.
//! Commitment + Stage 8 batching integration is handled separately (see `bytecode-commitment-progress.md`).

use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use allocative::Allocative;
use itertools::Itertools;
use rayon::prelude::*;

use crate::field::JoltField;
use crate::poly::commitment::dory::{DoryGlobals, DoryLayout};
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::bytecode::chunks::{build_bytecode_chunks, total_lanes};
use crate::zkvm::bytecode::read_raf_checking::BytecodeReadRafSumcheckParams;
use crate::zkvm::instruction::{
    CircuitFlags, InstructionFlags, NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS,
};
use crate::zkvm::lookup_table::LookupTables;
use crate::zkvm::program::ProgramPreprocessing;
use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use common::constants::{REGISTER_COUNT, XLEN};
use strum::EnumCount;

const DEGREE_BOUND: usize = 2;
const NUM_VAL_STAGES: usize = 5;

/// For `DoryLayout::AddressMajor`, committed bytecode chunks are stored in "cycle-major" index order
/// (cycle*K + address), which makes `BindingOrder::LowToHigh` bind **lane** bits first.
///
/// The claim reduction sumcheck needs to bind **cycle** bits first in Stage 6b, so we permute
/// dense coefficient vectors into the `DoryLayout::CycleMajor` order (address*T + cycle) when
/// running the reduction. This is a pure index permutation, i.e. a variable renaming, and the
/// resulting evaluations match the committed polynomial when the opening point is interpreted in
/// the unified `[lane || cycle]` order.
fn permute_address_major_to_cycle_major<F: JoltField>(
    coeffs: Vec<F>,
    k_chunk: usize,
    t_size: usize,
) -> Vec<F> {
    debug_assert_eq!(coeffs.len(), k_chunk * t_size);
    let mut out: Vec<F> = unsafe_allocate_zero_vec(k_chunk * t_size);
    for lane in 0..k_chunk {
        for k in 0..t_size {
            // AddressMajor: idx = cycle * K + address
            let idx_in = k * k_chunk + lane;
            // CycleMajor: idx = address * T + cycle
            let idx_out = lane * t_size + k;
            out[idx_out] = coeffs[idx_in];
        }
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Allocative)]
pub enum BytecodeReductionPhase {
    CycleVariables,
    LaneVariables,
}

#[derive(Clone, Allocative)]
pub struct BytecodeClaimReductionParams<F: JoltField> {
    pub phase: BytecodeReductionPhase,
    pub eta: F,
    pub eta_powers: [F; NUM_VAL_STAGES],
    pub log_k: usize,
    pub log_k_chunk: usize,
    pub num_chunks: usize,
    /// Bytecode address point (log_K bits, big-endian).
    pub r_bc: OpeningPoint<BIG_ENDIAN, F>,
    /// Per-chunk lane weight tables (length = k_chunk) for `W_eta`.
    pub chunk_lane_weights: Vec<Vec<F>>,
    /// (little-endian) challenges used in the cycle phase.
    pub cycle_var_challenges: Vec<F::Challenge>,
}

impl<F: JoltField> BytecodeClaimReductionParams<F> {
    pub fn new(
        bytecode_read_raf_params: &BytecodeReadRafSumcheckParams<F>,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let log_k = bytecode_read_raf_params.log_K;

        let eta: F = transcript.challenge_scalar();
        let mut eta_powers = [F::one(); NUM_VAL_STAGES];
        for i in 1..NUM_VAL_STAGES {
            eta_powers[i] = eta_powers[i - 1] * eta;
        }

        // r_bc comes from the Stage 6a BytecodeReadRaf address phase.
        let (r_bc, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::BytecodeReadRafAddrClaim,
            SumcheckId::BytecodeReadRafAddressPhase,
        );

        let log_k_chunk = bytecode_read_raf_params.one_hot_params.log_k_chunk;
        let k_chunk = 1 << log_k_chunk;
        let num_chunks = total_lanes().div_ceil(k_chunk);

        let chunk_lane_weights = compute_chunk_lane_weights(
            bytecode_read_raf_params,
            accumulator,
            &eta_powers,
            num_chunks,
            k_chunk,
        );

        Self {
            phase: BytecodeReductionPhase::CycleVariables,
            eta,
            eta_powers,
            log_k,
            log_k_chunk,
            num_chunks,
            r_bc,
            chunk_lane_weights,
            cycle_var_challenges: vec![],
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for BytecodeClaimReductionParams<F> {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        match self.phase {
            BytecodeReductionPhase::CycleVariables => (0..NUM_VAL_STAGES)
                .map(|stage| {
                    let (_, val_claim) = accumulator.get_virtual_polynomial_opening(
                        VirtualPolynomial::BytecodeValStage(stage),
                        SumcheckId::BytecodeReadRafAddressPhase,
                    );
                    self.eta_powers[stage] * val_claim
                })
                .sum(),
            BytecodeReductionPhase::LaneVariables => {
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
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        match self.phase {
            BytecodeReductionPhase::CycleVariables => self.log_k,
            BytecodeReductionPhase::LaneVariables => self.log_k_chunk,
        }
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        match self.phase {
            BytecodeReductionPhase::CycleVariables => {
                OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
            }
            BytecodeReductionPhase::LaneVariables => {
                // Full point: [lane || cycle] in big-endian.
                let full_le: Vec<F::Challenge> =
                    [self.cycle_var_challenges.as_slice(), challenges].concat();
                OpeningPoint::<LITTLE_ENDIAN, F>::new(full_le).match_endianness()
            }
        }
    }
}

#[derive(Allocative)]
pub struct BytecodeClaimReductionProver<F: JoltField> {
    pub params: BytecodeClaimReductionParams<F>,
    /// Chunk polynomials B_i(lane, k) (eventually committed).
    bytecode_chunks: Vec<MultilinearPolynomial<F>>,
    /// Weight polynomials W_i(lane, k) = W_eta(lane) * eq(r_bc, k) (multilinear).
    weight_chunks: Vec<MultilinearPolynomial<F>>,
    /// Batched-sumcheck scaling for trailing dummy rounds (see `round_offset`).
    #[allocative(skip)]
    batch_dummy_rounds: AtomicUsize,
}

impl<F: JoltField> BytecodeClaimReductionProver<F> {
    #[tracing::instrument(skip_all, name = "BytecodeClaimReductionProver::initialize")]
    pub fn initialize(
        params: BytecodeClaimReductionParams<F>,
        program: Arc<ProgramPreprocessing>,
    ) -> Self {
        let log_k = params.log_k;
        let t_size = 1 << log_k;
        let k_chunk = 1 << params.log_k_chunk;
        let layout = DoryGlobals::get_layout();

        // Eq table over the bytecode address point.
        let eq_r_bc = EqPolynomial::<F>::evals(&params.r_bc.r);
        debug_assert_eq!(eq_r_bc.len(), t_size);

        // Build per-chunk weight polynomials as an outer product (lane_weight ⊗ eq_r_bc).
        let weight_chunks: Vec<MultilinearPolynomial<F>> = (0..params.num_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let lane_weights = &params.chunk_lane_weights[chunk_idx];
                debug_assert_eq!(lane_weights.len(), k_chunk);
                let mut coeffs: Vec<F> = unsafe_allocate_zero_vec(k_chunk * t_size);
                for lane in 0..k_chunk {
                    let w = lane_weights[lane];
                    for k in 0..t_size {
                        // Claim reduction always uses CycleMajor ordering so that
                        // `BindingOrder::LowToHigh` binds cycle bits first in Stage 6b.
                        let idx =
                            DoryLayout::CycleMajor.address_cycle_to_index(lane, k, k_chunk, t_size);
                        coeffs[idx] = w * eq_r_bc[k];
                    }
                }
                MultilinearPolynomial::from(coeffs)
            })
            .collect();

        // Build per-chunk bytecode polynomials B_i(lane, k).
        let bytecode_len = program.bytecode_len();
        debug_assert_eq!(bytecode_len, t_size);
        let bytecode = program.as_bytecode();
        let mut bytecode_chunks = build_bytecode_chunks::<F>(&bytecode, params.log_k_chunk);
        if layout == DoryLayout::AddressMajor {
            // Permute committed AddressMajor coefficient order into CycleMajor for the reduction.
            for poly in bytecode_chunks.iter_mut() {
                if let MultilinearPolynomial::LargeScalars(p) = poly {
                    let old = std::mem::take(&mut p.Z);
                    p.Z = permute_address_major_to_cycle_major(old, k_chunk, t_size);
                } else {
                    unreachable!("bytecode chunks are dense field polynomials");
                }
            }
        }

        debug_assert_eq!(bytecode_chunks.len(), params.num_chunks);
        debug_assert_eq!(weight_chunks.len(), params.num_chunks);

        Self {
            params,
            bytecode_chunks,
            weight_chunks,
            batch_dummy_rounds: AtomicUsize::new(0),
        }
    }

    fn compute_message_impl(&self, previous_claim: F) -> UniPoly<F> {
        let half = self.bytecode_chunks[0].len() / 2;
        let mut evals: [F; DEGREE_BOUND] = (0..half)
            .into_par_iter()
            .map(|j| {
                let mut out = [F::zero(); DEGREE_BOUND];
                for (b, w) in self.bytecode_chunks.iter().zip(self.weight_chunks.iter()) {
                    let b_evals =
                        b.sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                    let w_evals =
                        w.sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                    for i in 0..DEGREE_BOUND {
                        out[i] += b_evals[i] * w_evals[i];
                    }
                }
                out
            })
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |mut acc, arr| {
                    acc.iter_mut().zip(arr.iter()).for_each(|(a, b)| *a += *b);
                    acc
                },
            );

        // If this instance is back-loaded in a batched sumcheck (i.e., it has trailing dummy
        // rounds), then `previous_claim` is scaled by 2^{dummy_rounds}. The per-round univariate
        // evaluations must be scaled by the same factor to satisfy the sumcheck consistency check.
        let dummy_rounds = self.batch_dummy_rounds.load(Ordering::Relaxed);
        if dummy_rounds != 0 {
            let scale = F::one().mul_pow_2(dummy_rounds);
            for e in evals.iter_mut() {
                *e *= scale;
            }
        }
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for BytecodeClaimReductionProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn round_offset(&self, max_num_rounds: usize) -> usize {
        // Bytecode claim reduction's cycle-phase rounds must align to the *start* of the
        // batched cycle challenge vector so that its (log_K) point is the suffix (LSB side)
        // of the full (log_T) cycle point used by other Stage 6b instances. This is required
        // for Stage 8's committed-bytecode embedding when log_T > log_K.
        //
        // This deviates from the default "front-loaded" batching offset, so we record the number
        // of trailing dummy rounds and scale univariate evaluations accordingly.
        let dummy_rounds = max_num_rounds.saturating_sub(self.params.num_rounds());
        self.batch_dummy_rounds
            .store(dummy_rounds, Ordering::Relaxed);
        0
    }

    #[tracing::instrument(skip_all, name = "BytecodeClaimReductionProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        self.compute_message_impl(previous_claim)
    }

    #[tracing::instrument(skip_all, name = "BytecodeClaimReductionProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        if self.params.phase == BytecodeReductionPhase::CycleVariables {
            self.params.cycle_var_challenges.push(r_j);
        }
        self.bytecode_chunks
            .iter_mut()
            .for_each(|p| p.bind_parallel(r_j, BindingOrder::LowToHigh));
        self.weight_chunks
            .iter_mut()
            .for_each(|p| p.bind_parallel(r_j, BindingOrder::LowToHigh));
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        match self.params.phase {
            BytecodeReductionPhase::CycleVariables => {
                // Cache intermediate claim for Stage 7.
                let opening_point = self.params.normalize_opening_point(sumcheck_challenges);

                let mut sum = F::zero();
                for (b, w) in self.bytecode_chunks.iter().zip(self.weight_chunks.iter()) {
                    debug_assert_eq!(b.len(), w.len());
                    for i in 0..b.len() {
                        sum += b.get_bound_coeff(i) * w.get_bound_coeff(i);
                    }
                }

                accumulator.append_virtual(
                    transcript,
                    VirtualPolynomial::BytecodeClaimReductionIntermediate,
                    SumcheckId::BytecodeClaimReductionCyclePhase,
                    opening_point,
                    sum,
                );
            }
            BytecodeReductionPhase::LaneVariables => {
                // Cache final openings of the bytecode chunk polynomials at the full point.
                let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
                let (r_lane, r_cycle) = opening_point.split_at(self.params.log_k_chunk);

                let polynomial_types: Vec<CommittedPolynomial> = (0..self.params.num_chunks)
                    .map(CommittedPolynomial::BytecodeChunk)
                    .collect();
                let claims: Vec<F> = self
                    .bytecode_chunks
                    .iter()
                    .map(|p| p.final_sumcheck_claim())
                    .collect();

                accumulator.append_sparse(
                    transcript,
                    polynomial_types,
                    SumcheckId::BytecodeClaimReduction,
                    r_lane.r,
                    r_cycle.r,
                    claims,
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

    fn round_offset(&self, _max_num_rounds: usize) -> usize {
        // Must mirror the prover: align this instance to the start of the batched challenge vector.
        0
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let params = self.params.borrow();
        match params.phase {
            BytecodeReductionPhase::CycleVariables => {
                accumulator
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::BytecodeClaimReductionIntermediate,
                        SumcheckId::BytecodeClaimReductionCyclePhase,
                    )
                    .1
            }
            BytecodeReductionPhase::LaneVariables => {
                let opening_point = params.normalize_opening_point(sumcheck_challenges);
                let (r_lane, r_cycle) = opening_point.split_at(params.log_k_chunk);

                let eq_eval = EqPolynomial::<F>::mle(&r_cycle.r, &params.r_bc.r);

                // Evaluate each chunk's lane-weight polynomial at r_lane and combine with chunk openings.
                let eq_lane = EqPolynomial::<F>::evals(&r_lane.r);
                let mut sum = F::zero();
                for chunk_idx in 0..params.num_chunks {
                    let (_, chunk_opening) = accumulator.get_committed_polynomial_opening(
                        CommittedPolynomial::BytecodeChunk(chunk_idx),
                        SumcheckId::BytecodeClaimReduction,
                    );
                    let w_eval: F = params.chunk_lane_weights[chunk_idx]
                        .iter()
                        .zip(eq_lane.iter())
                        .map(|(w, e)| *w * *e)
                        .sum();
                    sum += chunk_opening * w_eval;
                }

                sum * eq_eval
            }
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let mut params = self.params.borrow_mut();
        match params.phase {
            BytecodeReductionPhase::CycleVariables => {
                let opening_point = params.normalize_opening_point(sumcheck_challenges);
                accumulator.append_virtual(
                    transcript,
                    VirtualPolynomial::BytecodeClaimReductionIntermediate,
                    SumcheckId::BytecodeClaimReductionCyclePhase,
                    opening_point,
                );
                // Record LE challenges for phase 2 normalization.
                params.cycle_var_challenges = sumcheck_challenges.to_vec();
            }
            BytecodeReductionPhase::LaneVariables => {
                let opening_point = params.normalize_opening_point(sumcheck_challenges);
                let polynomial_types: Vec<CommittedPolynomial> = (0..params.num_chunks)
                    .map(CommittedPolynomial::BytecodeChunk)
                    .collect();
                accumulator.append_sparse(
                    transcript,
                    polynomial_types,
                    SumcheckId::BytecodeClaimReduction,
                    opening_point.r,
                );
            }
        }
    }
}

fn compute_chunk_lane_weights<F: JoltField>(
    bytecode_read_raf_params: &BytecodeReadRafSumcheckParams<F>,
    accumulator: &dyn OpeningAccumulator<F>,
    eta_powers: &[F; NUM_VAL_STAGES],
    num_chunks: usize,
    k_chunk: usize,
) -> Vec<Vec<F>> {
    let reg_count = REGISTER_COUNT as usize;
    let total = total_lanes();

    // Offsets (canonical lane ordering)
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

    // Eq tables for stage4/stage5 register selection weights.
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

    let mut weights = vec![F::zero(); total];

    // Stage 1
    {
        let coeff = eta_powers[0];
        let g = &bytecode_read_raf_params.stage1_gammas;
        weights[unexp_pc_idx] += coeff * g[0];
        weights[imm_idx] += coeff * g[1];
        for i in 0..NUM_CIRCUIT_FLAGS {
            weights[circuit_start + i] += coeff * g[2 + i];
        }
    }

    // Stage 2
    {
        let coeff = eta_powers[1];
        let g = &bytecode_read_raf_params.stage2_gammas;
        weights[circuit_start + (CircuitFlags::Jump as usize)] += coeff * g[0];
        weights[instr_start + (InstructionFlags::Branch as usize)] += coeff * g[1];
        weights[instr_start + (InstructionFlags::IsRdNotZero as usize)] += coeff * g[2];
        weights[circuit_start + (CircuitFlags::WriteLookupOutputToRD as usize)] += coeff * g[3];
    }

    // Stage 3
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

    // Stage 4
    {
        let coeff = eta_powers[3];
        let g = &bytecode_read_raf_params.stage4_gammas;
        for r in 0..reg_count {
            weights[rd_start + r] += coeff * g[0] * eq_r_register_4[r];
            weights[rs1_start + r] += coeff * g[1] * eq_r_register_4[r];
            weights[rs2_start + r] += coeff * g[2] * eq_r_register_4[r];
        }
    }

    // Stage 5
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

    // Chunk into k_chunk-sized blocks.
    (0..num_chunks)
        .map(|chunk_idx| {
            (0..k_chunk)
                .map(|lane| {
                    let global = chunk_idx * k_chunk + lane;
                    if global < total {
                        weights[global]
                    } else {
                        F::zero()
                    }
                })
                .collect_vec()
        })
        .collect_vec()
}
