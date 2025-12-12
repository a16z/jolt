//! Inc Polynomial Opening Reduction Sumcheck
//!
//! This module implements an optimized batch opening reduction specifically for the two
//! dense committed polynomials: `RamInc` and `RdInc`.
//!
//! ## Background
//!
//! Currently we have two dense committed polynomials that are opened at multiple points:
//!
//! 1. **RamInc**: Claims are emitted from:
//!    - `RamReadWriteChecking` (Stage 2): opened at `r_cycle_stage2`
//!    - `RamValEvaluation` (Stage 4): opened at `r_cycle_stage4`
//!    - `RamValFinalEvaluation` (Stage 4): opened at `r_cycle_stage4` (same as RamValEvaluation)
//!    
//!    Note: ValEvaluation and ValFinal share the same opening point because they're
//!    in the same batched sumcheck and both normalize using the same sumcheck challenges.
//!    So effectively RamInc has **2 distinct opening points**.
//!
//! 2. **RdInc**: Claims are emitted from:
//!    - `RegistersReadWriteChecking` (Stage 4): opened at `s_cycle_stage4`
//!    - `RegistersValEvaluation` (Stage 5): opened at `s_cycle_stage5`
//!    
//!    So RdInc has **2 distinct opening points**.
//!
//! ## Sumcheck Relation
//!
//! Let:
//!   - v_1 = RamInc(r_cycle_stage2)     from RamReadWriteChecking
//!   - v_2 = RamInc(r_cycle_stage4)     from RamValEvaluation (and RamValFinal)
//!   - w_1 = RdInc(s_cycle_stage4)      from RegistersReadWriteChecking  
//!   - w_2 = RdInc(s_cycle_stage5)      from RegistersValEvaluation
//!
//! Input claim:
//!   v_1 + γ·v_2 + γ²·w_1 + γ³·w_2
//!
//! Sumcheck proves (over log T rounds):
//!   Σ_j RamInc(j) · [eq(r_cycle_stage2, j) + γ·eq(r_cycle_stage4, j)]
//!     + γ² · Σ_j RdInc(j) · [eq(s_cycle_stage4, j) + γ·eq(s_cycle_stage5, j)]
//!   = v_1 + γ·v_2 + γ²·w_1 + γ³·w_2
//!
//! After log T rounds with sumcheck challenges ρ, the final claim is:
//!   RamInc(ρ) · [eq(r_cycle_stage2, ρ) + γ·eq(r_cycle_stage4, ρ)]
//!     + γ² · RdInc(ρ) · [eq(s_cycle_stage4, ρ) + γ·eq(s_cycle_stage5, ρ)]
//!
//! The verifier computes the eq terms and recovers two openings at the SAME point ρ:
//!   - RamInc(ρ)
//!   - RdInc(ρ)
//!
//! ## Degree Analysis
//!
//! Each round polynomial has degree 2:
//!   - eq(r_i, j) contributes degree 1 in the binding variable
//!   - RamInc(j) or RdInc(j) contributes degree 1 in the binding variable
//!   - Product is degree 2
//!
//! ## Prefix-Suffix Optimization
//!
//! We apply the prefix-suffix technique (see Section 8.2 of
//! https://eprint.iacr.org/2025/611.pdf Appendix A) to optimize the first half of
//! sumcheck rounds. The optimization splits the sumcheck into two phases:
//!   - Phase 1 (first log T / 2 rounds): Uses prefix-suffix P/Q buffers
//!   - Phase 2 (remaining rounds): Standard sumcheck on materialized polynomials
//!
//! ## Implementation Outline
//!
//! ### 1. Add SumcheckId variant
//!
//! In `opening_proof.rs`, add:
//!   SumcheckId::IncReduction
//!
//! ### 2. Define IncReductionSumcheckParams
//!
//! Fields:
//!   - gamma: F                           // batching challenge
//!   - gamma_powers: [F; 4]               // γ, γ², γ³, γ⁴ for efficiency
//!   - r_cycle_stage2: OpeningPoint       // RamInc opening from Stage 2
//!   - r_cycle_stage4: OpeningPoint       // RamInc opening from Stage 4
//!   - s_cycle_stage4: OpeningPoint       // RdInc opening from Stage 4
//!   - s_cycle_stage5: OpeningPoint       // RdInc opening from Stage 5
//!   - n_cycle_vars: usize                // log T
//!
//! Methods:
//!   - new(): Fetches opening points from accumulator, samples gamma from transcript
//!   - input_claim(): Returns v_1 + γ·v_2 + γ²·w_1 + γ³·w_2
//!   - degree(): Returns 2
//!   - num_rounds(): Returns n_cycle_vars
//!
//! ### 3. Define IncReductionSumcheckProver (two-phase enum)
//!
//! Phase1 (prefix-suffix):
//!   - P: MultilinearPolynomial     // Combined eq prefix evaluations
//!   - Q: MultilinearPolynomial     // Combined (RamInc + γ²·RdInc) suffix evaluations
//!   - sumcheck_challenges: Vec     // Accumulated challenges for phase transition
//!
//! Phase2 (standard sumcheck):
//!   - ram_inc: MultilinearPolynomial
//!   - rd_inc: MultilinearPolynomial  
//!   - eq_ram: MultilinearPolynomial  // eq(r_cycle_stage2, ·) + γ·eq(r_cycle_stage4, ·)
//!   - eq_rd: MultilinearPolynomial   // eq(s_cycle_stage4, ·) + γ·eq(s_cycle_stage5, ·)
//!
//! ### 4. Define IncReductionSumcheckVerifier
//!
//! expected_output_claim():
//!   - Compute eq_ram_eval = eq(r_cycle_stage2, ρ) + γ·eq(r_cycle_stage4, ρ)
//!   - Compute eq_rd_eval = eq(s_cycle_stage4, ρ) + γ·eq(s_cycle_stage5, ρ)
//!   - Fetch ram_inc_claim and rd_inc_claim from accumulator
//!   - Return ram_inc_claim · eq_ram_eval + γ² · rd_inc_claim · eq_rd_eval
//!
//! cache_openings():
//!   - Append RamInc(ρ) claim with SumcheckId::IncReduction
//!   - Append RdInc(ρ) claim with SumcheckId::IncReduction
//!
//! ### 5. Integration in Prover (Stage 6)
//!
//! IncReduction runs in Stage 6, batched with the RA-related sumchecks:
//!
//! ```text
//! Stage 6 batched sumcheck contains:
//!   - BytecodeReadRaf
//!   - BytecodeBooleanity
//!   - RamBooleanity (moved from Stage 4)
//!   - RamRaVirtualization (moved from Stage 5)
//!   - InstructionRaVirtualization
//!   - InstructionBooleanity
//!   - IncReduction  ← This sumcheck
//! ```
//!
//! All these sumchecks run for log(T) rounds (cycle variables).
//! They share the same sumcheck challenges, producing r_cycle_stage6.
//!
//! Since IncReduction is batched with the RA sumchecks, its challenges ARE `r_cycle_stage6`.
//! This means:
//!   - RamInc opens at r_cycle_stage6
//!   - RdInc opens at r_cycle_stage6
//!   - Each ra_i opens at (ρ_addr, r_cycle_stage6)
//!
//! The cycle component is SHARED across all committed polynomials. The Inc polynomials
//! are dense over cycles only (no address component), while RA polynomials have both
//! address and cycle components. But the cycle portion is aligned.
//!
//! ### 6. Coordination with Stage 7
//!
//! After Stage 6:
//!   - RamInc has 1 claim at r_cycle_stage6 (from IncReduction)
//!   - RdInc has 1 claim at r_cycle_stage6 (from IncReduction)
//!   - RA polynomials have 2 claims each at (r_addr_*, r_cycle_stage6)
//!
//! Stage 7 then runs RaAddressReduction to align the RA polynomials' address portions.
//! Inc polynomials are NOT touched in Stage 7 (they're already fully reduced).
//!
//! After Stage 7:
//!   - RamInc: 1 claim at r_cycle_stage6 (unchanged)
//!   - RdInc: 1 claim at r_cycle_stage6 (unchanged)
//!   - Each ra_i: 1 claim at (ρ_addr, r_cycle_stage6)
//!
//! All committed polynomials share r_cycle_stage6 as the cycle component of their
//! opening point. They go directly to Dory in Stage 8.
//!
//! ### 7. Benefits
//!
//!   - Prover efficiency: Avoids expensive generic opening reduction for dense Inc polynomials
//!   - Proof size: Reduces from 5 claims to 2 claims (both at r_cycle_stage6)
//!   - Alignment: RamInc and RdInc share r_cycle_stage6 with ALL RA polynomials
//!   - Maximum sharing: The cycle component is unified across all committed polynomials

use std::sync::Arc;

use allocative::Allocative;
use ark_ff::biginteger::S64;
use ark_std::Zero;
use rayon::prelude::*;
use tracer::instruction::{Cycle, RAMAccess};

use crate::field::{BarrettReduce, FMAdd, JoltField};
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
use crate::utils::accumulation::Acc6S;
use crate::utils::math::{s64_from_diff_u64s, Math};
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::witness::CommittedPolynomial;

const DEGREE_BOUND: usize = 2;

// ============================================================================
// PARAMS
// ============================================================================

#[derive(Allocative, Clone)]
pub struct IncReductionSumcheckParams<F: JoltField> {
    /// γ, γ², γ³ for batching
    pub gamma_powers: [F; 3],
    pub n_cycle_vars: usize,
    pub r_cycle_stage2: OpeningPoint<BIG_ENDIAN, F>, // RamInc from RamReadWriteChecking
    pub r_cycle_stage4: OpeningPoint<BIG_ENDIAN, F>, // RamInc from RamValEvaluation/RamValFinal
    pub s_cycle_stage4: OpeningPoint<BIG_ENDIAN, F>, // RdInc from RegistersReadWriteChecking
    pub s_cycle_stage5: OpeningPoint<BIG_ENDIAN, F>, // RdInc from RegistersValEvaluation
}

impl<F: JoltField> IncReductionSumcheckParams<F> {
    pub fn new(
        trace_len: usize,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let gamma: F = transcript.challenge_scalar();
        let gamma_sqr = gamma.square();
        let gamma_cub = gamma_sqr * gamma;

        // Fetch opening points from accumulator
        let (r_cycle_stage2, _) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_cycle_stage4, _) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::RamValEvaluation,
        );

        // Debug assert: ValEvaluation and ValFinal have same opening point
        #[cfg(debug_assertions)]
        {
            let (r_cycle_stage4_final, _) = accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamInc,
                SumcheckId::RamValFinalEvaluation,
            );
            debug_assert_eq!(
                r_cycle_stage4.r, r_cycle_stage4_final.r,
                "ValEvaluation and ValFinal should have same RamInc opening point"
            );
        }

        let (s_cycle_stage4, _) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (s_cycle_stage5, _) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersValEvaluation,
        );

        Self {
            gamma_powers: [gamma, gamma_sqr, gamma_cub],
            n_cycle_vars: trace_len.log_2(),
            r_cycle_stage2,
            r_cycle_stage4,
            s_cycle_stage4,
            s_cycle_stage5,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for IncReductionSumcheckParams<F> {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let [gamma, gamma_sqr, gamma_cub] = self.gamma_powers;

        let (_, v_1) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, v_2) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::RamValEvaluation,
        );
        // Note: v_2 already includes ValFinal claim (same point, combined)

        let (_, w_1) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, w_2) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersValEvaluation,
        );

        v_1 + gamma * v_2 + gamma_sqr * w_1 + gamma_cub * w_2
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.n_cycle_vars
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

// ============================================================================
// PROVER
// ============================================================================

#[derive(Allocative)]
#[allow(clippy::large_enum_variant, private_interfaces)]
pub enum IncReductionSumcheckProver<F: JoltField> {
    Phase1(IncReductionPhase1Prover<F>),
    Phase2(IncReductionPhase2Prover<F>),
}

impl<F: JoltField> IncReductionSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "IncReductionSumcheckProver::initialize")]
    pub fn initialize(params: IncReductionSumcheckParams<F>, trace: Arc<Vec<Cycle>>) -> Self {
        Self::Phase1(IncReductionPhase1Prover::initialize(trace, params))
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for IncReductionSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        match self {
            Self::Phase1(prover) => &prover.params,
            Self::Phase2(prover) => &prover.params,
        }
    }

    #[tracing::instrument(skip_all, name = "IncReductionSumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        match self {
            Self::Phase1(prover) => prover.compute_message(previous_claim),
            Self::Phase2(prover) => prover.compute_message(previous_claim),
        }
    }

    #[tracing::instrument(skip_all, name = "IncReductionSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        match self {
            Self::Phase1(prover) => {
                if prover.should_transition_to_phase2() {
                    let params = prover.params.clone();
                    let mut sumcheck_challenges = prover.sumcheck_challenges.clone();
                    sumcheck_challenges.push(r_j);
                    *self = Self::Phase2(IncReductionPhase2Prover::gen(
                        &prover.trace,
                        &sumcheck_challenges,
                        params,
                    ));
                    return;
                }
                prover.bind(r_j);
            }
            Self::Phase2(prover) => prover.bind(r_j),
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let Self::Phase2(prover) = self else {
            panic!("Should finish sumcheck on phase 2");
        };

        let opening_point = SumcheckInstanceProver::<F, T>::get_params(self)
            .normalize_opening_point(sumcheck_challenges);

        let ram_inc_claim = prover.ram_inc.final_sumcheck_claim();
        let rd_inc_claim = prover.rd_inc.final_sumcheck_claim();

        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RamInc,
            SumcheckId::IncReduction,
            opening_point.r.clone(),
            ram_inc_claim,
        );
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RdInc,
            SumcheckId::IncReduction,
            opening_point.r,
            rd_inc_claim,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        match self {
            Self::Phase1(prover) => flamegraph.visit_root(prover),
            Self::Phase2(prover) => flamegraph.visit_root(prover),
        }
    }
}

// ============================================================================
// PHASE 1: Prefix-Suffix Sumcheck
// ============================================================================

#[derive(Allocative)]
struct IncReductionPhase1Prover<F: JoltField> {
    // P buffers: prefix eq evaluations (one per opening point)
    // P_ram[0] = eq(r_cycle_stage2_lo, ·)
    // P_ram[1] = eq(r_cycle_stage4_lo, ·)
    P_ram: [MultilinearPolynomial<F>; 2],
    // P_rd[0] = eq(s_cycle_stage4_lo, ·)
    // P_rd[1] = eq(s_cycle_stage5_lo, ·)
    P_rd: [MultilinearPolynomial<F>; 2],

    // Q buffers: suffix-weighted polynomial evaluations
    // Q_ram[i](j_lo) = Σ_{j_hi} RamInc(j) · eq(r_i_hi, j_hi)
    Q_ram: [MultilinearPolynomial<F>; 2],
    // Q_rd[i](j_lo) = Σ_{j_hi} RdInc(j) · eq(s_i_hi, j_hi)
    Q_rd: [MultilinearPolynomial<F>; 2],

    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    sumcheck_challenges: Vec<F::Challenge>,
    params: IncReductionSumcheckParams<F>,
}

impl<F: JoltField> IncReductionPhase1Prover<F> {
    #[tracing::instrument(skip_all, name = "IncReductionPhase1Prover::initialize")]
    fn initialize(trace: Arc<Vec<Cycle>>, params: IncReductionSumcheckParams<F>) -> Self {
        let n_vars = params.n_cycle_vars;
        let prefix_n_vars = n_vars / 2;
        let suffix_n_vars = n_vars - prefix_n_vars;
        let prefix_len = 1 << prefix_n_vars;
        let suffix_len = 1 << suffix_n_vars;

        // Split each opening point into hi (suffix) and lo (prefix)
        // Big-endian: hi is first half, lo is second half
        let (r2_hi, r2_lo) = params.r_cycle_stage2.split_at(suffix_n_vars);
        let (r4_hi, r4_lo) = params.r_cycle_stage4.split_at(suffix_n_vars);
        let (s4_hi, s4_lo) = params.s_cycle_stage4.split_at(suffix_n_vars);
        let (s5_hi, s5_lo) = params.s_cycle_stage5.split_at(suffix_n_vars);

        // P buffers: prefix eq evaluations
        let P_ram_0 = EqPolynomial::evals(&r2_lo.r);
        let P_ram_1 = EqPolynomial::evals(&r4_lo.r);
        let P_rd_0 = EqPolynomial::evals(&s4_lo.r);
        let P_rd_1 = EqPolynomial::evals(&s5_lo.r);

        // Suffix eq evaluations (for computing Q)
        let eq_r2_hi = EqPolynomial::evals(&r2_hi.r);
        let eq_r4_hi = EqPolynomial::evals(&r4_hi.r);
        let eq_s4_hi = EqPolynomial::evals(&s4_hi.r);
        let eq_s5_hi = EqPolynomial::evals(&s5_hi.r);

        // Q buffers: sum over suffix indices
        let mut Q_ram_0 = unsafe_allocate_zero_vec(prefix_len);
        let mut Q_ram_1 = unsafe_allocate_zero_vec(prefix_len);
        let mut Q_rd_0 = unsafe_allocate_zero_vec(prefix_len);
        let mut Q_rd_1 = unsafe_allocate_zero_vec(prefix_len);

        // Right-size chunks based on number of threads
        // Use ceiling division to ensure roughly equal work per thread
        let num_threads = rayon::current_num_threads();
        let chunk_size = prefix_len.div_ceil(num_threads).max(1);

        (
            Q_ram_0.par_chunks_mut(chunk_size),
            Q_ram_1.par_chunks_mut(chunk_size),
            Q_rd_0.par_chunks_mut(chunk_size),
            Q_rd_1.par_chunks_mut(chunk_size),
        )
            .into_par_iter()
            .enumerate()
            .for_each(|(chunk_i, (q_ram_0, q_ram_1, q_rd_0, q_rd_1))| {
                for i in 0..q_ram_0.len() {
                    let x_lo = chunk_i * chunk_size + i;

                    let mut acc_ram_0: Acc6S<F> = Acc6S::zero();
                    let mut acc_ram_1: Acc6S<F> = Acc6S::zero();
                    let mut acc_rd_0: Acc6S<F> = Acc6S::zero();
                    let mut acc_rd_1: Acc6S<F> = Acc6S::zero();

                    for x_hi in 0..suffix_len {
                        let x = x_lo + (x_hi << prefix_n_vars);
                        let cycle = &trace[x];

                        // RamInc = post_value - pre_value for RAM writes
                        let ram_inc: S64 = match cycle.ram_access() {
                            RAMAccess::Write(w) => s64_from_diff_u64s(w.post_value, w.pre_value),
                            _ => S64::from(0i64),
                        };

                        // RdInc = post_value - pre_value for rd writes
                        let (_, pre_rd, post_rd) = cycle.rd_write();
                        let rd_inc: S64 = s64_from_diff_u64s(post_rd, pre_rd);

                        acc_ram_0.fmadd(&eq_r2_hi[x_hi], &ram_inc);
                        acc_ram_1.fmadd(&eq_r4_hi[x_hi], &ram_inc);
                        acc_rd_0.fmadd(&eq_s4_hi[x_hi], &rd_inc);
                        acc_rd_1.fmadd(&eq_s5_hi[x_hi], &rd_inc);
                    }

                    q_ram_0[i] = acc_ram_0.barrett_reduce();
                    q_ram_1[i] = acc_ram_1.barrett_reduce();
                    q_rd_0[i] = acc_rd_0.barrett_reduce();
                    q_rd_1[i] = acc_rd_1.barrett_reduce();
                }
            });

        Self {
            P_ram: [P_ram_0.into(), P_ram_1.into()],
            P_rd: [P_rd_0.into(), P_rd_1.into()],
            Q_ram: [Q_ram_0.into(), Q_ram_1.into()],
            Q_rd: [Q_rd_0.into(), Q_rd_1.into()],
            trace,
            sumcheck_challenges: Vec::new(),
            params,
        }
    }

    fn compute_message(&self, previous_claim: F) -> UniPoly<F> {
        let [gamma, gamma_sqr, gamma_cub] = self.params.gamma_powers;

        let mut evals = [F::zero(); DEGREE_BOUND];
        let mut eval_at_1 = F::zero();

        for j in 0..self.P_ram[0].len() / 2 {
            let p_ram_0 =
                self.P_ram[0].sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let p_ram_1 =
                self.P_ram[1].sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let p_rd_0 =
                self.P_rd[0].sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let p_rd_1 =
                self.P_rd[1].sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);

            let q_ram_0 =
                self.Q_ram[0].sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let q_ram_1 =
                self.Q_ram[1].sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let q_rd_0 =
                self.Q_rd[0].sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let q_rd_1 =
                self.Q_rd[1].sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);

            // P_ram[0]·Q_ram[0] + γ·P_ram[1]·Q_ram[1]
            //   + γ²·P_rd[0]·Q_rd[0] + γ³·P_rd[1]·Q_rd[1]
            for k in 0..DEGREE_BOUND {
                evals[k] += p_ram_0[k] * q_ram_0[k]
                    + gamma * p_ram_1[k] * q_ram_1[k]
                    + gamma_sqr * p_rd_0[k] * q_rd_0[k]
                    + gamma_cub * p_rd_1[k] * q_rd_1[k];
            }

            // eval_at_1 for the hint
            eval_at_1 += self.P_ram[0].get_bound_coeff(j * 2 + 1)
                * self.Q_ram[0].get_bound_coeff(j * 2 + 1)
                + gamma
                    * self.P_ram[1].get_bound_coeff(j * 2 + 1)
                    * self.Q_ram[1].get_bound_coeff(j * 2 + 1)
                + gamma_sqr
                    * self.P_rd[0].get_bound_coeff(j * 2 + 1)
                    * self.Q_rd[0].get_bound_coeff(j * 2 + 1)
                + gamma_cub
                    * self.P_rd[1].get_bound_coeff(j * 2 + 1)
                    * self.Q_rd[1].get_bound_coeff(j * 2 + 1);
        }

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn bind(&mut self, r_j: F::Challenge) {
        assert!(!self.should_transition_to_phase2());
        self.sumcheck_challenges.push(r_j);

        self.P_ram[0].bind(r_j, BindingOrder::LowToHigh);
        self.P_ram[1].bind(r_j, BindingOrder::LowToHigh);
        self.P_rd[0].bind(r_j, BindingOrder::LowToHigh);
        self.P_rd[1].bind(r_j, BindingOrder::LowToHigh);

        self.Q_ram[0].bind(r_j, BindingOrder::LowToHigh);
        self.Q_ram[1].bind(r_j, BindingOrder::LowToHigh);
        self.Q_rd[0].bind(r_j, BindingOrder::LowToHigh);
        self.Q_rd[1].bind(r_j, BindingOrder::LowToHigh);
    }

    fn should_transition_to_phase2(&self) -> bool {
        self.P_ram[0].len().log_2() == 1
    }
}

// ============================================================================
// PHASE 2: Standard Sumcheck
// ============================================================================

#[derive(Allocative)]
struct IncReductionPhase2Prover<F: JoltField> {
    ram_inc: MultilinearPolynomial<F>,
    rd_inc: MultilinearPolynomial<F>,
    // Combined eq polynomials
    eq_ram: MultilinearPolynomial<F>, // eq(r_stage2, ·) + γ·eq(r_stage4, ·)
    eq_rd: MultilinearPolynomial<F>,  // eq(s_stage4, ·) + γ·eq(s_stage5, ·)
    params: IncReductionSumcheckParams<F>,
}

impl<F: JoltField> IncReductionPhase2Prover<F> {
    #[tracing::instrument(skip_all, name = "IncReductionPhase2Prover::gen")]
    fn gen(
        trace: &[Cycle],
        sumcheck_challenges: &[F::Challenge],
        params: IncReductionSumcheckParams<F>,
    ) -> Self {
        let n_vars = params.n_cycle_vars;
        let prefix_n_vars = n_vars / 2;
        let n_remaining_rounds = n_vars - sumcheck_challenges.len();

        let r_prefix: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness();

        let gamma = params.gamma_powers[0];

        // Compute eq evaluations for prefix bound
        let (_, r2_lo) = params.r_cycle_stage2.split_at(n_vars - prefix_n_vars);
        let (_, r4_lo) = params.r_cycle_stage4.split_at(n_vars - prefix_n_vars);
        let (_, s4_lo) = params.s_cycle_stage4.split_at(n_vars - prefix_n_vars);
        let (_, s5_lo) = params.s_cycle_stage5.split_at(n_vars - prefix_n_vars);

        let eq_r2_prefix = EqPolynomial::mle_endian(&r_prefix, &r2_lo);
        let eq_r4_prefix = EqPolynomial::mle_endian(&r_prefix, &r4_lo);
        let eq_s4_prefix = EqPolynomial::mle_endian(&r_prefix, &s4_lo);
        let eq_s5_prefix = EqPolynomial::mle_endian(&r_prefix, &s5_lo);

        // Suffix eq evaluations scaled by prefix contributions
        let (r2_hi, _) = params.r_cycle_stage2.split_at(n_vars - prefix_n_vars);
        let (r4_hi, _) = params.r_cycle_stage4.split_at(n_vars - prefix_n_vars);
        let (s4_hi, _) = params.s_cycle_stage4.split_at(n_vars - prefix_n_vars);
        let (s5_hi, _) = params.s_cycle_stage5.split_at(n_vars - prefix_n_vars);

        // Combined eq polynomials: eq_ram = eq_r2 + γ·eq_r4, eq_rd = eq_s4 + γ·eq_s5
        let (eq_ram, eq_rd) = rayon::join(
            || {
                let (eq_r2, eq_r4) = rayon::join(
                    || EqPolynomial::evals_serial(&r2_hi.r, Some(eq_r2_prefix)),
                    || EqPolynomial::evals_serial(&r4_hi.r, Some(eq_r4_prefix)),
                );
                eq_r2
                    .par_iter()
                    .zip(eq_r4.par_iter())
                    .map(|(e2, e4)| *e2 + gamma * e4)
                    .collect::<Vec<F>>()
            },
            || {
                let (eq_s4, eq_s5) = rayon::join(
                    || EqPolynomial::evals_serial(&s4_hi.r, Some(eq_s4_prefix)),
                    || EqPolynomial::evals_serial(&s5_hi.r, Some(eq_s5_prefix)),
                );
                eq_s4
                    .par_iter()
                    .zip(eq_s5.par_iter())
                    .map(|(e4, e5)| *e4 + gamma * e5)
                    .collect::<Vec<F>>()
            },
        );

        // Materialize Inc polynomials
        let eq_prefix_evals = EqPolynomial::evals(&r_prefix.r);
        let prefix_len = eq_prefix_evals.len();
        let suffix_len = 1 << n_remaining_rounds;
        let mut ram_inc = unsafe_allocate_zero_vec(suffix_len);
        let mut rd_inc = unsafe_allocate_zero_vec(suffix_len);

        let num_threads = rayon::current_num_threads();
        let chunk_size = suffix_len.div_ceil(num_threads).max(1);

        (
            ram_inc.par_chunks_mut(chunk_size),
            rd_inc.par_chunks_mut(chunk_size),
        )
            .into_par_iter()
            .enumerate()
            .for_each(|(chunk_i, (ram_chunk, rd_chunk))| {
                for i in 0..ram_chunk.len() {
                    let x_hi = chunk_i * chunk_size + i;
                    let mut acc_ram: Acc6S<F> = Acc6S::zero();
                    let mut acc_rd: Acc6S<F> = Acc6S::zero();

                    for (x_lo, eq_val) in eq_prefix_evals.iter().enumerate() {
                        let x = x_lo + (x_hi << prefix_len.log_2());
                        let cycle = &trace[x];

                        let ram_inc_val: S64 = match cycle.ram_access() {
                            RAMAccess::Write(w) => s64_from_diff_u64s(w.post_value, w.pre_value),
                            _ => S64::from(0i64),
                        };
                        let (_, pre_rd, post_rd) = cycle.rd_write();
                        let rd_inc_val: S64 = s64_from_diff_u64s(post_rd, pre_rd);

                        acc_ram.fmadd(eq_val, &ram_inc_val);
                        acc_rd.fmadd(eq_val, &rd_inc_val);
                    }

                    ram_chunk[i] = acc_ram.barrett_reduce();
                    rd_chunk[i] = acc_rd.barrett_reduce();
                }
            });

        Self {
            ram_inc: ram_inc.into(),
            rd_inc: rd_inc.into(),
            eq_ram: eq_ram.into(),
            eq_rd: eq_rd.into(),
            params,
        }
    }

    fn compute_message(&mut self, previous_claim: F) -> UniPoly<F> {
        let gamma_sqr = self.params.gamma_powers[1];
        let half_n = self.ram_inc.len() / 2;

        let mut evals = [F::zero(); DEGREE_BOUND];
        for j in 0..half_n {
            let ram_evals = self
                .ram_inc
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let rd_evals = self
                .rd_inc
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let eq_ram = self
                .eq_ram
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let eq_rd = self
                .eq_rd
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);

            for k in 0..DEGREE_BOUND {
                evals[k] += ram_evals[k] * eq_ram[k] + gamma_sqr * rd_evals[k] * eq_rd[k];
            }
        }

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn bind(&mut self, r_j: F::Challenge) {
        self.ram_inc.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.rd_inc.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_ram.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_rd.bind_parallel(r_j, BindingOrder::LowToHigh);
    }
}

// ============================================================================
// VERIFIER
// ============================================================================

pub struct IncReductionSumcheckVerifier<F: JoltField> {
    params: IncReductionSumcheckParams<F>,
}

impl<F: JoltField> IncReductionSumcheckVerifier<F> {
    pub fn new(
        trace_len: usize,
        accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = IncReductionSumcheckParams::new(trace_len, accumulator, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for IncReductionSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let [gamma, gamma_sqr, _] = self.params.gamma_powers;

        let opening_point = SumcheckInstanceVerifier::<F, T>::get_params(self)
            .normalize_opening_point(sumcheck_challenges);

        // Compute eq evaluations at final point
        let eq_r2 = EqPolynomial::mle(&opening_point.r, &self.params.r_cycle_stage2.r);
        let eq_r4 = EqPolynomial::mle(&opening_point.r, &self.params.r_cycle_stage4.r);
        let eq_s4 = EqPolynomial::mle(&opening_point.r, &self.params.s_cycle_stage4.r);
        let eq_s5 = EqPolynomial::mle(&opening_point.r, &self.params.s_cycle_stage5.r);

        let eq_ram_combined = eq_r2 + gamma * eq_r4;
        let eq_rd_combined = eq_s4 + gamma * eq_s5;

        // Fetch final claims from accumulator
        let (_, ram_inc_claim) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::IncReduction,
        );
        let (_, rd_inc_claim) = accumulator
            .get_committed_polynomial_opening(CommittedPolynomial::RdInc, SumcheckId::IncReduction);

        ram_inc_claim * eq_ram_combined + gamma_sqr * rd_inc_claim * eq_rd_combined
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = SumcheckInstanceVerifier::<F, T>::get_params(self)
            .normalize_opening_point(sumcheck_challenges);

        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RamInc,
            SumcheckId::IncReduction,
            opening_point.r.clone(),
        );
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RdInc,
            SumcheckId::IncReduction,
            opening_point.r,
        );
    }
}
