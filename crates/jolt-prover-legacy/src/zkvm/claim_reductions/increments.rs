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
//!    - `RamValCheck` (Stage 4): opened at `r_cycle_stage4`
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
//!   - v_2 = RamInc(r_cycle_stage4)     from RamValCheck
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

use std::sync::Arc;

use allocative::Allocative;
use ark_ff::biginteger::S64;
use ark_std::Zero;
use rayon::prelude::*;
use tracer::instruction::{Cycle, RAMAccess};

use crate::field::{BarrettReduce, FMAdd, JoltField};
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
#[cfg(all(feature = "akita", not(feature = "zk")))]
use crate::poly::opening_proof::LatticeOpening;
#[cfg(any(feature = "zk", all(feature = "akita", not(feature = "zk"))))]
use crate::poly::opening_proof::OpeningId;
use crate::poly::opening_proof::{
    AbstractVerifierOpeningAccumulator, OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator,
    SumcheckId, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::unipoly::UniPoly;
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::{InputClaimConstraint, OutputClaimConstraint};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::transcripts::Transcript;
use crate::utils::accumulation::MedAccumS;
use crate::utils::math::{s64_from_diff_u64s, Math};
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::witness::CommittedPolynomial;

const DEGREE_BOUND: usize = 2;
#[cfg(all(feature = "akita", not(feature = "zk")))]
const INC_VIRTUALIZATION_DEGREE_BOUND: usize = 3;
#[cfg(all(feature = "akita", not(feature = "zk")))]
const UNSIGNED_INC_SHIFT: u128 = 1u128 << 64;

#[derive(Allocative, Clone)]
pub struct IncClaimReductionSumcheckParams<F: JoltField> {
    /// γ, γ², γ³ for batching
    pub gamma_powers: [F; 3],
    pub n_cycle_vars: usize,
    pub r_cycle_stage2: OpeningPoint<BIG_ENDIAN, F>, // RamInc from RamReadWriteChecking
    pub r_cycle_stage4: OpeningPoint<BIG_ENDIAN, F>, // RamInc from RamValCheck
    pub s_cycle_stage4: OpeningPoint<BIG_ENDIAN, F>, // RdInc from RegistersReadWriteChecking
    pub s_cycle_stage5: OpeningPoint<BIG_ENDIAN, F>, // RdInc from RegistersValEvaluation
}

impl<F: JoltField> IncClaimReductionSumcheckParams<F> {
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
        let (r_cycle_stage4, _) = accumulator
            .get_committed_polynomial_opening(CommittedPolynomial::RamInc, SumcheckId::RamValCheck);

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

impl<F: JoltField> SumcheckInstanceParams<F> for IncClaimReductionSumcheckParams<F> {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let [gamma, gamma_sqr, gamma_cub] = self.gamma_powers;

        let (_, v_1) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, v_2) = accumulator
            .get_committed_polynomial_opening(CommittedPolynomial::RamInc, SumcheckId::RamValCheck);
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

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::weighted_openings(&[
            OpeningId::committed(
                CommittedPolynomial::RamInc,
                SumcheckId::RamReadWriteChecking,
            ),
            OpeningId::committed(CommittedPolynomial::RamInc, SumcheckId::RamValCheck),
            OpeningId::committed(
                CommittedPolynomial::RdInc,
                SumcheckId::RegistersReadWriteChecking,
            ),
            OpeningId::committed(
                CommittedPolynomial::RdInc,
                SumcheckId::RegistersValEvaluation,
            ),
        ])
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(&self, _: &dyn OpeningAccumulator<F>) -> Vec<F> {
        let [gamma, gamma_sqr, gamma_cub] = self.gamma_powers;
        vec![gamma, gamma_sqr, gamma_cub]
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        Some(OutputClaimConstraint::all_weighted_openings(&[
            OpeningId::committed(CommittedPolynomial::RamInc, SumcheckId::IncClaimReduction),
            OpeningId::committed(CommittedPolynomial::RdInc, SumcheckId::IncClaimReduction),
        ]))
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        let [gamma, gamma_sqr, _] = self.gamma_powers;

        let opening_point = self.normalize_opening_point(sumcheck_challenges);

        let eq_r2: F = EqPolynomial::mle(&opening_point.r, &self.r_cycle_stage2.r);
        let eq_r4: F = EqPolynomial::mle(&opening_point.r, &self.r_cycle_stage4.r);
        let eq_s4: F = EqPolynomial::mle(&opening_point.r, &self.s_cycle_stage4.r);
        let eq_s5: F = EqPolynomial::mle(&opening_point.r, &self.s_cycle_stage5.r);

        let eq_ram_combined = eq_r2 + gamma * eq_r4;
        let eq_rd_combined = eq_s4 + gamma * eq_s5;

        vec![eq_ram_combined, gamma_sqr * eq_rd_combined]
    }
}

#[derive(Allocative)]
pub struct IncClaimReductionSumcheckProver<F: JoltField> {
    phase: IncClaimReductionPhase<F>,
    pub params: IncClaimReductionSumcheckParams<F>,
}

#[derive(Allocative)]
#[allow(clippy::large_enum_variant)]
enum IncClaimReductionPhase<F: JoltField> {
    Phase1(IncClaimReductionPhase1State<F>),
    Phase2(IncClaimReductionPhase2State<F>),
}

impl<F: JoltField> IncClaimReductionSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "IncClaimReductionSumcheckProver::initialize")]
    pub fn initialize(params: IncClaimReductionSumcheckParams<F>, trace: Arc<Vec<Cycle>>) -> Self {
        let phase = IncClaimReductionPhase::Phase1(IncClaimReductionPhase1State::initialize(
            trace, &params,
        ));
        Self { params, phase }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for IncClaimReductionSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "IncClaimReductionSumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        match &self.phase {
            IncClaimReductionPhase::Phase1(state) => {
                state.compute_message(&self.params, previous_claim)
            }
            IncClaimReductionPhase::Phase2(state) => {
                state.compute_message(&self.params, previous_claim)
            }
        }
    }

    #[tracing::instrument(skip_all, name = "IncClaimReductionSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        match &mut self.phase {
            IncClaimReductionPhase::Phase1(state) => {
                if state.should_transition_to_phase2() {
                    let mut sumcheck_challenges = state.sumcheck_challenges.clone();
                    sumcheck_challenges.push(r_j);
                    self.phase = IncClaimReductionPhase::Phase2(IncClaimReductionPhase2State::gen(
                        &state.trace,
                        &sumcheck_challenges,
                        &self.params,
                    ));
                    return;
                }
                state.bind(r_j);
            }
            IncClaimReductionPhase::Phase2(state) => state.bind(r_j),
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let IncClaimReductionPhase::Phase2(state) = &self.phase else {
            panic!("Should finish sumcheck on phase 2");
        };

        let opening_point = SumcheckInstanceProver::<F, T>::get_params(self)
            .normalize_opening_point(sumcheck_challenges);

        let ram_inc_claim = state.ram_inc.final_sumcheck_claim();
        let rd_inc_claim = state.rd_inc.final_sumcheck_claim();

        accumulator.append_dense(
            CommittedPolynomial::RamInc,
            SumcheckId::IncClaimReduction,
            opening_point.r.clone(),
            ram_inc_claim,
        );
        accumulator.append_dense(
            CommittedPolynomial::RdInc,
            SumcheckId::IncClaimReduction,
            opening_point.r,
            rd_inc_claim,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

#[derive(Allocative)]
struct IncClaimReductionPhase1State<F: JoltField> {
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
}

impl<F: JoltField> IncClaimReductionPhase1State<F> {
    #[tracing::instrument(skip_all, name = "IncClaimReductionPhase1State::initialize")]
    fn initialize(trace: Arc<Vec<Cycle>>, params: &IncClaimReductionSumcheckParams<F>) -> Self {
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

                    let mut acc_ram_0: MedAccumS<F> = MedAccumS::zero();
                    let mut acc_ram_1: MedAccumS<F> = MedAccumS::zero();
                    let mut acc_rd_0: MedAccumS<F> = MedAccumS::zero();
                    let mut acc_rd_1: MedAccumS<F> = MedAccumS::zero();

                    for x_hi in 0..suffix_len {
                        let x = x_lo + (x_hi << prefix_n_vars);
                        let cycle = &trace[x];

                        // RamInc = post_value - pre_value for RAM writes
                        let ram_inc: S64 = match cycle.ram_access() {
                            RAMAccess::Write(w) => s64_from_diff_u64s(w.post_value, w.pre_value),
                            _ => S64::from(0i64),
                        };

                        // RdInc = post_value - pre_value for rd writes
                        let (_, pre_rd, post_rd) = cycle.rd_write().unwrap_or_default();
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
        }
    }

    fn compute_message(
        &self,
        params: &IncClaimReductionSumcheckParams<F>,
        previous_claim: F,
    ) -> UniPoly<F> {
        let [gamma, gamma_sqr, gamma_cub] = params.gamma_powers;
        let half_n = self.P_ram[0].len() / 2;

        let evals = (0..half_n)
            .into_par_iter()
            .fold(
                || [F::zero(); DEGREE_BOUND],
                |mut acc, j| {
                    let p_ram_0 = self.P_ram[0]
                        .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                    let p_ram_1 = self.P_ram[1]
                        .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                    let p_rd_0 = self.P_rd[0]
                        .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                    let p_rd_1 = self.P_rd[1]
                        .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);

                    let q_ram_0 = self.Q_ram[0]
                        .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                    let q_ram_1 = self.Q_ram[1]
                        .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                    let q_rd_0 = self.Q_rd[0]
                        .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                    let q_rd_1 = self.Q_rd[1]
                        .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);

                    // P_ram[0]·Q_ram[0] + γ·P_ram[1]·Q_ram[1]
                    //   + γ²·P_rd[0]·Q_rd[0] + γ³·P_rd[1]·Q_rd[1]
                    for k in 0..DEGREE_BOUND {
                        acc[k] += p_ram_0[k] * q_ram_0[k]
                            + gamma * p_ram_1[k] * q_ram_1[k]
                            + gamma_sqr * p_rd_0[k] * q_rd_0[k]
                            + gamma_cub * p_rd_1[k] * q_rd_1[k];
                    }
                    acc
                },
            )
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |mut a, b| {
                    for k in 0..DEGREE_BOUND {
                        a[k] += b[k];
                    }
                    a
                },
            );

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn bind(&mut self, r_j: F::Challenge) {
        assert!(!self.should_transition_to_phase2());
        self.sumcheck_challenges.push(r_j);

        self.P_ram[0].bind_parallel(r_j, BindingOrder::LowToHigh);
        self.P_ram[1].bind_parallel(r_j, BindingOrder::LowToHigh);
        self.P_rd[0].bind_parallel(r_j, BindingOrder::LowToHigh);
        self.P_rd[1].bind_parallel(r_j, BindingOrder::LowToHigh);

        self.Q_ram[0].bind_parallel(r_j, BindingOrder::LowToHigh);
        self.Q_ram[1].bind_parallel(r_j, BindingOrder::LowToHigh);
        self.Q_rd[0].bind_parallel(r_j, BindingOrder::LowToHigh);
        self.Q_rd[1].bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn should_transition_to_phase2(&self) -> bool {
        self.P_ram[0].len().log_2() == 1
    }
}

#[derive(Allocative)]
struct IncClaimReductionPhase2State<F: JoltField> {
    ram_inc: MultilinearPolynomial<F>,
    rd_inc: MultilinearPolynomial<F>,
    // Combined eq polynomials
    eq_ram: MultilinearPolynomial<F>, // eq(r_stage2, ·) + γ·eq(r_stage4, ·)
    eq_rd: MultilinearPolynomial<F>,  // eq(s_stage4, ·) + γ·eq(s_stage5, ·)
}

impl<F: JoltField> IncClaimReductionPhase2State<F> {
    #[tracing::instrument(skip_all, name = "IncClaimReductionPhase2State::gen")]
    fn gen(
        trace: &[Cycle],
        sumcheck_challenges: &[F::Challenge],
        params: &IncClaimReductionSumcheckParams<F>,
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
                    let mut acc_ram: MedAccumS<F> = MedAccumS::zero();
                    let mut acc_rd: MedAccumS<F> = MedAccumS::zero();

                    for (x_lo, eq_val) in eq_prefix_evals.iter().enumerate() {
                        let x = x_lo + (x_hi << prefix_len.log_2());
                        let cycle = &trace[x];

                        let ram_inc_val: S64 = match cycle.ram_access() {
                            RAMAccess::Write(w) => s64_from_diff_u64s(w.post_value, w.pre_value),
                            _ => S64::from(0i64),
                        };
                        let (_, pre_rd, post_rd) = cycle.rd_write().unwrap_or_default();
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
        }
    }

    fn compute_message(
        &self,
        params: &IncClaimReductionSumcheckParams<F>,
        previous_claim: F,
    ) -> UniPoly<F> {
        let gamma_sqr = params.gamma_powers[1];
        let half_n = self.ram_inc.len() / 2;

        let evals = (0..half_n)
            .into_par_iter()
            .fold(
                || [F::zero(); DEGREE_BOUND],
                |mut acc, j| {
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
                        acc[k] += ram_evals[k] * eq_ram[k] + gamma_sqr * rd_evals[k] * eq_rd[k];
                    }
                    acc
                },
            )
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |mut a, b| {
                    for k in 0..DEGREE_BOUND {
                        a[k] += b[k];
                    }
                    a
                },
            );

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn bind(&mut self, r_j: F::Challenge) {
        self.ram_inc.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.rd_inc.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_ram.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_rd.bind_parallel(r_j, BindingOrder::LowToHigh);
    }
}

pub struct IncClaimReductionSumcheckVerifier<F: JoltField> {
    params: IncClaimReductionSumcheckParams<F>,
}

impl<F: JoltField> IncClaimReductionSumcheckVerifier<F> {
    pub fn new<A: AbstractVerifierOpeningAccumulator<F>>(
        trace_len: usize,
        accumulator: &A,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = IncClaimReductionSumcheckParams::new(trace_len, accumulator, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript, A: AbstractVerifierOpeningAccumulator<F>>
    SumcheckInstanceVerifier<F, T, A> for IncClaimReductionSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(&self, accumulator: &A, sumcheck_challenges: &[F::Challenge]) -> F {
        let [gamma, gamma_sqr, _] = self.params.gamma_powers;

        let opening_point = SumcheckInstanceVerifier::<F, T, A>::get_params(self)
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
            SumcheckId::IncClaimReduction,
        );
        let (_, rd_inc_claim) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::IncClaimReduction,
        );

        ram_inc_claim * eq_ram_combined + gamma_sqr * rd_inc_claim * eq_rd_combined
    }

    fn cache_openings(&self, accumulator: &mut A, sumcheck_challenges: &[F::Challenge]) {
        let opening_point = SumcheckInstanceVerifier::<F, T, A>::get_params(self)
            .normalize_opening_point(sumcheck_challenges);

        accumulator.append_dense(
            CommittedPolynomial::RamInc,
            SumcheckId::IncClaimReduction,
            opening_point.r.clone(),
        );
        accumulator.append_dense(
            CommittedPolynomial::RdInc,
            SumcheckId::IncClaimReduction,
            opening_point.r,
        );
    }
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
#[derive(Allocative, Clone)]
pub struct IncVirtualizationSumcheckParams<F: JoltField> {
    pub gamma_powers: [F; 3],
    pub n_cycle_vars: usize,
    pub r_cycle_stage2: OpeningPoint<BIG_ENDIAN, F>,
    pub r_cycle_stage4: OpeningPoint<BIG_ENDIAN, F>,
    pub s_cycle_stage4: OpeningPoint<BIG_ENDIAN, F>,
    pub s_cycle_stage5: OpeningPoint<BIG_ENDIAN, F>,
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
impl<F: JoltField> IncVirtualizationSumcheckParams<F> {
    pub fn new(
        trace_len: usize,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let gamma: F = transcript.challenge_scalar();
        let gamma_sqr = gamma.square();
        let gamma_cub = gamma_sqr * gamma;

        let (r_cycle_stage2, _) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_cycle_stage4, _) = accumulator
            .get_committed_polynomial_opening(CommittedPolynomial::RamInc, SumcheckId::RamValCheck);
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

#[cfg(all(feature = "akita", not(feature = "zk")))]
impl<F: JoltField> SumcheckInstanceParams<F> for IncVirtualizationSumcheckParams<F> {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let [gamma, gamma_sqr, gamma_cub] = self.gamma_powers;

        let (_, ram_read_write) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, ram_val_check) = accumulator
            .get_committed_polynomial_opening(CommittedPolynomial::RamInc, SumcheckId::RamValCheck);
        let (_, rd_read_write) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rd_val_evaluation) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersValEvaluation,
        );

        ram_read_write
            + gamma * ram_val_check
            + gamma_sqr * rd_read_write
            + gamma_cub * rd_val_evaluation
    }

    fn degree(&self) -> usize {
        INC_VIRTUALIZATION_DEGREE_BOUND
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

#[cfg(all(feature = "akita", not(feature = "zk")))]
#[derive(Allocative)]
pub struct IncVirtualizationSumcheckProver<F: JoltField> {
    pub params: IncVirtualizationSumcheckParams<F>,
    inc: MultilinearPolynomial<F>,
    store: MultilinearPolynomial<F>,
    eq_ram: MultilinearPolynomial<F>,
    eq_rd: MultilinearPolynomial<F>,
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
impl<F: JoltField> IncVirtualizationSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "IncVirtualizationSumcheckProver::initialize")]
    pub fn initialize(params: IncVirtualizationSumcheckParams<F>, trace: Arc<Vec<Cycle>>) -> Self {
        let gamma = params.gamma_powers[0];

        let (eq_ram, eq_rd) = rayon::join(
            || {
                let (eq_r2, eq_r4) = rayon::join(
                    || EqPolynomial::evals(&params.r_cycle_stage2.r),
                    || EqPolynomial::evals(&params.r_cycle_stage4.r),
                );
                eq_r2
                    .into_par_iter()
                    .zip(eq_r4)
                    .map(|(r2, r4)| r2 + gamma * r4)
                    .collect::<Vec<F>>()
            },
            || {
                let (eq_s4, eq_s5) = rayon::join(
                    || EqPolynomial::evals(&params.s_cycle_stage4.r),
                    || EqPolynomial::evals(&params.s_cycle_stage5.r),
                );
                eq_s4
                    .into_par_iter()
                    .zip(eq_s5)
                    .map(|(s4, s5)| s4 + gamma * s5)
                    .collect::<Vec<F>>()
            },
        );

        let (inc, store) = trace
            .par_iter()
            .map(|cycle| {
                let store = matches!(cycle.ram_access(), RAMAccess::Write(_));
                let inc = if store {
                    ram_inc_i128(cycle)
                } else {
                    rd_inc_i128(cycle)
                };
                (F::from_i128(inc), F::from_bool(store))
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();

        Self {
            params,
            inc: inc.into(),
            store: store.into(),
            eq_ram: eq_ram.into(),
            eq_rd: eq_rd.into(),
        }
    }
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for IncVirtualizationSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "IncVirtualizationSumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let gamma_sqr = self.params.gamma_powers[1];
        let half_n = self.inc.len() / 2;

        let evals = (0..half_n)
            .into_par_iter()
            .fold(
                || [F::zero(); INC_VIRTUALIZATION_DEGREE_BOUND],
                |mut acc, j| {
                    let inc = self
                        .inc
                        .sumcheck_evals_array::<INC_VIRTUALIZATION_DEGREE_BOUND>(
                            j,
                            BindingOrder::LowToHigh,
                        );
                    let store = self
                        .store
                        .sumcheck_evals_array::<INC_VIRTUALIZATION_DEGREE_BOUND>(
                            j,
                            BindingOrder::LowToHigh,
                        );
                    let eq_ram = self
                        .eq_ram
                        .sumcheck_evals_array::<INC_VIRTUALIZATION_DEGREE_BOUND>(
                            j,
                            BindingOrder::LowToHigh,
                        );
                    let eq_rd = self
                        .eq_rd
                        .sumcheck_evals_array::<INC_VIRTUALIZATION_DEGREE_BOUND>(
                            j,
                            BindingOrder::LowToHigh,
                        );

                    for k in 0..INC_VIRTUALIZATION_DEGREE_BOUND {
                        let store_coeff = eq_ram[k] * store[k];
                        let non_store_coeff = gamma_sqr * eq_rd[k] * (F::one() - store[k]);
                        acc[k] += inc[k] * (store_coeff + non_store_coeff);
                    }
                    acc
                },
            )
            .reduce(
                || [F::zero(); INC_VIRTUALIZATION_DEGREE_BOUND],
                |mut a, b| {
                    for k in 0..INC_VIRTUALIZATION_DEGREE_BOUND {
                        a[k] += b[k];
                    }
                    a
                },
            );

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    #[tracing::instrument(skip_all, name = "IncVirtualizationSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.inc.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.store.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_ram.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_rd.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = SumcheckInstanceProver::<F, T>::get_params(self)
            .normalize_opening_point(sumcheck_challenges);
        accumulator.append_lattice(
            LatticeOpening::IncVirtualizationInc,
            opening_point.clone(),
            self.inc.final_sumcheck_claim(),
        );
        accumulator.append_lattice(
            LatticeOpening::IncVirtualizationStore,
            opening_point,
            self.store.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
#[derive(Allocative, Clone)]
pub struct UnsignedIncClaimReductionSumcheckParams<F: JoltField> {
    pub n_cycle_vars: usize,
    pub input_point: OpeningPoint<BIG_ENDIAN, F>,
    pub input_claim: F,
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
impl<F: JoltField> UnsignedIncClaimReductionSumcheckParams<F> {
    pub fn new(trace_len: usize, accumulator: &ProverOpeningAccumulator<F>) -> Self {
        let Some((input_point, inc_claim)) = accumulator
            .openings
            .get(&OpeningId::Lattice(LatticeOpening::IncVirtualizationInc))
            .cloned()
        else {
            panic!("IncVirtualization inc opening must be available before unsigned inc reduction");
        };

        Self {
            n_cycle_vars: trace_len.log_2(),
            input_point,
            input_claim: inc_claim + F::from_u128(UNSIGNED_INC_SHIFT),
        }
    }
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
impl<F: JoltField> SumcheckInstanceParams<F> for UnsignedIncClaimReductionSumcheckParams<F> {
    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.input_claim
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

#[cfg(all(feature = "akita", not(feature = "zk")))]
#[derive(Allocative)]
pub struct UnsignedIncClaimReductionSumcheckProver<F: JoltField> {
    pub params: UnsignedIncClaimReductionSumcheckParams<F>,
    unsigned_inc: MultilinearPolynomial<F>,
    eq_input: MultilinearPolynomial<F>,
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
impl<F: JoltField> UnsignedIncClaimReductionSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "UnsignedIncClaimReductionSumcheckProver::initialize")]
    pub fn initialize(
        params: UnsignedIncClaimReductionSumcheckParams<F>,
        trace: Arc<Vec<Cycle>>,
    ) -> Self {
        let unsigned_inc = trace
            .par_iter()
            .map(|cycle| F::from_u128(unsigned_inc_u128(cycle)))
            .collect::<Vec<_>>();
        let eq_input = EqPolynomial::evals(&params.input_point.r);

        Self {
            params,
            unsigned_inc: unsigned_inc.into(),
            eq_input: eq_input.into(),
        }
    }
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for UnsignedIncClaimReductionSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(
        skip_all,
        name = "UnsignedIncClaimReductionSumcheckProver::compute_message"
    )]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let half_n = self.unsigned_inc.len() / 2;

        let evals = (0..half_n)
            .into_par_iter()
            .fold(
                || [F::zero(); DEGREE_BOUND],
                |mut acc, j| {
                    let unsigned_inc = self
                        .unsigned_inc
                        .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                    let eq_input = self
                        .eq_input
                        .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);

                    for k in 0..DEGREE_BOUND {
                        acc[k] += unsigned_inc[k] * eq_input[k];
                    }
                    acc
                },
            )
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |mut a, b| {
                    for k in 0..DEGREE_BOUND {
                        a[k] += b[k];
                    }
                    a
                },
            );

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    #[tracing::instrument(
        skip_all,
        name = "UnsignedIncClaimReductionSumcheckProver::ingest_challenge"
    )]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.unsigned_inc
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_input.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = SumcheckInstanceProver::<F, T>::get_params(self)
            .normalize_opening_point(sumcheck_challenges);
        accumulator.append_lattice(
            LatticeOpening::UnsignedInc,
            opening_point,
            self.unsigned_inc.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
#[derive(Allocative, Clone)]
pub struct UnsignedIncMsbBooleanitySumcheckParams {
    pub n_cycle_vars: usize,
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
impl UnsignedIncMsbBooleanitySumcheckParams {
    pub fn new(trace_len: usize) -> Self {
        Self {
            n_cycle_vars: trace_len.log_2(),
        }
    }
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
impl<F: JoltField> SumcheckInstanceParams<F> for UnsignedIncMsbBooleanitySumcheckParams {
    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        F::zero()
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

#[cfg(all(feature = "akita", not(feature = "zk")))]
#[derive(Allocative)]
pub struct UnsignedIncMsbBooleanitySumcheckProver<F: JoltField> {
    pub params: UnsignedIncMsbBooleanitySumcheckParams,
    msb: MultilinearPolynomial<F>,
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
impl<F: JoltField> UnsignedIncMsbBooleanitySumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "UnsignedIncMsbBooleanitySumcheckProver::initialize")]
    pub fn initialize(
        params: UnsignedIncMsbBooleanitySumcheckParams,
        trace: Arc<Vec<Cycle>>,
    ) -> Self {
        let msb = trace
            .par_iter()
            .map(|cycle| F::from_bool(unsigned_inc_msb(cycle)))
            .collect::<Vec<_>>();

        Self {
            params,
            msb: msb.into(),
        }
    }
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for UnsignedIncMsbBooleanitySumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(
        skip_all,
        name = "UnsignedIncMsbBooleanitySumcheckProver::compute_message"
    )]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let half_n = self.msb.len() / 2;

        let evals = (0..half_n)
            .into_par_iter()
            .fold(
                || [F::zero(); DEGREE_BOUND],
                |mut acc, j| {
                    let msb = self
                        .msb
                        .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);

                    for k in 0..DEGREE_BOUND {
                        acc[k] += msb[k].square() - msb[k];
                    }
                    acc
                },
            )
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |mut a, b| {
                    for k in 0..DEGREE_BOUND {
                        a[k] += b[k];
                    }
                    a
                },
            );

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    #[tracing::instrument(
        skip_all,
        name = "UnsignedIncMsbBooleanitySumcheckProver::ingest_challenge"
    )]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.msb.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = SumcheckInstanceProver::<F, T>::get_params(self)
            .normalize_opening_point(sumcheck_challenges);
        accumulator.append_lattice(
            LatticeOpening::UnsignedIncMsb,
            opening_point,
            self.msb.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
#[derive(Allocative, Clone)]
pub struct UnsignedIncChunkReconstructionSumcheckParams<F: JoltField> {
    pub gamma_powers: Vec<F>,
    pub places: Vec<F>,
    pub delta: F,
    pub input_claim: F,
    pub log_k_chunk: usize,
    pub r_addr_bool: Vec<F::Challenge>,
    pub cycle_point: OpeningPoint<BIG_ENDIAN, F>,
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
impl<F: JoltField> UnsignedIncChunkReconstructionSumcheckParams<F> {
    pub fn new(
        log_k_chunk: usize,
        accumulator: &ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let chunk_count = unsigned_inc_lower_chunk_count(log_k_chunk)
            .expect("unsigned increment chunk size must evenly divide 64 bits");
        let gamma: F = transcript.challenge_scalar();
        let mut gamma_powers = Vec::with_capacity(2 * chunk_count + 1);
        let mut power = F::one();
        for _ in 0..=2 * chunk_count {
            gamma_powers.push(power);
            power *= gamma;
        }
        let delta = gamma_powers[2 * chunk_count];

        let (cycle_point, unsigned_inc) = accumulator
            .openings
            .get(&OpeningId::Lattice(LatticeOpening::UnsignedInc))
            .cloned()
            .expect("unsigned inc opening must be available before chunk reconstruction");
        let unsigned_inc_msb =
            accumulator.get_opening(OpeningId::Lattice(LatticeOpening::UnsignedIncMsb));

        let mut chunk_claims = Vec::with_capacity(chunk_count);
        let mut r_addr_bool = None;
        for chunk_index in 0..chunk_count {
            let (chunk_point, chunk_claim) = accumulator
                .openings
                .get(&OpeningId::Lattice(LatticeOpening::UnsignedIncChunk(
                    chunk_index,
                )))
                .cloned()
                .expect("unsigned inc Booleanity chunk opening must be available");
            if r_addr_bool.is_none() {
                r_addr_bool = Some(chunk_point.r[..log_k_chunk].to_vec());
            }
            chunk_claims.push(chunk_claim);
        }

        let places = unsigned_inc_places::<F>(chunk_count, log_k_chunk);
        let lower_value = unsigned_inc - F::from_u128(UNSIGNED_INC_SHIFT) * unsigned_inc_msb;
        let mut input_claim = delta * lower_value;
        for chunk_index in 0..chunk_count {
            input_claim += gamma_powers[2 * chunk_index]
                + gamma_powers[2 * chunk_index + 1] * chunk_claims[chunk_index];
        }

        Self {
            gamma_powers,
            places,
            delta,
            input_claim,
            log_k_chunk,
            r_addr_bool: r_addr_bool.expect("at least one unsigned inc chunk is required"),
            cycle_point,
        }
    }

    fn chunk_count(&self) -> usize {
        self.places.len()
    }
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
impl<F: JoltField> SumcheckInstanceParams<F> for UnsignedIncChunkReconstructionSumcheckParams<F> {
    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.input_claim
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.log_k_chunk
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let r_address: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness();
        OpeningPoint::new([r_address.r.as_slice(), self.cycle_point.r.as_slice()].concat())
    }
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
#[derive(Allocative)]
pub struct UnsignedIncChunkReconstructionSumcheckProver<F: JoltField> {
    pub params: UnsignedIncChunkReconstructionSumcheckParams<F>,
    chunks: Vec<MultilinearPolynomial<F>>,
    eq_bool: MultilinearPolynomial<F>,
    identity: MultilinearPolynomial<F>,
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
impl<F: JoltField> UnsignedIncChunkReconstructionSumcheckProver<F> {
    #[tracing::instrument(
        skip_all,
        name = "UnsignedIncChunkReconstructionSumcheckProver::initialize"
    )]
    pub fn initialize(
        params: UnsignedIncChunkReconstructionSumcheckParams<F>,
        trace: Arc<Vec<Cycle>>,
    ) -> Self {
        let chunk_indices =
            unsigned_inc_chunk_indices(&trace, params.chunk_count(), params.log_k_chunk);
        let chunk_g = compute_unsigned_inc_chunk_g::<F>(
            &chunk_indices,
            1 << params.log_k_chunk,
            &params.cycle_point.r,
        );
        let chunks = chunk_g
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect();
        let eq_bool = MultilinearPolynomial::from(EqPolynomial::evals(&params.r_addr_bool));
        let identity = MultilinearPolynomial::from(
            (0..(1 << params.log_k_chunk))
                .map(|index| F::from_u64(index as u64))
                .collect::<Vec<_>>(),
        );

        Self {
            params,
            chunks,
            eq_bool,
            identity,
        }
    }
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for UnsignedIncChunkReconstructionSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(
        skip_all,
        name = "UnsignedIncChunkReconstructionSumcheckProver::compute_message"
    )]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let half_n = self.chunks[0].len() / 2;
        let evals = (0..half_n)
            .into_par_iter()
            .fold(
                || [F::zero(); DEGREE_BOUND],
                |mut acc, j| {
                    let eq_bool = self
                        .eq_bool
                        .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                    let identity = self
                        .identity
                        .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);

                    for chunk_index in 0..self.params.chunk_count() {
                        let chunk = self.chunks[chunk_index]
                            .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                        let gamma_even = self.params.gamma_powers[2 * chunk_index];
                        let gamma_odd = self.params.gamma_powers[2 * chunk_index + 1];
                        let place = self.params.places[chunk_index];
                        for k in 0..DEGREE_BOUND {
                            let coeff = gamma_even
                                + gamma_odd * eq_bool[k]
                                + self.params.delta * place * identity[k];
                            acc[k] += chunk[k] * coeff;
                        }
                    }
                    acc
                },
            )
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |mut a, b| {
                    for k in 0..DEGREE_BOUND {
                        a[k] += b[k];
                    }
                    a
                },
            );

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    #[tracing::instrument(
        skip_all,
        name = "UnsignedIncChunkReconstructionSumcheckProver::ingest_challenge"
    )]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.chunks
            .par_iter_mut()
            .for_each(|chunk| chunk.bind_parallel(r_j, BindingOrder::LowToHigh));
        self.eq_bool.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.identity.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = SumcheckInstanceProver::<F, T>::get_params(self)
            .normalize_opening_point(sumcheck_challenges);
        for (chunk_index, chunk) in self.chunks.iter().enumerate() {
            accumulator.append_lattice(
                LatticeOpening::UnsignedIncReconstructedChunk(chunk_index),
                opening_point.clone(),
                chunk.final_sumcheck_claim(),
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
fn ram_inc_i128(cycle: &Cycle) -> i128 {
    match cycle.ram_access() {
        RAMAccess::Write(write) => write.post_value as i128 - write.pre_value as i128,
        _ => 0,
    }
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
fn rd_inc_i128(cycle: &Cycle) -> i128 {
    let (_, pre_value, post_value) = cycle.rd_write().unwrap_or_default();
    post_value as i128 - pre_value as i128
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
pub(crate) fn selected_inc_i128(cycle: &Cycle) -> i128 {
    if matches!(cycle.ram_access(), RAMAccess::Write(_)) {
        ram_inc_i128(cycle)
    } else {
        rd_inc_i128(cycle)
    }
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
pub(crate) fn unsigned_inc_u128(cycle: &Cycle) -> u128 {
    (UNSIGNED_INC_SHIFT as i128 + selected_inc_i128(cycle)) as u128
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
pub(crate) fn unsigned_inc_msb(cycle: &Cycle) -> bool {
    (unsigned_inc_u128(cycle) >> 64) != 0
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
pub(crate) fn unsigned_inc_lower_chunk_count(log_k_chunk: usize) -> Option<usize> {
    (log_k_chunk != 0 && 64_usize.is_multiple_of(log_k_chunk)).then_some(64 / log_k_chunk)
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
fn unsigned_inc_places<F: JoltField>(chunk_count: usize, log_k_chunk: usize) -> Vec<F> {
    let mut places = Vec::with_capacity(chunk_count);
    let radix = F::from_u64(1_u64 << log_k_chunk);
    let mut place = F::one();
    for _ in 0..chunk_count {
        places.push(place);
        place *= radix;
    }
    places
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
pub(crate) fn unsigned_inc_chunk_index(
    cycle: &Cycle,
    chunk_index: usize,
    log_k_chunk: usize,
) -> usize {
    let mask = (1_u128 << log_k_chunk) - 1;
    ((unsigned_inc_u128(cycle) >> (chunk_index * log_k_chunk)) & mask) as usize
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
pub(crate) fn unsigned_inc_chunk_indices(
    trace: &[Cycle],
    chunk_count: usize,
    log_k_chunk: usize,
) -> Vec<Vec<usize>> {
    (0..chunk_count)
        .into_par_iter()
        .map(|chunk_index| {
            trace
                .par_iter()
                .map(|cycle| unsigned_inc_chunk_index(cycle, chunk_index, log_k_chunk))
                .collect()
        })
        .collect()
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
pub(crate) fn compute_unsigned_inc_chunk_g<F: JoltField>(
    chunk_indices: &[Vec<usize>],
    k_chunk: usize,
    r_cycle: &[F::Challenge],
) -> Vec<Vec<F>> {
    if chunk_indices.is_empty() {
        return Vec::new();
    }

    let chunk_count = chunk_indices.len();
    let trace_len = chunk_indices[0].len();
    let log_T = r_cycle.len();
    let lo_bits = log_T / 2;
    let hi_bits = log_T - lo_bits;
    let (r_hi, r_lo) = r_cycle.split_at(hi_bits);

    let (E_hi, E_lo) = rayon::join(
        || EqPolynomial::<F>::evals(r_hi),
        || EqPolynomial::<F>::evals(r_lo),
    );

    let in_len = E_lo.len();
    let num_threads = rayon::current_num_threads();
    let out_len = E_hi.len();
    let chunk_size = out_len.div_ceil(num_threads).max(1);

    E_hi.par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let mut partial: Vec<Vec<F>> =
                (0..chunk_count).map(|_| vec![F::zero(); k_chunk]).collect();
            let mut local: Vec<Vec<F::UnreducedMulU64>> = (0..chunk_count)
                .map(|_| vec![F::UnreducedMulU64::zero(); k_chunk])
                .collect();

            let chunk_start = chunk_idx * chunk_size;
            for (local_idx, &e_hi) in chunk.iter().enumerate() {
                for values in &mut local {
                    values.fill(F::UnreducedMulU64::zero());
                }

                let c_hi = chunk_start + local_idx;
                let c_hi_base = c_hi * in_len;
                for (c_lo, e_lo) in E_lo.iter().enumerate() {
                    let cycle_index = c_hi_base + c_lo;
                    if cycle_index >= trace_len {
                        break;
                    }

                    let add = e_lo.to_unreduced();
                    for chunk_index in 0..chunk_count {
                        let k = chunk_indices[chunk_index][cycle_index];
                        local[chunk_index][k] += add;
                    }
                }

                for chunk_index in 0..chunk_count {
                    for k in 0..k_chunk {
                        let reduced = F::reduce_mul_u64(local[chunk_index][k]);
                        if !reduced.is_zero() {
                            partial[chunk_index][k] += e_hi * reduced;
                        }
                    }
                }
            }
            partial
        })
        .reduce(
            || (0..chunk_count).map(|_| vec![F::zero(); k_chunk]).collect(),
            |mut a, b| {
                for (a_poly, b_poly) in a.iter_mut().zip(b.iter()) {
                    a_poly
                        .par_iter_mut()
                        .zip(b_poly.par_iter())
                        .for_each(|(a_val, b_val)| *a_val += *b_val);
                }
                a
            },
        )
}

#[cfg(all(test, feature = "host", feature = "akita", not(feature = "zk")))]
mod tests {
    #![expect(
        clippy::expect_used,
        clippy::unwrap_used,
        reason = "tests construct prover inputs and assert successful sumcheck execution"
    )]

    use ark_bn254::Fr;

    use super::*;
    use crate::{
        host,
        poly::{multilinear_polynomial::PolynomialEvaluation, opening_proof::OpeningId},
        subprotocols::sumcheck::BatchedSumcheck,
        transcripts::Blake2bTranscript,
    };

    #[test]
    fn inc_virtualization_sumcheck_matches_cached_lattice_openings() {
        let mut program = host::Program::new("muldiv-guest");
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
        let (_, mut trace, _, _) = program.trace(&inputs, &[], &[]);
        trace.resize(trace.len().next_power_of_two(), Cycle::NoOp);

        let trace = Arc::new(trace);
        let log_t = trace.len().log_2();
        let ram_inc = MultilinearPolynomial::from(
            trace
                .iter()
                .map(|cycle| Fr::from_i128(ram_inc_i128(cycle)))
                .collect::<Vec<_>>(),
        );
        let rd_inc = MultilinearPolynomial::from(
            trace
                .iter()
                .map(|cycle| Fr::from_i128(rd_inc_i128(cycle)))
                .collect::<Vec<_>>(),
        );

        let r_cycle_stage2 = challenge_point(3, log_t);
        let r_cycle_stage4 = challenge_point(7, log_t);
        let s_cycle_stage4 = challenge_point(11, log_t);
        let s_cycle_stage5 = challenge_point(17, log_t);

        let mut transcript = Blake2bTranscript::new(b"inc_virtualization_test");
        let mut accumulator = ProverOpeningAccumulator::new(log_t);
        accumulator.append_dense(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
            r_cycle_stage2.r.clone(),
            ram_inc.evaluate(&r_cycle_stage2.r),
        );
        accumulator.append_dense(
            CommittedPolynomial::RamInc,
            SumcheckId::RamValCheck,
            r_cycle_stage4.r.clone(),
            ram_inc.evaluate(&r_cycle_stage4.r),
        );
        accumulator.append_dense(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
            s_cycle_stage4.r.clone(),
            rd_inc.evaluate(&s_cycle_stage4.r),
        );
        accumulator.append_dense(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersValEvaluation,
            s_cycle_stage5.r.clone(),
            rd_inc.evaluate(&s_cycle_stage5.r),
        );
        accumulator.flush_to_transcript(&mut transcript);

        let params =
            IncVirtualizationSumcheckParams::new(trace.len(), &accumulator, &mut transcript);
        let gamma_sqr = params.gamma_powers[1];
        let mut prover = IncVirtualizationSumcheckProver::initialize(params, Arc::clone(&trace));
        let input_claim = prover.params.input_claim(&accumulator);

        let (proof, r_sumcheck, initial_claim) =
            BatchedSumcheck::prove(vec![&mut prover], &mut accumulator, &mut transcript);

        let batch_coeff = initial_claim * input_claim.inverse().unwrap();
        let final_claim = proof
            .compressed_polys
            .iter()
            .zip(&r_sumcheck)
            .fold(initial_claim, |claim, (poly, r_j)| {
                poly.decompress(&claim).evaluate(r_j)
            });

        let (opening_point, inc_claim) = accumulator
            .openings
            .get(&OpeningId::Lattice(LatticeOpening::IncVirtualizationInc))
            .cloned()
            .expect("inc virtualization should cache inc opening");
        let store_claim =
            accumulator.get_opening(OpeningId::Lattice(LatticeOpening::IncVirtualizationStore));

        let eq_r2 = EqPolynomial::<Fr>::mle(&opening_point.r, &r_cycle_stage2.r);
        let eq_r4 = EqPolynomial::<Fr>::mle(&opening_point.r, &r_cycle_stage4.r);
        let eq_s4 = EqPolynomial::<Fr>::mle(&opening_point.r, &s_cycle_stage4.r);
        let eq_s5 = EqPolynomial::<Fr>::mle(&opening_point.r, &s_cycle_stage5.r);
        let gamma = prover.params.gamma_powers[0];
        let ram_coeff = eq_r2 + gamma * eq_r4;
        let rd_coeff = eq_s4 + gamma * eq_s5;
        let expected = inc_claim
            * (ram_coeff * store_claim + gamma_sqr * rd_coeff * (Fr::from_u64(1) - store_claim));

        assert_eq!(final_claim, batch_coeff * expected);
    }

    #[test]
    fn unsigned_inc_claim_reduction_sumcheck_matches_cached_lattice_opening() {
        let mut program = host::Program::new("muldiv-guest");
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
        let (_, mut trace, _, _) = program.trace(&inputs, &[], &[]);
        trace.resize(trace.len().next_power_of_two(), Cycle::NoOp);

        let trace = Arc::new(trace);
        let log_t = trace.len().log_2();
        let selected_inc = MultilinearPolynomial::from(
            trace
                .iter()
                .map(|cycle| Fr::from_i128(selected_inc_i128(cycle)))
                .collect::<Vec<_>>(),
        );

        let input_point = challenge_point(23, log_t);
        let mut transcript = Blake2bTranscript::new(b"unsigned_inc_reduction_test");
        let mut accumulator = ProverOpeningAccumulator::new(log_t);
        accumulator.append_lattice(
            LatticeOpening::IncVirtualizationInc,
            input_point.clone(),
            selected_inc.evaluate(&input_point.r),
        );
        accumulator.flush_to_transcript(&mut transcript);

        let params = UnsignedIncClaimReductionSumcheckParams::new(trace.len(), &accumulator);
        let mut prover =
            UnsignedIncClaimReductionSumcheckProver::initialize(params, Arc::clone(&trace));
        let input_claim = prover.params.input_claim(&accumulator);

        let (proof, r_sumcheck, initial_claim) =
            BatchedSumcheck::prove(vec![&mut prover], &mut accumulator, &mut transcript);

        let batch_coeff = initial_claim * input_claim.inverse().unwrap();
        let final_claim = proof
            .compressed_polys
            .iter()
            .zip(&r_sumcheck)
            .fold(initial_claim, |claim, (poly, r_j)| {
                poly.decompress(&claim).evaluate(r_j)
            });

        let (opening_point, unsigned_inc_claim) = accumulator
            .openings
            .get(&OpeningId::Lattice(LatticeOpening::UnsignedInc))
            .cloned()
            .expect("unsigned inc reduction should cache unsigned inc opening");
        let eq_input = EqPolynomial::<Fr>::mle(&opening_point.r, &input_point.r);

        assert_eq!(final_claim, batch_coeff * eq_input * unsigned_inc_claim);
    }

    #[test]
    fn unsigned_inc_msb_booleanity_sumcheck_caches_lattice_opening() {
        let trace = Arc::new(vec![Cycle::NoOp; 8]);
        let log_t = trace.len().log_2();
        let mut transcript = Blake2bTranscript::new(b"unsigned_inc_msb_booleanity_test");
        let mut accumulator = ProverOpeningAccumulator::new(log_t);
        let params = UnsignedIncMsbBooleanitySumcheckParams::new(trace.len());
        let mut prover =
            UnsignedIncMsbBooleanitySumcheckProver::<Fr>::initialize(params, Arc::clone(&trace));

        let (proof, r_sumcheck, initial_claim) =
            BatchedSumcheck::prove(vec![&mut prover], &mut accumulator, &mut transcript);
        let final_claim = proof
            .compressed_polys
            .iter()
            .zip(&r_sumcheck)
            .fold(initial_claim, |claim, (poly, r_j)| {
                poly.decompress(&claim).evaluate(r_j)
            });
        let msb_claim = accumulator.get_opening(OpeningId::Lattice(LatticeOpening::UnsignedIncMsb));

        assert_eq!(final_claim, Fr::zero());
        assert_eq!(msb_claim, Fr::from_u64(1));
    }

    #[test]
    fn unsigned_inc_chunk_reconstruction_sumcheck_caches_reconstructed_chunks() {
        let mut program = host::Program::new("muldiv-guest");
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
        let (_, mut trace, _, _) = program.trace(&inputs, &[], &[]);
        trace.resize(trace.len().next_power_of_two(), Cycle::NoOp);

        let trace = Arc::new(trace);
        let log_t = trace.len().log_2();
        let log_k_chunk = 4;
        let chunk_count = unsigned_inc_lower_chunk_count(log_k_chunk).unwrap();
        let cycle_point = challenge_point(31, log_t);
        let r_addr_bool = challenge_point(41, log_k_chunk);
        let full_chunk_point =
            OpeningPoint::new([r_addr_bool.r.as_slice(), cycle_point.r.as_slice()].concat());

        let unsigned_inc = MultilinearPolynomial::from(
            trace
                .iter()
                .map(|cycle| Fr::from_u128(unsigned_inc_u128(cycle)))
                .collect::<Vec<_>>(),
        );
        let unsigned_inc_msb_poly = MultilinearPolynomial::from(
            trace
                .iter()
                .map(|cycle| Fr::from_bool(unsigned_inc_msb(cycle)))
                .collect::<Vec<_>>(),
        );
        let chunk_indices = unsigned_inc_chunk_indices(&trace, chunk_count, log_k_chunk);
        let chunk_g =
            compute_unsigned_inc_chunk_g::<Fr>(&chunk_indices, 1 << log_k_chunk, &cycle_point.r);
        let chunk_polys = chunk_g
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect::<Vec<_>>();

        let mut transcript = Blake2bTranscript::new(b"unsigned_inc_chunk_recon");
        let mut accumulator = ProverOpeningAccumulator::new(log_t);
        accumulator.append_lattice(
            LatticeOpening::UnsignedInc,
            cycle_point.clone(),
            unsigned_inc.evaluate(&cycle_point.r),
        );
        accumulator.append_lattice(
            LatticeOpening::UnsignedIncMsb,
            cycle_point.clone(),
            unsigned_inc_msb_poly.evaluate(&cycle_point.r),
        );
        for (chunk_index, chunk_poly) in chunk_polys.iter().enumerate() {
            accumulator.append_lattice(
                LatticeOpening::UnsignedIncChunk(chunk_index),
                full_chunk_point.clone(),
                chunk_poly.evaluate(&r_addr_bool.r),
            );
        }
        accumulator.flush_to_transcript(&mut transcript);

        let params = UnsignedIncChunkReconstructionSumcheckParams::new(
            log_k_chunk,
            &accumulator,
            &mut transcript,
        );
        let mut prover =
            UnsignedIncChunkReconstructionSumcheckProver::initialize(params, Arc::clone(&trace));
        let input_claim = prover.params.input_claim(&accumulator);

        let (proof, r_sumcheck, initial_claim) =
            BatchedSumcheck::prove(vec![&mut prover], &mut accumulator, &mut transcript);
        let final_claim = proof
            .compressed_polys
            .iter()
            .zip(&r_sumcheck)
            .fold(initial_claim, |claim, (poly, r_j)| {
                poly.decompress(&claim).evaluate(r_j)
            });

        let r_address: OpeningPoint<BIG_ENDIAN, Fr> =
            OpeningPoint::<LITTLE_ENDIAN, Fr>::new(r_sumcheck).match_endianness();
        let eq_bool = EqPolynomial::<Fr>::mle(&r_address.r, &r_addr_bool.r);
        let identity = MultilinearPolynomial::from(
            (0..(1 << log_k_chunk))
                .map(|index| Fr::from_u64(index as u64))
                .collect::<Vec<_>>(),
        )
        .evaluate(&r_address.r);
        let mut expected_output = Fr::zero();
        for chunk_index in 0..chunk_count {
            let reconstructed = accumulator.get_opening(OpeningId::Lattice(
                LatticeOpening::UnsignedIncReconstructedChunk(chunk_index),
            ));
            let coeff = prover.params.gamma_powers[2 * chunk_index]
                + prover.params.gamma_powers[2 * chunk_index + 1] * eq_bool
                + prover.params.delta * prover.params.places[chunk_index] * identity;
            expected_output += coeff * reconstructed;
        }

        let batch_coeff = initial_claim * input_claim.inverse().unwrap();
        assert_eq!(final_claim, batch_coeff * expected_output);
    }

    fn challenge_point(seed: u64, log_t: usize) -> OpeningPoint<BIG_ENDIAN, Fr> {
        OpeningPoint::new(
            (0..log_t)
                .map(|index| <Fr as JoltField>::Challenge::from(seed as u128 + index as u128))
                .collect(),
        )
    }
}
