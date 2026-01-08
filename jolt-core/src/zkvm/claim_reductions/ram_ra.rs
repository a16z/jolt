//! RAM RA claim reduction sumcheck.
//!
//! Consolidates the four RAM RA claims (from RafEvaluation, ReadWriteChecking, ValEvaluation,
//! ValFinal) into a single claim for the RA virtualization sumcheck. See `mod.rs` for claim
//! coincidence constraints.
//!
//! ## Sumcheck Identity
//!
//! Proves over `(k, c) ∈ {0,1}^{log_K} × {0,1}^{log_T}`:
//!
//! ```text
//! Σ_{k,c} eq_combined(k, c) · ra(k, c) = input_claim
//! ```
//!
//! where `eq_combined` batches the four claims with γ-powers:
//! ```text
//! eq_combined(k, c) = eq(r_addr_1, k)·(eq_raf(c) + γ·eq_val(c))
//!                   + γ²·eq(r_addr_2, k)·(eq_rw(c) + γ·eq_val(c))
//! input_claim = claim_raf + γ·claim_val_final + γ²·claim_rw + γ³·claim_val_eval
//! ```
//!
//! ## Prover Structure
//!
//! The prover is organized into three phases:
//! - **PhaseAddress**: First `log_K` rounds binding address variables
//! - **PhaseCycle1**: First `log_T/2` cycle rounds using prefix-suffix optimization
//! - **PhaseCycle2**: Remaining `log_T/2` cycle rounds using dense sumcheck

use std::array;
use std::sync::Arc;

use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::jolt_device::MemoryLayout;
use num_traits::Zero;
use rayon::prelude::*;
use tracer::instruction::Cycle;

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    utils::{expanding_table::ExpandingTable, math::Math, thread::unsafe_allocate_zero_vec},
    zkvm::{config::OneHotParams, ram::remap_address, witness::VirtualPolynomial},
};

/// Degree bound of the sumcheck round polynomials.
/// Degree 2: one from eq polynomial, one from ra (which is 0 or 1).
const DEGREE_BOUND: usize = 2;

// ============================================================================
// Main Prover Enum
// ============================================================================

/// RAM RA reduction sumcheck prover.
///
/// Reduces four RA claims (from RafEvaluation, ReadWriteChecking, ValEvaluation, ValFinal)
/// into a single claim that can be fed into the RA virtualization sumcheck.
///
/// Organized as a state machine with three phases:
/// - PhaseAddress: Address rounds (log_K rounds)
/// - PhaseCycle1: Prefix-suffix cycle rounds (log_T/2 rounds)
/// - PhaseCycle2: Dense suffix cycle rounds (log_T/2 rounds)
#[derive(Allocative)]
pub struct RamRaClaimReductionSumcheckProver<F: JoltField> {
    phase: RamRaClaimReductionPhase<F>,
    pub params: RaReductionParams<F>,
}

#[derive(Allocative)]
#[allow(clippy::large_enum_variant)]
enum RamRaClaimReductionPhase<F: JoltField> {
    PhaseAddress(PhaseAddressState<F>),
    PhaseCycle1(PhaseCycle1State<F>),
    PhaseCycle2(PhaseCycle2State<F>),
}

impl<F: JoltField> RamRaClaimReductionSumcheckProver<F> {
    /// Create a new RAM RA reduction sumcheck prover.
    #[tracing::instrument(skip_all, name = "RamRaClaimReductionSumcheckProver::initialize")]
    pub fn initialize(
        params: RaReductionParams<F>,
        trace: &[Cycle],
        memory_layout: &MemoryLayout,
        one_hot_params: &OneHotParams,
    ) -> Self {
        let phase = RamRaClaimReductionPhase::PhaseAddress(PhaseAddressState::gen(
            &params,
            trace,
            memory_layout,
            one_hot_params,
        ));
        Self { phase, params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for RamRaClaimReductionSumcheckProver<F>
{
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.params.input_claim()
    }

    #[tracing::instrument(skip_all, name = "RamRaClaimReductionSumcheckProver::compute_message")]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        match &self.phase {
            RamRaClaimReductionPhase::PhaseAddress(state) => {
                state.compute_message(&self.params, round, previous_claim)
            }
            RamRaClaimReductionPhase::PhaseCycle1(state) => {
                state.compute_message(&self.params, previous_claim)
            }
            RamRaClaimReductionPhase::PhaseCycle2(state) => state.compute_message(previous_claim),
        }
    }

    #[tracing::instrument(skip_all, name = "RamRaClaimReductionSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        match &mut self.phase {
            RamRaClaimReductionPhase::PhaseAddress(state) => {
                state.bind(r_j);
                if state.is_last_address_round(&self.params, round) {
                    // Transition to PhaseCycle1
                    self.phase = RamRaClaimReductionPhase::PhaseCycle1(PhaseCycle1State::gen(
                        state,
                        state.sumcheck_challenges.clone(),
                        &self.params,
                    ));
                }
            }
            RamRaClaimReductionPhase::PhaseCycle1(state) => {
                state.bind(r_j);
                if state.should_transition_to_phase2() {
                    // Transition to PhaseCycle2
                    self.phase = RamRaClaimReductionPhase::PhaseCycle2(PhaseCycle2State::gen(
                        state,
                        state.sumcheck_challenges.clone(),
                        &self.params,
                    ));
                }
            }
            RamRaClaimReductionPhase::PhaseCycle2(state) => state.bind(r_j),
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let RamRaClaimReductionPhase::PhaseCycle2(state) = &self.phase else {
            panic!("cache_openings should only be called in PhaseCycle2");
        };

        let log_K = self.params.log_K;
        let r_address_reduced = &sumcheck_challenges[..log_K];
        let r_cycle_reduced = &sumcheck_challenges[log_K..];

        let opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(
            [
                r_address_reduced.iter().rev().copied().collect::<Vec<_>>(),
                r_cycle_reduced.iter().rev().copied().collect::<Vec<_>>(),
            ]
            .concat(),
        );

        // The reduced RA claim is H_prime.final_sumcheck_claim()
        let ra_claim_reduced = state.H_prime.final_sumcheck_claim();

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamRaClaimReduction,
            opening_point,
            ra_claim_reduced,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

// ============================================================================
// Phase Address State
// ============================================================================

/// State for address rounds (first log_K rounds).
///
/// Proves: Σ_k [eq(r_addr_1, k)·G_A[k] + γ²·eq(r_addr_2, k)·G_B[k]]
#[derive(Allocative)]
struct PhaseAddressState<F: JoltField> {
    /// The trace of addresses accessed at each cycle.
    addresses: Arc<Vec<Option<usize>>>,

    /// eq(r_address_1, k) polynomial - bound during address rounds.
    B_1: MultilinearPolynomial<F>,
    /// eq(r_address_2, k) polynomial - bound during address rounds.
    B_2: MultilinearPolynomial<F>,
    /// Expanding table tracking eq(r_addr_reduced, k).
    F: ExpandingTable<F>,
    /// G_A[k] = Σ_{c: address[c]=k} eq_cycle_A(c)
    G_A: Vec<F>,
    /// G_B[k] = Σ_{c: address[c]=k} eq_cycle_B(c)
    G_B: Vec<F>,

    /// Sumcheck challenges bound so far.
    sumcheck_challenges: Vec<F::Challenge>,
}

impl<F: JoltField> PhaseAddressState<F> {
    /// Minimum number of k iterations to parallelize the inner loop.
    const MIN_INNER_PARALLEL_LEN: usize = 1 << 12;

    #[tracing::instrument(skip_all, name = "PhaseAddressState::gen")]
    fn gen(
        params: &RaReductionParams<F>,
        trace: &[Cycle],
        memory_layout: &MemoryLayout,
        one_hot_params: &OneHotParams,
    ) -> Self {
        // Extract addresses from trace
        let addresses: Arc<Vec<Option<usize>>> = Arc::new(
            trace
                .par_iter()
                .map(|cycle| {
                    remap_address(cycle.ram_access().address() as u64, memory_layout)
                        .map(|addr| addr as usize)
                })
                .collect(),
        );

        // Initialize eq tables for addresses
        let B_1 = MultilinearPolynomial::from(EqPolynomial::<F>::evals(&params.r_address_1));
        let B_2 = MultilinearPolynomial::from(EqPolynomial::<F>::evals(&params.r_address_2));

        // Initialize expanding table for tracking eq(r_addr_reduced, k)
        let mut F = ExpandingTable::new(one_hot_params.ram_k, BindingOrder::LowToHigh);
        F.reset(F::one());

        // Compute G_A and G_B arrays
        let (G_A, G_B) = Self::compute_G_arrays(
            &addresses,
            one_hot_params.ram_k,
            &params.r_cycle_raf,
            &params.r_cycle_rw,
            &params.r_cycle_val,
            params.gamma,
        );

        Self {
            addresses,
            B_1,
            B_2,
            F,
            G_A,
            G_B,
            sumcheck_challenges: Vec::new(),
        }
    }

    /// Compute G_A[k] = Σ_{c: address[c]=k} eq_cycle_A(c) and G_B similarly.
    ///
    /// Uses two-table split-eq: split r_cycle into MSB/LSB halves, compute E_hi and E_lo,
    /// then eq(r_cycle, c) = E_hi[c_hi] * E_lo[c_lo] where c = (c_hi << lo_bits) | c_lo.
    #[tracing::instrument(skip_all, name = "PhaseAddressProver::compute_G_arrays")]
    fn compute_G_arrays(
        addresses: &[Option<usize>],
        K: usize,
        r_cycle_raf: &[F::Challenge],
        r_cycle_rw: &[F::Challenge],
        r_cycle_val: &[F::Challenge],
        gamma: F,
    ) -> (Vec<F>, Vec<F>) {
        let T = addresses.len();

        // Two-table split-eq:
        // EqPolynomial::evals uses big-endian bit order: r[0] is MSB, r[last] is LSB.
        // To get contiguous blocks in the cycle index, we split off the LSB half (suffix) as E_lo.
        let log_T = r_cycle_raf.len();
        let lo_bits = log_T / 2;
        let hi_bits = log_T - lo_bits;

        let (r_raf_hi, r_raf_lo) = r_cycle_raf.split_at(hi_bits);
        let (r_rw_hi, r_rw_lo) = r_cycle_rw.split_at(hi_bits);
        let (r_val_hi, r_val_lo) = r_cycle_val.split_at(hi_bits);

        // Compute 5 eq tables in parallel (unscaled), plus E_val_hi_scaled separately
        let ([E_raf_hi, E_raf_lo, E_rw_hi, E_rw_lo, E_val_lo], E_val_hi_scaled) = rayon::join(
            || {
                [r_raf_hi, r_raf_lo, r_rw_hi, r_rw_lo, r_val_lo]
                    .into_par_iter()
                    .map(EqPolynomial::<F>::evals)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            },
            // Scale E_val_hi by gamma upfront to compute G_A and G_B directly
            || EqPolynomial::<F>::evals_with_scaling(r_val_hi, Some(gamma)),
        );

        let in_len = E_raf_lo.len(); // 2^lo_bits
        let out_len = E_raf_hi.len(); // 2^hi_bits

        // Divide work evenly among threads (by c_hi index)
        let num_threads = rayon::current_num_threads();
        let chunk_size = out_len.div_ceil(num_threads);

        // Each thread computes partial G_A and G_B directly
        // G_A = sum_raf + gamma * sum_val, G_B = sum_rw + gamma * sum_val
        // E_val_hi is pre-scaled by gamma, so we compute eq products inline
        let (G_A, G_B): (Vec<F>, Vec<F>) = E_raf_hi
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let mut partial_G_A: Vec<F> = unsafe_allocate_zero_vec(K);
                let mut partial_G_B: Vec<F> = unsafe_allocate_zero_vec(K);

                let chunk_start = chunk_idx * chunk_size;
                for (local_idx, &e_hi_raf) in chunk.iter().enumerate() {
                    let c_hi = chunk_start + local_idx;
                    let e_hi_rw = E_rw_hi[c_hi];
                    let e_hi_val_scaled = E_val_hi_scaled[c_hi]; // Already scaled by gamma
                    let c_hi_base = c_hi * in_len;

                    for c_lo in 0..in_len {
                        let j = c_hi_base + c_lo;
                        if j >= T {
                            break;
                        }

                        if let Some(k) = addresses[j] {
                            let eq_raf = e_hi_raf * E_raf_lo[c_lo];
                            let eq_rw = e_hi_rw * E_rw_lo[c_lo];
                            let eq_val = e_hi_val_scaled * E_val_lo[c_lo];

                            partial_G_A[k] += eq_raf + eq_val;
                            partial_G_B[k] += eq_rw + eq_val;
                        }
                    }
                }

                (partial_G_A, partial_G_B)
            })
            .reduce(
                || (unsafe_allocate_zero_vec(K), unsafe_allocate_zero_vec(K)),
                |(mut acc_A, mut acc_B), (p_A, p_B)| {
                    acc_A
                        .par_iter_mut()
                        .zip(p_A.par_iter())
                        .for_each(|(a, p)| *a += *p);
                    acc_B
                        .par_iter_mut()
                        .zip(p_B.par_iter())
                        .for_each(|(a, p)| *a += *p);
                    (acc_A, acc_B)
                },
            );

        (G_A, G_B)
    }

    fn compute_message(
        &self,
        params: &RaReductionParams<F>,
        round: usize,
        previous_claim: F,
    ) -> UniPoly<F> {
        let m = round + 1;
        let half_len = self.B_1.len() / 2;
        let inner_len = 1 << m;
        // Precompute mask for k % (1 << (m - 1)) -> k & f_index_mask
        let f_index_mask = (1 << (m - 1)) - 1;
        let gamma_squared = params.gamma_squared;

        let [eval_0, eval_c2] = (0..half_len)
            .into_par_iter()
            .map(|k_prime| {
                let B_1_evals = self
                    .B_1
                    .sumcheck_evals_array::<2>(k_prime, BindingOrder::LowToHigh);
                let B_2_evals = self
                    .B_2
                    .sumcheck_evals_array::<2>(k_prime, BindingOrder::LowToHigh);

                let k_start = k_prime << m;
                let k_end = k_start + inner_len;

                let (sum_A_0, sum_A_1, sum_B_0, sum_B_1) =
                    if inner_len >= Self::MIN_INNER_PARALLEL_LEN {
                        (k_start..k_end)
                            .into_par_iter()
                            .fold(
                                || {
                                    (
                                        F::Unreduced::<9>::zero(),
                                        F::Unreduced::<9>::zero(),
                                        F::Unreduced::<9>::zero(),
                                        F::Unreduced::<9>::zero(),
                                    )
                                },
                                |mut acc, k| {
                                    let k_m = (k >> (m - 1)) & 1;
                                    let F_k = self.F[k & f_index_mask];
                                    let G_A_k = self.G_A[k];
                                    let G_B_k = self.G_B[k];

                                    let contrib_A = G_A_k.mul_unreduced::<9>(F_k);
                                    let contrib_B = G_B_k.mul_unreduced::<9>(F_k);

                                    if k_m == 0 {
                                        acc.0 += contrib_A;
                                        acc.2 += contrib_B;
                                    } else {
                                        acc.1 += contrib_A;
                                        acc.3 += contrib_B;
                                    }
                                    acc
                                },
                            )
                            .reduce(
                                || {
                                    (
                                        F::Unreduced::<9>::zero(),
                                        F::Unreduced::<9>::zero(),
                                        F::Unreduced::<9>::zero(),
                                        F::Unreduced::<9>::zero(),
                                    )
                                },
                                |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3),
                            )
                    } else {
                        let mut sum_A_0 = F::Unreduced::<9>::zero();
                        let mut sum_A_1 = F::Unreduced::<9>::zero();
                        let mut sum_B_0 = F::Unreduced::<9>::zero();
                        let mut sum_B_1 = F::Unreduced::<9>::zero();

                        for k in k_start..k_end {
                            let k_m = (k >> (m - 1)) & 1;
                            let F_k = self.F[k & f_index_mask];
                            let G_A_k = self.G_A[k];
                            let G_B_k = self.G_B[k];

                            let contrib_A = G_A_k.mul_unreduced::<9>(F_k);
                            let contrib_B = G_B_k.mul_unreduced::<9>(F_k);

                            if k_m == 0 {
                                sum_A_0 += contrib_A;
                                sum_B_0 += contrib_B;
                            } else {
                                sum_A_1 += contrib_A;
                                sum_B_1 += contrib_B;
                            }
                        }
                        (sum_A_0, sum_A_1, sum_B_0, sum_B_1)
                    };

                let sum_A_0 = F::from_montgomery_reduce::<9>(sum_A_0);
                let sum_A_1 = F::from_montgomery_reduce::<9>(sum_A_1);
                let sum_B_0 = F::from_montgomery_reduce::<9>(sum_B_0);
                let sum_B_1 = F::from_montgomery_reduce::<9>(sum_B_1);

                let inner_A_0 = sum_A_0;
                let inner_A_c2 = sum_A_1 + sum_A_1 - sum_A_0;
                let inner_B_0 = sum_B_0;
                let inner_B_c2 = sum_B_1 + sum_B_1 - sum_B_0;

                [
                    B_1_evals[0] * inner_A_0 + gamma_squared * B_2_evals[0] * inner_B_0,
                    B_1_evals[1] * inner_A_c2 + gamma_squared * B_2_evals[1] * inner_B_c2,
                ]
            })
            .reduce(|| [F::zero(), F::zero()], |a, b| [a[0] + b[0], a[1] + b[1]]);

        UniPoly::from_evals_and_hint(previous_claim, &[eval_0, eval_c2])
    }

    fn bind(&mut self, r_j: F::Challenge) {
        self.sumcheck_challenges.push(r_j);
        self.B_1.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.B_2.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.F.update(r_j);
    }

    fn is_last_address_round(&self, params: &RaReductionParams<F>, round: usize) -> bool {
        round == params.log_K - 1
    }
}

// ============================================================================
// Phase Cycle 1 State (Prefix-Suffix)
// ============================================================================

/// State for first half of cycle rounds using prefix-suffix optimization.
///
/// Uses P/Q buffer structure where:
/// - P_x[c_lo] = eq(r_cycle_x_lo, c_lo)
/// - Q_x[c_lo] = Σ_{c_hi} H[c_lo, c_hi] · eq(r_cycle_x_hi, c_hi)
#[derive(Allocative)]
struct PhaseCycle1State<F: JoltField> {
    /// Prefix eq evaluations for each cycle point.
    P_raf: MultilinearPolynomial<F>,
    P_rw: MultilinearPolynomial<F>,
    P_val: MultilinearPolynomial<F>,

    /// Suffix sums: Q_x[c_lo] = Σ_{c_hi} H[c_lo, c_hi] · eq_x_hi(c_hi)
    Q_raf: MultilinearPolynomial<F>,
    Q_rw: MultilinearPolynomial<F>,
    Q_val: MultilinearPolynomial<F>,

    /// α_1 = eq(r_addr_1, r_addr_reduced) from address rounds
    alpha_1: F,
    /// α_2 = eq(r_addr_2, r_addr_reduced) from address rounds
    alpha_2: F,

    /// Needed for Phase 2 transition
    addresses: Arc<Vec<Option<usize>>>,
    F_values: Vec<F>,
    r_cycle_raf_hi: Vec<F::Challenge>,
    r_cycle_rw_hi: Vec<F::Challenge>,
    r_cycle_val_hi: Vec<F::Challenge>,

    sumcheck_challenges: Vec<F::Challenge>,
}

impl<F: JoltField> PhaseCycle1State<F> {
    /// Generate PhaseCycle1 from PhaseAddress after address rounds complete.
    #[tracing::instrument(skip_all, name = "PhaseCycle1State::gen")]
    fn gen(
        address_state: &mut PhaseAddressState<F>,
        address_challenges: Vec<F::Challenge>,
        params: &RaReductionParams<F>,
    ) -> Self {
        // Get α_1 and α_2 from final B polynomial claims
        let alpha_1 = address_state.B_1.final_sumcheck_claim();
        let alpha_2 = address_state.B_2.final_sumcheck_claim();

        // Get F_values = eq(r_addr_reduced, k) for each k
        let F_values = address_state.F.clone_values();
        let addresses = Arc::clone(&address_state.addresses);

        let log_T = params.log_T;
        let prefix_n_vars = log_T / 2;
        let suffix_n_vars = log_T - prefix_n_vars;

        // Split cycle randomness into suffix (high, first half) and prefix (low, second half)
        // Note: vectors are in BIG_ENDIAN order, so first half is high bits
        let (r_cycle_raf_hi, r_cycle_raf_lo) = params.r_cycle_raf.split_at(suffix_n_vars);
        let (r_cycle_rw_hi, r_cycle_rw_lo) = params.r_cycle_rw.split_at(suffix_n_vars);
        let (r_cycle_val_hi, r_cycle_val_lo) = params.r_cycle_val.split_at(suffix_n_vars);

        // P arrays: eq evaluations over prefix bits
        let P_raf = MultilinearPolynomial::from(EqPolynomial::<F>::evals(r_cycle_raf_lo));
        let P_rw = MultilinearPolynomial::from(EqPolynomial::<F>::evals(r_cycle_rw_lo));
        let P_val = MultilinearPolynomial::from(EqPolynomial::<F>::evals(r_cycle_val_lo));

        // Suffix eq evaluations
        let eq_raf_hi = EqPolynomial::<F>::evals(r_cycle_raf_hi);
        let eq_rw_hi = EqPolynomial::<F>::evals(r_cycle_rw_hi);
        let eq_val_hi = EqPolynomial::<F>::evals(r_cycle_val_hi);

        // Compute Q arrays by iterating over trace
        // Q_x[c_lo] = Σ_{c_hi} H[c_lo, c_hi] · eq_x_hi(c_hi)
        // where H[c] = F_values[addresses[c]]
        let prefix_size = 1 << prefix_n_vars;
        let suffix_size = 1 << suffix_n_vars;

        let (Q_raf, Q_rw, Q_val) = Self::compute_Q_arrays(
            &addresses,
            &F_values,
            &eq_raf_hi,
            &eq_rw_hi,
            &eq_val_hi,
            prefix_size,
            suffix_size,
        );

        Self {
            P_raf,
            P_rw,
            P_val,
            Q_raf: MultilinearPolynomial::from(Q_raf),
            Q_rw: MultilinearPolynomial::from(Q_rw),
            Q_val: MultilinearPolynomial::from(Q_val),
            alpha_1,
            alpha_2,
            addresses,
            F_values,
            r_cycle_raf_hi: r_cycle_raf_hi.to_vec(),
            r_cycle_rw_hi: r_cycle_rw_hi.to_vec(),
            r_cycle_val_hi: r_cycle_val_hi.to_vec(),
            sumcheck_challenges: address_challenges,
        }
    }

    /// Compute Q arrays by iterating over trace.
    #[tracing::instrument(skip_all, name = "PhaseCycle1State::compute_Q_arrays")]
    fn compute_Q_arrays(
        addresses: &[Option<usize>],
        F_values: &[F],
        eq_raf_hi: &[F],
        eq_rw_hi: &[F],
        eq_val_hi: &[F],
        prefix_size: usize,
        _suffix_size: usize,
    ) -> (Vec<F>, Vec<F>, Vec<F>) {
        let chunk_size = 1 << 14;

        addresses
            .par_chunks(chunk_size)
            .enumerate()
            .fold(
                || {
                    (
                        unsafe_allocate_zero_vec(prefix_size),
                        unsafe_allocate_zero_vec(prefix_size),
                        unsafe_allocate_zero_vec(prefix_size),
                    )
                },
                |(mut q_raf, mut q_rw, mut q_val), (chunk_idx, chunk)| {
                    let base_c = chunk_idx * chunk_size;
                    for (i, addr) in chunk.iter().enumerate() {
                        if let Some(k) = addr {
                            let c = base_c + i;
                            let c_lo = c & (prefix_size - 1);
                            let c_hi = c >> prefix_size.trailing_zeros();
                            let h_c = F_values[*k];

                            q_raf[c_lo] += h_c * eq_raf_hi[c_hi];
                            q_rw[c_lo] += h_c * eq_rw_hi[c_hi];
                            q_val[c_lo] += h_c * eq_val_hi[c_hi];
                        }
                    }
                    (q_raf, q_rw, q_val)
                },
            )
            .reduce(
                || {
                    (
                        unsafe_allocate_zero_vec(prefix_size),
                        unsafe_allocate_zero_vec(prefix_size),
                        unsafe_allocate_zero_vec(prefix_size),
                    )
                },
                |(mut acc_raf, mut acc_rw, mut acc_val), (q_raf, q_rw, q_val)| {
                    for (a, q) in acc_raf.iter_mut().zip(q_raf) {
                        *a += q;
                    }
                    for (a, q) in acc_rw.iter_mut().zip(q_rw) {
                        *a += q;
                    }
                    for (a, q) in acc_val.iter_mut().zip(q_val) {
                        *a += q;
                    }
                    (acc_raf, acc_rw, acc_val)
                },
            )
    }

    fn compute_message(&self, params: &RaReductionParams<F>, previous_claim: F) -> UniPoly<F> {
        // Coefficients: α_1, γ²·α_2, (γ·α_1 + γ³·α_2)
        let coeff_raf = self.alpha_1;
        let coeff_rw = params.gamma_squared * self.alpha_2;
        let coeff_val = params.gamma * self.alpha_1 + params.gamma_cubed * self.alpha_2;

        let half_len = self.P_raf.len() / 2;

        let evals = (0..half_len)
            .into_par_iter()
            .map(|j| {
                let p_raf = self
                    .P_raf
                    .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let q_raf = self
                    .Q_raf
                    .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let p_rw = self
                    .P_rw
                    .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let q_rw = self
                    .Q_rw
                    .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let p_val = self
                    .P_val
                    .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let q_val = self
                    .Q_val
                    .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);

                array::from_fn::<_, DEGREE_BOUND, _>(|i| {
                    coeff_raf * p_raf[i] * q_raf[i]
                        + coeff_rw * p_rw[i] * q_rw[i]
                        + coeff_val * p_val[i] * q_val[i]
                })
            })
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |a, b| array::from_fn(|i| a[i] + b[i]),
            );

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn bind(&mut self, r_j: F::Challenge) {
        self.sumcheck_challenges.push(r_j);
        self.P_raf.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.P_rw.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.P_val.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.Q_raf.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.Q_rw.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.Q_val.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn should_transition_to_phase2(&self) -> bool {
        self.P_raf.len() == 1
    }
}

// ============================================================================
// Phase Cycle 2 State (Dense Suffix)
// ============================================================================

/// State for second half of cycle rounds using dense sumcheck.
///
/// After prefix rounds, we have:
/// - H'[c_hi] = Σ_{c_lo} H[c_lo, c_hi] · eq(r_prefix_reduced, c_lo)
/// - eq_x_hi polynomials for the suffix variables
#[derive(Allocative)]
struct PhaseCycle2State<F: JoltField> {
    /// Folded H polynomial: H'[c_hi] = Σ_{c_lo} H[c_lo,c_hi] · eq(r_prefix, c_lo)
    H_prime: MultilinearPolynomial<F>,

    /// Suffix eq evaluations
    eq_raf_hi: MultilinearPolynomial<F>,
    eq_rw_hi: MultilinearPolynomial<F>,
    eq_val_hi: MultilinearPolynomial<F>,

    /// Coefficients
    coeff_raf: F,
    coeff_rw: F,
    coeff_val: F,
}

impl<F: JoltField> PhaseCycle2State<F> {
    /// Generate PhaseCycle2 from PhaseCycle1 after prefix rounds complete.
    #[tracing::instrument(skip_all, name = "PhaseCycle2State::gen")]
    fn gen(
        cycle1_state: &mut PhaseCycle1State<F>,
        sumcheck_challenges: Vec<F::Challenge>,
        params: &RaReductionParams<F>,
    ) -> Self {
        let log_K = params.log_K;
        let log_T = params.log_T;
        let prefix_n_vars = log_T / 2;
        let suffix_n_vars = log_T - prefix_n_vars;

        // Extract cycle prefix challenges (those after address rounds)
        // sumcheck_challenges are in LITTLE_ENDIAN order (low-to-high binding),
        // so we reverse them.
        let r_cycle_prefix: Vec<_> = sumcheck_challenges[log_K..].iter().rev().copied().collect();
        debug_assert_eq!(r_cycle_prefix.len(), prefix_n_vars);

        // Compute eq(r_prefix_reduced, c_lo) evaluations
        // Use LITTLE_ENDIAN to match c_lo iteration pattern (c_lo = 0, 1, 2, ...)
        let eq_prefix = EqPolynomial::<F>::evals(&r_cycle_prefix);

        // Compute H'[c_hi] = Σ_{c_lo} H[c_lo, c_hi] · eq_prefix[c_lo]
        // where H[c] = F_values[addresses[c]]
        let H_prime = Self::compute_H_prime(
            &cycle1_state.addresses,
            &cycle1_state.F_values,
            &eq_prefix,
            prefix_n_vars,
            suffix_n_vars,
        );

        // Suffix eq evaluations scaled by eq(r_prefix_x, r_cycle_prefix_reduced)
        let eq_raf_hi = EqPolynomial::<F>::evals(&cycle1_state.r_cycle_raf_hi);
        let eq_rw_hi = EqPolynomial::<F>::evals(&cycle1_state.r_cycle_rw_hi);
        let eq_val_hi = EqPolynomial::<F>::evals(&cycle1_state.r_cycle_val_hi);

        // Compute scaling factors: eq(r_cycle_x_lo, r_cycle_prefix_reduced)
        // r_cycle_*_lo are in BIG_ENDIAN, so reverse r_cycle_prefix for mle
        let r_cycle_raf_lo: Vec<_> = params.r_cycle_raf[suffix_n_vars..].to_vec();
        let r_cycle_rw_lo: Vec<_> = params.r_cycle_rw[suffix_n_vars..].to_vec();
        let r_cycle_val_lo: Vec<_> = params.r_cycle_val[suffix_n_vars..].to_vec();

        let scale_raf = EqPolynomial::<F>::mle(&r_cycle_raf_lo, &r_cycle_prefix);
        let scale_rw = EqPolynomial::<F>::mle(&r_cycle_rw_lo, &r_cycle_prefix);
        let scale_val = EqPolynomial::<F>::mle(&r_cycle_val_lo, &r_cycle_prefix);

        // Coefficients: α_1·scale_raf, γ²·α_2·scale_rw, (γ·α_1 + γ³·α_2)·scale_val
        let alpha_1 = cycle1_state.alpha_1;
        let alpha_2 = cycle1_state.alpha_2;
        let coeff_raf = alpha_1 * scale_raf;
        let coeff_rw = params.gamma_squared * alpha_2 * scale_rw;
        let coeff_val = (params.gamma * alpha_1 + params.gamma_cubed * alpha_2) * scale_val;

        Self {
            H_prime: MultilinearPolynomial::from(H_prime),
            eq_raf_hi: MultilinearPolynomial::from(eq_raf_hi),
            eq_rw_hi: MultilinearPolynomial::from(eq_rw_hi),
            eq_val_hi: MultilinearPolynomial::from(eq_val_hi),
            coeff_raf,
            coeff_rw,
            coeff_val,
        }
    }

    /// Compute H'[c_hi] = Σ_{c_lo} H[c_lo, c_hi] · eq_prefix[c_lo]
    #[tracing::instrument(skip_all, name = "PhaseCycle2State::compute_H_prime")]
    fn compute_H_prime(
        addresses: &[Option<usize>],
        F_values: &[F],
        eq_prefix: &[F],
        prefix_n_vars: usize,
        suffix_n_vars: usize,
    ) -> Vec<F> {
        let prefix_size = 1 << prefix_n_vars;
        let suffix_size = 1 << suffix_n_vars;
        let chunk_size = 1 << 14;

        addresses
            .par_chunks(chunk_size)
            .enumerate()
            .fold(
                || unsafe_allocate_zero_vec(suffix_size),
                |mut h_prime, (chunk_idx, chunk)| {
                    let base_c = chunk_idx * chunk_size;
                    for (i, addr) in chunk.iter().enumerate() {
                        if let Some(k) = addr {
                            let c = base_c + i;
                            let c_lo = c & (prefix_size - 1);
                            let c_hi = c >> prefix_n_vars;
                            let h_c = F_values[*k];
                            h_prime[c_hi] += h_c * eq_prefix[c_lo];
                        }
                    }
                    h_prime
                },
            )
            .reduce(
                || unsafe_allocate_zero_vec(suffix_size),
                |mut acc, h_prime| {
                    for (a, h) in acc.iter_mut().zip(h_prime) {
                        *a += h;
                    }
                    acc
                },
            )
    }

    fn compute_message(&self, previous_claim: F) -> UniPoly<F> {
        let half_len = self.H_prime.len() / 2;

        let evals = (0..half_len)
            .into_par_iter()
            .map(|j| {
                let h_evals = self
                    .H_prime
                    .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let eq_raf = self
                    .eq_raf_hi
                    .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let eq_rw = self
                    .eq_rw_hi
                    .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let eq_val = self
                    .eq_val_hi
                    .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);

                array::from_fn::<_, DEGREE_BOUND, _>(|i| {
                    h_evals[i]
                        * (self.coeff_raf * eq_raf[i]
                            + self.coeff_rw * eq_rw[i]
                            + self.coeff_val * eq_val[i])
                })
            })
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |a, b| array::from_fn(|i| a[i] + b[i]),
            );

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn bind(&mut self, r_j: F::Challenge) {
        self.H_prime.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_raf_hi.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_rw_hi.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_val_hi.bind_parallel(r_j, BindingOrder::LowToHigh);
    }
}

// ============================================================================
// Shared Parameters
// ============================================================================

/// Shared parameters between prover and verifier.
#[derive(Clone, Allocative)]
pub struct RaReductionParams<F: JoltField> {
    /// γ coefficient for combining claims
    pub gamma: F,
    /// γ² coefficient
    pub gamma_squared: F,
    /// γ³ coefficient
    pub gamma_cubed: F,

    /// r_address_1 = r_address_raf (from RafEvaluation/OutputCheck)
    #[allocative(skip)]
    pub r_address_1: Vec<F::Challenge>,
    /// r_address_2 = r_address_rw (from ReadWriteChecking)
    #[allocative(skip)]
    pub r_address_2: Vec<F::Challenge>,

    /// r_cycle_raf (from SpartanOuter via RafEvaluation)
    #[allocative(skip)]
    pub r_cycle_raf: Vec<F::Challenge>,
    /// r_cycle_rw (from ReadWriteChecking phase 1)
    #[allocative(skip)]
    pub r_cycle_rw: Vec<F::Challenge>,
    /// r_cycle_val (from ValEvaluation/ValFinal in Stage 4)
    #[allocative(skip)]
    pub r_cycle_val: Vec<F::Challenge>,

    /// The four input claims
    pub claim_raf: F,
    pub claim_val_final: F,
    pub claim_rw: F,
    pub claim_val_eval: F,

    /// log_2(K) - number of address rounds
    pub log_K: usize,
    /// log_2(T) - number of cycle rounds
    pub log_T: usize,
}

impl<F: JoltField> RaReductionParams<F> {
    /// Create params from the opening accumulator.
    pub fn new(
        trace_len: usize,
        one_hot_params: &OneHotParams,
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let log_K = one_hot_params.ram_k.log_2();
        let log_T = trace_len.log_2();

        // Get the four RA claims from the accumulator
        let (r_raf, claim_raf) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamRafEvaluation);
        let (r_rw, claim_rw) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_val_eval, claim_val_eval) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamValEvaluation);
        let (r_val_final, claim_val_final) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamValFinalEvaluation,
        );

        // Extract r_address and r_cycle from each opening point
        let (r_address_1, r_cycle_raf) = r_raf.split_at_r(log_K);
        let (r_address_2, r_cycle_rw) = r_rw.split_at_r(log_K);
        let (_, r_cycle_val) = r_val_eval.split_at_r(log_K);

        // Verify coincidences (these should hold by construction)
        debug_assert_eq!(r_address_1, r_val_final.split_at_r(log_K).0);
        debug_assert_eq!(r_address_2, r_val_eval.split_at_r(log_K).0);
        debug_assert_eq!(r_cycle_val, r_val_final.split_at_r(log_K).1);

        // Sample γ for combining claims
        let gamma: F = transcript.challenge_scalar();
        let gamma_squared = gamma * gamma;
        let gamma_cubed = gamma_squared * gamma;

        Self {
            gamma,
            gamma_squared,
            gamma_cubed,
            r_address_1: r_address_1.to_vec(),
            r_address_2: r_address_2.to_vec(),
            r_cycle_raf: r_cycle_raf.to_vec(),
            r_cycle_rw: r_cycle_rw.to_vec(),
            r_cycle_val: r_cycle_val.to_vec(),
            claim_raf,
            claim_val_final,
            claim_rw,
            claim_val_eval,
            log_K,
            log_T,
        }
    }

    fn num_rounds(&self) -> usize {
        self.log_K + self.log_T
    }

    fn input_claim(&self) -> F {
        self.claim_raf
            + self.gamma * self.claim_val_final
            + self.gamma_squared * self.claim_rw
            + self.gamma_cubed * self.claim_val_eval
    }
}

// ============================================================================
// Verifier
// ============================================================================

/// RAM RA reduction sumcheck verifier.
pub struct RamRaClaimReductionSumcheckVerifier<F: JoltField> {
    params: RaReductionParams<F>,
}

impl<F: JoltField> RamRaClaimReductionSumcheckVerifier<F> {
    /// Create a new RAM RA reduction sumcheck verifier.
    pub fn new(
        trace_len: usize,
        one_hot_params: &OneHotParams,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params =
            RaReductionParams::new(trace_len, one_hot_params, opening_accumulator, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for RamRaClaimReductionSumcheckVerifier<F>
{
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<F>) -> F {
        self.params.input_claim()
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r_address_reduced: Vec<_> = sumcheck_challenges[..self.params.log_K]
            .iter()
            .rev()
            .copied()
            .collect();
        let r_cycle_reduced: Vec<_> = sumcheck_challenges[self.params.log_K..]
            .iter()
            .rev()
            .copied()
            .collect();

        // Compute eq_combined(r_address_reduced, r_cycle_reduced)
        let eq_addr_1 = EqPolynomial::<F>::mle(&self.params.r_address_1, &r_address_reduced);
        let eq_addr_2 = EqPolynomial::<F>::mle(&self.params.r_address_2, &r_address_reduced);

        let eq_cycle_raf = EqPolynomial::<F>::mle(&self.params.r_cycle_raf, &r_cycle_reduced);
        let eq_cycle_rw = EqPolynomial::<F>::mle(&self.params.r_cycle_rw, &r_cycle_reduced);
        let eq_cycle_val = EqPolynomial::<F>::mle(&self.params.r_cycle_val, &r_cycle_reduced);

        let eq_cycle_A = eq_cycle_raf + self.params.gamma * eq_cycle_val;
        let eq_cycle_B = eq_cycle_rw + self.params.gamma * eq_cycle_val;

        let eq_combined =
            eq_addr_1 * eq_cycle_A + self.params.gamma_squared * eq_addr_2 * eq_cycle_B;

        // Get the reduced ra claim that was cached by the prover
        let (_, ra_claim_reduced) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamRaClaimReduction,
        );

        eq_combined * ra_claim_reduced
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        // Cache the reduced RA opening point for RA virtualization
        let r_address_reduced = &sumcheck_challenges[..self.params.log_K];
        let r_cycle_reduced = &sumcheck_challenges[self.params.log_K..];

        let opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(
            [
                r_address_reduced.iter().rev().copied().collect::<Vec<_>>(),
                r_cycle_reduced.iter().rev().copied().collect::<Vec<_>>(),
            ]
            .concat(),
        );

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamRaClaimReduction,
            opening_point,
        );
    }
}
