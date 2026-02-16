//! RAM RA claim reduction sumcheck.
//!
//! Consolidates multiple RAM RA claims into a single claim for the RA virtualization sumcheck.
//!
//! **Key design point:** Stage 2 *purposefully aligns* the global rounds so that
//! `RamReadWriteChecking`, `RamRafEvaluation`, and `OutputCheck` all derive the **same**
//! RAM address challenge vector `r_address`, independent of `ReadWriteConfig`.
//! Stage 4 (`RamValCheck`) reuses that same `r_address`.
//!
//! ## Sumcheck Identity
//!
//! Fix the (already-sampled) aligned RAM address point `r_address`, and prove over
//! `c ∈ {0,1}^{log_T}`:
//!
//! ```text
//! Σ_c ( eq_raf(c) + γ·eq_rw(c) + γ²·eq_val(c) ) · ra(r_address, c) = input_claim
//! ```
//!
//! where:
//! ```text
//! input_claim = claim_raf + γ·claim_rw + γ²·claim_val
//! ```
//!
//! Equivalently, since `ra(r_address, c) = Σ_k eq(r_address, k)·ra(k, c)`, this is the same as
//! proving the full identity over `(k, c)` but without explicitly binding the `k` variables.
//!
//! - `claim_raf` is the `RamRa` opening claim from `SumcheckId::RamRafEvaluation`.
//! - `claim_rw` is the `RamRa` opening claim from `SumcheckId::RamReadWriteChecking`.
//! - `claim_val` is the `RamRa` opening claim from `SumcheckId::RamValCheck`.
//!
//! ## Prover Structure
//!
//! The prover is organized into two phases (cycle-only):
//! - **Phase1**: First `log_T/2` cycle rounds using prefix-suffix optimization
//! - **Phase2**: Remaining `log_T/2` cycle rounds using dense sumcheck

use std::array;
use std::sync::Arc;

use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::jolt_device::MemoryLayout;
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
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
    zkvm::{config::OneHotParams, ram::remap_address, witness::VirtualPolynomial},
};

/// Degree bound of the sumcheck round polynomials.
/// Degree 2: one from eq polynomial, one from ra (which is 0 or 1).
const DEGREE_BOUND: usize = 2;

/// RAM RA reduction sumcheck prover.
///
/// Organized as a state machine with two phases over the cycle variables (log_T rounds total):
/// - Phase1: Prefix-suffix rounds (log_T/2 rounds)
/// - Phase2: Dense suffix rounds (log_T/2 rounds)
#[derive(Allocative)]
pub struct RamRaClaimReductionSumcheckProver<F: JoltField> {
    phase: RamRaClaimReductionPhase<F>,
    pub params: RaReductionParams<F>,
}

#[derive(Allocative)]
#[allow(clippy::large_enum_variant)]
enum RamRaClaimReductionPhase<F: JoltField> {
    Phase1(Phase1State<F>),
    Phase2(Phase2State<F>),
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
        let phase = RamRaClaimReductionPhase::Phase1(Phase1State::initialize_from_aligned_address(
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
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "RamRaClaimReductionSumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        match &self.phase {
            RamRaClaimReductionPhase::Phase1(state) => state.compute_message(&self.params, previous_claim),
            RamRaClaimReductionPhase::Phase2(state) => state.compute_message(previous_claim),
        }
    }

    #[tracing::instrument(skip_all, name = "RamRaClaimReductionSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        match &mut self.phase {
            RamRaClaimReductionPhase::Phase1(state) => {
                state.bind(r_j);
                if state.should_transition_to_phase2() {
                    // Transition to Phase2
                    self.phase = RamRaClaimReductionPhase::Phase2(Phase2State::gen(
                        state,
                        state.sumcheck_challenges.clone(),
                        &self.params,
                    ));
                }
            }
            RamRaClaimReductionPhase::Phase2(state) => state.bind(r_j),
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let RamRaClaimReductionPhase::Phase2(state) = &self.phase else {
            panic!("cache_openings should only be called in Phase2");
        };
        let r_cycle_be: Vec<_> = sumcheck_challenges.iter().rev().copied().collect();
        let opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(
            [self.params.r_address.clone(), r_cycle_be].concat(),
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

/// State for Phase1 (first half of cycle rounds) using prefix-suffix optimization.
///
/// Uses P/Q buffer structure where:
/// - P_x[c_lo] = eq(r_cycle_x_lo, c_lo)
/// - Q_x[c_lo] = Σ_{c_hi} H[c_lo, c_hi] · eq(r_cycle_x_hi, c_hi)
#[derive(Allocative)]
struct Phase1State<F: JoltField> {
    /// Prefix eq evaluations for each cycle point.
    P_raf: MultilinearPolynomial<F>,
    P_rw: MultilinearPolynomial<F>,
    P_val: MultilinearPolynomial<F>,

    /// Suffix sums: Q_x[c_lo] = Σ_{c_hi} H[c_lo, c_hi] · eq_x_hi(c_hi)
    Q_raf: MultilinearPolynomial<F>,
    Q_rw: MultilinearPolynomial<F>,
    Q_val: MultilinearPolynomial<F>,

    /// Needed for Phase 2 transition
    addresses: Arc<Vec<Option<usize>>>,
    F_values: Vec<F>,
    r_cycle_raf_hi: Vec<F::Challenge>,
    r_cycle_rw_hi: Vec<F::Challenge>,
    r_cycle_val_hi: Vec<F::Challenge>,

    sumcheck_challenges: Vec<F::Challenge>,
}

impl<F: JoltField> Phase1State<F> {
    /// Initialize Phase1 directly from the aligned `r_address` (cycle-only reduction).
    #[tracing::instrument(skip_all, name = "Phase1State::initialize_from_aligned_address")]
    fn initialize_from_aligned_address(
        params: &RaReductionParams<F>,
        trace: &[Cycle],
        memory_layout: &MemoryLayout,
        one_hot_params: &OneHotParams,
    ) -> Self {
        // Extract addresses from the trace.
        let addresses: Arc<Vec<Option<usize>>> = Arc::new(
            trace
                .par_iter()
                .map(|cycle| {
                    remap_address(cycle.ram_access().address() as u64, memory_layout)
                        .map(|addr| addr as usize)
                })
                .collect(),
        );

        // F_values[k] = eq(r_address, k) for each k.
        let F_values = EqPolynomial::<F>::evals(&params.r_address);
        debug_assert_eq!(F_values.len(), one_hot_params.ram_k);

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
            addresses,
            F_values,
            r_cycle_raf_hi: r_cycle_raf_hi.to_vec(),
            r_cycle_rw_hi: r_cycle_rw_hi.to_vec(),
            r_cycle_val_hi: r_cycle_val_hi.to_vec(),
            sumcheck_challenges: Vec::new(),
        }
    }

    /// Compute Q arrays by iterating over trace.
    #[tracing::instrument(skip_all, name = "Phase1State::compute_Q_arrays")]
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
        // Coefficients: 1, γ, γ² (address is fixed to the aligned `r_address`).
        let coeff_raf = F::one();
        let coeff_rw = params.gamma;
        let coeff_val = params.gamma_squared;

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

/// State for second half of cycle rounds using dense sumcheck.
///
/// After prefix rounds, we have:
/// - H'[c_hi] = Σ_{c_lo} H[c_lo, c_hi] · eq(r_prefix_reduced, c_lo)
/// - eq_x_hi polynomials for the suffix variables
#[derive(Allocative)]
struct Phase2State<F: JoltField> {
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

impl<F: JoltField> Phase2State<F> {
    /// Generate Phase2 from Phase1 after prefix rounds complete.
    #[tracing::instrument(skip_all, name = "Phase2State::gen")]
    fn gen(
        cycle1_state: &mut Phase1State<F>,
        sumcheck_challenges: Vec<F::Challenge>,
        params: &RaReductionParams<F>,
    ) -> Self {
        let log_T = params.log_T;
        let prefix_n_vars = log_T / 2;
        let suffix_n_vars = log_T - prefix_n_vars;

        // Extract cycle prefix challenges.
        // Sumcheck challenges are in LITTLE_ENDIAN order (low-to-high binding), so we reverse them.
        let r_cycle_prefix: Vec<_> = sumcheck_challenges.iter().rev().copied().collect();
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

        // Coefficients: scale_raf, γ·scale_rw, γ²·scale_val (address is fixed).
        let coeff_raf = scale_raf;
        let coeff_rw = params.gamma * scale_rw;
        let coeff_val = params.gamma_squared * scale_val;

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
    #[tracing::instrument(skip_all, name = "Phase2State::compute_H_prime")]
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

/// Shared parameters between prover and verifier.
#[derive(Clone, Allocative)]
pub struct RaReductionParams<F: JoltField> {
    /// γ coefficient for combining claims
    pub gamma: F,
    /// γ² coefficient
    pub gamma_squared: F,

    /// Unified RAM address point (after Stage 2 alignment).
    #[allocative(skip)]
    pub r_address: Vec<F::Challenge>,

    /// r_cycle_raf (from SpartanOuter via RafEvaluation)
    #[allocative(skip)]
    pub r_cycle_raf: Vec<F::Challenge>,
    /// r_cycle_rw (from ReadWriteChecking phase 1)
    #[allocative(skip)]
    pub r_cycle_rw: Vec<F::Challenge>,
    /// r_cycle_val (from ValEvaluation/ValFinal in Stage 4)
    #[allocative(skip)]
    pub r_cycle_val: Vec<F::Challenge>,

    /// The three input claims
    pub claim_raf: F,
    pub claim_rw: F,
    pub claim_val: F,

    /// log_2(K) - number of address bits
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

        // Get the three RA claims from the accumulator.
        let (r_raf, claim_raf) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamRafEvaluation);
        let (r_rw, claim_rw) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_val, claim_val) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamValCheck,
        );

        // Extract r_address and r_cycle from each opening point.
        let (r_address_raf, r_cycle_raf) = r_raf.split_at_r(log_K);
        let (r_address_rw, r_cycle_rw) = r_rw.split_at_r(log_K);
        let (r_address_val, r_cycle_val) = r_val.split_at_r(log_K);

        // Verify unified address (these should hold by construction after Stage 2 alignment).
        debug_assert_eq!(r_address_raf, r_address_rw);
        debug_assert_eq!(r_address_raf, r_address_val);

        // Sample γ for combining claims
        let gamma: F = transcript.challenge_scalar();
        let gamma_squared = gamma * gamma;

        Self {
            gamma,
            gamma_squared,
            r_address: r_address_raf.to_vec(),
            r_cycle_raf: r_cycle_raf.to_vec(),
            r_cycle_rw: r_cycle_rw.to_vec(),
            r_cycle_val: r_cycle_val.to_vec(),
            claim_raf,
            claim_rw,
            claim_val,
            log_K,
            log_T,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for RaReductionParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        // Cycle-only reduction: address is fixed to the aligned `r_address`.
        self.log_T
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.claim_raf + self.gamma * self.claim_rw + self.gamma_squared * self.claim_val
    }

    fn normalize_opening_point(
        &self,
        sumcheck_challenges: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        debug_assert_eq!(sumcheck_challenges.len(), self.num_rounds());
        let r_cycle_be: Vec<_> = sumcheck_challenges.iter().rev().copied().collect();
        OpeningPoint::<BIG_ENDIAN, F>::new([self.r_address.clone(), r_cycle_be].concat())
    }
}

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
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r_cycle_reduced: Vec<_> = sumcheck_challenges
            .iter()
            .rev()
            .copied()
            .collect();

        // Compute eq_combined(r_cycle_reduced) at the fixed aligned address point.
        let eq_cycle_raf = EqPolynomial::<F>::mle(&self.params.r_cycle_raf, &r_cycle_reduced);
        let eq_cycle_rw = EqPolynomial::<F>::mle(&self.params.r_cycle_rw, &r_cycle_reduced);
        let eq_cycle_val = EqPolynomial::<F>::mle(&self.params.r_cycle_val, &r_cycle_reduced);

        let eq_cycle_combined = eq_cycle_raf + self.params.gamma * eq_cycle_rw
            + self.params.gamma_squared * eq_cycle_val;
        let eq_combined = eq_cycle_combined;

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
        // Cache the reduced RA opening point for RA virtualization.
        // The address part is fixed to the aligned `r_address`; the cycle part is the sumcheck's
        // reduced cycle point.
        let r_cycle_be: Vec<_> = sumcheck_challenges.iter().rev().copied().collect();
        let opening_point =
            OpeningPoint::<BIG_ENDIAN, F>::new([self.params.r_address.clone(), r_cycle_be].concat());

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamRaClaimReduction,
            opening_point,
        );
    }
}
