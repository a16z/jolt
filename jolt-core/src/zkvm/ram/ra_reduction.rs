//! RAM RA claim reduction sumcheck
//!
//! This sumcheck consolidates the four different RAM RA claims into a single claim.
//! The single claim is then fed into the RA virtualization sumcheck, which decomposes
//! it into claims about the individual `ra_i` polynomials.
//!
//! ## The Four RA Claims
//!
//! The following sumchecks emit RA claims that need to be consolidated:
//!
//! | Sumcheck | Opening Point | Stage |
//! |----------|---------------|-------|
//! | RamReadWriteChecking | `ra(r_address_rw, r_cycle_rw)` | Stage 2 |
//! | RamRafEvaluation | `ra(r_address_raf, r_cycle_raf)` | Stage 2 |
//! | RamValEvaluation | `ra(r_address_rw, r_cycle_val)` | Stage 4 |
//! | RamValFinal | `ra(r_address_raf, r_cycle_val)` | Stage 4 |
//!
//! ## Confirmed Coincidences
//!
//! The following equalities hold due to the stage structure:
//!
//! - `r_address_raf = r_address_val_final` — Both derive from first `log_K` challenges of Stage 2.
//!   OutputCheck and RafEvaluation are in the same batched sumcheck.
//! - `r_address_val_eval = r_address_rw` — ValEvaluation inherits `r_address` from RamReadWriteChecking.
//! - `r_cycle_val_eval = r_cycle_val_final` — Both are in Stage 4 with `log_T` rounds each.
//!
//! ## Constraints for Coincidences to Hold
//!
//! 1. OutputCheck and RafEvaluation MUST be in the same batched sumcheck (Stage 2),
//!    and both must use challenges `[0 .. log_K]` for `r_address`.
//! 2. ValEvaluation and ValFinal MUST be in the same batched sumcheck (Stage 4),
//!    and have the same `num_rounds = log_T`.
//! 3. ValEvaluation MUST read `r_address` from RamReadWriteChecking's opening.
//! 4. ValFinal MUST read `r_address` from OutputCheck's opening.
//!
//! ## Address Groups
//!
//! The four claims naturally group into two address groups:
//!
//! - **Group A** (`r_address_1 = r_address_raf`):
//!   - `ra(r_address_1, r_cycle_raf) = claim_raf`
//!   - `ra(r_address_1, r_cycle_val) = claim_val_final`
//!
//! - **Group B** (`r_address_2 = r_address_rw`):
//!   - `ra(r_address_2, r_cycle_rw) = claim_rw`
//!   - `ra(r_address_2, r_cycle_val) = claim_val_eval`
//!
//! ## The Reduction Sumcheck
//!
//! We prove the following identity via sumcheck over `(k, c) ∈ {0,1}^{log_K} × {0,1}^{log_T}`:
//!
//! ```text
//! Σ_{k,c} eq_combined(k, c) · ra(k, c) = input_claim
//! ```
//!
//! where:
//!
//! ```text
//! eq_combined(k, c) =
//!     eq(r_address_1, k) · (eq(r_cycle_raf, c) + γ · eq(r_cycle_val, c))
//!   + γ² · eq(r_address_2, k) · (eq(r_cycle_rw, c) + γ · eq(r_cycle_val, c))
//! ```
//!
//! and:
//!
//! ```text
//! input_claim = claim_raf + γ · claim_val_final + γ² · claim_rw + γ³ · claim_val_eval
//! ```
//!
//! After binding `k → r_address_reduced` and `c → r_cycle_reduced`, the expected output is:
//!
//! ```text
//! eq_combined(r_address_reduced, r_cycle_reduced) · ra(r_address_reduced, r_cycle_reduced)
//! ```
//!
//! This yields the single reduced claim:
//!
//! ```text
//! ra(r_address_reduced, r_cycle_reduced) = ra_claim_reduced
//! ```
//!
//! ## Binding Order
//!
//! The sumcheck binds variables in the order: **address variables first** (low-to-high),
//! then **cycle variables** (low-to-high). This matches the polynomial layout `ra(address, cycle)`
//! and the structure used in `OneHotPolynomialProverOpening`.
//!
//! ## Implementation Structure (Two Phases)
//!
//! ### Phase 1: Address Rounds (log_K rounds)
//!
//! During these rounds, we bind address variables while maintaining:
//! - `B_1`, `B_2`: eq polynomials at `r_address_1`, `r_address_2`
//! - `F_1`, `F_2`: expanding tables tracking partial eq evaluations
//! - `G_A[k] = Σ_{c: address[c]=k} eq_cycle_A(c)` where `eq_cycle_A = eq(r_cycle_raf, ·) + γ·eq(r_cycle_val, ·)`
//! - `G_B[k] = Σ_{c: address[c]=k} eq_cycle_B(c)` where `eq_cycle_B = eq(r_cycle_rw, ·) + γ·eq(r_cycle_val, ·)`
//!
//! The round polynomial sums contributions from both address groups:
//! ```text
//! inner_sum = Σ_k B_1[k] · G_A[k] + γ² · Σ_k B_2[k] · G_B[k]
//! ```
//!
//! After all address rounds, `F_1` and `F_2` contain `eq(r_address_1, k)` and `eq(r_address_2, k)` for all k.
//!
//! ### Phase 2: Cycle Rounds (log_T rounds)
//!
//! For each cycle c, we look up `address[c]` and compute:
//! ```text
//! contribution = F_1[address[c]] · eq_cycle_A(c) + γ² · F_2[address[c]] · eq_cycle_B(c)
//! ```
//!
//! This uses Gruen's optimization on `eq_cycle_A` and `eq_cycle_B`.
//!
//! ## Output
//!
//! After this sumcheck, the RA virtualization sumcheck proves:
//!
//! ```text
//! Σ_c eq(r_cycle_reduced, c) · Π_{i=0}^{d−1} ra_i(r_address_reduced_i, c) = ra_claim_reduced
//! ```
//!
//! where `r_address_reduced` is split into chunks `r_address_reduced_i` according to the
//! one-hot decomposition parameters.

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
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    utils::{expanding_table::ExpandingTable, math::Math, thread::unsafe_allocate_zero_vec},
    zkvm::{
        config::OneHotParams,
        ram::remap_address,
        witness::VirtualPolynomial,
    },
};

/// Degree bound of the sumcheck round polynomials.
/// Degree 2: one from eq polynomial, one from ra (which is 0 or 1).
const DEGREE_BOUND: usize = 2;

/// RAM RA reduction sumcheck prover.
///
/// Reduces four RA claims (from RafEvaluation, ReadWriteChecking, ValEvaluation, ValFinal)
/// into a single claim that can be fed into the RA virtualization sumcheck.
#[derive(Allocative)]
pub struct RamRaReductionSumcheckProver<F: JoltField> {
    /// The trace of addresses accessed at each cycle.
    /// `addresses[c] = Some(k)` if address k was accessed at cycle c.
    addresses: Arc<Vec<Option<usize>>>,

    // ========== Group A state (r_address_1 = r_address_raf) ==========
    /// eq(r_address_1, k) polynomial - bound during address rounds
    B_1: MultilinearPolynomial<F>,
    /// Expanding table for r_address_1, tracking partial eq evaluations
    F_1: ExpandingTable<F>,
    /// G_A[k] = Σ_{c: address[c]=k} eq_cycle_A(c)
    G_A: Vec<F>,

    // ========== Group B state (r_address_2 = r_address_rw) ==========
    /// eq(r_address_2, k) polynomial - bound during address rounds
    B_2: MultilinearPolynomial<F>,
    /// Expanding table for r_address_2, tracking partial eq evaluations
    F_2: ExpandingTable<F>,
    /// G_B[k] = Σ_{c: address[c]=k} eq_cycle_B(c)
    G_B: Vec<F>,

    // ========== Phase 2 state (cycle rounds) ==========
    /// H_1[c] = F_1[address[c]] - set after address rounds complete
    /// During cycle rounds, this stores the combined coefficient for each cycle.
    H_combined: MultilinearPolynomial<F>,
    /// eq_cycle_A = eq(r_cycle_raf, ·) + γ · eq(r_cycle_val, ·)
    /// Using Gruen optimization for the underlying eq polynomial
    eq_cycle_raf: GruenSplitEqPolynomial<F>,
    /// eq_cycle_rw: eq(r_cycle_rw, ·)
    eq_cycle_rw: GruenSplitEqPolynomial<F>,
    /// eq_cycle_val: eq(r_cycle_val, ·)
    eq_cycle_val: GruenSplitEqPolynomial<F>,

    #[allocative(skip)]
    params: RaReductionParams<F>,
}

/// RAM RA reduction sumcheck verifier.
pub struct RamRaReductionSumcheckVerifier<F: JoltField> {
    params: RaReductionParams<F>,
}

/// Shared parameters between prover and verifier.
#[derive(Clone)]
struct RaReductionParams<F: JoltField> {
    /// γ coefficient for combining claims
    gamma: F,
    /// γ² coefficient
    gamma_squared: F,
    /// γ³ coefficient
    gamma_cubed: F,

    /// r_address_1 = r_address_raf (from RafEvaluation/OutputCheck)
    r_address_1: Vec<F::Challenge>,
    /// r_address_2 = r_address_rw (from ReadWriteChecking)
    r_address_2: Vec<F::Challenge>,

    /// r_cycle_raf (from SpartanOuter via RafEvaluation)
    r_cycle_raf: Vec<F::Challenge>,
    /// r_cycle_rw (from ReadWriteChecking phase 1)
    r_cycle_rw: Vec<F::Challenge>,
    /// r_cycle_val (from ValEvaluation/ValFinal in Stage 4)
    r_cycle_val: Vec<F::Challenge>,

    /// The four input claims
    claim_raf: F,
    claim_val_final: F,
    claim_rw: F,
    claim_val_eval: F,

    /// log_2(K) - number of address rounds
    log_K: usize,
    /// log_2(T) - number of cycle rounds
    log_T: usize,
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
        let (r_raf, claim_raf) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamRafEvaluation,
        );
        let (r_rw, claim_rw) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_val_eval, claim_val_eval) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamValEvaluation,
        );
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

impl<F: JoltField> RamRaReductionSumcheckProver<F> {
    /// Create a new RAM RA reduction sumcheck prover.
    #[tracing::instrument(skip_all, name = "RamRaReductionSumcheckProver::gen")]
    pub fn gen(
        trace: &[Cycle],
        memory_layout: &MemoryLayout,
        one_hot_params: &OneHotParams,
        opening_accumulator: &ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = RaReductionParams::new(
            trace.len(),
            one_hot_params,
            opening_accumulator,
            transcript,
        );

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

        // Initialize expanding tables for tracking partial eq evaluations
        let mut F_1 = ExpandingTable::new(one_hot_params.ram_k, BindingOrder::LowToHigh);
        F_1.reset(F::one());
        let mut F_2 = ExpandingTable::new(one_hot_params.ram_k, BindingOrder::LowToHigh);
        F_2.reset(F::one());

        // Compute G_A[k] = Σ_{c: address[c]=k} eq_cycle_A(c)
        // and G_B[k] = Σ_{c: address[c]=k} eq_cycle_B(c)
        // where eq_cycle_A = eq(r_cycle_raf, ·) + γ·eq(r_cycle_val, ·)
        // and eq_cycle_B = eq(r_cycle_rw, ·) + γ·eq(r_cycle_val, ·)
        let (G_A, G_B) = Self::compute_G_arrays(
            &addresses,
            one_hot_params.ram_k,
            &params.r_cycle_raf,
            &params.r_cycle_rw,
            &params.r_cycle_val,
            params.gamma,
        );

        // Initialize Gruen eq polynomials for cycle rounds
        let eq_cycle_raf = GruenSplitEqPolynomial::new(&params.r_cycle_raf, BindingOrder::LowToHigh);
        let eq_cycle_rw = GruenSplitEqPolynomial::new(&params.r_cycle_rw, BindingOrder::LowToHigh);
        let eq_cycle_val = GruenSplitEqPolynomial::new(&params.r_cycle_val, BindingOrder::LowToHigh);

        // H_combined will be initialized after address rounds complete
        let H_combined = MultilinearPolynomial::from(Vec::<F>::new());

        Self {
            addresses,
            B_1,
            F_1,
            G_A,
            B_2,
            F_2,
            G_B,
            H_combined,
            eq_cycle_raf,
            eq_cycle_rw,
            eq_cycle_val,
            params,
        }
    }

    /// Compute G_A[k] = Σ_{c: address[c]=k} eq_cycle_A(c) and G_B similarly.
    ///
    /// eq_cycle_A(c) = eq(r_cycle_raf, c) + γ·eq(r_cycle_val, c)
    /// eq_cycle_B(c) = eq(r_cycle_rw, c) + γ·eq(r_cycle_val, c)
    #[tracing::instrument(skip_all, name = "RamRaReductionSumcheckProver::compute_G_arrays")]
    fn compute_G_arrays(
        addresses: &[Option<usize>],
        K: usize,
        r_cycle_raf: &[F::Challenge],
        r_cycle_rw: &[F::Challenge],
        r_cycle_val: &[F::Challenge],
        gamma: F,
    ) -> (Vec<F>, Vec<F>) {
        // Materialize full eq tables for each cycle point
        let eq_raf = EqPolynomial::<F>::evals(r_cycle_raf);
        let eq_rw = EqPolynomial::<F>::evals(r_cycle_rw);
        let eq_val = EqPolynomial::<F>::evals(r_cycle_val);

        // Compute G arrays in parallel using fold-reduce pattern
        let chunk_size = 1 << 14; // Process 16K cycles at a time
        let (G_A, G_B) = addresses
            .par_chunks(chunk_size)
            .enumerate()
            .fold(
                || (unsafe_allocate_zero_vec(K), unsafe_allocate_zero_vec(K)),
                |(mut partial_A, mut partial_B), (chunk_idx, chunk)| {
                    let base_c = chunk_idx * chunk_size;
                    for (i, addr) in chunk.iter().enumerate() {
                        if let Some(k) = addr {
                            let c = base_c + i;
                            // eq_cycle_A(c) = eq_raf[c] + γ·eq_val[c]
                            partial_A[*k] += eq_raf[c] + gamma * eq_val[c];
                            // eq_cycle_B(c) = eq_rw[c] + γ·eq_val[c]
                            partial_B[*k] += eq_rw[c] + gamma * eq_val[c];
                        }
                    }
                    (partial_A, partial_B)
                },
            )
            .reduce(
                || (unsafe_allocate_zero_vec(K), unsafe_allocate_zero_vec(K)),
                |(mut acc_A, mut acc_B), (partial_A, partial_B)| {
                    for (a, p) in acc_A.iter_mut().zip(partial_A) {
                        *a += p;
                    }
                    for (b, p) in acc_B.iter_mut().zip(partial_B) {
                        *b += p;
                    }
                    (acc_A, acc_B)
                },
            );

        (G_A, G_B)
    }

    /// Compute the round polynomial for address rounds (phase 1).
    ///
    /// The identity during address rounds is:
    /// Σ_k [eq(r_address_1, k) · G_A[k] + γ² · eq(r_address_2, k) · G_B[k]]
    ///
    /// At round m (0-indexed), we bind the first m variables and sum over the rest.
    fn address_round_compute_message(&self, round: usize, previous_claim: F) -> UniPoly<F> {
        let m = round + 1;
        let half_len = self.B_1.len() / 2;

        // For each k' in {0,1}^{log_K - m}, compute contribution to round polynomial
        let evals = (0..half_len)
            .into_par_iter()
            .map(|k_prime| {
                // Get sumcheck evals for B_1 and B_2 at position k'
                let B_1_evals = self.B_1.sumcheck_evals_array::<2>(k_prime, BindingOrder::LowToHigh);
                let B_2_evals = self.B_2.sumcheck_evals_array::<2>(k_prime, BindingOrder::LowToHigh);

                // Sum over all k that share prefix k'
                // k ranges from k' << m to (k' + 1) << m
                let mut inner_A = [F::zero(); 2];
                let mut inner_B = [F::zero(); 2];

                for k in (k_prime << m)..((k_prime + 1) << m) {
                    // Extract the m-th bit (the variable being bound in this round)
                    // k is a global index, so we need to mask to get just bit (m-1)
                    let k_m = (k >> (m - 1)) & 1;
                    let F_1_k = self.F_1[k % (1 << (m - 1))];
                    let F_2_k = self.F_2[k % (1 << (m - 1))];
                    let G_A_k = self.G_A[k];
                    let G_B_k = self.G_B[k];

                    let contrib_A = G_A_k * F_1_k;
                    let contrib_B = G_B_k * F_2_k;

                    // For eval at 0: only k_m = 0 contributes
                    // For eval at 2: linear extrapolation from 0 and 1
                    if k_m == 0 {
                        inner_A[0] += contrib_A;
                        inner_A[1] += -contrib_A; // c2 term
                        inner_B[0] += contrib_B;
                        inner_B[1] += -contrib_B;
                    } else {
                        inner_A[1] += contrib_A + contrib_A; // c2 term
                        inner_B[1] += contrib_B + contrib_B;
                    }
                }

                // Combine with B evals: B[k'] · inner[k'] + γ² · B2[k'] · inner_B[k']
                [
                    B_1_evals[0] * inner_A[0]
                        + self.params.gamma_squared * B_2_evals[0] * inner_B[0],
                    B_1_evals[1] * inner_A[1]
                        + self.params.gamma_squared * B_2_evals[1] * inner_B[1],
                ]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        UniPoly::from_evals_and_hint(previous_claim, &evals.to_vec())
    }

    /// Compute the round polynomial for cycle rounds (phase 2).
    ///
    /// After address rounds, we have:
    /// Σ_c [H_combined[c] · (eq_raf(c) + γ·eq_val(c)) + γ² · H_combined[c] · (eq_rw(c) + γ·eq_val(c))]
    /// = Σ_c H_combined[c] · (eq_raf(c) + γ·eq_val(c) + γ²·eq_rw(c) + γ³·eq_val(c))
    ///
    /// But we need to track Group A and B separately, so H_combined stores:
    /// H_combined[c] = F_1[address[c]] + γ² · F_2[address[c]]
    fn cycle_round_compute_message(&self, _round: usize, previous_claim: F) -> UniPoly<F> {
        // The identity is:
        // Σ_c H_combined[c] · combined_eq_cycle(c)
        // where combined_eq_cycle(c) = eq_raf(c) + γ·eq_val(c) + γ²·eq_rw(c) + γ³·eq_val(c)
        //                            = eq_raf(c) + γ²·eq_rw(c) + (γ + γ³)·eq_val(c)
        let gamma_coeff_val = self.params.gamma + self.params.gamma_cubed;

        // For Gruen optimization, we need the final eq_address claims
        let eq_address_claim = self.B_1.final_sumcheck_claim()
            + self.params.gamma_squared * self.B_2.final_sumcheck_claim();

        // Use Gruen's poly_deg_2 for each eq polynomial component
        // Compute contribution from each component
        let [gruen_eval_raf_0] = self.eq_cycle_raf.par_fold_out_in_unreduced::<9, 1>(&|g| {
            [self.H_combined.get_bound_coeff(2 * g)]
        });
        let [gruen_eval_rw_0] = self.eq_cycle_rw.par_fold_out_in_unreduced::<9, 1>(&|g| {
            [self.H_combined.get_bound_coeff(2 * g)]
        });
        let [gruen_eval_val_0] = self.eq_cycle_val.par_fold_out_in_unreduced::<9, 1>(&|g| {
            [self.H_combined.get_bound_coeff(2 * g)]
        });

        // Get individual round polynomials
        let poly_raf = self.eq_cycle_raf.gruen_poly_deg_2(
            gruen_eval_raf_0,
            previous_claim / eq_address_claim,
        );
        let poly_rw = self.eq_cycle_rw.gruen_poly_deg_2(
            gruen_eval_rw_0,
            previous_claim / eq_address_claim,
        );
        let poly_val = self.eq_cycle_val.gruen_poly_deg_2(
            gruen_eval_val_0,
            previous_claim / eq_address_claim,
        );

        // Combine: poly_raf + γ²·poly_rw + (γ + γ³)·poly_val
        // Use references for addition since Add is implemented for &UniPoly
        let mut combined = poly_raf;
        combined += &(poly_rw * self.params.gamma_squared);
        combined += &(poly_val * gamma_coeff_val);

        combined * eq_address_claim
    }

    /// Initialize H_combined after address rounds complete.
    /// H_combined[c] = F_1[address[c]] + γ² · F_2[address[c]]
    fn initialize_H_combined(&mut self) {
        let F_1_values = self.F_1.clone_values();
        let F_2_values = self.F_2.clone_values();
        let gamma_squared = self.params.gamma_squared;
        let addresses = &self.addresses;

        let H: Vec<F> = addresses
            .par_iter()
            .map(|addr| {
                addr.map_or(F::zero(), |k| {
                    F_1_values[k] + gamma_squared * F_2_values[k]
                })
            })
            .collect();

        self.H_combined = MultilinearPolynomial::from(H);

        // Clear G arrays as they're no longer needed
        self.G_A.clear();
        self.G_B.clear();
    }

    /// Bind variables during address rounds (phase 1).
    fn address_round_bind(&mut self, r_j: F::Challenge, round: usize) {
        self.B_1.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.B_2.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.F_1.update(r_j);
        self.F_2.update(r_j);

        // If this is the last address round, initialize H_combined
        if round == self.params.log_K - 1 {
            self.initialize_H_combined();
        }
    }

    /// Bind variables during cycle rounds (phase 2).
    fn cycle_round_bind(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_cycle_raf.bind(r_j);
        self.eq_cycle_rw.bind(r_j);
        self.eq_cycle_val.bind(r_j);
        self.H_combined.bind_parallel(r_j, BindingOrder::LowToHigh);
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for RamRaReductionSumcheckProver<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.params.input_claim()
    }

    #[tracing::instrument(skip_all, name = "RamRaReductionSumcheckProver::compute_message")]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        if round < self.params.log_K {
            self.address_round_compute_message(round, previous_claim)
        } else {
            self.cycle_round_compute_message(round, previous_claim)
        }
    }

    #[tracing::instrument(skip_all, name = "RamRaReductionSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        if round < self.params.log_K {
            self.address_round_bind(r_j, round);
        } else {
            self.cycle_round_bind(r_j, round);
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        // Cache the reduced RA opening for use by RA virtualization sumcheck
        let r_address_reduced = &sumcheck_challenges[..self.params.log_K];
        let r_cycle_reduced = &sumcheck_challenges[self.params.log_K..];

        let opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(
            [
                r_address_reduced.iter().rev().copied().collect::<Vec<_>>(),
                r_cycle_reduced.iter().rev().copied().collect::<Vec<_>>(),
            ]
            .concat(),
        );

        // Compute the reduced RA claim.
        // After sumcheck, we have: eq_combined(r_reduced) · ra(r_reduced) = final_claim
        // So: ra_claim_reduced = final_claim / eq_combined(r_reduced)
        //
        // eq_combined(r_reduced) = eq_addr_1 · eq_cycle_A + γ² · eq_addr_2 · eq_cycle_B
        // where eq_cycle_A = eq_raf + γ·eq_val and eq_cycle_B = eq_rw + γ·eq_val
        let eq_addr_1 = self.B_1.final_sumcheck_claim();
        let eq_addr_2 = self.B_2.final_sumcheck_claim();

        // Get final eq_cycle values from Gruen polynomials
        let eq_raf = self.eq_cycle_raf.current_scalar;
        let eq_rw = self.eq_cycle_rw.current_scalar;
        let eq_val = self.eq_cycle_val.current_scalar;

        let eq_cycle_A = eq_raf + self.params.gamma * eq_val;
        let eq_cycle_B = eq_rw + self.params.gamma * eq_val;

        let eq_combined = eq_addr_1 * eq_cycle_A + self.params.gamma_squared * eq_addr_2 * eq_cycle_B;

        // ra_claim_reduced = H_combined[final] / eq_addr_combined
        // But actually we need to recover ra from the identity:
        // H_combined[c] · (eq_cycle_A + γ²·eq_cycle_B) = eq_combined · ra(r_reduced)
        // So ra_claim_reduced = H_combined.final_claim / eq_combined
        let ra_claim_reduced = self.H_combined.final_sumcheck_claim() / eq_combined;

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamRaVirtualization, // The RA virtualization will use this
            opening_point,
            ra_claim_reduced,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

impl<F: JoltField> RamRaReductionSumcheckVerifier<F> {
    /// Create a new RAM RA reduction sumcheck verifier.
    pub fn new(
        trace_len: usize,
        one_hot_params: &OneHotParams,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = RaReductionParams::new(
            trace_len,
            one_hot_params,
            opening_accumulator,
            transcript,
        );
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for RamRaReductionSumcheckVerifier<F>
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
            SumcheckId::RamRaVirtualization,
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
            SumcheckId::RamRaVirtualization,
            opening_point,
        );
    }
}

