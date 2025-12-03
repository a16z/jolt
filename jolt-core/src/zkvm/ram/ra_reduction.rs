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
        split_eq_poly::GruenSplitEqPolynomial,
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

/// RAM RA reduction sumcheck prover.
///
/// Reduces four RA claims (from RafEvaluation, ReadWriteChecking, ValEvaluation, ValFinal)
/// into a single claim that can be fed into the RA virtualization sumcheck.
#[derive(Allocative)]
pub struct RamRaReductionSumcheckProver<F: JoltField> {
    /// The trace of addresses accessed at each cycle.
    /// `addresses[c] = Some(k)` if address k was accessed at cycle c.
    addresses: Arc<Vec<Option<usize>>>,

    // ========== Address round state ==========
    /// eq(r_address_1, k) polynomial - bound during address rounds.
    /// After all address rounds, B_1.final_sumcheck_claim() = eq(r_address_1, r_addr_reduced) = α_1
    B_1: MultilinearPolynomial<F>,
    /// eq(r_address_2, k) polynomial - bound during address rounds.
    /// After all address rounds, B_2.final_sumcheck_claim() = eq(r_address_2, r_addr_reduced) = α_2
    B_2: MultilinearPolynomial<F>,
    /// Expanding table tracking eq(r_addr_reduced, k) where r_addr_reduced is
    /// the vector of sumcheck challenges bound so far.
    F: ExpandingTable<F>,
    /// G_A[k] = Σ_{c: address[c]=k} eq_cycle_A(c)
    G_A: Vec<F>,
    /// G_B[k] = Σ_{c: address[c]=k} eq_cycle_B(c)
    G_B: Vec<F>,

    // ========== Cycle round state (Phase 2) ==========
    /// H[c] = eq(r_addr_reduced, address[c]) = ra(r_addr_reduced, c)
    /// This is the multilinear extension of the one-hot polynomial ra evaluated
    /// at r_addr_reduced for the address variables.
    /// Initialized to None, set to Some after address rounds complete.
    H: Option<MultilinearPolynomial<F>>,
    /// eq(r_cycle_raf, ·) using Gruen optimization
    eq_cycle_raf: GruenSplitEqPolynomial<F>,
    /// eq(r_cycle_rw, ·) using Gruen optimization
    eq_cycle_rw: GruenSplitEqPolynomial<F>,
    /// eq(r_cycle_val, ·) using Gruen optimization
    eq_cycle_val: GruenSplitEqPolynomial<F>,

    // ========== Cycle round Gruen state ==========
    /// Previous claim for eq_raf: Σ_c H[c] * eq_raf(c)
    prev_claim_raf: Option<F>,
    /// Previous claim for eq_rw: Σ_c H[c] * eq_rw(c)
    prev_claim_rw: Option<F>,
    /// Previous claim for eq_val: Σ_c H[c] * eq_val(c)
    prev_claim_val: Option<F>,
    /// Previous round polynomial for eq_raf
    prev_round_poly_raf: Option<UniPoly<F>>,
    /// Previous round polynomial for eq_rw
    prev_round_poly_rw: Option<UniPoly<F>>,
    /// Previous round polynomial for eq_val
    prev_round_poly_val: Option<UniPoly<F>>,

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
        let params =
            RaReductionParams::new(trace.len(), one_hot_params, opening_accumulator, transcript);

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
        let eq_cycle_raf =
            GruenSplitEqPolynomial::new(&params.r_cycle_raf, BindingOrder::LowToHigh);
        let eq_cycle_rw = GruenSplitEqPolynomial::new(&params.r_cycle_rw, BindingOrder::LowToHigh);
        let eq_cycle_val =
            GruenSplitEqPolynomial::new(&params.r_cycle_val, BindingOrder::LowToHigh);

        Self {
            addresses,
            B_1,
            B_2,
            F,
            G_A,
            G_B,
            H: None, // Initialized after address rounds complete
            eq_cycle_raf,
            eq_cycle_rw,
            eq_cycle_val,
            prev_claim_raf: None,
            prev_claim_rw: None,
            prev_claim_val: None,
            prev_round_poly_raf: None,
            prev_round_poly_rw: None,
            prev_round_poly_val: None,
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

    /// Minimum number of k iterations to parallelize the inner loop.
    /// Below this threshold, sequential iteration is faster due to parallel overhead.
    const MIN_INNER_PARALLEL_LEN: usize = 1 << 12; // 4096

    /// Compute the round polynomial for address rounds (phase 1).
    ///
    /// The identity during address rounds is:
    /// Σ_k [eq(r_address_1, k) · G_A[k] + γ² · eq(r_address_2, k) · G_B[k]]
    ///
    /// At round m (0-indexed), we bind the first m variables and sum over the rest.
    fn address_round_compute_message(&self, round: usize, previous_claim: F) -> UniPoly<F> {
        let m = round + 1;
        let half_len = self.B_1.len() / 2;
        let inner_len = 1 << m; // Number of k values per k_prime

        // For each k' in {0,1}^{log_K - m}, compute contribution to round polynomial
        // Use unreduced arithmetic for inner loop accumulation only.
        // The c2 coefficient is computed as 2*sum_1 - sum_0 after reduction.
        let [eval_0, eval_c2] = (0..half_len)
            .into_par_iter()
            .map(|k_prime| {
                // Get sumcheck evals for B_1 and B_2 at position k'
                let B_1_evals = self
                    .B_1
                    .sumcheck_evals_array::<2>(k_prime, BindingOrder::LowToHigh);
                let B_2_evals = self
                    .B_2
                    .sumcheck_evals_array::<2>(k_prime, BindingOrder::LowToHigh);

                // Sum over all k that share prefix k'
                // k ranges from k' << m to (k' + 1) << m
                // Track sums for k_m=0 and k_m=1 separately using unreduced arithmetic
                let k_start = k_prime << m;
                let k_end = k_start + inner_len;

                // Parallelize inner loop when it's large enough
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
                                    let F_k = self.F[k % (1 << (m - 1))];
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
                        // Sequential path for small inner loops
                        let mut sum_A_0 = F::Unreduced::<9>::zero();
                        let mut sum_A_1 = F::Unreduced::<9>::zero();
                        let mut sum_B_0 = F::Unreduced::<9>::zero();
                        let mut sum_B_1 = F::Unreduced::<9>::zero();

                        for k in k_start..k_end {
                            let k_m = (k >> (m - 1)) & 1;
                            let F_k = self.F[k % (1 << (m - 1))];
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

                // Reduce to field elements
                let sum_A_0 = F::from_montgomery_reduce::<9>(sum_A_0);
                let sum_A_1 = F::from_montgomery_reduce::<9>(sum_A_1);
                let sum_B_0 = F::from_montgomery_reduce::<9>(sum_B_0);
                let sum_B_1 = F::from_montgomery_reduce::<9>(sum_B_1);

                // Compute inner_A[0] = sum_0 (eval at X=0)
                // Compute inner_A[1] = 2*sum_1 - sum_0 (c2 coefficient for quadratic interpolation)
                let inner_A_0 = sum_A_0;
                let inner_A_c2 = sum_A_1 + sum_A_1 - sum_A_0;
                let inner_B_0 = sum_B_0;
                let inner_B_c2 = sum_B_1 + sum_B_1 - sum_B_0;

                // Combine with B evals: B[k'] · inner[k'] + γ² · B2[k'] · inner_B[k']
                [
                    B_1_evals[0] * inner_A_0 + self.params.gamma_squared * B_2_evals[0] * inner_B_0,
                    B_1_evals[1] * inner_A_c2
                        + self.params.gamma_squared * B_2_evals[1] * inner_B_c2,
                ]
            })
            .reduce(|| [F::zero(), F::zero()], |a, b| [a[0] + b[0], a[1] + b[1]]);

        UniPoly::from_evals_and_hint(previous_claim, &[eval_0, eval_c2])
    }

    /// Compute the round polynomial for cycle rounds (phase 2).
    ///
    /// After address rounds, the identity we're proving is:
    /// ```text
    /// Σ_c H[c] · [α_1·(eq_raf(c) + γ·eq_val(c)) + γ²·α_2·(eq_rw(c) + γ·eq_val(c))]
    /// = α_1 · Σ_c H[c]·eq_raf(c) + γ²·α_2 · Σ_c H[c]·eq_rw(c) + (γ·α_1 + γ³·α_2) · Σ_c H[c]·eq_val(c)
    /// ```
    ///
    /// We track three separate sums and compute three separate round polynomials using Gruen,
    /// then combine them: combined = coeff_raf * poly_raf + coeff_rw * poly_rw + coeff_val * poly_val
    fn cycle_round_compute_message(&mut self, _round: usize, _previous_claim: F) -> UniPoly<F> {
        // Get α_1 and α_2 from the final B polynomial claims
        let alpha_1 = self.B_1.final_sumcheck_claim();
        let alpha_2 = self.B_2.final_sumcheck_claim();

        // Compute coefficients for each eq_cycle polynomial
        let coeff_raf = alpha_1;
        let coeff_rw = self.params.gamma_squared * alpha_2;
        let coeff_val = self.params.gamma * alpha_1 + self.params.gamma_cubed * alpha_2;

        // Lockstep invariant for split-eq structures
        debug_assert_eq!(
            self.eq_cycle_raf.E_out_current_len(),
            self.eq_cycle_rw.E_out_current_len()
        );
        debug_assert_eq!(
            self.eq_cycle_raf.E_out_current_len(),
            self.eq_cycle_val.E_out_current_len()
        );
        debug_assert_eq!(
            self.eq_cycle_raf.E_in_current_len(),
            self.eq_cycle_rw.E_in_current_len()
        );
        debug_assert_eq!(
            self.eq_cycle_raf.E_in_current_len(),
            self.eq_cycle_val.E_in_current_len()
        );

        let H = self
            .H
            .as_ref()
            .expect("H must be initialized before cycle rounds");

        // Compute eval_at_0 for each eq polynomial using Gruen folds
        // gruen_poly_deg_2 only needs q_0 (eval at X=0), not eval at infinity
        let [eval_at_0_raf] = self
            .eq_cycle_raf
            .par_fold_out_in_unreduced::<9, 1>(&|g| [H.get_bound_coeff(2 * g)]);
        let [eval_at_0_rw] = self
            .eq_cycle_rw
            .par_fold_out_in_unreduced::<9, 1>(&|g| [H.get_bound_coeff(2 * g)]);
        let [eval_at_0_val] = self
            .eq_cycle_val
            .par_fold_out_in_unreduced::<9, 1>(&|g| [H.get_bound_coeff(2 * g)]);

        // Compute individual Gruen round polynomials using the tracked prev_claims
        let round_poly_raf = self.eq_cycle_raf.gruen_poly_deg_2(
            eval_at_0_raf,
            self.prev_claim_raf.expect("prev_claim_raf must be set"),
        );
        let round_poly_rw = self.eq_cycle_rw.gruen_poly_deg_2(
            eval_at_0_rw,
            self.prev_claim_rw.expect("prev_claim_rw must be set"),
        );
        let round_poly_val = self.eq_cycle_val.gruen_poly_deg_2(
            eval_at_0_val,
            self.prev_claim_val.expect("prev_claim_val must be set"),
        );

        // Store for use in ingest_challenge
        self.prev_round_poly_raf = Some(round_poly_raf.clone());
        self.prev_round_poly_rw = Some(round_poly_rw.clone());
        self.prev_round_poly_val = Some(round_poly_val.clone());

        // Combine: coeff_raf * poly_raf + coeff_rw * poly_rw + coeff_val * poly_val
        let mut combined = round_poly_raf * coeff_raf;
        combined += &(round_poly_rw * coeff_rw);
        combined += &(round_poly_val * coeff_val);

        combined
    }

    /// Initialize H and compute initial prev_claims after address rounds complete.
    /// H[c] = F[address[c]] = eq(r_addr_reduced, address[c]) = ra(r_addr_reduced, c)
    ///
    /// Also computes initial claims for each eq polynomial:
    /// - prev_claim_raf = Σ_c H[c] * eq_raf(c)
    /// - prev_claim_rw = Σ_c H[c] * eq_rw(c)
    /// - prev_claim_val = Σ_c H[c] * eq_val(c)
    #[tracing::instrument(skip_all, name = "RamRaReductionSumcheckProver::initialize_H")]
    fn initialize_H(&mut self) {
        let F_values = self.F.clone_values();
        let addresses = &self.addresses;

        let H_vec: Vec<F> = addresses
            .par_iter()
            .map(|addr| addr.map_or(F::zero(), |k| F_values[k]))
            .collect();

        // Compute initial prev_claims using Gruen's E_out/E_in structure
        // Similar pattern to read_raf_checking.rs init_log_t_rounds
        let e_out_raf = self.eq_cycle_raf.E_out_current();
        let e_out_rw = self.eq_cycle_rw.E_out_current();
        let e_out_val = self.eq_cycle_val.E_out_current();
        debug_assert_eq!(e_out_raf.len(), e_out_rw.len());
        debug_assert_eq!(e_out_raf.len(), e_out_val.len());

        let in_len = self.eq_cycle_raf.E_in_current_len();
        debug_assert_eq!(in_len, self.eq_cycle_rw.E_in_current_len());
        debug_assert_eq!(in_len, self.eq_cycle_val.E_in_current_len());
        let x_in_bits = in_len.log_2();

        // Precompute merged inner coeffs
        let merged_raf = self.eq_cycle_raf.merged_in_with_current_w();
        let merged_rw = self.eq_cycle_rw.merged_in_with_current_w();
        let merged_val = self.eq_cycle_val.merged_in_with_current_w();

        let (prev_claim_raf_unr, prev_claim_rw_unr, prev_claim_val_unr) = (0..e_out_raf.len())
            .into_par_iter()
            .map(|x_out| {
                let high_raf = e_out_raf[x_out];
                let high_rw = e_out_rw[x_out];
                let high_val = e_out_val[x_out];

                let mut inner_raf = F::Unreduced::<9>::zero();
                let mut inner_rw = F::Unreduced::<9>::zero();
                let mut inner_val = F::Unreduced::<9>::zero();

                for x_in in 0..in_len {
                    let base_index = (x_out << (x_in_bits + 1)) + (x_in << 1);
                    let off = 2 * x_in;

                    // j = base_index (c_0 = 0)
                    {
                        let j0 = base_index;
                        let h_j0 = H_vec[j0];
                        inner_raf += merged_raf[off].mul_unreduced::<9>(h_j0);
                        inner_rw += merged_rw[off].mul_unreduced::<9>(h_j0);
                        inner_val += merged_val[off].mul_unreduced::<9>(h_j0);
                    }
                    // j = base_index + 1 (c_0 = 1)
                    {
                        let j1 = base_index + 1;
                        let h_j1 = H_vec[j1];
                        inner_raf += merged_raf[off + 1].mul_unreduced::<9>(h_j1);
                        inner_rw += merged_rw[off + 1].mul_unreduced::<9>(h_j1);
                        inner_val += merged_val[off + 1].mul_unreduced::<9>(h_j1);
                    }
                }

                let scaled_raf =
                    high_raf.mul_unreduced::<9>(F::from_montgomery_reduce::<9>(inner_raf));
                let scaled_rw =
                    high_rw.mul_unreduced::<9>(F::from_montgomery_reduce::<9>(inner_rw));
                let scaled_val =
                    high_val.mul_unreduced::<9>(F::from_montgomery_reduce::<9>(inner_val));
                (scaled_raf, scaled_rw, scaled_val)
            })
            .reduce(
                || {
                    (
                        F::Unreduced::<9>::zero(),
                        F::Unreduced::<9>::zero(),
                        F::Unreduced::<9>::zero(),
                    )
                },
                |(r0, w0, v0), (r1, w1, v1)| (r0 + r1, w0 + w1, v0 + v1),
            );

        self.prev_claim_raf = Some(F::from_montgomery_reduce::<9>(prev_claim_raf_unr));
        self.prev_claim_rw = Some(F::from_montgomery_reduce::<9>(prev_claim_rw_unr));
        self.prev_claim_val = Some(F::from_montgomery_reduce::<9>(prev_claim_val_unr));

        self.H = Some(MultilinearPolynomial::from(H_vec));

        // Clear G arrays as they're no longer needed
        self.G_A.clear();
        self.G_B.clear();
    }

    /// Bind variables during address rounds (phase 1).
    fn address_round_bind(&mut self, r_j: F::Challenge, round: usize) {
        self.B_1.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.B_2.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.F.update(r_j);

        // If this is the last address round, initialize H
        if round == self.params.log_K - 1 {
            self.initialize_H();
        }
    }

    /// Bind variables during cycle rounds (phase 2).
    fn cycle_round_bind(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_cycle_raf.bind(r_j);
        self.eq_cycle_rw.bind(r_j);
        self.eq_cycle_val.bind(r_j);
        self.H
            .as_mut()
            .expect("H must be initialized before cycle rounds")
            .bind_parallel(r_j, BindingOrder::LowToHigh);

        // Update prev_claims by evaluating round polynomials at the challenge
        self.prev_claim_raf = Some(
            self.prev_round_poly_raf
                .take()
                .expect("prev_round_poly_raf must be set")
                .evaluate(&r_j),
        );
        self.prev_claim_rw = Some(
            self.prev_round_poly_rw
                .take()
                .expect("prev_round_poly_rw must be set")
                .evaluate(&r_j),
        );
        self.prev_claim_val = Some(
            self.prev_round_poly_val
                .take()
                .expect("prev_round_poly_val must be set")
                .evaluate(&r_j),
        );
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

        // The reduced RA claim is simply H.final_sumcheck_claim().
        // H[c] = eq(r_addr_reduced, address[c]) = ra(r_addr_reduced, c)
        // After binding all cycle variables, H.final_sumcheck_claim() = ra(r_addr_reduced, r_cycle_reduced)
        let ra_claim_reduced = self
            .H
            .as_ref()
            .expect("H must be initialized before cache_openings")
            .final_sumcheck_claim();

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamRaReduction, // Output of RA reduction, consumed by RA virtualization
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
        let params =
            RaReductionParams::new(trace_len, one_hot_params, opening_accumulator, transcript);
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
        let (_, ra_claim_reduced) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamRaReduction);

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
            SumcheckId::RamRaReduction,
            opening_point,
        );
    }
}
