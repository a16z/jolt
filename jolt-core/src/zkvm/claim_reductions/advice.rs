//! Advice Polynomial Claim Reduction Sumcheck
//!
//! This module implements a claim reduction sumcheck that consolidates multiple advice
//! polynomial claims into single claims that can be batched into the unified Dory opening
//! in Stage 8.
//!
//! ## Background
//!
//! Jolt supports two types of "advice" inputs that are committed separately from the
//! main witness polynomials:
//!
//! - **Trusted advice**: Committed before proving, verifier receives the commitment
//! - **Untrusted advice**: Committed during proving, commitment is part of the proof
//!
//! These advice polynomials are much smaller than the main polynomials:
//! - Advice has `advice_vars = log2((max_advice_size / 8).next_power_of_two())` variables
//! - Typically 7-9 variables (128-512 words, i.e., 1-4 KB)
//! - Compare to main polynomials with `log_k_chunk + log_T` variables (e.g., 24 vars)
//!
//! ## Current Advice Claims (Before This Sumcheck)
//!
//! Advice claims are generated in Stage 4 via `prover_accumulate_advice()`:
//!
//! | Advice Type      | Sumcheck Source        | Opening Point                |
//! |------------------|------------------------|------------------------------|
//! | trusted_advice   | RamValEvaluation       | suffix of r_address_rw       |
//! | trusted_advice   | RamValFinalEvaluation  | suffix of r_address_raf      |
//! | untrusted_advice | RamValEvaluation       | suffix of r_address_rw       |
//! | untrusted_advice | RamValFinalEvaluation  | suffix of r_address_raf      |
//!
//! Where:
//! - `r_address_rw` comes from `RamReadWriteChecking` (Stage 2)
//! - `r_address_raf` comes from `RamOutputCheck` (Stage 2)
//! - The suffix is the last `advice_vars` components of the address point
//!
//! Note: When `needs_single_advice_opening(T)` is true, only `RamValEvaluation` claims
//! exist (the two points are identical in that case).
//!
//! ## Claim Reduction Strategy
//!
//! This sumcheck reduces the 2-4 advice claims to 2 claims (one per advice type) at
//! a unified point that aligns with `r_cycle_stage6`.
//!
//! ### Alignment with r_cycle_stage6
//!
//! The key insight is that this sumcheck runs in Stage 6 for `advice_vars` rounds.
//! Since all Stage 6 sumchecks are batched together and run for `log_T` rounds total,
//! the advice sumcheck contributes 0 for the first `(log_T - advice_vars)` rounds
//! and only becomes active in the last `advice_vars` rounds.
//!
//! ```text
//! Stage 6 rounds:  [0, 1, ..., log_T - advice_vars - 1 | log_T - advice_vars, ..., log_T - 1]
//!                   └──────────────────────────────────┘ └────────────────────────────────────┘
//!                    Advice contributes 0                 Advice reduction runs here
//!                    (poly is constant in these vars)     (normal sumcheck behavior)
//! ```
//!
//! After Stage 6:
//! - `r_cycle_stage6 = [r_0, r_1, ..., r_{log_T-1}]`
//! - Advice claims are at `r_cycle_stage6[log_T - advice_vars ..]` (last `advice_vars`)
//!
//! ### Sumcheck Relation
//!
//! Let:
//! - `T_1 = trusted_advice(r_advice_rw)` from RamValEvaluation
//! - `T_2 = trusted_advice(r_advice_raf)` from RamValFinalEvaluation (if exists)
//! - `U_1 = untrusted_advice(r_advice_rw)` from RamValEvaluation
//! - `U_2 = untrusted_advice(r_advice_raf)` from RamValFinalEvaluation (if exists)
//!
//! The sumcheck proves:
//!
//! ```text
//! Σ_a [ trusted(a) · (eq(r_advice_rw, a) + γ·eq(r_advice_raf, a))
//!     + γ²·untrusted(a) · (eq(r_advice_rw', a) + γ·eq(r_advice_raf', a)) ]
//! = T_1 + γ·T_2 + γ²·U_1 + γ³·U_2
//! ```
//!
//! After `advice_vars` rounds with challenges ρ, the final claims are:
//! - `trusted_advice(ρ)` with combined eq factor
//! - `untrusted_advice(ρ)` with combined eq factor
//!
//! Where ρ = last `advice_vars` components of `r_cycle_stage6`.
//!
//! ### Degree Analysis
//!
//! Each round polynomial has degree 2:
//! - advice(a) contributes degree 1
//! - eq(r, a) contributes degree 1
//! - Product is degree 2
//!
//! ## Integration with Stage 8 Batch Opening
//!
//! After this sumcheck, advice claims are at a point that is a suffix of `r_cycle_stage6`.
//! The unified Dory opening point in Stage 8 is `(r_addr_stage7 || r_cycle_stage6)`.
//!
//! To include advice in the batch opening, we apply a Lagrange factor:
//!
//! ```text
//! Unified point: [r_0, r_1, ..., r_{log_k + log_T - 1}]
//!                 └─────────────────────────────────────┘
//!                           log_k_chunk + log_T vars
//!
//! Advice is at: last advice_vars
//!
//! Lagrange factor: ∏_{i=0}^{log_k_chunk + log_T - advice_vars - 1} (1 - r_i)
//! ```
//!
//! This factor accounts for the "zero-prefix" padding: advice is conceptually a
//! polynomial that is zero everywhere except when all prefix variables are 0.
//!
//! ## Implementation Notes
//!
//! This module is part of the advice batching infrastructure:
//!
//! 1. **Commitment with Main context**: Advice polynomials are now committed using
//!    the Main `DoryContext` (with the same dimensions as other polynomials). This
//!    enables the batch opening to work correctly. The SDK's `commit_trusted_advice`
//!    function and the prover's `generate_and_commit_untrusted_advice` both use
//!    the Main context.
//!
//! 2. **Hint storage and combination**: Advice hints are stored in `JoltAdvice` and
//!    combined with other polynomial hints in `build_streaming_rlc`. Shorter hints
//!    are zero-padded by `combine_hints` to match the maximum row count.
//!
//! 3. **Lagrange factors**: Since advice polynomials have fewer variables than the
//!    unified opening point, Lagrange factors are applied in Stage 7 to account
//!    for the "zero-prefix" padding.

use crate::field::JoltField;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding,
};
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use allocative::Allocative;
use common::jolt_device::MemoryLayout;
use rayon::prelude::*;

const DEGREE_BOUND: usize = 2;

// ============================================================================
// PARAMS
// ============================================================================

/// Parameters for the advice claim reduction sumcheck.
#[derive(Clone, Allocative)]
pub struct AdviceClaimReductionParams<F: JoltField> {
    /// γ, γ², γ³ for batching
    pub gamma_powers: [F; 3],
    /// Number of variables in the advice polynomials
    pub advice_vars: usize,
    /// Opening point for trusted advice from RamValEvaluation (advice_vars dimensions)
    pub r_trusted_val_eval: Option<OpeningPoint<BIG_ENDIAN, F>>,
    /// Opening point for trusted advice from RamValFinalEvaluation (advice_vars dimensions)
    /// None if single_opening mode
    pub r_trusted_val_final: Option<OpeningPoint<BIG_ENDIAN, F>>,
    /// Opening point for untrusted advice from RamValEvaluation (advice_vars dimensions)
    pub r_untrusted_val_eval: Option<OpeningPoint<BIG_ENDIAN, F>>,
    /// Opening point for untrusted advice from RamValFinalEvaluation (advice_vars dimensions)
    /// None if single_opening mode
    pub r_untrusted_val_final: Option<OpeningPoint<BIG_ENDIAN, F>>,
    /// Whether we have trusted advice
    pub has_trusted_advice: bool,
    /// Whether we have untrusted advice
    pub has_untrusted_advice: bool,
    /// Whether we need single opening (both points are identical)
    pub single_opening: bool,
}

impl<F: JoltField> AdviceClaimReductionParams<F> {
    /// Create parameters for the advice claim reduction sumcheck.
    ///
    /// Returns None if there is no advice to reduce.
    pub fn new(
        memory_layout: &MemoryLayout,
        trace_len: usize,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Option<Self> {
        let has_trusted_advice = memory_layout.max_trusted_advice_size > 0;
        let has_untrusted_advice = memory_layout.max_untrusted_advice_size > 0;

        if !has_trusted_advice && !has_untrusted_advice {
            return None;
        }

        let gamma: F = transcript.challenge_scalar();
        let gamma_sqr = gamma.square();
        let gamma_cub = gamma_sqr * gamma;

        let single_opening =
            crate::zkvm::ram::read_write_checking::needs_single_advice_opening(trace_len);

        // Determine advice_vars from the larger of the two advice sizes
        let max_advice_size = std::cmp::max(
            memory_layout.max_trusted_advice_size,
            memory_layout.max_untrusted_advice_size,
        ) as usize;
        let advice_vars = (max_advice_size / 8).next_power_of_two().log_2();

        // Fetch opening points from accumulator
        let r_trusted_val_eval = if has_trusted_advice {
            accumulator
                .get_trusted_advice_opening(SumcheckId::RamValEvaluation)
                .map(|(point, _)| point)
        } else {
            None
        };

        let r_trusted_val_final = if has_trusted_advice && !single_opening {
            accumulator
                .get_trusted_advice_opening(SumcheckId::RamValFinalEvaluation)
                .map(|(point, _)| point)
        } else {
            None
        };

        let r_untrusted_val_eval = if has_untrusted_advice {
            accumulator
                .get_untrusted_advice_opening(SumcheckId::RamValEvaluation)
                .map(|(point, _)| point)
        } else {
            None
        };

        let r_untrusted_val_final = if has_untrusted_advice && !single_opening {
            accumulator
                .get_untrusted_advice_opening(SumcheckId::RamValFinalEvaluation)
                .map(|(point, _)| point)
        } else {
            None
        };

        Some(Self {
            gamma_powers: [gamma, gamma_sqr, gamma_cub],
            advice_vars,
            r_trusted_val_eval,
            r_trusted_val_final,
            r_untrusted_val_eval,
            r_untrusted_val_final,
            has_trusted_advice,
            has_untrusted_advice,
            single_opening,
        })
    }

    /// Create parameters for verifier (same logic as prover)
    pub fn new_verifier(
        memory_layout: &MemoryLayout,
        trace_len: usize,
        accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Option<Self> {
        Self::new(memory_layout, trace_len, accumulator, transcript)
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for AdviceClaimReductionParams<F> {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let [gamma, gamma_sqr, gamma_cub] = self.gamma_powers;
        let mut claim = F::zero();

        // Trusted advice claims
        if self.has_trusted_advice {
            if let Some((_, t1)) =
                accumulator.get_trusted_advice_opening(SumcheckId::RamValEvaluation)
            {
                claim += t1;
            }
            if !self.single_opening {
                if let Some((_, t2)) =
                    accumulator.get_trusted_advice_opening(SumcheckId::RamValFinalEvaluation)
                {
                    claim += gamma * t2;
                }
            }
        }

        // Untrusted advice claims
        if self.has_untrusted_advice {
            let offset = if self.single_opening { gamma } else { gamma_sqr };
            if let Some((_, u1)) =
                accumulator.get_untrusted_advice_opening(SumcheckId::RamValEvaluation)
            {
                claim += offset * u1;
            }
            if !self.single_opening {
                if let Some((_, u2)) =
                    accumulator.get_untrusted_advice_opening(SumcheckId::RamValFinalEvaluation)
                {
                    claim += gamma_cub * u2;
                }
            }
        }

        claim
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.advice_vars
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        // The challenges are the LAST advice_vars of r_cycle_stage6
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

// ============================================================================
// PROVER
// ============================================================================

/// Prover for the advice claim reduction sumcheck.
#[derive(Allocative)]
pub struct AdviceClaimReductionProver<F: JoltField> {
    params: AdviceClaimReductionParams<F>,
    /// Trusted advice polynomial (if present)
    trusted_advice: Option<MultilinearPolynomial<F>>,
    /// Untrusted advice polynomial (if present)
    untrusted_advice: Option<MultilinearPolynomial<F>>,
    /// Combined eq polynomial for trusted advice: eq(r_val_eval, ·) + γ·eq(r_val_final, ·)
    eq_trusted: Option<MultilinearPolynomial<F>>,
    /// Combined eq polynomial for untrusted advice: eq(r_val_eval, ·) + γ·eq(r_val_final, ·)
    eq_untrusted: Option<MultilinearPolynomial<F>>,
}

impl<F: JoltField> AdviceClaimReductionProver<F> {
    #[tracing::instrument(skip_all, name = "AdviceClaimReductionProver::initialize")]
    pub fn initialize(
        params: AdviceClaimReductionParams<F>,
        trusted_advice_poly: Option<MultilinearPolynomial<F>>,
        untrusted_advice_poly: Option<MultilinearPolynomial<F>>,
    ) -> Self {
        let gamma = params.gamma_powers[0];

        // Build combined eq polynomial for trusted advice
        let eq_trusted = if params.has_trusted_advice {
            let r_val_eval = params.r_trusted_val_eval.as_ref().unwrap();
            let eq_val_eval = EqPolynomial::evals(&r_val_eval.r);

            if params.single_opening {
                Some(MultilinearPolynomial::from(eq_val_eval))
            } else {
                let r_val_final = params.r_trusted_val_final.as_ref().unwrap();
                let eq_val_final = EqPolynomial::evals(&r_val_final.r);

                // Combine: eq_val_eval + γ·eq_val_final
                let combined: Vec<F> = eq_val_eval
                    .par_iter()
                    .zip(eq_val_final.par_iter())
                    .map(|(e1, e2)| *e1 + gamma * e2)
                    .collect();
                Some(MultilinearPolynomial::from(combined))
            }
        } else {
            None
        };

        // Build combined eq polynomial for untrusted advice
        let eq_untrusted = if params.has_untrusted_advice {
            let r_val_eval = params.r_untrusted_val_eval.as_ref().unwrap();
            let eq_val_eval = EqPolynomial::evals(&r_val_eval.r);

            if params.single_opening {
                Some(MultilinearPolynomial::from(eq_val_eval))
            } else {
                let r_val_final = params.r_untrusted_val_final.as_ref().unwrap();
                let eq_val_final = EqPolynomial::evals(&r_val_final.r);

                // Combine: eq_val_eval + γ·eq_val_final
                let combined: Vec<F> = eq_val_eval
                    .par_iter()
                    .zip(eq_val_final.par_iter())
                    .map(|(e1, e2)| *e1 + gamma * e2)
                    .collect();
                Some(MultilinearPolynomial::from(combined))
            }
        } else {
            None
        };

        Self {
            params,
            trusted_advice: trusted_advice_poly,
            untrusted_advice: untrusted_advice_poly,
            eq_trusted,
            eq_untrusted,
        }
    }

    fn compute_message(&mut self, previous_claim: F) -> UniPoly<F> {
        let gamma_sqr = self.params.gamma_powers[1];
        let untrusted_coeff = if self.params.single_opening {
            self.params.gamma_powers[0] // γ
        } else {
            gamma_sqr // γ²
        };

        let half_n = self.params.advice_vars.checked_sub(1).map_or(1, |v| 1 << v);
        let mut evals = [F::zero(); DEGREE_BOUND];

        // Trusted advice contribution: trusted(a) · eq_trusted(a)
        if let (Some(ref trusted), Some(ref eq_trusted)) =
            (&self.trusted_advice, &self.eq_trusted)
        {
            for j in 0..half_n.min(trusted.len() / 2) {
                let t_evals = trusted.sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let eq_evals =
                    eq_trusted.sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);

                for k in 0..DEGREE_BOUND {
                    evals[k] += t_evals[k] * eq_evals[k];
                }
            }
        }

        // Untrusted advice contribution: γ² · untrusted(a) · eq_untrusted(a)
        if let (Some(ref untrusted), Some(ref eq_untrusted)) =
            (&self.untrusted_advice, &self.eq_untrusted)
        {
            for j in 0..half_n.min(untrusted.len() / 2) {
                let u_evals =
                    untrusted.sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let eq_evals =
                    eq_untrusted.sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);

                for k in 0..DEGREE_BOUND {
                    evals[k] += untrusted_coeff * u_evals[k] * eq_evals[k];
                }
            }
        }

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn bind(&mut self, r_j: F::Challenge) {
        if let Some(ref mut trusted) = self.trusted_advice {
            trusted.bind(r_j, BindingOrder::LowToHigh);
        }
        if let Some(ref mut untrusted) = self.untrusted_advice {
            untrusted.bind(r_j, BindingOrder::LowToHigh);
        }
        if let Some(ref mut eq_trusted) = self.eq_trusted {
            eq_trusted.bind(r_j, BindingOrder::LowToHigh);
        }
        if let Some(ref mut eq_untrusted) = self.eq_untrusted {
            eq_untrusted.bind(r_j, BindingOrder::LowToHigh);
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for AdviceClaimReductionProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "AdviceClaimReductionProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        self.compute_message(previous_claim)
    }

    #[tracing::instrument(skip_all, name = "AdviceClaimReductionProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.bind(r_j);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = SumcheckInstanceProver::<F, T>::get_params(self)
            .normalize_opening_point(sumcheck_challenges);

        // Cache trusted advice opening
        if let Some(ref trusted) = self.trusted_advice {
            let claim = trusted.final_sumcheck_claim();
            accumulator.append_trusted_advice(
                transcript,
                SumcheckId::AdviceClaimReduction,
                opening_point.clone(),
                claim,
            );
        }

        // Cache untrusted advice opening
        if let Some(ref untrusted) = self.untrusted_advice {
            let claim = untrusted.final_sumcheck_claim();
            accumulator.append_untrusted_advice(
                transcript,
                SumcheckId::AdviceClaimReduction,
                opening_point,
                claim,
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

// ============================================================================
// VERIFIER
// ============================================================================

/// Verifier for the advice claim reduction sumcheck.
pub struct AdviceClaimReductionVerifier<F: JoltField> {
    params: AdviceClaimReductionParams<F>,
}

impl<F: JoltField> AdviceClaimReductionVerifier<F> {
    pub fn new(
        memory_layout: &MemoryLayout,
        trace_len: usize,
        accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Option<Self> {
        let params =
            AdviceClaimReductionParams::new_verifier(memory_layout, trace_len, accumulator, transcript)?;
        Some(Self { params })
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for AdviceClaimReductionVerifier<F>
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
        let untrusted_coeff = if self.params.single_opening {
            gamma
        } else {
            gamma_sqr
        };

        let opening_point = SumcheckInstanceVerifier::<F, T>::get_params(self)
            .normalize_opening_point(sumcheck_challenges);

        let mut claim = F::zero();

        // Trusted advice contribution
        if self.params.has_trusted_advice {
            let (_, trusted_claim) = accumulator
                .get_trusted_advice_opening(SumcheckId::AdviceClaimReduction)
                .expect("Trusted advice claim not found");

            // Compute eq factors
            let r_val_eval = self.params.r_trusted_val_eval.as_ref().unwrap();
            let eq_val_eval = EqPolynomial::mle(&opening_point.r, &r_val_eval.r);

            if self.params.single_opening {
                claim += trusted_claim * eq_val_eval;
            } else {
                let r_val_final = self.params.r_trusted_val_final.as_ref().unwrap();
                let eq_val_final = EqPolynomial::mle(&opening_point.r, &r_val_final.r);
                claim += trusted_claim * (eq_val_eval + gamma * eq_val_final);
            }
        }

        // Untrusted advice contribution
        if self.params.has_untrusted_advice {
            let (_, untrusted_claim) = accumulator
                .get_untrusted_advice_opening(SumcheckId::AdviceClaimReduction)
                .expect("Untrusted advice claim not found");

            // Compute eq factors
            let r_val_eval = self.params.r_untrusted_val_eval.as_ref().unwrap();
            let eq_val_eval = EqPolynomial::mle(&opening_point.r, &r_val_eval.r);

            if self.params.single_opening {
                claim += untrusted_coeff * untrusted_claim * eq_val_eval;
            } else {
                let r_val_final = self.params.r_untrusted_val_final.as_ref().unwrap();
                let eq_val_final = EqPolynomial::mle(&opening_point.r, &r_val_final.r);
                claim += untrusted_coeff * untrusted_claim * (eq_val_eval + gamma * eq_val_final);
            }
        }

        claim
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = SumcheckInstanceVerifier::<F, T>::get_params(self)
            .normalize_opening_point(sumcheck_challenges);

        // Cache trusted advice opening
        if self.params.has_trusted_advice {
            accumulator.append_trusted_advice(
                transcript,
                SumcheckId::AdviceClaimReduction,
                opening_point.clone(),
            );
        }

        // Cache untrusted advice opening
        if self.params.has_untrusted_advice {
            accumulator.append_untrusted_advice(
                transcript,
                SumcheckId::AdviceClaimReduction,
                opening_point,
            );
        }
    }
}
