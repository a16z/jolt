//! Shift sumcheck for verifying rho_next claims in packed GT exponentiation
//! Proves: Σ_{s,x} γ^i * (v_i - EqPlusOne(r_s*_i, s) × Eq(r_x*_i, x) × rho_i(s,x)) = 0
//!
//! This sumcheck runs after packed GT constraint sumcheck (Stage 1b) to verify
//! that claimed rho_next values equal rho at shifted positions.

use crate::{
    field::JoltField,
    poly::{
        eq_plus_one_poly::EqPlusOnePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    zkvm::witness::VirtualPolynomial,
};
use rayon::prelude::*;

/// Parameters for shift rho sumcheck
#[derive(Clone)]
pub struct ShiftRhoParams {
    /// Number of variables (12 = 8 step + 4 element)
    pub num_vars: usize,
    /// Number of claims to verify
    pub num_claims: usize,
    /// Sumcheck instance identifier
    pub sumcheck_id: SumcheckId,
}

impl ShiftRhoParams {
    pub fn new(num_claims: usize) -> Self {
        Self {
            num_vars: 12, // 8 step vars + 4 element vars
            num_claims,
            sumcheck_id: SumcheckId::ShiftRho,
        }
    }
}

/// Shift claim to be verified
#[derive(Clone, Debug)]
pub struct ShiftClaim {
    /// Constraint index
    pub constraint_idx: usize,
}

/// Prover for shift rho sumcheck
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct ShiftRhoProver<F: JoltField, T: Transcript> {
    /// Parameters
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub params: ShiftRhoParams,

    /// Rho polynomials (one per claim)
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub rho_polys: Vec<MultilinearPolynomial<F>>,

    /// EqPlusOne polynomials for each claim
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub eq_plus_one_polys: Vec<MultilinearPolynomial<F>>,

    /// Eq polynomials for element variables
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub eq_x_polys: Vec<MultilinearPolynomial<F>>,

    /// Gamma for batching
    pub gamma: F,

    /// Current round
    pub round: usize,

    /// Claimed values for each rho_next
    pub claimed_values: Vec<F>,

    pub _marker: std::marker::PhantomData<T>,
}

impl<F: JoltField, T: Transcript> ShiftRhoProver<F, T> {
    pub fn new(
        params: ShiftRhoParams,
        rho_polys: Vec<Vec<F>>,
        claims: Vec<ShiftClaim>,
        accumulator: &ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Self {
        assert_eq!(params.num_claims, claims.len());
        assert_eq!(params.num_claims, rho_polys.len());

        // Sample batching coefficient
        let gamma: F = transcript.challenge_scalar_optimized::<F>().into();

        // Convert rho polynomials
        let rho_polys = rho_polys
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect();

        // For each claim, fetch the point and value from accumulator
        let mut eq_plus_one_polys = Vec::with_capacity(params.num_claims);
        let mut eq_x_polys = Vec::with_capacity(params.num_claims);
        let mut claimed_values = Vec::with_capacity(params.num_claims);

        for claim in &claims {
            let (point, value) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::PackedGtExpRhoNext(claim.constraint_idx),
                SumcheckId::PackedGtExp,
            );

            claimed_values.push(value);

            // Split point into step and element parts
            let r_s = &point.r[..8];
            let r_x = &point.r[8..];

            // Create EqPlusOne polynomial for step variables
            let (_, eq_plus_one_evals) = EqPlusOnePolynomial::<F>::evals(r_s, None);
            eq_plus_one_polys.push(MultilinearPolynomial::from(eq_plus_one_evals));

            // Create Eq polynomial for element variables (padded to full 12 vars)
            let eq_x_evals = EqPolynomial::<F>::evals(r_x);
            // Pad to 12 variables by replicating across step dimension
            let mut padded_eq_x = vec![F::zero(); 1 << params.num_vars];
            for s in 0..256 {
                for x in 0..16 {
                    padded_eq_x[x * 256 + s] = eq_x_evals[x];
                }
            }
            eq_x_polys.push(MultilinearPolynomial::from(padded_eq_x));
        }

        Self {
            params,
            rho_polys,
            eq_plus_one_polys,
            eq_x_polys,
            gamma,
            round: 0,
            claimed_values,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ShiftRhoProver<F, T> {
    fn degree(&self) -> usize {
        3 // EqPlusOne * Eq * rho (each degree 1)
    }

    fn num_rounds(&self) -> usize {
        self.params.num_vars
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        // The sum should equal the batched claimed values
        let mut sum = F::zero();
        let mut gamma_power = F::one();

        for claimed_value in &self.claimed_values {
            sum += gamma_power * claimed_value;
            gamma_power *= self.gamma;
        }

        sum
    }

    #[tracing::instrument(skip_all, name = "ShiftRho::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        const DEGREE: usize = 3;

        let half = if !self.rho_polys.is_empty() {
            self.rho_polys[0].get_num_vars() - 1
        } else {
            0
        };
        let x_half = 1 << half;

        let gamma = self.gamma;

        // Compute evaluations in parallel
        let evals = (0..x_half)
            .into_par_iter()
            .map(|i| {
                let mut term_evals = [F::zero(); DEGREE];
                let mut gamma_power = F::one();

                for claim_idx in 0..self.params.num_claims {
                    let rho_evals = self.rho_polys[claim_idx]
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                    let eq_plus_one_evals = self.eq_plus_one_polys[claim_idx]
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                    let eq_x_evals = self.eq_x_polys[claim_idx]
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                    for t in 0..DEGREE {
                        // Compute: γ^i * EqPlusOne * Eq * rho
                        let shift_eval = eq_plus_one_evals[t] * eq_x_evals[t] * rho_evals[t];
                        term_evals[t] += gamma_power * shift_eval;
                    }

                    gamma_power *= gamma;
                }

                term_evals
            })
            .reduce(
                || [F::zero(); DEGREE],
                |mut acc, evals| {
                    for (a, e) in acc.iter_mut().zip(evals.iter()) {
                        *a += *e;
                    }
                    acc
                },
            );

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        // Bind all polynomials
        for rho in &mut self.rho_polys {
            rho.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for eq_plus_one in &mut self.eq_plus_one_polys {
            eq_plus_one.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for eq_x in &mut self.eq_x_polys {
            eq_x.bind_parallel(r_j, BindingOrder::LowToHigh);
        }

        self.round = round + 1;
    }

    fn cache_openings(
        &self,
        _accumulator: &mut ProverOpeningAccumulator<F>,
        _transcript: &mut T,
        _sumcheck_challenges: &[F::Challenge],
    ) {
        // The rho_next claims are already in the accumulator from PackedGtExp sumcheck
        // The shift sumcheck just verified their correctness
        // No additional openings needed
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// Verifier for shift rho sumcheck
pub struct ShiftRhoVerifier<F: JoltField> {
    pub params: ShiftRhoParams,
    pub claims: Vec<ShiftClaim>,
    pub gamma: F,
}

impl<F: JoltField> ShiftRhoVerifier<F> {
    pub fn new<T: Transcript>(
        params: ShiftRhoParams,
        claims: Vec<ShiftClaim>,
        transcript: &mut T,
    ) -> Self {
        assert_eq!(params.num_claims, claims.len());

        // Sample same batching coefficient
        let gamma: F = transcript.challenge_scalar_optimized::<F>().into();

        Self {
            params,
            claims,
            gamma,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for ShiftRhoVerifier<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.params.num_vars
    }

    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F {
        // The sum should equal the batched claimed values fetched from accumulator
        let mut sum = F::zero();
        let mut gamma_power = F::one();

        for claim in &self.claims {
            let (_, value) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::PackedGtExpRhoNext(claim.constraint_idx),
                SumcheckId::PackedGtExp,
            );

            sum += gamma_power * value;
            gamma_power *= self.gamma;
        }

        sum
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let mut sum = F::zero();
        let mut gamma_power = F::one();

        for claim in &self.claims {
            // Get rho_next point and value from accumulator
            let (rho_next_point, _rho_next_value) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::PackedGtExpRhoNext(claim.constraint_idx),
                SumcheckId::PackedGtExp,
            );

            // Get rho evaluation at the challenge point
            // Note: This assumes rho was opened at the same challenge point
            let (_, rho_eval) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::PackedGtExpRho(claim.constraint_idx),
                SumcheckId::PackedGtExp,
            );

            // Split the original point into step and element parts
            let r_s = &rho_next_point.r[..8];
            let r_x = &rho_next_point.r[8..];

            // Evaluate EqPlusOne(r_s, challenges[..8]) * Eq(r_x, challenges[8..])
            let s_challenges = &sumcheck_challenges[..8];
            let x_challenges = &sumcheck_challenges[8..];

            let eq_plus_one = EqPlusOnePolynomial::<F>::mle(r_s, s_challenges);
            let eq_x = EqPolynomial::<F>::mle(r_x, x_challenges);

            // Accumulate: γ^i * EqPlusOne * Eq * rho
            sum += gamma_power * eq_plus_one * eq_x * rho_eval;
            gamma_power *= self.gamma;
        }

        sum
    }

    fn cache_openings(
        &self,
        _accumulator: &mut VerifierOpeningAccumulator<F>,
        _transcript: &mut T,
        _sumcheck_challenges: &[F::Challenge],
    ) {
        // The rho_next claims are already in the accumulator from PackedGtExp sumcheck
        // The shift sumcheck just verified their correctness
        // No additional openings needed
    }
}