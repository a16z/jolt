//! Virtualization sumcheck for recursion protocol (Phase 2)
//!
//! Protocol:
//! 1. V→P: ρ ← F^m (random challenge)
//! 2. Both compute: v = Σ_i eq(ρ,i) · v_i (where v_i are virtual claims from Phase 1)
//! 3. Sumcheck on: Σ_s eq(ρ,s) · M(s,r_x) = v
//! 4. Output claim: M(r_s,r_x) = c_m / eq(ρ,r_s)
//!
//! Where:
//! - M(s,x) is the packed MLE: M(i,b) = p_i(b) for all virtual polynomials p_i
//! - s is a unified index that selects both constraint and polynomial type
//! - r_x is the evaluation point from Phase 1 (what we call x*)
//! - ρ is the random challenge for virtualization (what we call r_s)

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
        recursion_constraints::ConstraintSystem, sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    zkvm::witness::{CommittedPolynomial, VirtualPolynomial},
};
use ark_bn254::Fq;
use ark_ff::Zero;
use rayon::prelude::*;

/// Parameters for virtualization sumcheck
#[derive(Clone)]
pub struct RecursionVirtualizationParams {
    /// Number of s variables (unified index for constraint and poly type)
    pub num_s_vars: usize,

    /// Number of constraints (actual, no padding)
    pub num_constraints: usize,

    /// Number of constraints padded to power of 2
    pub num_constraints_padded: usize,

    /// Sumcheck instance identifier
    pub sumcheck_id: SumcheckId,

    /// Committed polynomial for M
    pub polynomial: CommittedPolynomial,
}

impl RecursionVirtualizationParams {
    pub fn new(
        num_s_vars: usize,
        num_constraints: usize,
        num_constraints_padded: usize,
        polynomial: CommittedPolynomial,
    ) -> Self {
        Self {
            num_s_vars,
            num_constraints,
            num_constraints_padded,
            sumcheck_id: SumcheckId::RecursionVirtualization,
            polynomial,
        }
    }

    /// Total sumcheck rounds: all s variables
    pub fn num_rounds(&self) -> usize {
        self.num_s_vars
    }
}

/// Prover for recursion virtualization sumcheck
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct RecursionVirtualizationProver {
    /// Parameters
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub params: RecursionVirtualizationParams,

    /// Materialized M(s, x) as a multilinear polynomial (bound to x*)
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub m_poly: MultilinearPolynomial<Fq>,

    /// Equality polynomial eq(ρ, s)
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub eq_rho_s: MultilinearPolynomial<Fq>,

    /// Random challenge ρ for virtualization
    pub rho: Vec<<Fq as JoltField>::Challenge>,

    /// Evaluation point x* from Phase 1 (reversed for big-endian)
    pub x_star: Vec<<Fq as JoltField>::Challenge>,

    /// Individual virtual claims from Phase 1 for each constraint
    pub base_claims: Vec<Fq>,
    pub rho_prev_claims: Vec<Fq>,
    pub rho_curr_claims: Vec<Fq>,
    pub quotient_claims: Vec<Fq>,

    /// Gamma coefficient from Phase 1
    pub gamma: Fq,

    /// Current round
    pub round: usize,

    /// Number of constraint variables (x) - fixed at 4 for Fq12
    pub num_constraint_vars: usize,
}

impl RecursionVirtualizationProver {
    pub fn new<T: Transcript>(
        params: RecursionVirtualizationParams,
        constraint_system: &ConstraintSystem,
        transcript: &mut T,
        x_star: Vec<<Fq as JoltField>::Challenge>,
        phase1_accumulator: &ProverOpeningAccumulator<Fq>,
        gamma: Fq,
    ) -> Self {
        // V→P: ρ ← F^m (get random virtualization challenge)
        let rho: Vec<<Fq as JoltField>::Challenge> = (0..params.num_rounds())
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();

        // Compute eq(ρ, s) polynomial
        let eq_rho_s = MultilinearPolynomial::from(EqPolynomial::<Fq>::evals(&rho));

        // Create M polynomial and bind x variables to x*
        let mut m_poly = MultilinearPolynomial::from(constraint_system.matrix.evaluations.clone());

        // Matrix layout is [x_bits, s_bits] in little-endian
        // We need to bind the low-order x bits to x*
        // x_star comes in challenge order, convert to field elements
        let x_star_fq: Vec<Fq> = x_star.iter().map(|c| (*c).into()).collect();

        // Bind x variables (first num_constraint_vars variables in little-endian order)
        for i in 0..constraint_system.matrix.num_constraint_vars {
            m_poly.bind_parallel(x_star_fq[i].into(), BindingOrder::LowToHigh);
        }

        // Now m_poly only has s variables
        assert_eq!(
            m_poly.get_num_vars(),
            params.num_s_vars,
            "M polynomial should only have s variables after binding x*"
        );

        // Get individual virtual claims from Phase 1 for each constraint
        let mut base_claims = Vec::new();
        let mut rho_prev_claims = Vec::new();
        let mut rho_curr_claims = Vec::new();
        let mut quotient_claims = Vec::new();

        for i in 0..params.num_constraints {
            let (_, base_claim) = phase1_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RecursionBase(i),
                SumcheckId::SquareAndMultiply,
            );
            let (_, rho_prev_claim) = phase1_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RecursionRhoPrev(i),
                SumcheckId::SquareAndMultiply,
            );
            let (_, rho_curr_claim) = phase1_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RecursionRhoCurr(i),
                SumcheckId::SquareAndMultiply,
            );
            let (_, quotient_claim) = phase1_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RecursionQuotient(i),
                SumcheckId::SquareAndMultiply,
            );

            base_claims.push(base_claim);
            rho_prev_claims.push(rho_prev_claim);
            rho_curr_claims.push(rho_curr_claim);
            quotient_claims.push(quotient_claim);
        }

        Self {
            params,
            m_poly,
            eq_rho_s,
            rho,
            x_star,
            base_claims,
            rho_prev_claims,
            rho_curr_claims,
            quotient_claims,
            gamma,
            round: 0,
            num_constraint_vars: constraint_system.matrix.num_constraint_vars,
        }
    }
}

impl<T: Transcript> SumcheckInstanceProver<Fq, T> for RecursionVirtualizationProver {
    fn degree(&self) -> usize {
        2 // Degree 2 because eq(r_s, s) * M(s, x) in the s variables
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<Fq>) -> Fq {
        // Step 2: Compute v = Σ_i eq(ρ,i) · v_i
        // where v_i are the virtual polynomial claims from Phase 1
        let eq_evals = match &self.eq_rho_s {
            MultilinearPolynomial::LargeScalars(poly) => poly.Z.clone(),
            _ => panic!("Expected eq_rho_s to be LargeScalars variant"),
        };

        // Build μ evaluations from Phase 1 claims
        let mu_size = 1 << self.params.num_s_vars;
        let mut mu_evals = vec![Fq::zero(); mu_size];

        // Fill μ following the matrix layout
        for i in 0..self.params.num_constraints {
            mu_evals[0 * self.params.num_constraints_padded + i] = self.base_claims[i];
            mu_evals[1 * self.params.num_constraints_padded + i] = self.rho_prev_claims[i];
            mu_evals[2 * self.params.num_constraints_padded + i] = self.rho_curr_claims[i];
            mu_evals[3 * self.params.num_constraints_padded + i] = self.quotient_claims[i];
        }

        // Compute the inner product: Σ_s eq(ρ, s) * μ(s)
        // This gives us v = Σ_i eq(ρ,i) · v_i from the spec
        eq_evals
            .iter()
            .zip(mu_evals.iter())
            .map(|(eq_val, mu_val)| *eq_val * *mu_val)
            .sum()
    }

    #[tracing::instrument(skip_all, name = "RecursionVirtualization::compute_message")]
    fn compute_message(&mut self, round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        const DEGREE: usize = 2;
        let num_s_remaining = self.eq_rho_s.get_num_vars();
        let s_half = 1 << (num_s_remaining - 1);

        if num_s_remaining == 1 {
            eprintln!("=== FINAL ROUND DEBUG (round {}) ===", round);
            eprintln!("num_s_remaining = {}", num_s_remaining);
            eprintln!("previous_claim = {:?}", previous_claim);
        }

        // Step 3: Sumcheck on Σ_s eq(ρ,s) · M(s,r_x) = v
        // M is already bound by r_x (what we call x*), so we're summing over s variables
        let total_evals = (0..s_half)
            .into_par_iter()
            .map(|s_idx| {
                // Get evaluations using hints (up to DEGREE)
                let eq_rho_evals = self
                    .eq_rho_s
                    .sumcheck_evals_array::<DEGREE>(s_idx, BindingOrder::LowToHigh);

                // Get M evaluations at this s index (already bound to r_x)
                let m_evals = self
                    .m_poly
                    .sumcheck_evals_array::<DEGREE>(s_idx, BindingOrder::LowToHigh);

                let mut s_evals = [Fq::zero(); DEGREE];

                // Compute eq(ρ, s) * M(s, r_x)
                for t in 0..DEGREE {
                    s_evals[t] = eq_rho_evals[t] * m_evals[t];
                }

                if num_s_remaining == 1 && s_idx == 0 {
                    eprintln!("Final round polynomial computation:");
                    eprintln!("  s_idx = {}", s_idx);
                    eprintln!("  eq_rho_evals = {:?}", eq_rho_evals);
                    eprintln!("  m_evals = {:?}", m_evals);
                    eprintln!("  s_evals = {:?}", s_evals);
                }

                s_evals
            })
            .reduce(
                || [Fq::zero(); DEGREE],
                |mut acc, evals| {
                    for (a, e) in acc.iter_mut().zip(evals.iter()) {
                        *a += *e;
                    }
                    acc
                },
            );

        // Use eval_with_hint to interpolate and get the DEGREE+1 evaluation using previous claim
        let poly = UniPoly::from_evals_and_hint(previous_claim, &total_evals);

        if num_s_remaining == 1 {
            eprintln!("total_evals = {:?}", total_evals);
            eprintln!("poly coeffs = {:?}", poly.coeffs);
            eprintln!("===========================");
        }

        poly
    }

    #[tracing::instrument(skip_all, name = "RecursionVirtualization::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: <Fq as JoltField>::Challenge, round: usize) {
        // Bind s variable in polynomials
        self.eq_rho_s.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.m_poly.bind_parallel(r_j, BindingOrder::LowToHigh);

        self.round = round + 1;
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // Construct opening point for M: (x*, s*)
        let mut opening_point_vec = self.x_star.clone();
        opening_point_vec.extend(sumcheck_challenges.to_vec());

        // Reverse entire point for big-endian
        opening_point_vec.reverse();

        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(opening_point_vec);
        let m_claim = self.m_poly.get_bound_coeff(0);

        accumulator.append_dense(
            transcript,
            self.params.polynomial,
            self.params.sumcheck_id,
            opening_point.r,
            m_claim,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// Verifier for recursion virtualization sumcheck
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct RecursionVirtualizationVerifier {
    pub params: RecursionVirtualizationParams,
    pub gamma: Fq,
    pub x_star: Vec<<Fq as JoltField>::Challenge>,
    pub constraint_bits: Vec<bool>,
    pub rho: Vec<<Fq as JoltField>::Challenge>,
}

impl RecursionVirtualizationVerifier {
    pub fn new<T: Transcript>(
        params: RecursionVirtualizationParams,
        constraint_bits: Vec<bool>,
        transcript: &mut T,
        x_star: Vec<<Fq as JoltField>::Challenge>,
        gamma: Fq,
    ) -> Self {
        // V→P: ρ ← F^m (get random virtualization challenge)
        let rho: Vec<<Fq as JoltField>::Challenge> = (0..params.num_rounds())
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();

        Self {
            params,
            gamma,
            x_star,
            constraint_bits,
            rho,
        }
    }
}

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for RecursionVirtualizationVerifier {
    fn degree(&self) -> usize {
        2 // Degree 2 because eq(r_s, s) * M(s, x) in the s variables
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<Fq>) -> Fq {
        // Step 2: Compute v = Σ_i eq(ρ,i) · v_i
        // where v_i are the virtual polynomial claims from Phase 1
        let mu_size = 1 << self.params.num_s_vars;
        let mut mu_evals = vec![Fq::zero(); mu_size];

        // Fill μ following the matrix layout (μ(i) = v_i)
        for i in 0..self.params.num_constraints {
            let (_, claim) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RecursionBase(i),
                SumcheckId::SquareAndMultiply,
            );
            mu_evals[0 * self.params.num_constraints_padded + i] = claim;

            let (_, claim) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RecursionRhoPrev(i),
                SumcheckId::SquareAndMultiply,
            );
            mu_evals[1 * self.params.num_constraints_padded + i] = claim;

            let (_, claim) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RecursionRhoCurr(i),
                SumcheckId::SquareAndMultiply,
            );
            mu_evals[2 * self.params.num_constraints_padded + i] = claim;

            let (_, claim) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RecursionQuotient(i),
                SumcheckId::SquareAndMultiply,
            );
            mu_evals[3 * self.params.num_constraints_padded + i] = claim;
        }

        // Compute eq(ρ, s) polynomial evals
        let rho_fq: Vec<Fq> = self.rho.iter().map(|c| (*c).into()).collect();
        let eq_evals = EqPolynomial::<Fq>::evals(&rho_fq);

        // Compute inner product: v = Σ_s eq(ρ, s) * μ(s) = Σ_i eq(ρ,i) · v_i
        eq_evals
            .iter()
            .zip(mu_evals.iter())
            .map(|(eq_val, mu_val)| *eq_val * *mu_val)
            .sum()
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<Fq>,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> Fq {
        // Step 4: Output claim M(r_s,r_x) = c_m / eq(ρ,r_s)
        // Get M opening at (r_s, r_x) where r_s = sumcheck_challenges
        let (_, m_claim) = accumulator
            .get_committed_polynomial_opening(self.params.polynomial, self.params.sumcheck_id);

        // Compute eq(ρ, r_s)
        let r_s: Vec<Fq> = sumcheck_challenges
            .iter()
            .rev()
            .map(|c| (*c).into())
            .collect();
        let rho_fq: Vec<Fq> = self.rho.iter().map(|c| (*c).into()).collect();
        let eq_eval = EqPolynomial::mle(&rho_fq, &r_s);

        eprintln!("=== PHASE 2 EXPECTED OUTPUT DEBUG ===");
        eprintln!("m_claim = M(r_s, r_x) = {:?}", m_claim);
        eprintln!("eq_eval = eq(ρ, r_s) = {:?}", eq_eval);
        eprintln!(
            "expected output = eq(ρ, r_s) * M(r_s, r_x) = {:?}",
            eq_eval * m_claim
        );
        eprintln!("=====================================");

        // Expected output: eq(ρ, r_s) * M(r_s, r_x)
        // This matches the spec: if output_claim = c_m, then M(r_s,r_x) = c_m / eq(ρ,r_s)
        eq_eval * m_claim
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // Construct opening point for M: (x*, s*)
        let mut opening_point_vec = self.x_star.clone();
        opening_point_vec.extend(sumcheck_challenges.to_vec());

        // Reverse entire point for big-endian
        opening_point_vec.reverse();

        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(opening_point_vec);

        // Register the expected opening for M
        accumulator.append_dense(
            transcript,
            self.params.polynomial,
            self.params.sumcheck_id,
            opening_point.r,
        );
    }
}
