//! Virtualization sumcheck for recursion protocol (Phase 2)
//!
//! Protocol:
//! 1. V→P: r_s ← F^m (random challenge for virtualization)
//! 2. Both compute: v = Σ_i eq(r_s,i) · v_i (where v_i are virtual claims from Phase 1)
//! 3. Sumcheck on: Σ_s eq(r_s,s) · M(s,r_x) = v
//! 4. Output claim: M(r_s_final,r_x) = c_m / eq(r_s,r_s_final)
//!
//! Where:
//! - M(s,x) is the packed MLE: M(i,b) = p_i(b) for all virtual polynomials p_i
//! - s is a unified index that selects both constraint and polynomial type
//! - r_x is the evaluation point from Phase 1 (sumcheck challenge)
//! - r_s is the random challenge for virtualization
//! - r_s_final is the final sumcheck challenge point

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

/// Compute v = Σ_i eq(r_s,i) · v_i for the virtualization protocol
/// This is shared by both prover and verifier
fn compute_virtualization_claim(
    params: &RecursionVirtualizationParams,
    eq_evals: &[Fq],
    base_claims: &[Fq],
    rho_prev_claims: &[Fq],
    rho_curr_claims: &[Fq],
    quotient_claims: &[Fq],
) -> Fq {
    // Build μ evaluations from Phase 1 claims
    let mu_size = 1 << params.num_s_vars;
    let mut mu_evals = vec![Fq::zero(); mu_size];

    // Fill μ following the matrix layout
    for i in 0..params.num_constraints {
        mu_evals[0 * params.num_constraints_padded + i] = base_claims[i];
        mu_evals[1 * params.num_constraints_padded + i] = rho_prev_claims[i];
        mu_evals[2 * params.num_constraints_padded + i] = rho_curr_claims[i];
        mu_evals[3 * params.num_constraints_padded + i] = quotient_claims[i];
    }

    // Compute the inner product: Σ_s eq(r_s, s) * μ(s)
    eq_evals
        .iter()
        .zip(mu_evals.iter())
        .map(|(eq_val, mu_val)| *eq_val * *mu_val)
        .sum()
}

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

    /// Equality polynomial eq(r_s, s)
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub eq_r_s: MultilinearPolynomial<Fq>,

    /// Evaluation point from Phase 1 sumcheck (r_x from square-and-multiply)
    pub r_x_prev: Vec<<Fq as JoltField>::Challenge>,

    /// Random challenge r_s for virtualization
    pub r_s: Vec<<Fq as JoltField>::Challenge>,

    /// Gamma coefficient from Phase 1
    pub gamma: Fq,

    /// Individual virtual claims from Phase 1 for each constraint
    pub base_claims: Vec<Fq>,
    pub rho_prev_claims: Vec<Fq>,
    pub rho_curr_claims: Vec<Fq>,
    pub quotient_claims: Vec<Fq>,

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
        r_x_prev: Vec<<Fq as JoltField>::Challenge>,
        phase1_accumulator: &ProverOpeningAccumulator<Fq>,
        gamma: Fq,
    ) -> Self {
        let r_s: Vec<<Fq as JoltField>::Challenge> = (0..params.num_rounds())
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();

        let eq_r_s = MultilinearPolynomial::from(EqPolynomial::<Fq>::evals(&r_s));
        let mut m_poly = MultilinearPolynomial::from(constraint_system.matrix.evaluations.clone());

        // Matrix layout is [x_bits, s_bits] in little-endian
        // We need to bind the low-order x bits to r_x_prev
        let r_x_prev_fq: Vec<Fq> = r_x_prev.iter().map(|c| (*c).into()).collect();

        for i in 0..constraint_system.matrix.num_constraint_vars {
            m_poly.bind_parallel(r_x_prev_fq[i].into(), BindingOrder::LowToHigh);
        }
        assert_eq!(
            m_poly.get_num_vars(),
            params.num_s_vars,
            "M polynomial should only have s variables after binding r_x_prev"
        );

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
            eq_r_s,
            r_x_prev,
            r_s,
            gamma,
            base_claims,
            rho_prev_claims,
            rho_curr_claims,
            quotient_claims,
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
        let eq_evals = match &self.eq_r_s {
            MultilinearPolynomial::LargeScalars(poly) => poly.Z.clone(),
            _ => panic!("Expected eq_r_s to be LargeScalars variant"),
        };

        compute_virtualization_claim(
            &self.params,
            &eq_evals,
            &self.base_claims,
            &self.rho_prev_claims,
            &self.rho_curr_claims,
            &self.quotient_claims,
        )
    }

    #[tracing::instrument(skip_all, name = "RecursionVirtualization::compute_message")]
    fn compute_message(&mut self, round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        const DEGREE: usize = 2;
        let num_s_remaining = self.eq_r_s.get_num_vars();
        let s_half = 1 << (num_s_remaining - 1);

        if num_s_remaining == 1 {
            eprintln!("=== FINAL ROUND DEBUG (round {}) ===", round);
            eprintln!("num_s_remaining = {}", num_s_remaining);
            eprintln!("previous_claim = {:?}", previous_claim);
        }

        // Step 3: Sumcheck on Σ_s eq(r_s,s) · M(s,r_x) = v
        // M is already bound by r_x, so we're summing over s variables
        let total_evals = (0..s_half)
            .into_par_iter()
            .map(|s_idx| {
                let eq_r_s_evals = self
                    .eq_r_s
                    .sumcheck_evals_array::<DEGREE>(s_idx, BindingOrder::LowToHigh);

                let m_evals = self
                    .m_poly
                    .sumcheck_evals_array::<DEGREE>(s_idx, BindingOrder::LowToHigh);

                let mut s_evals = [Fq::zero(); DEGREE];

                for t in 0..DEGREE {
                    s_evals[t] = eq_r_s_evals[t] * m_evals[t];
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

        UniPoly::from_evals_and_hint(previous_claim, &total_evals)
    }

    #[tracing::instrument(skip_all, name = "RecursionVirtualization::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: <Fq as JoltField>::Challenge, round: usize) {
        self.eq_r_s.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.m_poly.bind_parallel(r_j, BindingOrder::LowToHigh);

        self.round = round + 1;
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // Construct opening point for M: (r_x_prev, r_s_final) in big-endian order
        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(
            sumcheck_challenges
                .iter()
                .rev()
                .chain(self.r_x_prev.iter().rev())
                .cloned()
                .collect(),
        );
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
    pub r_x_prev: Vec<<Fq as JoltField>::Challenge>,
    pub r_s: Vec<<Fq as JoltField>::Challenge>,
    pub gamma: Fq,
    pub constraint_bits: Vec<bool>,
}

impl RecursionVirtualizationVerifier {
    pub fn new<T: Transcript>(
        params: RecursionVirtualizationParams,
        constraint_bits: Vec<bool>,
        transcript: &mut T,
        r_x_prev: Vec<<Fq as JoltField>::Challenge>,
        gamma: Fq,
    ) -> Self {
        let r_s: Vec<<Fq as JoltField>::Challenge> = (0..params.num_rounds())
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();

        Self {
            params,
            r_x_prev,
            r_s,
            gamma,
            constraint_bits,
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
        let mut base_claims = Vec::with_capacity(self.params.num_constraints);
        let mut rho_prev_claims = Vec::with_capacity(self.params.num_constraints);
        let mut rho_curr_claims = Vec::with_capacity(self.params.num_constraints);
        let mut quotient_claims = Vec::with_capacity(self.params.num_constraints);

        for i in 0..self.params.num_constraints {
            let (_, base_claim) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RecursionBase(i),
                SumcheckId::SquareAndMultiply,
            );
            base_claims.push(base_claim);

            let (_, rho_prev_claim) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RecursionRhoPrev(i),
                SumcheckId::SquareAndMultiply,
            );
            rho_prev_claims.push(rho_prev_claim);

            let (_, rho_curr_claim) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RecursionRhoCurr(i),
                SumcheckId::SquareAndMultiply,
            );
            rho_curr_claims.push(rho_curr_claim);

            let (_, quotient_claim) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RecursionQuotient(i),
                SumcheckId::SquareAndMultiply,
            );
            quotient_claims.push(quotient_claim);
        }

        let r_s_fq: Vec<Fq> = self.r_s.iter().map(|c| (*c).into()).collect();
        let eq_evals = EqPolynomial::<Fq>::evals(&r_s_fq);

        compute_virtualization_claim(
            &self.params,
            &eq_evals,
            &base_claims,
            &rho_prev_claims,
            &rho_curr_claims,
            &quotient_claims,
        )
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<Fq>,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> Fq {
        // Step 4: Output claim M(r_s_final,r_x) = c_m / eq(r_s,r_s_final)
        let (_, m_claim) = accumulator
            .get_committed_polynomial_opening(self.params.polynomial, self.params.sumcheck_id);

        let r_s_final: Vec<Fq> = sumcheck_challenges
            .iter()
            .rev()
            .map(|c| (*c).into())
            .collect();
        let r_s_fq: Vec<Fq> = self.r_s.iter().map(|c| (*c).into()).collect();
        let eq_eval = EqPolynomial::mle(&r_s_fq, &r_s_final);

        // Expected output: eq(r_s, r_s_final) * M(r_s_final, r_x)
        // This matches the spec: if output_claim = c_m, then M(r_s_final,r_x) = c_m / eq(r_s,r_s_final)
        eq_eval * m_claim
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // Construct opening point for M: (r_x_prev, r_s_final) in big-endian order
        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(
            sumcheck_challenges
                .iter()
                .rev()
                .chain(self.r_x_prev.iter().rev())
                .cloned()
                .collect(),
        );

        accumulator.append_dense(
            transcript,
            self.params.polynomial,
            self.params.sumcheck_id,
            opening_point.r,
        );
    }
}
