//! Stage 2: Direct Evaluation Protocol
//!
//! This module implements the optimized Stage 2 protocol that directly evaluates
//! M(r_s, r_x) without running a sumcheck. The key insight is that M is the
//! multilinear extension of the virtual claims v_i from Stage 1.
//!
//! Protocol flow:
//! 1. Sample r_s directly from the transcript
//! 2. Prover evaluates M(r_s, r_x) where r_x comes from Stage 1
//! 3. Verifier computes expected value: Σ_i eq(r_s, i) · v_i
//! 4. Verify that the two values match

use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation},
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
        },
    },
    transcripts::Transcript,
    zkvm::{
        recursion::constraints_sys::ConstraintType,
        witness::VirtualPolynomial,
    },
};
use ark_bn254::Fq;
use ark_ff::Zero;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

/// Parameters for the direct evaluation protocol
#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct DirectEvaluationParams {
    /// Number of s-variables (log of matrix rows)
    pub num_s_vars: usize,
    /// Number of constraints
    pub num_constraints: usize,
    /// Padded number of constraints (next power of 2)
    pub num_constraints_padded: usize,
    /// Number of constraint variables (x variables)
    pub num_constraint_vars: usize,
    /// Number of polynomial types (15 for all constraint types)
    pub num_poly_types: usize,
}

impl DirectEvaluationParams {
    pub fn new(
        num_s_vars: usize,
        num_constraints: usize,
        num_constraints_padded: usize,
        num_constraint_vars: usize,
    ) -> Self {
        Self {
            num_s_vars,
            num_constraints,
            num_constraints_padded,
            num_constraint_vars,
            num_poly_types: 15, // Fixed for the 15 polynomial types
        }
    }
}

/// Prover for the direct evaluation protocol
pub struct DirectEvaluationProver {
    /// Protocol parameters
    pub params: DirectEvaluationParams,
    /// The constraint matrix M bound to r_x from Stage 1
    pub matrix_bound: MultilinearPolynomial<Fq>,
    /// Virtual claims from Stage 1
    pub virtual_claims: Vec<Fq>,
    /// The r_x point from Stage 1
    pub r_x: Vec<Fq>,
}

impl DirectEvaluationProver {
    /// Create a new prover
    pub fn new(
        params: DirectEvaluationParams,
        matrix_evals: Vec<Fq>,
        virtual_claims: Vec<Fq>,
        r_x: Vec<Fq>,
    ) -> Self {
        // The matrix has layout [x_vars, s_vars] in little-endian
        // We need to bind the x variables to r_x
        let mut matrix_poly = MultilinearPolynomial::LargeScalars(
            DensePolynomial::new(matrix_evals)
        );

        // Bind x variables (first num_constraint_vars variables)
        for i in 0..params.num_constraint_vars {
            matrix_poly.bind_parallel(r_x[i].into(), BindingOrder::LowToHigh);
        }

        assert_eq!(
            matrix_poly.get_num_vars(),
            params.num_s_vars,
            "After binding x vars, should only have s vars left"
        );

        Self {
            params,
            matrix_bound: matrix_poly,
            virtual_claims,
            r_x,
        }
    }

    /// Run the prover protocol
    pub fn prove<T: Transcript>(
        &self,
        transcript: &mut T,
        accumulator: &mut ProverOpeningAccumulator<Fq>,
    ) -> (Vec<Fq>, Fq) {
        // Sample r_s from the transcript
        let r_s: Vec<Fq> = (0..self.params.num_s_vars)
            .map(|_| transcript.challenge_scalar::<Fq>())
            .collect();

        // Evaluate M(r_s, r_x)
        let m_eval = PolynomialEvaluation::evaluate(&self.matrix_bound, &r_s);

        eprintln!("Prover Stage 2:");
        eprintln!("  m_eval = {:?}", m_eval);
        eprintln!("  num_s_vars = {}", self.params.num_s_vars);
        eprintln!("  r_s.len() = {}", r_s.len());
        eprintln!("  matrix_bound num_vars = {}", self.matrix_bound.get_num_vars());

        // Note: m_eval is passed in the proof structure, but we still append
        // to transcript to maintain Fiat-Shamir soundness
        transcript.append_scalar(&m_eval);

        // Store the opening in the accumulator for Stage 3
        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(
            r_s.iter()
                .rev()
                .chain(self.r_x.iter().rev())
                .cloned()
                .map(|f| f.into())
                .collect(),
        );

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::DorySparseConstraintMatrix,
            SumcheckId::RecursionVirtualization,
            opening_point,
            m_eval,
        );

        (r_s, m_eval)
    }
}

/// Verifier for the direct evaluation protocol
pub struct DirectEvaluationVerifier {
    /// Protocol parameters
    pub params: DirectEvaluationParams,
    /// Virtual claims from Stage 1
    pub virtual_claims: Vec<Fq>,
    /// The r_x point from Stage 1
    pub r_x: Vec<Fq>,
}

impl DirectEvaluationVerifier {
    /// Create a new verifier
    pub fn new(
        params: DirectEvaluationParams,
        virtual_claims: Vec<Fq>,
        r_x: Vec<Fq>,
    ) -> Self {
        Self {
            params,
            virtual_claims,
            r_x,
        }
    }

    /// Run the verifier protocol
    pub fn verify<T: Transcript>(
        &self,
        transcript: &mut T,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        m_eval_claimed: Fq,
    ) -> Result<Vec<Fq>, String> {
        // Sample the same r_s as the prover
        let r_s: Vec<Fq> = (0..self.params.num_s_vars)
            .map(|_| transcript.challenge_scalar::<Fq>())
            .collect();

        // Compute the expected value: Σ_i eq(r_s, i) · v_i
        let eq_evals = EqPolynomial::<Fq>::evals(&r_s);
        let m_eval_expected = self.compute_expected_evaluation(&eq_evals);

        // Verify the claim
        if m_eval_claimed != m_eval_expected {
            eprintln!("Direct evaluation verification failed:");
            eprintln!("  m_eval_claimed   = {:?}", m_eval_claimed);
            eprintln!("  m_eval_expected  = {:?}", m_eval_expected);
            eprintln!("  num_claims       = {}", self.virtual_claims.len());
            eprintln!("  num_s_vars       = {}", self.params.num_s_vars);
            eprintln!("  num_constraints  = {}", self.params.num_constraints);
            return Err("Direct evaluation verification failed: M(r_s, r_x) mismatch".to_string());
        }

        // Append to transcript to maintain Fiat-Shamir soundness
        transcript.append_scalar(&m_eval_claimed);

        // Store the opening in the accumulator for Stage 3
        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(
            r_s.iter()
                .rev()
                .chain(self.r_x.iter().rev())
                .cloned()
                .map(|f| f.into())
                .collect(),
        );

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::DorySparseConstraintMatrix,
            SumcheckId::RecursionVirtualization,
            opening_point,
        );

        Ok(r_s)
    }

    /// Compute Σ_i eq(r_s, i) · v_i
    fn compute_expected_evaluation(&self, eq_evals: &[Fq]) -> Fq {
        let mut result = Fq::zero();

        eprintln!("Computing expected evaluation:");
        eprintln!("  eq_evals.len() = {}", eq_evals.len());
        eprintln!("  virtual_claims.len() = {}", self.virtual_claims.len());
        eprintln!("  num_constraints = {}", self.params.num_constraints);
        eprintln!("  num_constraints_padded = {}", self.params.num_constraints_padded);
        eprintln!("  num_poly_types = {}", self.params.num_poly_types);

        // The virtual claims are laid out as:
        // [constraint_0_poly_0, constraint_1_poly_0, ..., constraint_0_poly_1, ...]
        // We need to match this with the eq evaluations

        for constraint_idx in 0..self.params.num_constraints {
            for poly_idx in 0..self.params.num_poly_types {
                let claim_idx = constraint_idx * self.params.num_poly_types + poly_idx;
                let s_idx = poly_idx * self.params.num_constraints_padded + constraint_idx;

                if claim_idx < self.virtual_claims.len() && s_idx < eq_evals.len() {
                    let term = eq_evals[s_idx] * self.virtual_claims[claim_idx];
                    if !term.is_zero() {
                        eprintln!("  Adding term: eq[{}] * claim[{}] = {:?} * {:?} = {:?}",
                            s_idx, claim_idx, eq_evals[s_idx], self.virtual_claims[claim_idx], term);
                    }
                    result += term;
                }
            }
        }

        eprintln!("  Final result = {:?}", result);
        result
    }
}

/// Extract virtual claims from Stage 1 accumulator in the correct order
pub fn extract_virtual_claims_from_accumulator<F: JoltField>(
    accumulator: &ProverOpeningAccumulator<F>,
    constraint_types: &[ConstraintType],
) -> Vec<F> {
    use crate::poly::opening_proof::OpeningAccumulator;

    let mut claims = Vec::new();

    // Process each constraint
    for (idx, constraint_type) in constraint_types.iter().enumerate() {
        // For each constraint, we need to extract claims for all 15 polynomial types
        // in the correct order matching the PolyType enum

        let mut constraint_claims = vec![F::zero(); 15];

        match constraint_type {
            ConstraintType::GtExp { .. } => {
                // GT Exp uses polynomials 0-3 (Base, RhoPrev, RhoCurr, Quotient)
                let (_, base) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionBase(idx),
                    SumcheckId::SquareAndMultiply,
                );
                let (_, rho_prev) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionRhoPrev(idx),
                    SumcheckId::SquareAndMultiply,
                );
                let (_, rho_curr) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionRhoCurr(idx),
                    SumcheckId::SquareAndMultiply,
                );
                let (_, quotient) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionQuotient(idx),
                    SumcheckId::SquareAndMultiply,
                );

                constraint_claims[0] = base;
                constraint_claims[1] = rho_prev;
                constraint_claims[2] = rho_curr;
                constraint_claims[3] = quotient;
            }
            ConstraintType::GtMul { .. } => {
                // GT Mul uses polynomials 4-7 (MulLhs, MulRhs, MulResult, MulQuotient)
                let (_, lhs) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionMulLhs(idx),
                    SumcheckId::GtMul,
                );
                let (_, rhs) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionMulRhs(idx),
                    SumcheckId::GtMul,
                );
                let (_, result) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionMulResult(idx),
                    SumcheckId::GtMul,
                );
                let (_, quotient) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionMulQuotient(idx),
                    SumcheckId::GtMul,
                );

                constraint_claims[4] = lhs;
                constraint_claims[5] = rhs;
                constraint_claims[6] = result;
                constraint_claims[7] = quotient;
            }
            ConstraintType::G1ScalarMul { .. } => {
                // G1 Scalar Mul uses polynomials 8-14
                let (_, x_a) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionG1ScalarMulXA(idx),
                    SumcheckId::G1ScalarMul,
                );
                let (_, y_a) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionG1ScalarMulYA(idx),
                    SumcheckId::G1ScalarMul,
                );
                let (_, x_t) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionG1ScalarMulXT(idx),
                    SumcheckId::G1ScalarMul,
                );
                let (_, y_t) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionG1ScalarMulYT(idx),
                    SumcheckId::G1ScalarMul,
                );
                let (_, x_a_next) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionG1ScalarMulXANext(idx),
                    SumcheckId::G1ScalarMul,
                );
                let (_, y_a_next) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionG1ScalarMulYANext(idx),
                    SumcheckId::G1ScalarMul,
                );
                let (_, indicator) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionG1ScalarMulIndicator(idx),
                    SumcheckId::G1ScalarMul,
                );

                constraint_claims[8] = x_a;
                constraint_claims[9] = y_a;
                constraint_claims[10] = x_t;
                constraint_claims[11] = y_t;
                constraint_claims[12] = x_a_next;
                constraint_claims[13] = y_a_next;
                constraint_claims[14] = indicator;
            }
        }

        claims.extend(constraint_claims);
    }

    claims
}

/// Extract virtual claims from verifier accumulator
pub fn extract_virtual_claims_from_verifier_accumulator<F: JoltField>(
    accumulator: &VerifierOpeningAccumulator<F>,
    constraint_types: &[ConstraintType],
) -> Vec<F> {
    use crate::poly::opening_proof::OpeningAccumulator;

    let mut claims = Vec::new();

    // Same logic as prover version, but for verifier accumulator
    for (idx, constraint_type) in constraint_types.iter().enumerate() {
        let mut constraint_claims = vec![F::zero(); 15];

        match constraint_type {
            ConstraintType::GtExp { .. } => {
                let (_, base) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionBase(idx),
                    SumcheckId::SquareAndMultiply,
                );
                let (_, rho_prev) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionRhoPrev(idx),
                    SumcheckId::SquareAndMultiply,
                );
                let (_, rho_curr) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionRhoCurr(idx),
                    SumcheckId::SquareAndMultiply,
                );
                let (_, quotient) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionQuotient(idx),
                    SumcheckId::SquareAndMultiply,
                );

                constraint_claims[0] = base;
                constraint_claims[1] = rho_prev;
                constraint_claims[2] = rho_curr;
                constraint_claims[3] = quotient;
            }
            ConstraintType::GtMul { .. } => {
                let (_, lhs) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionMulLhs(idx),
                    SumcheckId::GtMul,
                );
                let (_, rhs) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionMulRhs(idx),
                    SumcheckId::GtMul,
                );
                let (_, result) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionMulResult(idx),
                    SumcheckId::GtMul,
                );
                let (_, quotient) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionMulQuotient(idx),
                    SumcheckId::GtMul,
                );

                constraint_claims[4] = lhs;
                constraint_claims[5] = rhs;
                constraint_claims[6] = result;
                constraint_claims[7] = quotient;
            }
            ConstraintType::G1ScalarMul { .. } => {
                let (_, x_a) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionG1ScalarMulXA(idx),
                    SumcheckId::G1ScalarMul,
                );
                let (_, y_a) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionG1ScalarMulYA(idx),
                    SumcheckId::G1ScalarMul,
                );
                let (_, x_t) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionG1ScalarMulXT(idx),
                    SumcheckId::G1ScalarMul,
                );
                let (_, y_t) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionG1ScalarMulYT(idx),
                    SumcheckId::G1ScalarMul,
                );
                let (_, x_a_next) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionG1ScalarMulXANext(idx),
                    SumcheckId::G1ScalarMul,
                );
                let (_, y_a_next) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionG1ScalarMulYANext(idx),
                    SumcheckId::G1ScalarMul,
                );
                let (_, indicator) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionG1ScalarMulIndicator(idx),
                    SumcheckId::G1ScalarMul,
                );

                constraint_claims[8] = x_a;
                constraint_claims[9] = y_a;
                constraint_claims[10] = x_t;
                constraint_claims[11] = y_t;
                constraint_claims[12] = x_a_next;
                constraint_claims[13] = y_a_next;
                constraint_claims[14] = indicator;
            }
        }

        claims.extend(constraint_claims);
    }

    claims
}