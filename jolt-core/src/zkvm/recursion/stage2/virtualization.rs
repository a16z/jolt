//! Stage 2: Direct Evaluation Protocol for Recursion SNARK
//!
//! This module implements the optimized Stage 2 protocol that directly evaluates
//! M(r_s, r_x) without running a sumcheck. The key insight is that M is the
//! multilinear extension of the virtual claims v_i from Stage 1.
//!
//! ## Mathematical Foundation
//!
//! The matrix M is defined such that M(i, r_x) = v_i for all i, where v_i are
//! the virtual claims from Stage 1. The direct evaluation protocol uses the fact
//! that for the MLE of M:
//!
//! M(r_s, r_x) = Σ_i eq(r_s, i) · M(i, r_x) = Σ_i eq(r_s, i) · v_i
//!
//! ## Data Layout
//!
//! ### Virtual Claims Layout
//! Virtual claims from Stage 1 are organized by constraint then polynomial type:
//! [c0_p0, c0_p1, ..., c0_p14, c1_p0, c1_p1, ..., c1_p14, ...]
//!
//! ### Matrix S Layout
//! The matrix S rows are indexed differently for mathematical efficiency:
//! Row index = poly_type * num_constraints_padded + constraint_idx
//!
//! This transposed layout ensures proper alignment for the virtualization sumcheck.
//!
//! ## Protocol Flow
//!
//! 1. **Sampling**: Sample r_s directly from the Fiat-Shamir transcript
//! 2. **Prover Evaluation**: Prover evaluates M(r_s, r_x) where r_x comes from Stage 1
//! 3. **Verifier Computation**: Verifier computes Σ_i eq(r_s, i) · v_i independently
//! 4. **Verification**: Check that prover's evaluation matches verifier's computation

use thiserror::Error;

/// Errors that can occur in Stage 2 direct evaluation protocol
#[derive(Debug, Error)]
pub enum Stage2Error {
    #[error("Direct evaluation mismatch: expected {expected}, got {actual}")]
    EvaluationMismatch { expected: String, actual: String },

    #[error("Invalid accumulator state: {0}")]
    InvalidAccumulator(String),

    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
}

use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
        },
    },
    transcripts::Transcript,
    zkvm::{
        recursion::{
            constraints_sys::ConstraintType,
            stage1::packed_gt_exp::PackedGtExpPublicInputs,
        },
        witness::VirtualPolynomial,
    },
};
use ark_bn254::Fq;
use ark_ff::Zero;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

/// Number of polynomial types in the constraint system
/// Note: This was reduced from 16 to 14 after removing Base and Bit polynomials,
/// then to 13 after removing RhoCurr (rho_next is verified via shift sumcheck)
const NUM_POLY_TYPES: usize = 13;

/// Helper function to compute the index in the virtual claims array
///
/// Virtual claims are laid out as:
/// [constraint_0_poly_0, constraint_0_poly_1, ..., constraint_0_poly_13,
///  constraint_1_poly_0, constraint_1_poly_1, ..., constraint_1_poly_13, ...]
///
/// So for constraint i and polynomial type j, the index is: i * NUM_POLY_TYPES + j
#[inline]
pub fn virtual_claim_index(constraint_idx: usize, poly_idx: usize) -> usize {
    constraint_idx * NUM_POLY_TYPES + poly_idx
}

/// Helper function to compute the index in the matrix S evaluations
///
/// The matrix S is laid out with a different pattern than virtual claims:
/// - Rows are indexed by polynomial type first, then constraint
/// - This layout is: poly_type * num_constraints_padded + constraint_idx
///
/// This is the transpose of how virtual claims are laid out, which is important
/// for the mathematical properties of the virtualization protocol.
#[inline]
pub fn matrix_s_index(poly_idx: usize, constraint_idx: usize, num_constraints_padded: usize) -> usize {
    poly_idx * num_constraints_padded + constraint_idx
}

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

    /// Run the prover protocol at a provided r_s.
    pub fn prove<T: Transcript>(
        &self,
        transcript: &mut T,
        accumulator: &mut ProverOpeningAccumulator<Fq>,
        r_s: Vec<Fq>,
    ) -> (Vec<Fq>, Fq) {
        debug_assert_eq!(r_s.len(), self.params.num_s_vars);

        // Evaluate M(r_s, r_x)
        let m_eval = PolynomialEvaluation::evaluate(&self.matrix_bound, &r_s);

        // Note: m_eval is passed in the proof structure, but we still append
        // to transcript to maintain Fiat-Shamir soundness
        transcript.append_scalar(&m_eval);

        // Store the opening in the accumulator for Stage 3
        // Note: We reverse r_s and r_x because OpeningPoint expects big-endian ordering
        // while our polynomials use little-endian variable ordering internally.
        // The matrix has variables ordered as [x_vars, s_vars] in little-endian.
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

    /// Run the verifier protocol at a provided r_s.
    pub fn verify<T: Transcript>(
        &self,
        transcript: &mut T,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        m_eval_claimed: Fq,
        r_s: Vec<Fq>,
    ) -> Result<Vec<Fq>, Stage2Error> {
        debug_assert_eq!(r_s.len(), self.params.num_s_vars);

        // Compute the expected value: Σ_i eq(r_s, i) · v_i
        let eq_evals = EqPolynomial::<Fq>::evals(&r_s);
        let m_eval_expected = self.compute_expected_evaluation(&eq_evals);

        // Verify the claim
        if m_eval_claimed != m_eval_expected {
            return Err(Stage2Error::EvaluationMismatch {
                expected: format!("{m_eval_expected:?}"),
                actual: format!("{m_eval_claimed:?}"),
            });
        }

        // Append to transcript to maintain Fiat-Shamir soundness
        transcript.append_scalar(&m_eval_claimed);

        // Store the opening in the accumulator for Stage 3
        // Note: We reverse r_s and r_x because OpeningPoint expects big-endian ordering
        // while our polynomials use little-endian variable ordering internally.
        // The matrix has variables ordered as [x_vars, s_vars] in little-endian.
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

        // The virtual claims are laid out as:
        // [constraint_0_poly_0, constraint_1_poly_0, ..., constraint_0_poly_1, ...]
        // We need to match this with the eq evaluations

        for constraint_idx in 0..self.params.num_constraints {
            for poly_idx in 0..NUM_POLY_TYPES {
                let claim_idx = virtual_claim_index(constraint_idx, poly_idx);
                let s_idx = matrix_s_index(poly_idx, constraint_idx, self.params.num_constraints_padded);

                if claim_idx < self.virtual_claims.len() && s_idx < eq_evals.len() {
                    result += eq_evals[s_idx] * self.virtual_claims[claim_idx];
                }
            }
        }

        result
    }
}

/// Extract virtual claims from any accumulator (Prover or Verifier) in the correct order
///
/// This function extracts the virtual polynomial claims from Stage 1 accumulators
/// and organizes them in the standard layout expected by Stage 2:
/// [constraint_0_poly_0, constraint_0_poly_1, ..., constraint_0_poly_12,
///  constraint_1_poly_0, constraint_1_poly_1, ..., constraint_1_poly_12, ...]
///
/// For PackedGtExp constraints, base and bit evaluations are computed directly from
/// public inputs rather than being extracted from the accumulator.
///
/// # Type Parameters
/// - `F`: The field type
/// - `A`: The accumulator type (ProverOpeningAccumulator or VerifierOpeningAccumulator)
///
/// # Arguments
/// - `accumulator`: The Stage 1 opening accumulator
/// - `constraint_types`: The types of constraints in order
/// - `packed_gt_exp_public_inputs`: Public inputs for each packed GT exp (base, scalar_bits)
///
/// # Returns
/// A vector of virtual claims organized by constraint then polynomial type
pub fn extract_virtual_claims_from_accumulator<F: JoltField, A: OpeningAccumulator<F>>(
    accumulator: &A,
    constraint_types: &[ConstraintType],
    _packed_gt_exp_public_inputs: &[PackedGtExpPublicInputs],
) -> Vec<F> {
    let mut claims = Vec::new();

    // Track separate indices for each constraint type
    let mut packed_gt_exp_idx = 0;
    let mut gt_mul_idx = 0;
    let mut g1_scalar_mul_idx = 0;

    // Process each constraint
    for (idx, constraint_type) in constraint_types.iter().enumerate() {
        // For each constraint, we need to extract claims for all 13 polynomial types
        // in the correct order matching the PolyType enum (0-12)
        // Note: Base and Bit are NOT in this array - they're computed from public inputs
        // and are not stored in the matrix

        let mut constraint_claims = vec![F::zero(); NUM_POLY_TYPES];

        match constraint_type {
            ConstraintType::PackedGtExp => {
                // Packed GT Exp uses polynomials 0-1 (RhoPrev, Quotient)
                // Base, Bit, and RhoNext are public inputs or virtual polynomials, not in the matrix
                tracing::debug!("[extract_constraint_claims] Getting PackedGtExp({}) openings for constraint {}", packed_gt_exp_idx, idx);

                // Get committed polynomial claims
                let (_, rho_prev) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::PackedGtExpRho(packed_gt_exp_idx),
                    SumcheckId::PackedGtExpClaimReduction,
                );
                let (_, quotient) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::PackedGtExpQuotient(packed_gt_exp_idx),
                    SumcheckId::PackedGtExpClaimReduction,
                );

                // New PolyType values after removing RhoCurr: RhoPrev=0, Quotient=1
                constraint_claims[0] = rho_prev;
                constraint_claims[1] = quotient;
                // Note: rho_next is verified separately via shift sumcheck, not included here

                packed_gt_exp_idx += 1;
            }
            ConstraintType::GtMul { .. } => {
                // GT Mul uses polynomials 2-5 (MulLhs, MulRhs, MulResult, MulQuotient)
                let (_, lhs) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionMulLhs(gt_mul_idx),
                    SumcheckId::GtMul,
                );
                let (_, rhs) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionMulRhs(gt_mul_idx),
                    SumcheckId::GtMul,
                );
                let (_, result) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionMulResult(gt_mul_idx),
                    SumcheckId::GtMul,
                );
                let (_, quotient) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionMulQuotient(gt_mul_idx),
                    SumcheckId::GtMul,
                );

                // New PolyType values after removing RhoCurr: MulLhs=2, MulRhs=3, MulResult=4, MulQuotient=5
                constraint_claims[2] = lhs;
                constraint_claims[3] = rhs;
                constraint_claims[4] = result;
                constraint_claims[5] = quotient;

                gt_mul_idx += 1;
            }
            ConstraintType::G1ScalarMul { .. } => {
                // G1 Scalar Mul uses polynomials 6-12
                let (_, x_a) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionG1ScalarMulXA(g1_scalar_mul_idx),
                    SumcheckId::G1ScalarMul,
                );
                let (_, y_a) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionG1ScalarMulYA(g1_scalar_mul_idx),
                    SumcheckId::G1ScalarMul,
                );
                let (_, x_t) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionG1ScalarMulXT(g1_scalar_mul_idx),
                    SumcheckId::G1ScalarMul,
                );
                let (_, y_t) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionG1ScalarMulYT(g1_scalar_mul_idx),
                    SumcheckId::G1ScalarMul,
                );
                let (_, x_a_next) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionG1ScalarMulXANext(g1_scalar_mul_idx),
                    SumcheckId::G1ScalarMul,
                );
                let (_, y_a_next) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionG1ScalarMulYANext(g1_scalar_mul_idx),
                    SumcheckId::G1ScalarMul,
                );
                let (_, indicator) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::RecursionG1ScalarMulIndicator(g1_scalar_mul_idx),
                    SumcheckId::G1ScalarMul,
                );

                // New PolyType values after removing RhoCurr: G1ScalarMulXA=6, YA=7, XT=8, YT=9, XANext=10, YANext=11, Indicator=12
                constraint_claims[6] = x_a;
                constraint_claims[7] = y_a;
                constraint_claims[8] = x_t;
                constraint_claims[9] = y_t;
                constraint_claims[10] = x_a_next;
                constraint_claims[11] = y_a_next;
                constraint_claims[12] = indicator;

                g1_scalar_mul_idx += 1;
            }
        }

        claims.extend(constraint_claims);
    }

    claims
}
