//! Unified verifier for the two-stage recursion SNARK protocol
//!
//! This module provides a high-level verifier that verifies:
//! - Stage 1: Constraint sumchecks (GT exp, GT mul, G1 scalar mul)
//! - Stage 2: Virtualization sumcheck
//!
//! The verifier returns an opening accumulator for PCS verification.

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        opening_proof::VerifierOpeningAccumulator,
    },
    transcripts::Transcript,
    zkvm::witness::CommittedPolynomial,
};
use ark_bn254::Fq;

use super::{
    constraints_sys::ConstraintType,
    recursion_prover::RecursionProof,
    stage1::{
        g1_scalar_mul::{G1ScalarMulParams, G1ScalarMulVerifier},
        gt_mul::{GtMulParams, GtMulVerifier},
        square_and_multiply::{SquareAndMultiplyParams, SquareAndMultiplyVerifier},
    },
    stage2::virtualization::{
        RecursionVirtualizationParams, RecursionVirtualizationVerifier,
    },
};
use crate::subprotocols::{
    sumcheck::BatchedSumcheck,
    sumcheck_verifier::SumcheckInstanceVerifier,
};

/// Input required by the verifier
#[derive(Clone, Debug)]
pub struct RecursionVerifierInput {
    /// Constraint types to verify
    pub constraint_types: Vec<ConstraintType>,
    /// Number of variables in the constraint system
    pub num_vars: usize,
    /// Number of s-variables for virtualization
    pub num_s_vars: usize,
    /// Total number of constraints
    pub num_constraints: usize,
    /// Padded number of constraints
    pub num_constraints_padded: usize,
}

/// Unified verifier for the recursion SNARK
pub struct RecursionVerifier<F: JoltField = Fq> {
    /// Input parameters for verification
    input: RecursionVerifierInput,
    /// Phantom data for the field type
    _phantom: std::marker::PhantomData<F>,
}

impl<F: JoltField> RecursionVerifier<F> {
    /// Create a new recursion verifier
    pub fn new(input: RecursionVerifierInput) -> Self {
        Self {
            input,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Verify the full two-stage recursion proof and PCS opening
    pub fn verify<T: Transcript, PCS: CommitmentScheme<Field = F>>(
        &self,
        proof: &RecursionProof<F, T, PCS>,
        transcript: &mut T,
        matrix_commitment: &PCS::Commitment,
        verifier_setup: &PCS::VerifierSetup,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        use std::any::TypeId;

        // Runtime check that F = Fq for recursion SNARK
        if TypeId::of::<F>() != TypeId::of::<Fq>() {
            panic!("Recursion SNARK requires F = Fq");
        }
        // Initialize opening accumulator
        let mut accumulator = VerifierOpeningAccumulator::<F>::new(self.input.num_vars);

        // Populate accumulator with opening claims from proof
        for (key, value) in &proof.opening_claims {
            accumulator.openings.insert(key.clone(), value.clone());
        }

        // ============ STAGE 1: Verify Constraint Sumchecks ============
        let r_stage1 = self.verify_stage1(
            &proof.stage1_proof,
            transcript,
            &mut accumulator,
            proof.gamma,
            proof.delta,
        )?;

        // ============ STAGE 2: Verify Virtualization Sumcheck ============
        let r_stage2 = self.verify_stage2(
            &proof.stage2_proof,
            transcript,
            &mut accumulator,
            &r_stage1,
            proof.gamma,
        )?;

        // ============ PCS OPENING VERIFICATION ============
        // Verify opening proof using PCS
        accumulator
            .verify_single::<T, PCS>(
                &proof.opening_proof,
                matrix_commitment.clone(),
                verifier_setup,
                transcript,
            )?;

        Ok(true)
    }

    /// Verify Stage 1: Constraint sumchecks
    fn verify_stage1<T: Transcript>(
        &self,
        proof: &crate::subprotocols::sumcheck::SumcheckInstanceProof<F, T>,
        transcript: &mut T,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        gamma: F,
        delta: F,
    ) -> Result<Vec<<F as crate::field::JoltField>::Challenge>, Box<dyn std::error::Error>> {
        use std::any::TypeId;

        // Runtime check that F = Fq for recursion SNARK
        if TypeId::of::<F>() != TypeId::of::<Fq>() {
            panic!("Recursion SNARK requires F = Fq");
        }
        // Create verifiers for each constraint type
        let mut verifiers: Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> = Vec::new();

        // Count constraints by type
        let mut num_gt_exp = 0;
        let mut num_gt_mul = 0;
        let mut num_g1_scalar_mul = 0;

        // Collect constraint information
        let mut gt_exp_bits = Vec::new();
        let mut gt_exp_indices = Vec::new();
        let mut gt_mul_indices = Vec::new();
        let mut g1_scalar_mul_base_points = Vec::new();
        let mut g1_scalar_mul_indices = Vec::new();

        for constraint in &self.input.constraint_types {
            match constraint {
                ConstraintType::GtExp { bit } => {
                    gt_exp_bits.push(*bit);
                    gt_exp_indices.push(num_gt_exp);
                    num_gt_exp += 1;
                }
                ConstraintType::GtMul => {
                    gt_mul_indices.push(num_gt_exp + num_gt_mul);
                    num_gt_mul += 1;
                }
                ConstraintType::G1ScalarMul { base_point } => {
                    g1_scalar_mul_base_points.push(*base_point);
                    g1_scalar_mul_indices.push(num_gt_exp + num_gt_mul + num_g1_scalar_mul);
                    num_g1_scalar_mul += 1;
                }
            }
        }

        // Add GT exp verifier if we have GT exp constraints
        if num_gt_exp > 0 {
            let params = SquareAndMultiplyParams::new(num_gt_exp);
            let verifier = SquareAndMultiplyVerifier::new(
                params,
                gt_exp_bits,
                gt_exp_indices,
                transcript,
            );
            verifiers.push(Box::new(verifier));
        }

        // Add GT mul verifier if we have GT mul constraints
        if num_gt_mul > 0 {
            let params = GtMulParams::new(num_gt_mul);
            let verifier = GtMulVerifier::new(
                params,
                gt_mul_indices,
                transcript,
            );
            verifiers.push(Box::new(verifier));
        }

        // Add G1 scalar mul verifier if we have G1 scalar mul constraints
        if num_g1_scalar_mul > 0 {
            let params = G1ScalarMulParams::new(num_g1_scalar_mul);
            let verifier = G1ScalarMulVerifier::new(
                params,
                g1_scalar_mul_base_points,
                g1_scalar_mul_indices,
                transcript,
            );
            verifiers.push(Box::new(verifier));
        }

        if verifiers.is_empty() {
            return Err("No constraints to verify in Stage 1".into());
        }

        // Run batched sumcheck verification for all verifiers
        let verifier_refs: Vec<&dyn SumcheckInstanceVerifier<F, T>> =
            verifiers.iter().map(|v| &**v).collect();

        let r_stage1 = BatchedSumcheck::verify(
            proof,
            verifier_refs,
            accumulator,
            transcript,
        )?;

        Ok(r_stage1)
    }

    /// Verify Stage 2: Virtualization sumcheck
    fn verify_stage2<T: Transcript>(
        &self,
        proof: &crate::subprotocols::sumcheck::SumcheckInstanceProof<F, T>,
        transcript: &mut T,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        r_stage1: &[<F as crate::field::JoltField>::Challenge],
        gamma: F,
    ) -> Result<Vec<<F as crate::field::JoltField>::Challenge>, Box<dyn std::error::Error>> {
        use std::any::TypeId;

        // Runtime check that F = Fq for recursion SNARK
        if TypeId::of::<F>() != TypeId::of::<Fq>() {
            panic!("Recursion SNARK requires F = Fq");
        }
        // Create virtualization parameters
        let params = RecursionVirtualizationParams::new(
            self.input.num_s_vars,
            self.input.num_constraints,
            self.input.num_constraints_padded,
            CommittedPolynomial::DoryConstraintMatrix,
        );

        // Create virtualization verifier
        let verifier = RecursionVirtualizationVerifier::new(
            params,
            self.input.constraint_types.clone(),
            transcript,
            r_stage1.to_vec(),
            gamma,
        );

        // Run virtualization sumcheck verification
        let r_stage2 = BatchedSumcheck::verify(
            proof,
            vec![&verifier],
            accumulator,
            transcript,
        )?;

        Ok(r_stage2)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_recursion_verifier_creation() {
        // TODO: Add test for verifier creation
    }
}