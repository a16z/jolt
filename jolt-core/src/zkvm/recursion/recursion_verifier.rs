//! Unified verifier for the recursion SNARK protocol
//!
//! This module provides a high-level verifier that verifies:
//! - Stage 1: Constraint sumchecks (GT exp)
//! - Stage 2: Batched constraint sumchecks (shift + reduction + GT mul + G1 scalar mul)
//! - Stage 3: Virtualization direct evaluation
//! - Stage 4: Jagged transform sumcheck
//! - Stage 5: Jagged assist sumcheck
//!
//! The verifier returns an opening accumulator for PCS verification.

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        opening_proof::{OpeningAccumulator, OpeningId, SumcheckId, VerifierOpeningAccumulator},
    },
    transcripts::Transcript,
    zkvm::witness::VirtualPolynomial,
};
use ark_bn254::Fq;

use super::{
    bijection::{ConstraintMapping, VarCountJaggedBijection},
    constraints_sys::ConstraintType,
    recursion_prover::RecursionProof,
    stage1::packed_gt_exp::{PackedGtExpParams, PackedGtExpPublicInputs, PackedGtExpVerifier},
    stage2::{
        g1_scalar_mul::{G1ScalarMulParams, G1ScalarMulVerifier},
        gt_mul::{GtMulParams, GtMulVerifier},
        packed_gt_exp_reduction::{
            PackedGtExpClaimReductionParams, PackedGtExpClaimReductionVerifier,
        },
        shift_rho::{ShiftRhoParams, ShiftRhoVerifier},
    },
    stage3::virtualization::{
        extract_virtual_claims_from_accumulator, DirectEvaluationParams,
        DirectEvaluationVerifier,
    },
    stage4::{jagged::JaggedSumcheckParams, jagged::JaggedSumcheckVerifier},
    stage5::jagged_assist::JaggedAssistVerifier,
};
use crate::subprotocols::{sumcheck::BatchedSumcheck, sumcheck_verifier::SumcheckInstanceVerifier};

/// Input required by the verifier
#[derive(Clone, Debug)]
pub struct RecursionVerifierInput {
    /// Constraint types to verify
    pub constraint_types: Vec<ConstraintType>,
    /// Number of variables in the constraint system
    pub num_vars: usize,
    /// Number of constraint variables (x variables) in the matrix
    pub num_constraint_vars: usize,
    /// Number of s-variables for virtualization
    pub num_s_vars: usize,
    /// Total number of constraints
    pub num_constraints: usize,
    /// Padded number of constraints
    pub num_constraints_padded: usize,
    /// Jagged bijection for Stage 4
    pub jagged_bijection: VarCountJaggedBijection,
    /// Mapping for decoding polynomial indices to matrix rows
    pub jagged_mapping: ConstraintMapping,
    /// Precomputed matrix row indices for each polynomial index
    pub matrix_rows: Vec<usize>,
    /// Public inputs for packed GT exp (base Fq12 and scalar bits)
    pub packed_gt_exp_public_inputs: Vec<PackedGtExpPublicInputs>,
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

    /// Verify the full recursion proof and PCS opening
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify")]
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
        let (_r_stage1, _num_gt_exp) =
            tracing::info_span!("verify_recursion_stage1").in_scope(|| {
                tracing::info!("Verifying Stage 1: Constraint sumchecks");
                self.verify_stage1(
                    &proof.stage1_proof,
                    transcript,
                    &mut accumulator,
                    proof.gamma,
                    proof.delta,
                )
            })?;

        // ============ STAGE 2: Verify Batched Constraint Sumchecks ============
        let r_stage2 = tracing::info_span!("verify_recursion_stage2").in_scope(|| {
            tracing::info!("Verifying Stage 2: Batched constraint sumchecks");
            let num_packed_witnesses = self.input.packed_gt_exp_public_inputs.len();
            self.verify_stage2(
                &proof.stage2_proof,
                transcript,
                &mut accumulator,
                num_packed_witnesses,
            )
        })?;

        // ============ STAGE 3: Verify Virtualization Direct Evaluation ============
        let r_stage3_s = tracing::info_span!("verify_recursion_stage3").in_scope(|| {
            tracing::info!("Verifying Stage 3: Direct evaluation");
            self.verify_stage3(transcript, &mut accumulator, &r_stage2, proof.stage3_m_eval)
        })?;

        // ============ STAGE 4: Verify Jagged Transform Sumcheck ============
        let r_stage4 = tracing::info_span!("verify_recursion_stage4").in_scope(|| {
            tracing::info!("Verifying Stage 4: Jagged transform sumcheck");
            self.verify_stage4(
                &proof.stage4_proof,
                &proof.stage5_proof,
                transcript,
                &mut accumulator,
                &r_stage3_s,
                &r_stage2,
            )
        })?;

        // ============ STAGE 5: Verify Jagged Assist ============
        tracing::info_span!("verify_recursion_stage5").in_scope(|| {
            tracing::info!("Verifying Stage 5: Jagged assist");
            self.verify_stage5(
                &proof.stage5_proof,
                transcript,
                &mut accumulator,
                &r_stage4,
                &r_stage2,
            )
        })?;

        // ============ PCS OPENING VERIFICATION ============
        tracing::info_span!("verify_recursion_pcs_opening").in_scope(|| {
            tracing::info!("Verifying PCS opening proof");
            // Verify opening proof using PCS
            accumulator.verify_single::<T, PCS>(
                &proof.opening_proof,
                matrix_commitment.clone(),
                verifier_setup,
                transcript,
            )
        })?;

        Ok(true)
    }

    /// Verify Stage 1: Constraint sumchecks
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage1")]
    fn verify_stage1<T: Transcript>(
        &self,
        proof: &crate::subprotocols::sumcheck::SumcheckInstanceProof<F, T>,
        transcript: &mut T,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        _gamma: F,
        _delta: F,
    ) -> Result<(Vec<<F as crate::field::JoltField>::Challenge>, usize), Box<dyn std::error::Error>>
    {
        use std::any::TypeId;

        // Runtime check that F = Fq for recursion SNARK
        if TypeId::of::<F>() != TypeId::of::<Fq>() {
            panic!("Recursion SNARK requires F = Fq");
        }
        // Create verifiers for each constraint type
        let mut verifiers: Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> = Vec::new();

        // Count constraints by type
        let mut num_gt_exp = 0;
        for constraint in &self.input.constraint_types {
            if matches!(constraint, ConstraintType::PackedGtExp) {
                num_gt_exp += 1;
            }
        }

        // Add packed GT exp verifier if we have packed GT exp constraints
        // Each PackedGtExp constraint = 1 witness (covers all 254 steps)
        if num_gt_exp > 0 {
            let params = PackedGtExpParams::new();
            let verifier = PackedGtExpVerifier::new(
                params,
                self.input.packed_gt_exp_public_inputs.clone(),
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

        let r_stage1 = BatchedSumcheck::verify(proof, verifier_refs, accumulator, transcript)?;

        Ok((r_stage1, num_gt_exp))
    }

    /// Verify Stage 2: PackedGtExp claim reduction + shift sumcheck
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage2")]
    fn verify_stage2<T: Transcript>(
        &self,
        proof: &crate::subprotocols::sumcheck::SumcheckInstanceProof<F, T>,
        transcript: &mut T,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        num_packed_witnesses: usize,
    ) -> Result<Vec<<F as crate::field::JoltField>::Challenge>, Box<dyn std::error::Error>> {
        tracing::info!(
            "[Stage 2] Verifying PackedGtExp claim reduction for {} GT exp witnesses",
            num_packed_witnesses
        );

        let mut claim_indices: Vec<usize> = accumulator
            .openings
            .keys()
            .filter_map(|opening_id| match opening_id {
                OpeningId::Virtual(
                    VirtualPolynomial::PackedGtExpRhoNext(idx),
                    SumcheckId::PackedGtExp,
                ) => Some(*idx),
                _ => None,
            })
            .collect();
        claim_indices.sort_unstable();
        debug_assert_eq!(claim_indices.len(), num_packed_witnesses);

        // Create shift verifier
        let shift_verifier = ShiftRhoVerifier::new(
            ShiftRhoParams::new(claim_indices.len()),
            claim_indices.clone(),
            transcript,
        );

        let reduction_verifier = PackedGtExpClaimReductionVerifier::new(
            PackedGtExpClaimReductionParams::new(claim_indices.len() * 2),
            claim_indices,
            transcript,
        );

        // Collect constraint information for GT mul / G1 scalar mul
        let mut num_gt_mul = 0;
        let mut num_g1_scalar_mul = 0;
        let mut gt_mul_indices = Vec::new();
        let mut g1_scalar_mul_base_points = Vec::new();
        let mut g1_scalar_mul_indices = Vec::new();

        for (global_idx, constraint) in self.input.constraint_types.iter().enumerate() {
            match constraint {
                ConstraintType::GtMul => {
                    gt_mul_indices.push(global_idx);
                    num_gt_mul += 1;
                }
                ConstraintType::G1ScalarMul { base_point } => {
                    g1_scalar_mul_base_points.push(*base_point);
                    g1_scalar_mul_indices.push(global_idx);
                    num_g1_scalar_mul += 1;
                }
                _ => {}
            }
        }

        let mut verifier_refs: Vec<&dyn SumcheckInstanceVerifier<F, T>> =
            vec![&shift_verifier, &reduction_verifier];

        let gt_mul_verifier = if num_gt_mul > 0 {
            Some(GtMulVerifier::new(
                GtMulParams::new(num_gt_mul),
                gt_mul_indices,
                transcript,
            ))
        } else {
            None
        };
        if let Some(ref verifier) = gt_mul_verifier {
            verifier_refs.push(verifier);
        }

        let g1_scalar_mul_verifier = if num_g1_scalar_mul > 0 {
            Some(G1ScalarMulVerifier::new(
                G1ScalarMulParams::new(num_g1_scalar_mul),
                g1_scalar_mul_base_points,
                g1_scalar_mul_indices,
                transcript,
            ))
        } else {
            None
        };
        if let Some(ref verifier) = g1_scalar_mul_verifier {
            verifier_refs.push(verifier);
        }

        // Run batched sumcheck verification with shared challenges
        let r_stage2 = BatchedSumcheck::verify(proof, verifier_refs, accumulator, transcript)?;

        Ok(r_stage2)
    }

    /// Verify Stage 3: Direct evaluation protocol
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage3")]
    fn verify_stage3<T: Transcript>(
        &self,
        transcript: &mut T,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        r_stage2: &[<F as crate::field::JoltField>::Challenge],
        stage3_m_eval: F,
    ) -> Result<Vec<<F as crate::field::JoltField>::Challenge>, Box<dyn std::error::Error>> {
        use std::any::TypeId;

        // Runtime check that F = Fq for recursion SNARK
        if TypeId::of::<F>() != TypeId::of::<Fq>() {
            panic!("Recursion SNARK requires F = Fq");
        }

        // Since we know F = Fq, we can work directly with Fq types
        let accumulator_fq: &mut VerifierOpeningAccumulator<Fq> =
            unsafe { std::mem::transmute(accumulator) };

        // Convert r_stage2 challenges to Fq field elements
        // SAFETY: We verified F = Fq above, so F::Challenge = Fq::Challenge
        let r_x: Vec<Fq> = unsafe {
            let r_stage2_fq: &[<Fq as JoltField>::Challenge] = std::mem::transmute(r_stage2);
            r_stage2_fq.iter().map(|c| (*c).into()).collect()
        };

        // Extract virtual claims from Stage 1
        let virtual_claims = extract_virtual_claims_from_accumulator(
            accumulator_fq,
            &self.input.constraint_types,
            &self.input.packed_gt_exp_public_inputs,
        );

        // Create parameters
        let params = DirectEvaluationParams::new(
            self.input.num_s_vars,
            self.input.num_constraints,
            self.input.num_constraints_padded,
            self.input.num_constraint_vars,
        );

        // Create and run verifier
        let verifier = DirectEvaluationVerifier::new(params, virtual_claims, r_x);

        // Convert stage3_m_eval from F to Fq
        // SAFETY: We verified F = Fq above
        let m_eval_fq: Fq = unsafe { std::mem::transmute_copy(&stage3_m_eval) };

        let r_s: Vec<Fq> = (0..self.input.num_s_vars)
            .map(|_| transcript.challenge_scalar::<Fq>())
            .collect();

        verifier
            .verify(transcript, accumulator_fq, m_eval_fq, r_s.clone())
            .map_err(|e| Box::<dyn std::error::Error>::from(e))?;

        let r_stage3_s: Vec<<F as JoltField>::Challenge> = unsafe {
            let r_s_challenges: Vec<<Fq as JoltField>::Challenge> =
                r_s.into_iter().rev().map(|f| f.into()).collect();
            std::mem::transmute(r_s_challenges)
        };

        Ok(r_stage3_s)
    }

    /// Verify Stage 4: Jagged transform sumcheck
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage4")]
    fn verify_stage4<T: Transcript>(
        &self,
        stage4_proof: &crate::subprotocols::sumcheck::SumcheckInstanceProof<F, T>,
        stage5_proof: &super::stage5::jagged_assist::JaggedAssistProof<F, T>,
        transcript: &mut T,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        r_stage3_s: &[<F as crate::field::JoltField>::Challenge],
        r_stage2_x: &[<F as crate::field::JoltField>::Challenge],
    ) -> Result<Vec<<F as crate::field::JoltField>::Challenge>, Box<dyn std::error::Error>> {
        use std::any::TypeId;

        // Runtime check that F = Fq for recursion SNARK
        if TypeId::of::<F>() != TypeId::of::<Fq>() {
            panic!("Recursion SNARK requires F = Fq");
        }

        let _get_claim_span = tracing::info_span!("stage4_get_sparse_claim").entered();
        // Get the Stage 2 opening claim (sparse matrix claim)
        let (_, sparse_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::DorySparseConstraintMatrix,
            SumcheckId::RecursionVirtualization,
        );
        drop(_get_claim_span);

        let _convert_challenges_span = tracing::info_span!("stage4_convert_challenges").entered();
        // Convert challenges to field elements
        let r_s_final: Vec<F> = r_stage3_s
            .iter()
            .take(self.input.num_s_vars)
            .map(|c| (*c).into())
            .collect();
        let r_x_prev: Vec<F> = r_stage2_x.iter().map(|c| (*c).into()).collect();
        drop(_convert_challenges_span);

        let _dense_size_span = tracing::info_span!("stage4_compute_dense_size").entered();
        // Calculate number of dense variables based on the true dense size
        let dense_size = <VarCountJaggedBijection as crate::zkvm::recursion::bijection::JaggedTransform<Fq>>::dense_size(&self.input.jagged_bijection);
        let num_dense_vars = dense_size.next_power_of_two().trailing_zeros() as usize;
        drop(_dense_size_span);

        let _create_params_span = tracing::info_span!("stage4_create_params").entered();
        // Create jagged sumcheck parameters
        let params = JaggedSumcheckParams::new(
            self.input.num_s_vars,
            self.input.num_constraint_vars,
            num_dense_vars,
        );
        drop(_create_params_span);

        // Convert per-polynomial claimed evaluations to per-row claimed evaluations
        // stage5_proof.claimed_evaluations[k] = v_k = ĝ(r_x, r_dense, t_{k-1}, t_k)
        // We need: claimed_evaluations[y] = Σ_{k: matrix_row[k]==y} v_k
        let _poly_to_row_span = tracing::info_span!(
            "stage4_poly_to_row_conversion",
            num_polys = stage5_proof.claimed_evaluations.len()
        )
        .entered();
        let num_rows = 1usize << self.input.num_s_vars;
        let mut claimed_evaluations = vec![F::zero(); num_rows];

        for (poly_idx, claimed_eval) in stage5_proof.claimed_evaluations.iter().enumerate() {
            let matrix_row = self.input.matrix_rows[poly_idx];
            if matrix_row < num_rows {
                claimed_evaluations[matrix_row] += *claimed_eval;
            }
        }
        drop(_poly_to_row_span);

        let _create_verifier_span = tracing::info_span!(
            "stage4_create_verifier",
            num_polys = self.input.jagged_bijection.num_polynomials(),
            num_matrix_rows = self.input.matrix_rows.len()
        )
        .entered();
        // Create jagged sumcheck verifier with claimed evaluations for cheap f̂_jagged
        let verifier = JaggedSumcheckVerifier::new(
            (r_s_final.clone(), r_x_prev.clone()),
            sparse_claim,
            self.input.jagged_bijection.clone(),
            self.input.jagged_mapping.clone(),
            self.input.matrix_rows.clone(),
            params,
            claimed_evaluations,
        );
        drop(_create_verifier_span);

        let _batched_sumcheck_span =
            tracing::info_span!("stage4_batched_sumcheck_verify").entered();
        let r_stage4 =
            BatchedSumcheck::verify(stage4_proof, vec![&verifier], accumulator, transcript)?;
        drop(_batched_sumcheck_span);

        Ok(r_stage4)
    }

    /// Verify Stage 5: Jagged Assist (Batch MLE Verification)
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage5")]
    fn verify_stage5<T: Transcript>(
        &self,
        stage5_proof: &super::stage5::jagged_assist::JaggedAssistProof<F, T>,
        transcript: &mut T,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        r_stage4: &[<F as crate::field::JoltField>::Challenge],
        r_stage2_x: &[<F as crate::field::JoltField>::Challenge],
    ) -> Result<Vec<<F as crate::field::JoltField>::Challenge>, Box<dyn std::error::Error>> {
        // Convert r_stage4 (dense challenges) to F
        let r_dense: Vec<F> = r_stage4.iter().map(|c| (*c).into()).collect();
        let r_x_prev: Vec<F> = r_stage2_x.iter().map(|c| (*c).into()).collect();

        let dense_size = <VarCountJaggedBijection as crate::zkvm::recursion::bijection::JaggedTransform<Fq>>::dense_size(&self.input.jagged_bijection);
        let num_dense_vars = dense_size.next_power_of_two().trailing_zeros() as usize;
        // Compute num_bits for branching program
        let num_bits = std::cmp::max(self.input.num_constraint_vars, num_dense_vars);

        // Create Jagged Assist verifier - iterates over K polynomials (not rows!)
        let assist_verifier = JaggedAssistVerifier::<F, T>::new(
            stage5_proof.claimed_evaluations.clone(),
            r_x_prev,
            r_dense,
            &self.input.jagged_bijection,
            num_bits,
            transcript,
        );

        // Verify Jagged Assist sumcheck
        let _r_assist = BatchedSumcheck::verify(
            &stage5_proof.sumcheck_proof,
            vec![&assist_verifier],
            accumulator,
            transcript,
        )?;
        Ok(_r_assist)
    }
}
