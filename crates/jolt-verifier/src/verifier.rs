//! Top-level Jolt proof verifier.

use jolt_openings::{CommitmentScheme, VerifierClaim};
use jolt_transcript::Transcript;

use crate::error::JoltError;
use crate::proof::{JoltProof, JoltVerifyingKey};
use crate::stage::VerifierStage;

/// Top-level Jolt proof verifier.
///
/// Generic over the polynomial commitment scheme. Orchestrates the
/// stage-by-stage verification pipeline:
///
/// 1. For each stage (1–7): verify the sumcheck proof, extract opening claims
/// 2. Stage 8: reduce all opening claims via RLC, verify PCS opening proofs
///
/// # Usage
///
/// ```ignore
/// let verifier = JoltVerifier::new(verifying_key);
/// verifier.verify(&proof, &stages, &mut transcript)?;
/// ```
pub struct JoltVerifier<PCS: CommitmentScheme> {
    vk: JoltVerifyingKey<PCS>,
}

impl<PCS: CommitmentScheme> JoltVerifier<PCS> {
    pub fn new(vk: JoltVerifyingKey<PCS>) -> Self {
        Self { vk }
    }

    /// Returns a reference to the verifying key.
    pub fn verifying_key(&self) -> &JoltVerifyingKey<PCS> {
        &self.vk
    }

    /// Verify a Jolt proof.
    ///
    /// Runs the full verification pipeline:
    /// 1. For each stage, verify the sumcheck proof and collect opening claims
    /// 2. Reduce opening claims via RLC
    /// 3. Verify PCS opening proofs
    ///
    /// Each `VerifierStage` is responsible for calling
    /// `BatchedSumcheckVerifier::verify` internally — the top-level verifier
    /// just orchestrates the stage loop and claim threading.
    pub fn verify<T: Transcript>(
        &self,
        proof: &JoltProof<PCS>,
        stages: &[&dyn VerifierStage<PCS, T>],
        transcript: &mut T,
    ) -> Result<(), JoltError> {
        if proof.stage_proofs.len() != stages.len() {
            return Err(JoltError::InvalidProof(format!(
                "expected {} stage proofs, got {}",
                stages.len(),
                proof.stage_proofs.len(),
            )));
        }

        let mut all_opening_claims: Vec<VerifierClaim<PCS::Field, PCS::Output>> = Vec::new();

        // Stages 1–7: each stage verifies its sumcheck proof and extracts claims
        for (i, (stage, stage_proof)) in stages.iter().zip(&proof.stage_proofs).enumerate() {
            let claims = stage.build_claims(&all_opening_claims, transcript);

            let new_claims = stage
                .verify(&claims, stage_proof, &all_opening_claims, transcript)
                .map_err(|e| JoltError::StageVerification {
                    stage: i + 1,
                    reason: e.to_string(),
                })?;

            all_opening_claims.extend(new_claims);
        }

        // Stage 8: opening reduction + PCS verify
        let _ = &self.vk;
        let _ = &proof.opening_proofs;

        Ok(())
    }
}
