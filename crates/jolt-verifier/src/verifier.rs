//! Top-level Jolt proof verification.
//!
//! [`verify`] orchestrates the full verification pipeline:
//!
//! 1. **S1 (Spartan)**: Verify R1CS satisfiability via [`SpartanVerifier`]
//! 2. **S2–S7**: For each stage, verify the batched sumcheck proof, check
//!    claimed polynomial evaluations, and accumulate opening claims
//! 3. **S8 (Openings)**: Reduce all opening claims via RLC and verify PCS
//!    opening proofs

use jolt_field::Field;
use jolt_openings::{
    AdditivelyHomomorphic, OpeningReduction, OpeningsError, RlcReduction, VerifierClaim,
};
use jolt_spartan::{SpartanError, SpartanKey, SpartanProof, SpartanVerifier};
use jolt_sumcheck::BatchedSumcheckVerifier;
use jolt_transcript::Transcript;

use crate::error::JoltError;
use crate::key::JoltVerifyingKey;
use crate::proof::{BatchOpeningProofs, JoltProof, SumcheckStageProof};
use crate::stage::VerifierStage;

/// Verifies the Spartan R1CS proof (stage 1) and returns the challenge vectors.
///
/// The returned `(r_x, r_y)` are the outer and inner sumcheck challenge points,
/// needed by downstream stages to construct eq-weighted sumcheck claims.
///
/// The transcript is left in the same state as the prover's transcript after S1,
/// so subsequent stages can sample consistent challenges.
#[allow(clippy::type_complexity)]
#[tracing::instrument(skip_all, name = "verify_spartan")]
pub fn verify_spartan<PCS, T>(
    key: &SpartanKey<PCS::Field>,
    proof: &SpartanProof<PCS::Field, PCS>,
    verifier_setup: &PCS::VerifierSetup,
    transcript: &mut T,
) -> Result<(Vec<PCS::Field>, Vec<PCS::Field>), SpartanError>
where
    PCS: jolt_openings::CommitmentScheme,
    T: Transcript<Challenge = u128>,
{
    SpartanVerifier::verify_with_challenges::<PCS, T>(key, proof, verifier_setup, transcript)
}

/// Verifies one sumcheck stage (S2–S7).
///
/// 1. Delegates to the stage's [`build_claims`](VerifierStage::build_claims) to
///    construct sumcheck claims from prior opening data
/// 2. Verifies the batched sumcheck proof
/// 3. Delegates to [`check_and_extract`](VerifierStage::check_and_extract) to
///    verify claimed evaluations and produce opening claims
///
/// Returns the new opening claims for this stage.
fn verify_sumcheck_stage<F, C, T>(
    stage_index: usize,
    stage: &mut dyn VerifierStage<F, C, T>,
    stage_proof: &SumcheckStageProof<F>,
    commitments: &[C],
    prior_claims: &[VerifierClaim<F, C>],
    transcript: &mut T,
    challenge_fn: impl Fn(T::Challenge) -> F,
) -> Result<Vec<VerifierClaim<F, C>>, JoltError>
where
    F: Field,
    C: Clone,
    T: Transcript,
{
    let claims = stage.build_claims(prior_claims, transcript);

    let (final_eval, challenges) = BatchedSumcheckVerifier::verify(
        &claims,
        &stage_proof.sumcheck_proof,
        transcript,
        challenge_fn,
    )
    .map_err(|e| JoltError::StageVerification {
        stage: stage_index,
        reason: e.to_string(),
    })?;

    stage.check_and_extract(
        final_eval,
        &challenges,
        &stage_proof.evaluations,
        commitments,
    )
}

/// Verifies batch PCS opening proofs (stage 8).
///
/// Reduces all accumulated opening claims via random linear combination,
/// then verifies each reduced claim against the corresponding PCS proof.
#[tracing::instrument(skip_all, name = "verify_openings")]
pub fn verify_openings<PCS, T>(
    claims: Vec<VerifierClaim<PCS::Field, PCS::Output>>,
    opening_proofs: &BatchOpeningProofs<PCS>,
    verifier_setup: &PCS::VerifierSetup,
    transcript: &mut T,
    challenge_fn: impl Fn(T::Challenge) -> PCS::Field,
) -> Result<(), JoltError>
where
    PCS: AdditivelyHomomorphic,
    T: Transcript,
{
    let reduced = <RlcReduction as OpeningReduction<PCS>>::reduce_verifier(
        claims,
        &(),
        transcript,
        &challenge_fn,
    )
    .map_err(JoltError::Opening)?;

    if reduced.len() != opening_proofs.proofs.len() {
        return Err(JoltError::Opening(OpeningsError::VerificationFailed));
    }

    for (claim, proof) in reduced.iter().zip(opening_proofs.proofs.iter()) {
        PCS::verify(
            &claim.commitment,
            &claim.point,
            claim.eval,
            proof,
            verifier_setup,
            transcript,
        )
        .map_err(JoltError::Opening)?;
    }

    Ok(())
}

/// Verifies a complete Jolt proof.
///
/// Orchestrates the full verification pipeline:
///
/// 1. **S1**: Verify Spartan R1CS proof, extract challenge vectors `(r_x, r_y)`
/// 2. **S2–S7**: For each stage, verify sumcheck and accumulate opening claims
/// 3. **S8**: Reduce opening claims via RLC and verify PCS opening proofs
///
/// The caller provides [`VerifierStage`] implementations for S2–S7. These are
/// config-driven — constructed using [`ClaimDefinition`](jolt_ir::ClaimDefinition)s
/// and stage metadata from the proof configuration.
///
/// # Arguments
///
/// * `proof` — complete Jolt proof
/// * `vk` — verification key (Spartan key + PCS setup)
/// * `stages` — verifier-side stage implementations for S2–S7
/// * `transcript` — Fiat-Shamir transcript, initialized with the same label as
///   the prover's transcript
/// * `challenge_fn` — converts transcript challenges to field elements
///
/// # Returns
///
/// `(r_x, r_y)` — the Spartan challenge vectors, useful for the caller to
/// inspect the Spartan output if needed. Returns `Err(JoltError)` if any
/// verification step fails.
#[allow(clippy::type_complexity)]
#[tracing::instrument(skip_all, name = "verify")]
pub fn verify<PCS, T>(
    proof: &JoltProof<PCS::Field, PCS>,
    vk: &JoltVerifyingKey<PCS::Field, PCS>,
    stages: &mut [Box<dyn VerifierStage<PCS::Field, PCS::Output, T>>],
    transcript: &mut T,
    challenge_fn: impl Fn(T::Challenge) -> PCS::Field + Copy,
) -> Result<(Vec<PCS::Field>, Vec<PCS::Field>), JoltError>
where
    PCS: AdditivelyHomomorphic,
    T: Transcript<Challenge = u128>,
{
    if proof.stage_proofs.len() != stages.len() {
        return Err(JoltError::InvalidProof(format!(
            "expected {} stage proofs, got {}",
            stages.len(),
            proof.stage_proofs.len(),
        )));
    }

    // S1: Spartan
    let (r_x, r_y) = verify_spartan::<PCS, T>(
        &vk.spartan_key,
        &proof.spartan_proof,
        &vk.pcs_setup,
        transcript,
    )?;

    // S2–S7: Sumcheck stages
    let mut all_opening_claims: Vec<VerifierClaim<PCS::Field, PCS::Output>> = Vec::new();

    for (i, (stage, stage_proof)) in stages.iter_mut().zip(&proof.stage_proofs).enumerate() {
        let new_claims = verify_sumcheck_stage(
            i + 2, // stages are numbered S2–S7
            stage.as_mut(),
            stage_proof,
            &proof.commitments,
            &all_opening_claims,
            transcript,
            challenge_fn,
        )?;

        all_opening_claims.extend(new_claims);
    }

    // S8: Batch opening proofs
    verify_openings::<PCS, T>(
        all_opening_claims,
        &proof.opening_proofs,
        &vk.pcs_setup,
        transcript,
        challenge_fn,
    )?;

    Ok((r_x, r_y))
}
