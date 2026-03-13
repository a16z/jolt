//! Top-level Jolt proof verification.
//!
//! [`verify`] orchestrates the full verification pipeline:
//!
//! 1. **S1 (Spartan)**: Verify R1CS satisfiability via [`UniformSpartanVerifier`]
//! 2. **S2–S7**: For each stage descriptor, verify the batched sumcheck proof,
//!    check claimed polynomial evaluations via expression evaluation, and
//!    accumulate opening claims
//! 3. **S8 (Openings)**: Reduce all opening claims via RLC and verify PCS
//!    opening proofs

use jolt_field::Field;
use jolt_openings::{
    AdditivelyHomomorphic, OpeningReduction, OpeningsError, RlcReduction, VerifierClaim,
};
use jolt_poly::EqPolynomial;
use jolt_spartan::{SpartanError, UniformSpartanKey, UniformSpartanProof, UniformSpartanVerifier};
use jolt_sumcheck::{BatchedSumcheckVerifier, SumcheckClaim};
use jolt_transcript::{AppendToTranscript, Transcript};

use crate::error::JoltError;
use crate::key::JoltVerifyingKey;
use crate::proof::{JoltProof, SumcheckStageProof};
use crate::stage::StageDescriptor;

/// Verifies the uniform Spartan R1CS proof (PIOP only — no PCS).
///
/// The returned `(r_x, r_y)` are the outer and inner sumcheck challenge
/// points, needed by downstream stages to construct eq-weighted sumcheck
/// claims.
///
/// The caller must append the witness commitment to the transcript before
/// calling this, and verify the witness opening proof afterward.
#[allow(clippy::type_complexity)]
#[tracing::instrument(skip_all, name = "verify_spartan")]
pub fn verify_spartan<F, T>(
    key: &UniformSpartanKey<F>,
    proof: &UniformSpartanProof<F>,
    transcript: &mut T,
) -> Result<(Vec<F>, Vec<F>), SpartanError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    UniformSpartanVerifier::verify_with_challenges(key, proof, transcript)
}

/// Verifies batch PCS opening proofs (stage 8).
///
/// Reduces all accumulated opening claims via random linear combination,
/// then verifies each reduced claim against the corresponding PCS proof.
#[tracing::instrument(skip_all, name = "verify_openings")]
pub fn verify_openings<PCS, T>(
    claims: Vec<VerifierClaim<PCS::Field, PCS::Output>>,
    opening_proofs: &[PCS::Proof],
    verifier_setup: &PCS::VerifierSetup,
    transcript: &mut T,
) -> Result<(), JoltError>
where
    PCS: AdditivelyHomomorphic,
    T: Transcript<Challenge = PCS::Field>,
{
    let reduced = <RlcReduction as OpeningReduction<PCS>>::reduce_verifier(claims, &(), transcript)
        .map_err(JoltError::Opening)?;

    if reduced.len() != opening_proofs.len() {
        return Err(JoltError::Opening(OpeningsError::VerificationFailed));
    }

    for (claim, proof) in reduced.iter().zip(opening_proofs.iter()) {
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

/// Verifies one sumcheck stage using its descriptor.
///
/// Returns the opening claims produced by this stage.
fn verify_stage<F, C, T>(
    stage_index: usize,
    desc: &StageDescriptor<F>,
    stage_proof: &SumcheckStageProof<F>,
    commitments: &[C],
    transcript: &mut T,
) -> Result<Vec<VerifierClaim<F, C>>, JoltError>
where
    F: Field,
    C: Clone,
    T: Transcript<Challenge = F>,
{
    let _span = tracing::info_span!("verify_stage", stage = stage_index).entered();

    let claims = [SumcheckClaim {
        num_vars: desc.num_vars,
        degree: desc.degree,
        claimed_sum: desc.claimed_sum,
    }];

    let (final_eval, challenges) =
        BatchedSumcheckVerifier::verify(&claims, &stage_proof.sumcheck_proof, transcript).map_err(
            |e| JoltError::StageVerification {
                stage: stage_index,
                reason: e.to_string(),
            },
        )?;

    let eval_point: Vec<F> = if desc.reverse_challenges {
        challenges.iter().rev().copied().collect()
    } else {
        challenges.clone()
    };

    if stage_proof.evaluations.len() != desc.commitment_indices.len() {
        return Err(JoltError::InvalidProof(format!(
            "stage {stage_index}: expected {} evaluations, got {}",
            desc.commitment_indices.len(),
            stage_proof.evaluations.len(),
        )));
    }

    // Check: eq(eq_point, eval_point) × g(evaluations, challenges) == final_eval
    let eq_eval = EqPolynomial::new(desc.eq_point.clone()).evaluate(&eval_point);

    let g_eval: F = desc
        .output_expr
        .evaluate(&stage_proof.evaluations, &desc.output_challenges);

    let expected = eq_eval * g_eval;
    if expected != final_eval {
        return Err(JoltError::EvaluationMismatch {
            stage: stage_index,
            reason: format!("eq * g = {expected:?}, final_eval = {final_eval:?}"),
        });
    }

    desc.commitment_indices
        .iter()
        .zip(stage_proof.evaluations.iter())
        .map(|(&idx, &eval)| {
            let commitment = commitments.get(idx).ok_or_else(|| {
                JoltError::InvalidProof(format!(
                    "stage {stage_index}: commitment index {idx} out of bounds ({})",
                    commitments.len(),
                ))
            })?;
            Ok(VerifierClaim {
                commitment: commitment.clone(),
                point: eval_point.clone(),
                eval,
            })
        })
        .collect()
}

/// Verifies a complete Jolt proof.
///
/// Orchestrates the full verification pipeline:
///
/// 1. Append witness commitment to Fiat-Shamir transcript
/// 2. **S1**: Verify uniform Spartan R1CS proof, extract `(r_x, r_y)`
/// 3. Build stage descriptors via `build_descriptors(r_x, r_y)`
/// 4. **S2–S7**: For each descriptor, verify sumcheck and accumulate opening claims
/// 5. Add witness opening claim (from Spartan)
/// 6. **S8**: Reduce opening claims via RLC and verify PCS opening proofs
///
/// The `build_descriptors` closure receives the Spartan challenge vectors and
/// transcript, and returns [`StageDescriptor`]s for S2–S7. Transcript access
/// allows squeezing batching challenges (e.g., γ) to match the prover's
/// Fiat-Shamir state. Each descriptor fully encodes the stage's verification
/// logic via its output expression — no per-stage trait implementations needed.
#[allow(clippy::type_complexity)]
#[tracing::instrument(skip_all, name = "verify")]
pub fn verify<PCS, T>(
    proof: &JoltProof<PCS::Field, PCS>,
    vk: &JoltVerifyingKey<PCS::Field, PCS>,
    build_descriptors: impl FnOnce(
        &[PCS::Field],
        &[PCS::Field],
        &mut T,
    ) -> Vec<StageDescriptor<PCS::Field>>,
    transcript: &mut T,
) -> Result<(Vec<PCS::Field>, Vec<PCS::Field>), JoltError>
where
    PCS: AdditivelyHomomorphic,
    T: Transcript<Challenge = PCS::Field>,
{
    // Append witness commitment to transcript (matches prover's commit step).
    transcript.append_bytes(format!("{:?}", proof.witness_commitment).as_bytes());

    let (r_x, r_y) = verify_spartan(&vk.spartan_key, &proof.spartan_proof, transcript)?;

    let descriptors = build_descriptors(&r_x, &r_y, transcript);

    if proof.stage_proofs.len() != descriptors.len() {
        return Err(JoltError::InvalidProof(format!(
            "expected {} stage proofs, got {}",
            descriptors.len(),
            proof.stage_proofs.len(),
        )));
    }

    let mut all_opening_claims: Vec<VerifierClaim<PCS::Field, PCS::Output>> = Vec::new();

    for (i, (desc, stage_proof)) in descriptors.iter().zip(&proof.stage_proofs).enumerate() {
        let new_claims = verify_stage(i + 2, desc, stage_proof, &proof.commitments, transcript)?;

        // Fiat-Shamir: absorb opening claim evaluations before the next
        // stage derives its challenges. Must match the prover's flush.
        for claim in &new_claims {
            claim.eval.append_to_transcript(transcript);
        }

        all_opening_claims.extend(new_claims);
    }

    // Witness opening claim from Spartan — must be added last to match prover ordering.
    all_opening_claims.push(VerifierClaim {
        commitment: proof.witness_commitment.clone(),
        point: r_y.clone(),
        eval: proof.spartan_proof.witness_eval,
    });

    verify_openings::<PCS, T>(
        all_opening_claims,
        &proof.opening_proofs[..],
        &vk.pcs_setup,
        transcript,
    )?;

    Ok((r_x, r_y))
}
