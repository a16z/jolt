//! Stage 8 verifier entry point: the final PCS opening.
//!
//! One [`verify`] per build, same signature, selected by the `akita` feature:
//!
//! - **Homomorphic** (default): every final claim is embedded into one
//!   unified opening point (per-polynomial Lagrange
//!   `commitment_embedding_scale` factors), then the clear arm discharges the
//!   same-point statement through [`HomomorphicBatch`]'s RLC-combined PCS
//!   opening, while the ZK arm combines the commitments and checks
//!   `PCS::verify_zk`, handing its pieces to BlindFold. Statement assembly
//!   lives in [`super::homomorphic`].
//! - **Akita**: Wjolt's uniform one-hot members open natively at one point;
//!   auxiliary advice/program objects use the generic packed-opening
//!   reduction. Statement assembly lives in [`super::packed`].

#[cfg(not(feature = "akita"))]
use super::homomorphic::{final_opening_entries, require_commitment_layout};
use super::outputs::Stage8Output;
#[cfg(not(feature = "akita"))]
use super::outputs::Stage8ZkOutput;
#[cfg(not(feature = "akita"))]
use super::precommitted::precommitted_final_openings;
use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{stage6b::Stage6bOutput, stage7::Stage7Output},
    verifier::CheckedInputs,
    VerifierError,
};
use jolt_claims::protocols::jolt::geometry::dimensions::JoltFormulaDimensions;
#[cfg(not(feature = "akita"))]
use jolt_claims::protocols::jolt::{
    geometry::committed_openings::{final_opening_point, FinalOpeningPointInputs},
    JoltOpeningId, JoltRelationId,
};
#[cfg(not(feature = "akita"))]
use jolt_crypto::HomomorphicCommitment;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
#[cfg(not(feature = "akita"))]
use jolt_openings::{
    AdditivelyHomomorphic, BatchOpeningScheme, EvaluationClaim, HomomorphicBatch,
    VerifierOpeningClaim, ZkEvaluationClaim, ZkOpeningScheme,
};
#[cfg(not(feature = "akita"))]
use jolt_poly::Point;
use jolt_transcript::{AppendToTranscript, Transcript};

#[cfg(not(feature = "akita"))]
#[expect(
    clippy::too_many_arguments,
    reason = "Stage 8 takes the shared formula dimensions, trusted-advice commitment, and the two upstream stage outputs it batches; bundling them would add indirection."
)]
pub fn verify<F, PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    formula_dimensions: &JoltFormulaDimensions,
    trusted_advice_commitment: Option<&PCS::Output>,
    transcript: &mut T,
    stage6: &Stage6bOutput<F, VC::Output>,
    stage7: &Stage7Output<F, VC::Output>,
) -> Result<Stage8Output<F, PCS::Output, VC::Output>, VerifierError>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>
        + AdditivelyHomomorphic
        + ZkOpeningScheme<HidingCommitment = VC::Output>,
    PCS::Output: Clone + HomomorphicCommitment<F>,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    let log_t = formula_dimensions.trace.log_t();
    let layout = formula_dimensions.ra_layout;

    // Stage 7's produced opening points, and (clear mode) the stage-7 and stage-6b
    // output claims. The hamming-weight opening point and precommitted finals are
    // resolved off these — before any transcript operation — since the finals'
    // points anchor the unified opening point.
    let (stage7_points, clear) = match (stage6, stage7) {
        (Stage6bOutput::Clear(stage6), Stage7Output::Clear(stage7)) => (
            &stage7.output_points,
            Some((&stage7.output_values, &stage6.output_values)),
        ),
        (Stage6bOutput::Zk(_), Stage7Output::Zk(stage7)) => (&stage7.output_points, None),
        (Stage6bOutput::Clear(_), Stage7Output::Zk(_)) => {
            return Err(VerifierError::ExpectedClearProof { field: "stage7" });
        }
        (Stage6bOutput::Zk(_), Stage7Output::Clear(_)) => {
            return Err(VerifierError::ExpectedCommittedProof { field: "stage7" });
        }
    };
    let stage6_points = stage6.output_points();
    let inc_opening_point = stage6_points.inc_opening_point();
    // `final_opening_entries` reads the clear claims in (stage6, stage7) order.
    let clear_claims = clear.map(|(stage7_values, stage6_values)| (stage6_values, stage7_values));
    require_commitment_layout(&proof.commitments, layout)?;

    let hamming_opening_point = stage7_points
        .hamming_weight_opening_point()
        .map(<[F]>::to_vec)
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::HammingWeightClaimReduction,
            reason: "stage 7 produced no hamming-weight openings".to_string(),
        })?;
    let precommitted_finals =
        precommitted_final_openings(&checked.precommitted, stage7_points, stage6_points, clear)?;

    let anchor_points: Vec<&[F]> = precommitted_finals
        .iter()
        .map(|opening| opening.point.as_slice())
        .collect();
    let opening_point = final_opening_point(FinalOpeningPointInputs {
        log_t,
        log_k_chunk: proof.one_hot_config.committed_chunk_bits(),
        trace_order: proof.trace_polynomial_order,
        hamming_weight_opening_point: hamming_opening_point.as_slice(),
        inc_claim_reduction_opening_point: inc_opening_point,
        precommitted_anchor_points: &anchor_points,
    })
    .map_err(|error| VerifierError::FinalOpeningBatchFailed {
        reason: error.to_string(),
    })?;
    let pcs_opening_point = Point::high_to_low(opening_point.clone());

    let entries = final_opening_entries(
        preprocessing,
        proof,
        layout,
        trusted_advice_commitment,
        &opening_point,
        hamming_opening_point.as_slice(),
        inc_opening_point,
        &precommitted_finals,
        clear_claims,
    )?;

    if !checked.zk {
        let opening_claims = entries
            .iter()
            .map(|entry| {
                let opening_claim =
                    entry
                        .opening_claim
                        .ok_or_else(|| VerifierError::FinalOpeningBatchFailed {
                            reason: "missing clear opening claim in final batch".to_string(),
                        })?;
                Ok(VerifierOpeningClaim {
                    commitment: entry.commitment.clone(),
                    evaluation: EvaluationClaim::new(
                        pcs_opening_point.clone(),
                        opening_claim * entry.scale,
                    ),
                })
            })
            .collect::<Result<Vec<_>, VerifierError>>()?;

        HomomorphicBatch::<PCS>::verify_batch(
            &preprocessing.pcs_setup,
            &opening_claims,
            &proof.joint_opening_proof,
            transcript,
        )
        .map_err(|error| VerifierError::FinalOpeningVerificationFailed {
            reason: error.to_string(),
        })?;

        return Ok(Stage8Output::Clear);
    }

    let opening_ids: Vec<JoltOpeningId> = entries.iter().map(|entry| entry.id).collect();
    let gamma_powers = transcript.challenge_scalar_powers(entries.len());
    let commitments: Vec<PCS::Output> = entries
        .iter()
        .map(|entry| entry.commitment.clone())
        .collect();
    let joint_commitment = PCS::combine(&commitments, &gamma_powers);
    let constraint_coefficients = gamma_powers
        .iter()
        .zip(&entries)
        .map(|(gamma, entry)| *gamma * entry.scale)
        .collect::<Vec<_>>();

    let hiding_evaluation_commitment = PCS::verify_zk(
        &joint_commitment,
        pcs_opening_point.as_slice(),
        &proof.joint_opening_proof,
        &preprocessing.pcs_setup,
        transcript,
    )
    .map_err(|error| VerifierError::FinalOpeningVerificationFailed {
        reason: error.to_string(),
    })?;
    ZkEvaluationClaim::new(pcs_opening_point.as_slice(), &hiding_evaluation_commitment)
        .append_to_transcript(transcript);

    Ok(Stage8Output::Zk(Stage8ZkOutput {
        opening_ids,
        constraint_coefficients,
        pcs_opening_point,
        joint_commitment,
        hiding_evaluation_commitment,
    }))
}

#[cfg(feature = "akita")]
#[expect(
    clippy::too_many_arguments,
    reason = "same signature as the homomorphic build's verify"
)]
pub fn verify<F, PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    formula_dimensions: &JoltFormulaDimensions,
    trusted_advice_commitment: Option<&PCS::Output>,
    transcript: &mut T,
    stage6: &Stage6bOutput<F, VC::Output>,
    stage7: &Stage7Output<F, VC::Output>,
) -> Result<Stage8Output<F, PCS::Output, VC::Output>, VerifierError>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    PCS::Output: Clone + AppendToTranscript + super::WJoltCommitmentMetadata,
    PCS::VerifierSetup: super::WJoltSetupMetadata,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    // The reconstruction phase settles auxiliary word/chunk claims against
    // their committed one-hot decompositions.
    let reconstruction = super::reconstruction::verify(
        checked,
        proof.stages.reconstruction_sumcheck_proof.as_ref(),
        &proof.clear_claims()?.reconstruction,
        transcript,
        stage6.clear()?,
        stage7.clear()?,
    )?;

    // Wjolt then opens natively at its shared point; reconstruction leaves are
    // discharged by separate auxiliary packed openings.
    super::packed::verify(
        formula_dimensions,
        proof.one_hot_config,
        preprocessing,
        &proof.commitments,
        proof.untrusted_advice_commitment.as_ref(),
        trusted_advice_commitment,
        &proof.joint_opening_proof,
        transcript,
        stage7.clear()?,
        &reconstruction,
    )?;

    Ok(Stage8Output::Clear)
}
