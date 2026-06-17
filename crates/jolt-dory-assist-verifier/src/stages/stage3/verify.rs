use jolt_claims::protocols::dory_assist::DoryAssistOpeningId;
use jolt_dory::DoryScheme;
use jolt_field::Fq;
use jolt_hyrax::HyraxDimensions;
use jolt_openings::CommitmentScheme;
use jolt_poly::EqPolynomial;
use jolt_transcript::{Label, LabelWithCount, Transcript, U64Word};

use super::{inputs::Stage3Inputs, outputs::Stage3Output};
use crate::{
    derive_hyrax_verifier_setup, proof::DoryAssistOpeningClaim, verifier::squeeze_fq_challenge,
    DoryAssistHyrax, DoryAssistStage, DoryAssistVerifierError,
};

pub fn verify<T>(
    inputs: Stage3Inputs<'_, '_>,
    transcript: &mut T,
) -> Result<Stage3Output, DoryAssistVerifierError>
where
    T: Transcript<Challenge = <DoryScheme as CommitmentScheme>::Field>,
{
    if inputs.opening_proof.combined_row.is_empty() {
        return Err(DoryAssistVerifierError::InvalidProofShape {
            component: "stage3.opening_proof.combined_row",
            reason: "combined_row must be nonempty".to_string(),
        });
    }
    if inputs.claims.opening.packed_point.is_empty() {
        return Err(DoryAssistVerifierError::InvalidProofShape {
            component: "stage3.claims.opening.packed_point",
            reason: "packed_point must be nonempty".to_string(),
        });
    }
    if inputs.dense_commitment.rows.is_empty() {
        return Err(DoryAssistVerifierError::InvalidProofShape {
            component: "stage3.dense_commitment.rows",
            reason: "dense commitment must contain at least one row".to_string(),
        });
    }
    if inputs.proof.reduced_openings.is_empty() {
        return Err(DoryAssistVerifierError::InvalidProofShape {
            component: "stage3.reduced_openings",
            reason: "reduced_openings must be nonempty".to_string(),
        });
    }
    if inputs.proof.packed_eval != inputs.claims.opening.packed_eval {
        return Err(DoryAssistVerifierError::StageOutputMismatch {
            stage: DoryAssistStage::Stage3,
            reason: "stage packed_eval must match opening claim packed_eval".to_string(),
        });
    }

    let expected_openings = canonical_reduced_openings(inputs);
    if inputs.proof.reduced_openings != expected_openings {
        return Err(DoryAssistVerifierError::StageClaimMismatch {
            stage: DoryAssistStage::Stage3,
            reason: format!(
                "stage 3 reduced openings must match the canonical verified Stage 1 opening order: expected {expected_openings:?}, got {:?}",
                inputs.proof.reduced_openings
            ),
        });
    }

    let dimensions = infer_hyrax_dimensions(
        inputs.dense_commitment.rows.len(),
        inputs.opening_proof.combined_row.len(),
        inputs.claims.opening.packed_point.len(),
    )?;
    let reduced_claims = resolve_reduced_claims(inputs, &expected_openings)?;
    let expected_packed_eval =
        evaluate_packed_claim(&inputs.claims.opening.packed_point, &reduced_claims)?;
    if inputs.proof.packed_eval != expected_packed_eval {
        return Err(DoryAssistVerifierError::StageOutputMismatch {
            stage: DoryAssistStage::Stage3,
            reason: "packed eval must equal the prefix-weighted fold of reduced opening claims"
                .to_string(),
        });
    }

    absorb_stage3_inputs(&inputs, transcript);
    let challenge = squeeze_fq_challenge(transcript, b"dory_stage3_challenge");

    let hyrax_setup = derive_hyrax_verifier_setup(dimensions)?;
    DoryAssistHyrax::verify_opening_proof(
        &hyrax_setup,
        inputs.dense_commitment,
        &inputs.claims.opening.packed_point,
        inputs.claims.opening.packed_eval,
        inputs.opening_proof,
    )?;

    Ok(Stage3Output {
        packed_eval: inputs.proof.packed_eval,
        reduced_claims,
        expected_packed_eval,
        challenge,
    })
}

fn canonical_reduced_openings(inputs: Stage3Inputs<'_, '_>) -> Vec<DoryAssistOpeningId> {
    let mut openings = Vec::new();
    for opening_claim in inputs
        .stage1
        .relation_outputs
        .iter()
        .flat_map(|relation| &relation.opening_claims)
    {
        if !openings.contains(&opening_claim.id) {
            openings.push(opening_claim.id);
        }
    }
    openings
}

fn resolve_reduced_claims(
    inputs: Stage3Inputs<'_, '_>,
    openings: &[DoryAssistOpeningId],
) -> Result<Vec<DoryAssistOpeningClaim>, DoryAssistVerifierError> {
    openings
        .iter()
        .map(|opening| {
            let value = inputs
                .stage1
                .relation_outputs
                .iter()
                .flat_map(|relation| &relation.opening_claims)
                .find(|claim| claim.id == *opening)
                .map(|claim| claim.value)
                .ok_or(DoryAssistVerifierError::MissingOpeningClaim { id: *opening })?;
            Ok(DoryAssistOpeningClaim {
                id: *opening,
                value,
            })
        })
        .collect()
}

fn evaluate_packed_claim(
    packed_point: &[Fq],
    reduced_claims: &[DoryAssistOpeningClaim],
) -> Result<Fq, DoryAssistVerifierError> {
    let weights = EqPolynomial::new(packed_point.to_vec()).evaluations();
    if reduced_claims.len() > weights.len() {
        return Err(DoryAssistVerifierError::InvalidProofShape {
            component: "stage3.claims.opening.packed_point",
            reason: format!(
                "packed point has {} variables, which only supports {} reduced claims; proof needs {}",
                packed_point.len(),
                weights.len(),
                reduced_claims.len()
            ),
        });
    }

    Ok(reduced_claims
        .iter()
        .zip(weights)
        .fold(Fq::default(), |acc, (claim, weight)| {
            acc + claim.value * weight
        }))
}

fn absorb_stage3_inputs<T>(inputs: &Stage3Inputs<'_, '_>, transcript: &mut T)
where
    T: Transcript<Challenge = <DoryScheme as CommitmentScheme>::Field>,
{
    transcript.append(&Label(b"dory_assist_stage3"));
    transcript.append(&Label(inputs.checked.mode_name().as_bytes()));
    transcript.append(&Label(b"stage1_relations"));
    transcript.append(&U64Word(inputs.stage1.relation_count as u64));
    transcript.append(&Label(b"stage2_relations"));
    transcript.append(&U64Word(inputs.stage2.relation_count as u64));
    transcript.append(&Label(b"stage3_reduced_openings"));
    transcript.append(&U64Word(inputs.proof.reduced_openings.len() as u64));
    transcript.append(&Label(b"stage3_packed_eval"));
    transcript.append(&inputs.proof.packed_eval);
    transcript.append(&LabelWithCount(
        b"stage3_claim_point",
        inputs.claims.opening.packed_point.len() as u64,
    ));
    for point_coordinate in &inputs.claims.opening.packed_point {
        transcript.append(point_coordinate);
    }
    transcript.append(&Label(b"stage3_claim_eval"));
    transcript.append(&inputs.claims.opening.packed_eval);
    transcript.append(&Label(b"stage3_opening_proof"));
    transcript.append(inputs.opening_proof);
    transcript.append(&Label(b"stage3_dense_commitment"));
    transcript.append(inputs.dense_commitment);
    transcript.append(&Label(b"stage3_public_output"));
    transcript.append(&inputs.public_outputs.pre_final_exponentiation);
}

fn infer_hyrax_dimensions(
    row_count: usize,
    row_len: usize,
    point_len: usize,
) -> Result<HyraxDimensions, DoryAssistVerifierError> {
    let row_vars = checked_log2("stage3.dense_commitment.rows", row_count)?;
    let col_vars = checked_log2("stage3.opening_proof.combined_row", row_len)?;
    HyraxDimensions::new(point_len, row_vars, col_vars).map_err(|error| {
        DoryAssistVerifierError::InvalidProofShape {
            component: "stage3.hyrax_dimensions",
            reason: error.to_string(),
        }
    })
}

fn checked_log2(component: &'static str, value: usize) -> Result<usize, DoryAssistVerifierError> {
    if !value.is_power_of_two() {
        return Err(DoryAssistVerifierError::InvalidProofShape {
            component,
            reason: format!("{value} is not a power of two"),
        });
    }
    Ok(value.trailing_zeros() as usize)
}
