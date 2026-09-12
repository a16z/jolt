//! Akita's final opening: one heterogeneous precommitted/main-trace opening
//! over the canonical group order `[UntrustedAdvice?, TrustedAdvice?,
//! FieldIncLimbs (field-inline builds), BytecodeChunk(0..C),
//! ProgramImageInit, OneHotTrace]`.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::lattice::packing::{OneHotTraceShape, PrefixPackedObjectPlan};
use jolt_claims::protocols::jolt::lattice::strategy::ONE_HOT_TRACE_LAYOUT;
use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltRelationId};
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_openings::{CommitmentScheme, EvaluationClaim, GroupOpeningClaim, PrecommittedClaim};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::stages::stage4::outputs::Stage4ClearOutput;
use jolt_verifier::stages::stage6b::outputs::Stage6bClearOutput;
use jolt_verifier::stages::stage7::outputs::Stage7ClearOutput;
#[cfg(feature = "field-inline")]
use jolt_verifier::stages::stage8::field_inline_packed::FieldIncLimbClaims;
use jolt_verifier::stages::stage8::packed::{
    leaf_claims, object_leaf_claims, one_hot_trace_packed_claims,
};
use jolt_verifier::{CheckedInputs, VerifierError};

#[cfg(feature = "field-inline")]
use super::field_inline::FieldIncLimbsObject;
use super::witness::{AdviceObject, DirectProgramObjects};
use crate::{JoltProverPreprocessing, ProverConfig, ProverError};

fn batch_failed<F: JoltField>(reason: impl ToString) -> ProverError<F> {
    ProverError::Verifier(VerifierError::FinalOpeningBatchFailed {
        reason: reason.to_string(),
    })
}

fn reduce_precommitted<F, T>(
    plan: &PrefixPackedObjectPlan,
    leaves: &BTreeMap<JoltCommittedPolynomial, EvaluationClaim<F>>,
    transcript: &mut T,
) -> Result<EvaluationClaim<F>, ProverError<F>>
where
    F: JoltField,
    T: Transcript<Challenge = F>,
{
    let claims = object_leaf_claims(plan, leaves).map_err(ProverError::Verifier)?;
    let semantic = plan.packed_claims(&claims).map_err(batch_failed::<F>)?;
    plan.packing()
        .reduce_claims(&semantic, transcript)
        .map_err(batch_failed::<F>)
}

/// The stage-8 wire artifacts: the joint opening proof and (FR builds) the
/// limb-group claims the proof carries beside it.
pub struct Stage8Artifacts<PCS>
where
    PCS: CommitmentScheme,
{
    pub joint_opening_proof: PCS::Proof,
    #[cfg(feature = "field-inline")]
    pub field_inc_limbs: FieldIncLimbClaims<PCS::Field>,
}

#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
#[tracing::instrument(skip_all)]
pub fn prove_stage8<F, PCS, VC, T>(
    checked: &CheckedInputs,
    config: &ProverConfig,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    one_hot_trace_commitment: &PCS::Output,
    one_hot_trace_hint: PCS::OpeningHint,
    untrusted_advice: Option<&AdviceObject<PCS>>,
    trusted_advice: Option<&AdviceObject<PCS>>,
    #[cfg(feature = "field-inline")] field_inc_limbs: &FieldIncLimbsObject<PCS>,
    program: Option<&DirectProgramObjects<PCS>>,
    stage4: &Stage4ClearOutput<F>,
    stage6b: &Stage6bClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
    transcript: &mut T,
) -> Result<Stage8Artifacts<PCS>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    PCS::Output: Clone + AppendToTranscript,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    let log_t = checked.trace_length.ilog2() as usize;
    let chunk_width = config.one_hot_config.committed_chunk_bits();
    let formula_dimensions = crate::stages::formula_dimensions(
        checked,
        config,
        preprocessing.verifier.program.bytecode_len(),
        JoltRelationId::HammingWeightClaimReduction,
    )?;
    let plan = ONE_HOT_TRACE_LAYOUT
        .plan(&OneHotTraceShape {
            ra_layout: formula_dimensions.ra_layout,
            log_t,
            log_k_chunk: chunk_width,
        })
        .map_err(batch_failed::<F>)?;

    let leaves = leaf_claims(&checked.precommitted, stage4, stage6b, stage7)?;

    let packed_claims =
        one_hot_trace_packed_claims(&plan, chunk_width, &leaves).map_err(ProverError::Verifier)?;
    let packed_claim = plan
        .packing()
        .reduce_claims(&packed_claims, transcript)
        .map_err(batch_failed::<F>)?;

    let untrusted_physical = untrusted_advice
        .map(|object| reduce_precommitted(&object.plan, &leaves, transcript))
        .transpose()?;
    let trusted_physical = trusted_advice
        .map(|object| reduce_precommitted(&object.plan, &leaves, transcript))
        .transpose()?;

    // Canonical public batch order: advice, (field-inline) the FR limb group,
    // then the direct committed-program objects, then OneHotTrace.
    let mut precommitted = Vec::with_capacity(
        2 + usize::from(cfg!(feature = "field-inline")) + program.map_or(0, |p| p.objects.len()),
    );
    for (object, claim) in [
        (untrusted_advice, untrusted_physical.as_ref()),
        (trusted_advice, trusted_physical.as_ref()),
    ] {
        if let (Some(object), Some(claim)) = (object, claim) {
            precommitted.push((
                PrecommittedClaim::new(
                    object.plan.precommitted_role(),
                    GroupOpeningClaim::new(
                        object.commitment.clone(),
                        claim.point.as_slice().to_vec(),
                        vec![claim.value],
                    ),
                ),
                object.hint.clone(),
            ));
        }
    }
    #[cfg(feature = "field-inline")]
    let field_inc_limb_claims = {
        let (claim, claims) =
            super::field_inline::stage8_batch_entry(field_inc_limbs, stage6b, transcript)?;
        precommitted.push((claim, field_inc_limbs.hint.clone()));
        claims
    };

    if let Some(program) = program {
        for object in &program.objects {
            let physical = reduce_precommitted(&object.plan, &leaves, transcript)?;
            precommitted.push((
                PrecommittedClaim::new(
                    object.plan.precommitted_role(),
                    GroupOpeningClaim::new(
                        object.commitment.clone(),
                        physical.point.as_slice().to_vec(),
                        vec![physical.value],
                    ),
                ),
                object.hint.clone(),
            ));
        }
    }

    let main_group = GroupOpeningClaim::new(
        one_hot_trace_commitment.clone(),
        packed_claim.point.as_slice().to_vec(),
        vec![packed_claim.value],
    );
    let joint_opening_proof = tracing::info_span!("akita_main_batched_prove").in_scope(|| {
        PCS::prove_batch(
            &preprocessing.pcs_setup,
            precommitted,
            main_group,
            one_hot_trace_hint,
            transcript,
        )
        .map_err(batch_failed::<F>)
    })?;
    Ok(Stage8Artifacts {
        joint_opening_proof,
        #[cfg(feature = "field-inline")]
        field_inc_limbs: field_inc_limb_claims,
    })
}
