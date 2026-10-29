//! Akita's final opening combines auxiliary groups and the trace in canonical
//! order `[UntrustedAdvice?, TrustedAdvice?,
//! FieldInc (field-inline) OR ProgramBytecode, ProgramImageInit (committed),
//! FusedInc, OneHotTrace]`. Field-inline and committed-program suffixes are exclusive.

use std::collections::BTreeMap;

use jolt_algebra::JoltField;
use jolt_claims::protocols::jolt::lattice::packing::PrefixPackedObjectPlan;
use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
use jolt_crypto::VectorCommitment;
use jolt_kernels::akita::commitment::FusedIncObject;
use jolt_openings::{
    CommitmentScheme, EvaluationClaim, GroupOpeningClaim, TaggedGroupOpeningClaim,
};
use jolt_transcript::{Channel, ProverTranscript, Sponge};
use jolt_verifier::sites::STAGE8;
use jolt_verifier::stages::stage4::outputs::Stage4ClearOutput;
use jolt_verifier::stages::stage6b::outputs::Stage6bClearOutput;
use jolt_verifier::stages::stage7::outputs::Stage7ClearOutput;
#[cfg(feature = "field-inline")]
use jolt_verifier::stages::stage8::akita::field_inc_claim;
use jolt_verifier::stages::stage8::akita::{fused_inc_claim, leaf_claims, object_leaf_claims};
use jolt_verifier::CheckedInputs;

use super::spine::mark_region;
use super::trace_group::{batch_failed, trace_group_claim};
use super::witness::{AdviceObject, DirectProgramObjects};
use crate::{JoltProverPreprocessing, ProverConfig, ProverError};
#[cfg(feature = "field-inline")]
use jolt_kernels::akita::commitment::FieldIncObject;

fn reduce_precommitted<F, C>(
    plan: &PrefixPackedObjectPlan,
    leaves: &BTreeMap<JoltCommittedPolynomial, EvaluationClaim<F>>,
    transcript: &mut C,
) -> Result<EvaluationClaim<F>, ProverError<F>>
where
    F: JoltField,
    C: Channel,
{
    let claims = object_leaf_claims(plan, leaves).map_err(ProverError::Verifier)?;
    let semantic = plan.packed_claims(&claims).map_err(batch_failed::<F>)?;
    plan.packing()
        .reduce_claims(&semantic, transcript)
        .map_err(batch_failed::<F>)
}

#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
#[tracing::instrument(skip_all)]
pub fn prove_stage8<F, PCS, VC, H>(
    checked: &CheckedInputs,
    config: &ProverConfig,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    one_hot_trace_commitment: &PCS::Output,
    one_hot_trace_hint: PCS::OpeningHint,
    fused_inc: &FusedIncObject<PCS>,
    untrusted_advice: Option<&AdviceObject<PCS>>,
    trusted_advice: Option<&AdviceObject<PCS>>,
    #[cfg(feature = "field-inline")] field_inc: &FieldIncObject<PCS>,
    program: Option<&DirectProgramObjects<PCS>>,
    stage4: &Stage4ClearOutput<F>,
    stage6b: &Stage6bClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
    transcript: &mut ProverTranscript<H>,
) -> Result<(), ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    PCS::Output: Clone,
    VC: VectorCommitment<Field = F>,
    H: Sponge,
{
    transcript.site(STAGE8);
    let leaves = leaf_claims(&checked.precommitted, stage4, stage6b, stage7)?;
    let main_group = trace_group_claim(
        checked,
        config,
        preprocessing,
        &leaves,
        one_hot_trace_commitment,
    )?;

    let untrusted_physical = untrusted_advice
        .map(|object| reduce_precommitted(&object.plan, &leaves, transcript))
        .transpose()?;
    let trusted_physical = trusted_advice
        .map(|object| reduce_precommitted(&object.plan, &leaves, transcript))
        .transpose()?;

    // Canonical public batch order: advice, (field-inline) the field increment polynomial,
    // or the direct committed-program objects, then FusedInc and OneHotTrace. The suffixes
    // are exclusive: field-inline rejects committed-program preprocessing.
    let mut auxiliary_groups = Vec::with_capacity(
        3 + usize::from(cfg!(feature = "field-inline"))
            + program.map_or(0, |objects| objects.objects().count()),
    );
    for (object, claim) in [
        (untrusted_advice, untrusted_physical.as_ref()),
        (trusted_advice, trusted_physical.as_ref()),
    ] {
        if let (Some(object), Some(claim)) = (object, claim) {
            auxiliary_groups.push((
                TaggedGroupOpeningClaim::new(
                    object.plan.group_role(),
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
    auxiliary_groups.push((
        field_inc_claim(&field_inc.commitment, stage6b).map_err(ProverError::Verifier)?,
        field_inc.hint.clone(),
    ));

    if let Some(program) = program {
        for object in program.objects() {
            let physical = reduce_precommitted(&object.plan, &leaves, transcript)?;
            auxiliary_groups.push((
                TaggedGroupOpeningClaim::new(
                    object.plan.group_role(),
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

    auxiliary_groups.push((
        fused_inc_claim(&fused_inc.commitment, stage6b).map_err(ProverError::Verifier)?,
        fused_inc.hint.clone(),
    ));

    mark_region("stage8_reductions", transcript);
    tracing::info_span!("akita_main_batched_prove").in_scope(|| {
        PCS::prove_batch(
            &preprocessing.pcs_setup,
            auxiliary_groups,
            main_group,
            one_hot_trace_hint,
            transcript,
        )
        .map_err(batch_failed::<F>)
    })
}
