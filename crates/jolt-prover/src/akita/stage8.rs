//! Akita's final opening combines auxiliary groups and the trace in canonical
//! order `[UntrustedAdvice?, TrustedAdvice?,
//! FieldInc (field-inline) OR ProgramBytecode, ProgramImageInit (committed),
//! OneHotTrace]`. Field-inline and committed-program suffixes are exclusive.

use std::borrow::BorrowMut;
use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::lattice::packing::{OneHotTraceShape, PrefixPackedObjectPlan};
use jolt_claims::protocols::jolt::lattice::strategy::ONE_HOT_TRACE_LAYOUT;
use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltRelationId};
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_openings::{
    CommitmentScheme, EvaluationClaim, GroupBatchOpening, GroupOpeningClaim,
    TaggedGroupOpeningClaim,
};
use jolt_transcript::{Encoding, FieldTranscript, ProverState, ProverTranscript};
use jolt_verifier::stages::pow::PowSite;

use crate::stages::grind_pow;
use jolt_verifier::stages::stage4::outputs::Stage4ClearOutput;
use jolt_verifier::stages::stage6b::outputs::Stage6bClearOutput;
use jolt_verifier::stages::stage7::outputs::Stage7ClearOutput;
#[cfg(feature = "field-inline")]
use jolt_verifier::stages::stage8::akita::field_inc_claim;
use jolt_verifier::stages::stage8::akita::{
    leaf_claims, object_leaf_claims, one_hot_trace_claim, selector_loss,
};
use jolt_verifier::{CheckedInputs, VerifierError};

use super::witness::{AdviceObject, DirectProgramObjects};
use crate::{JoltProverPreprocessing, ProverConfig, ProverError};
#[cfg(feature = "field-inline")]
use jolt_kernels::akita::commitment::FieldIncObject;

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
    T: FieldTranscript<F> + ProverTranscript,
{
    let claims = object_leaf_claims(plan, leaves).map_err(ProverError::Verifier)?;
    let semantic = plan.packed_claims(&claims).map_err(batch_failed::<F>)?;
    plan.packing()
        .reduce_claims(&semantic, transcript)
        .map_err(batch_failed::<F>)
}

/// Returns the joint opening proof and the nonce of the selector points.
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
    #[cfg(feature = "field-inline")] field_inc: &FieldIncObject<PCS>,
    program: Option<&DirectProgramObjects<PCS>>,
    stage4: &Stage4ClearOutput<F>,
    stage6b: &Stage6bClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
    transcript: &mut T,
) -> Result<(PCS::Proof, Option<u32>), ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + GroupBatchOpening,
    PCS::Output: Clone + Encoding,
    VC: VectorCommitment<Field = F>,
    T: FieldTranscript<F> + ProverTranscript + BorrowMut<ProverState<PCS::Sponge>>,
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

    let main_group = one_hot_trace_claim(&plan, chunk_width, &leaves, one_hot_trace_commitment)
        .map_err(ProverError::Verifier)?;

    let pow_nonce = grind_pow::<F, _>(
        PowSite::Stage8,
        selector_loss(
            [untrusted_advice, trusted_advice]
                .into_iter()
                .flatten()
                .map(|object| &object.plan)
                .chain(
                    program
                        .into_iter()
                        .flat_map(|objects| objects.objects().map(|object| &object.plan)),
                ),
        ),
        transcript,
    )?;
    let untrusted_physical = untrusted_advice
        .map(|object| reduce_precommitted(&object.plan, &leaves, transcript))
        .transpose()?;
    let trusted_physical = trusted_advice
        .map(|object| reduce_precommitted(&object.plan, &leaves, transcript))
        .transpose()?;

    // Canonical public batch order: advice, (field-inline) the field increment polynomial,
    // or the direct committed-program objects, then OneHotTrace. The suffixes
    // are exclusive: field-inline rejects committed-program preprocessing.
    let mut auxiliary_groups = Vec::with_capacity(
        2 + usize::from(cfg!(feature = "field-inline"))
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

    let joint_opening_proof = tracing::info_span!("akita_main_batched_prove").in_scope(|| {
        PCS::prove_batch(
            &preprocessing.pcs_setup,
            auxiliary_groups,
            main_group,
            one_hot_trace_hint,
            BorrowMut::<ProverState<PCS::Sponge>>::borrow_mut(transcript),
        )
        .map_err(batch_failed::<F>)
    })?;
    Ok((joint_opening_proof, pow_nonce))
}
