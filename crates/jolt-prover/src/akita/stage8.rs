//! Akita's final opening: one heterogeneous advice/main-trace opening over the
//! canonical group order `[UntrustedAdvice, TrustedAdvice, OneHotTrace]`,
//! followed by independently pointed program objects.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::lattice::packing::{OneHotTraceShape, PrefixPackedObjectPlan};
use jolt_claims::protocols::jolt::lattice::strategy::ONE_HOT_TRACE_LAYOUT;
use jolt_claims::protocols::jolt::{JoltAdviceKind, JoltCommittedPolynomial, JoltRelationId};
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_openings::{CommitmentScheme, EvaluationClaim, GroupOpeningClaim, PrecommittedClaim};
use jolt_poly::MultilinearPoly;
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::proof::AkitaJointOpeningProof;
use jolt_verifier::stages::stage6b::outputs::Stage6bClearOutput;
use jolt_verifier::stages::stage7::outputs::Stage7ClearOutput;
use jolt_verifier::stages::stage8::packed::{
    leaf_claims, object_leaf_claims, one_hot_trace_packed_claims,
};
use jolt_verifier::stages::stage8::reconstruction::ReconstructionClearOutput;
use jolt_verifier::{CheckedInputs, VerifierError};

#[cfg(feature = "field-inline")]
use super::field_inline::FieldIncLimbsObject;
use super::witness::{AdviceObject, ProgramOneHot};
use crate::{JoltProverPreprocessing, ProverConfig, ProverError};
#[cfg(feature = "field-inline")]
use jolt_verifier::stages::stage8::field_inline_packed::FieldIncLimbClaims;

fn batch_failed<F: JoltField>(reason: impl ToString) -> ProverError<F> {
    ProverError::Verifier(VerifierError::FinalOpeningBatchFailed {
        reason: reason.to_string(),
    })
}

fn reduce_auxiliary<F, T>(
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

fn open_reduced_auxiliary<F, PCS, T, P>(
    polynomial: &P,
    setup: &PCS::ProverSetup,
    hint: PCS::OpeningHint,
    physical: &EvaluationClaim<F>,
    transcript: &mut T,
) -> Result<PCS::Proof, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    P: MultilinearPoly<F> + ?Sized,
    T: Transcript<Challenge = F>,
{
    PCS::open(
        polynomial,
        physical.point.as_slice(),
        physical.value,
        setup,
        Some(hint),
        transcript,
    )
    .map_err(batch_failed::<F>)
}

/// The stage-8 wire artifacts: the joint opening proof and (FR builds) the
/// limb-group claims the proof carries beside it.
pub struct Stage8Artifacts<PCS>
where
    PCS: CommitmentScheme,
{
    pub joint_opening_proof: AkitaJointOpeningProof<PCS::Proof>,
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
    program: Option<&ProgramOneHot<PCS>>,
    stage6b: &Stage6bClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
    reconstruction: &ReconstructionClearOutput<F>,
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

    let leaves = leaf_claims(&checked.precommitted, stage6b, stage7, reconstruction)?;

    let packed_claims =
        one_hot_trace_packed_claims(&plan, chunk_width, &leaves).map_err(ProverError::Verifier)?;
    let packed_claim = plan
        .packing()
        .reduce_claims(&packed_claims, transcript)
        .map_err(batch_failed::<F>)?;

    let untrusted_physical = untrusted_advice
        .map(|object| reduce_auxiliary(&object.plan, &leaves, transcript))
        .transpose()?;
    let trusted_physical = trusted_advice
        .map(|object| reduce_auxiliary(&object.plan, &leaves, transcript))
        .transpose()?;

    // Canonical public batch order:
    // [UntrustedAdvice, TrustedAdvice, FieldIncLimbs, OneHotTrace].
    let mut precommitted = Vec::with_capacity(3);
    for (role, object, claim) in [
        (
            JoltAdviceKind::Untrusted.precommitted_role(),
            untrusted_advice,
            untrusted_physical.as_ref(),
        ),
        (
            JoltAdviceKind::Trusted.precommitted_role(),
            trusted_advice,
            trusted_physical.as_ref(),
        ),
    ] {
        if let (Some(object), Some(claim)) = (object, claim) {
            precommitted.push((
                PrecommittedClaim::new(
                    role,
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

    let main_group = GroupOpeningClaim::new(
        one_hot_trace_commitment.clone(),
        packed_claim.point.as_slice().to_vec(),
        vec![packed_claim.value],
    );
    let main_batch = tracing::info_span!("akita_main_batched_prove").in_scope(|| {
        PCS::prove_batch(
            &preprocessing.pcs_setup,
            precommitted,
            main_group,
            one_hot_trace_hint,
            transcript,
        )
        .map_err(batch_failed::<F>)
    })?;

    let mut auxiliary = Vec::new();
    if let Some(program) = program {
        for object in &program.objects {
            let physical = reduce_auxiliary(&object.plan, &leaves, transcript)?;
            auxiliary.push(open_reduced_auxiliary::<F, PCS, T, _>(
                &object.witness,
                &object.setup,
                object.hint.clone(),
                &physical,
                transcript,
            )?);
        }
    }

    Ok(Stage8Artifacts {
        joint_opening_proof: AkitaJointOpeningProof::new(main_batch, auxiliary),
        #[cfg(feature = "field-inline")]
        field_inc_limbs: field_inc_limb_claims,
    })
}
