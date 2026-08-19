//! Packed stage 8: the final opening — the `OneHotTrace` prefix-packed
//! claim reduction and its one native Akita opening, then one packed opening
//! per auxiliary commitment object (advice byte columns, the `ProgramOneHot`
//! objects).
//!
//! Pure orchestration mirroring `stage8::packed::verify`: the per-column
//! leaf claims come from the verifier's promoted `leaf_claims` resolution
//! over the stage-7 and reconstruction outputs, the packed `OneHotTrace`
//! statement from the promoted `one_hot_trace_packed_claims` assembly, and
//! each auxiliary object's claims from the promoted `object_leaf_claims`.
//! The prover-only work is supplying the packed polynomial's retained
//! committed object (stage 0's commit hint) and the auxiliary objects'
//! witnesses.

use jolt_claims::protocols::jolt::lattice::{OneHotTraceShape, ONE_HOT_TRACE_LAYOUT};
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_poly::MultilinearPoly;
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::proof::AkitaJointOpeningProof;
use jolt_verifier::stages::stage7::outputs::Stage7ClearOutput;
use jolt_verifier::stages::stage8::packed::{
    leaf_claims, object_leaf_claims, one_hot_trace_packed_claims,
};
use jolt_verifier::stages::stage8::reconstruction::ReconstructionClearOutput;
use jolt_verifier::{CheckedInputs, VerifierError};

use super::witness::{AdviceOneHot, ProgramOneHot};
use crate::{JoltProverPreprocessing, ProverConfig, ProverError};

fn batch_failed<F: Field>(reason: impl ToString) -> ProverError<F> {
    ProverError::Verifier(VerifierError::FinalOpeningBatchFailed {
        reason: reason.to_string(),
    })
}

/// Prove packed stage 8 on `transcript` (positioned at the reconstruction
/// boundary): the `OneHotTrace` prefix-packed reduction and its native
/// opening from the group hint, then the auxiliary packed openings in
/// canonical object order (untrusted advice, trusted advice, the
/// `ProgramOneHot` objects).
#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
#[tracing::instrument(skip_all)]
pub fn prove_stage8<F, PCS, VC, T>(
    checked: &CheckedInputs,
    config: &ProverConfig,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    one_hot_trace_hint: PCS::OpeningHint,
    untrusted_advice: Option<&AdviceOneHot<PCS>>,
    trusted_advice: Option<&AdviceOneHot<PCS>>,
    program: Option<&ProgramOneHot<PCS>>,
    stage7: &Stage7ClearOutput<F>,
    reconstruction: &ReconstructionClearOutput<F>,
    transcript: &mut T,
) -> Result<AkitaJointOpeningProof<PCS::Proof>, ProverError<F>>
where
    F: Field,
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

    // Every packed column's single leaf claim, resolved exactly as the
    // verifier resolves them.
    let leaves = leaf_claims(stage7, reconstruction);

    // OneHotTrace: assemble the shared-point packed statement, reduce it to
    // one physical claim on the transcript, and open it natively from the
    // retained stage-0 commit state (the hint is the committed object; it
    // owns the witness forms and the backend opening data).
    let packed_claims =
        one_hot_trace_packed_claims(&plan, chunk_width, &leaves).map_err(ProverError::Verifier)?;
    let packed_claim = plan
        .packing()
        .reduce_claims(&packed_claims, transcript)
        .map_err(batch_failed::<F>)?;
    let one_hot_trace = tracing::info_span!(
        "CommitmentScheme::open_batch_from_hint",
        packed_num_vars = plan.packing().packed_num_vars()
    )
    .in_scope(|| {
        PCS::open_batch_from_hint(
            packed_claim.point.as_slice(),
            std::slice::from_ref(&packed_claim.value),
            &preprocessing.pcs_setup,
            one_hot_trace_hint,
            transcript,
        )
    })
    .map_err(batch_failed::<F>)?;

    // The auxiliary packed objects in canonical order: untrusted advice,
    // trusted advice, the ProgramOneHot objects (bytecode, then image). Each
    // reduces its own claims on the transcript and opens natively.
    let open_object = |plan: &jolt_claims::protocols::jolt::lattice::PrefixPackedObjectPlan,
                       polynomial: &dyn MultilinearPoly<F>,
                       setup: &PCS::ProverSetup,
                       hint: PCS::OpeningHint,
                       transcript: &mut T|
     -> Result<PCS::Proof, ProverError<F>> {
        let claims = object_leaf_claims(plan, &leaves).map_err(ProverError::Verifier)?;
        let semantic_claims = plan.packed_claims(&claims).map_err(batch_failed::<F>)?;
        let physical_claim = plan
            .packing()
            .reduce_claims(&semantic_claims, transcript)
            .map_err(batch_failed::<F>)?;
        PCS::open(
            polynomial,
            physical_claim.point.as_slice(),
            physical_claim.value,
            setup,
            Some(hint),
            transcript,
        )
        .map_err(batch_failed::<F>)
    };

    let mut auxiliary = Vec::new();
    for object in [untrusted_advice, trusted_advice].into_iter().flatten() {
        auxiliary.push(open_object(
            &object.plan,
            &object.byte_column,
            &object.setup,
            object.hint.clone(),
            transcript,
        )?);
    }
    if let Some(program) = program {
        for object in &program.objects {
            auxiliary.push(open_object(
                &object.plan,
                &object.witness,
                &object.setup,
                object.hint.clone(),
                transcript,
            )?);
        }
    }

    Ok(AkitaJointOpeningProof::new(one_hot_trace, auxiliary))
}
