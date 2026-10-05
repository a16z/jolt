//! Akita's final opening: one heterogeneous advice/main-trace opening over the
//! canonical group order `[UntrustedAdvice?, TrustedAdvice?,
//! BytecodeChunk(0..C), ProgramImageInit, OneHotTrace]`.

use std::collections::BTreeMap;
use std::sync::Arc;

use jolt_akita::{TraceOneHotCommitment, TraceOneHotRows};
use jolt_claims::protocols::jolt::lattice::packing::PrefixPackedObjectPlan;
use jolt_claims::protocols::jolt::lattice::strategy::OneHotTraceLayoutPlan;
use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_openings::{CommitmentScheme, EvaluationClaim, GroupOpeningClaim, PrecommittedClaim};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::stages::stage4::outputs::Stage4ClearOutput;
use jolt_verifier::stages::stage6b::outputs::Stage6bClearOutput;
use jolt_verifier::stages::stage7::outputs::Stage7ClearOutput;
use jolt_verifier::stages::stage8::packed::{
    leaf_claims, object_leaf_claims, one_hot_trace_packed_claims,
};
use jolt_verifier::{CheckedInputs, VerifierError};
use jolt_witness::JoltWitnessPlane;

use super::witness::{assemble_one_hot_trace_rows, AdviceObject, DirectProgramObjects};
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

/// `assembled_rows` are the `OneHotTrace` rows already assembled from
/// `witness`; without them the opening assembles its own.
#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
#[tracing::instrument(skip_all)]
pub fn prove_stage8<F, PCS, VC, T>(
    checked: &CheckedInputs,
    config: &ProverConfig,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    plan: &OneHotTraceLayoutPlan,
    assembled_rows: Option<Arc<dyn TraceOneHotRows>>,
    witness: &dyn JoltWitnessPlane<F>,
    one_hot_trace_commitment: &PCS::Output,
    mut one_hot_trace_hint: PCS::OpeningHint,
    untrusted_advice: Option<&AdviceObject<PCS>>,
    trusted_advice: Option<&AdviceObject<PCS>>,
    program: Option<&DirectProgramObjects<PCS>>,
    stage4: &Stage4ClearOutput<F>,
    stage6b: &Stage6bClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
    transcript: &mut T,
) -> Result<PCS::Proof, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + TraceOneHotCommitment,
    PCS::Output: Clone + AppendToTranscript,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    let chunk_width = config.one_hot_config.committed_chunk_bits();
    let rows = if let Some(rows) = assembled_rows {
        rows
    } else {
        let log_t = checked.trace_length.ilog2() as usize;
        assemble_one_hot_trace_rows(witness, plan, chunk_width, log_t)?
    };
    PCS::restore_trace_rows(&mut one_hot_trace_hint, rows).map_err(batch_failed::<F>)?;

    let leaves = leaf_claims(&checked.precommitted, stage4, stage6b, stage7)?;

    let packed_claims =
        one_hot_trace_packed_claims(plan, chunk_width, &leaves).map_err(ProverError::Verifier)?;
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

    let mut precommitted = Vec::with_capacity(2 + program.map_or(0, |p| p.objects.len()));
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
    tracing::info_span!("akita_main_batched_prove").in_scope(|| {
        PCS::prove_batch(
            &preprocessing.pcs_setup,
            precommitted,
            main_group,
            one_hot_trace_hint,
            transcript,
        )
        .map_err(batch_failed::<F>)
    })
}
