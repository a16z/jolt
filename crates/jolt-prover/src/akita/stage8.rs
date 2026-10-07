//! Akita's final opening: one heterogeneous advice/main-trace opening over the
//! canonical group order `[UntrustedAdvice?, TrustedAdvice?,
//! BytecodeChunk(0..C), ProgramImageInit, OneHotTrace]`, or under the byte
//! link `[.., ProgramImageInit, W triples, W RAM, Q]`.

use std::collections::BTreeMap;
#[cfg(not(feature = "akita-byte-link"))]
use std::sync::Arc;

use jolt_akita::TraceOneHotCommitment;
#[cfg(not(feature = "akita-byte-link"))]
use jolt_akita::TraceOneHotRows;
use jolt_claims::protocols::jolt::lattice::packing::PrefixPackedObjectPlan;
#[cfg(feature = "akita-byte-link")]
use jolt_claims::protocols::jolt::lattice::strategy::ByteTraceLayoutPlan;
#[cfg(not(feature = "akita-byte-link"))]
use jolt_claims::protocols::jolt::lattice::strategy::OneHotTraceLayoutPlan;
use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_openings::{CommitmentScheme, EvaluationClaim, GroupOpeningClaim, PrecommittedClaim};
use jolt_transcript::{AppendToTranscript, Transcript};
#[cfg(feature = "akita-byte-link")]
use jolt_verifier::stages::byte_link::ByteLinkOpenings;
use jolt_verifier::stages::stage4::outputs::Stage4ClearOutput;
use jolt_verifier::stages::stage6b::outputs::Stage6bClearOutput;
use jolt_verifier::stages::stage7::outputs::Stage7ClearOutput;
#[cfg(not(feature = "akita-byte-link"))]
use jolt_verifier::stages::stage8::packed::one_hot_trace_packed_claims;
#[cfg(feature = "akita-byte-link")]
use jolt_verifier::stages::stage8::packed::{byte_trace_packed_claims, histogram_claims};
use jolt_verifier::stages::stage8::packed::{leaf_claims, object_leaf_claims};
use jolt_verifier::{CheckedInputs, VerifierError};
#[cfg(not(feature = "akita-byte-link"))]
use jolt_witness::JoltWitnessPlane;

#[cfg(not(feature = "akita-byte-link"))]
use super::witness::assemble_one_hot_trace_rows;
use super::witness::{AdviceObject, DirectProgramObjects};
#[cfg(not(feature = "akita-byte-link"))]
use crate::ProverConfig;
use crate::{JoltProverPreprocessing, ProverError};

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
/// `witness`; without them the opening assembles its own. Under the byte link
/// `Q` opens at the link's source point and both histogram groups at their
/// query points.
#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
#[tracing::instrument(skip_all)]
pub fn prove_stage8<F, PCS, VC, T>(
    checked: &CheckedInputs,
    #[cfg(not(feature = "akita-byte-link"))] config: &ProverConfig,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    #[cfg(not(feature = "akita-byte-link"))] plan: &OneHotTraceLayoutPlan,
    #[cfg(not(feature = "akita-byte-link"))] assembled_rows: Option<Arc<dyn TraceOneHotRows>>,
    #[cfg(not(feature = "akita-byte-link"))] witness: &dyn JoltWitnessPlane<F>,
    #[cfg(feature = "akita-byte-link")] plan: &ByteTraceLayoutPlan,
    #[cfg(feature = "akita-byte-link")] link: &ByteLinkOpenings<F>,
    #[cfg(feature = "akita-byte-link")] histograms: (&[PCS::Output; 2], [PCS::OpeningHint; 2]),
    trace_commitment: &PCS::Output,
    trace_hint: PCS::OpeningHint,
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
    #[cfg(not(feature = "akita-byte-link"))]
    let (trace_hint, chunk_width) = {
        let mut trace_hint = trace_hint;
        let chunk_width = config.one_hot_config.committed_chunk_bits();
        let rows = if let Some(rows) = assembled_rows {
            rows
        } else {
            let log_t = checked.trace_length.ilog2() as usize;
            assemble_one_hot_trace_rows(witness, plan, chunk_width, log_t)?
        };
        PCS::restore_trace_rows(&mut trace_hint, rows).map_err(batch_failed::<F>)?;
        (trace_hint, chunk_width)
    };

    let leaves = leaf_claims(&checked.precommitted, stage4, stage6b, stage7)?;

    #[cfg(not(feature = "akita-byte-link"))]
    let packed_claims =
        one_hot_trace_packed_claims(plan, chunk_width, &leaves).map_err(ProverError::Verifier)?;
    #[cfg(feature = "akita-byte-link")]
    let packed_claims = byte_trace_packed_claims(plan, &link.source);
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

    #[cfg(feature = "akita-byte-link")]
    {
        let (commitments, hints) = histograms;
        precommitted.extend(histogram_claims(link, commitments).into_iter().zip(hints));
    }

    let main_group = GroupOpeningClaim::new(
        trace_commitment.clone(),
        packed_claim.point.as_slice().to_vec(),
        vec![packed_claim.value],
    );
    tracing::info_span!("akita_main_batched_prove").in_scope(|| {
        PCS::prove_batch(
            &preprocessing.pcs_setup,
            precommitted,
            main_group,
            trace_hint,
            transcript,
        )
        .map_err(batch_failed::<F>)
    })
}
