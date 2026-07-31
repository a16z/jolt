//! Akita's final opening: one prefix-packed trace polynomial followed by the
//! independently pointed advice and committed-program objects.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::lattice::packing::{OneHotTraceShape, PrefixPackedObjectPlan};
use jolt_claims::protocols::jolt::lattice::strategy::ONE_HOT_TRACE_LAYOUT;
use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltRelationId};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::{CommitmentScheme, EvaluationClaim};
use jolt_poly::MultilinearPoly;
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::proof::AkitaJointOpeningProof;
use jolt_verifier::stages::stage7::outputs::Stage7ClearOutput;
use jolt_verifier::stages::stage8::packed::leaf_claims;
use jolt_verifier::stages::stage8::reconstruction::ReconstructionClearOutput;
use jolt_verifier::{CheckedInputs, VerifierError};

use super::witness::{AdviceOneHot, CommittedOneHotShape, ProgramOneHot};
use crate::{JoltProverPreprocessing, ProverConfig, ProverError};

fn batch_failed<F: Field>(reason: impl ToString) -> ProverError<F> {
    ProverError::Verifier(VerifierError::FinalOpeningBatchFailed {
        reason: reason.to_string(),
    })
}

fn open_auxiliary<F, PCS, T, P>(
    plan: &PrefixPackedObjectPlan,
    polynomial: &P,
    setup: &PCS::ProverSetup,
    hint: PCS::OpeningHint,
    leaves: &BTreeMap<JoltCommittedPolynomial, EvaluationClaim<F>>,
    transcript: &mut T,
) -> Result<PCS::Proof, ProverError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    P: MultilinearPoly<F> + ?Sized,
    T: Transcript<Challenge = F>,
{
    let claims = plan
        .packing()
        .ids()
        .iter()
        .map(|id| {
            leaves
                .get(id)
                .cloned()
                .map(|claim| (*id, claim))
                .ok_or_else(|| batch_failed::<F>(format!("missing final claim for {id:?}")))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let semantic = plan.packed_claims(&claims).map_err(batch_failed::<F>)?;
    let physical = plan
        .packing()
        .reduce_claims(&semantic, transcript)
        .map_err(batch_failed::<F>)?;
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

#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
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
    let leaves = leaf_claims(stage7, reconstruction);

    let mut common_point = None;
    let mut evaluations = Vec::with_capacity(plan.packing().ids().len());
    for polynomial in plan.packing().ids() {
        let claim = leaves.get(polynomial).ok_or_else(|| {
            batch_failed::<F>(format!(
                "missing final OneHotTrace claim for {polynomial:?}"
            ))
        })?;
        let point = ONE_HOT_TRACE_LAYOUT
            .column_point(*polynomial, chunk_width, claim.point.as_slice())
            .map_err(batch_failed::<F>)?;
        if let Some(expected) = &common_point {
            if expected != &point {
                return Err(batch_failed::<F>(format!(
                    "OneHotTrace column {polynomial:?} has a different opening point"
                )));
            }
        } else {
            common_point = Some(point);
        }
        evaluations.push(claim.value);
    }
    let common_point = common_point.ok_or_else(|| batch_failed::<F>("OneHotTrace is empty"))?;
    let semantic = plan.packed_claims(common_point, evaluations);
    let physical = plan
        .packing()
        .reduce_claims(&semantic, transcript)
        .map_err(batch_failed::<F>)?;
    let shape = CommittedOneHotShape {
        num_vars: plan.packing().packed_num_vars(),
    };
    let polynomials: [&dyn MultilinearPoly<F>; 1] = [&shape];
    let one_hot_trace = PCS::open_batch(
        &polynomials,
        physical.point.as_slice(),
        std::slice::from_ref(&physical.value),
        &preprocessing.pcs_setup,
        one_hot_trace_hint,
        transcript,
    )
    .map_err(batch_failed::<F>)?;

    let mut auxiliary = Vec::new();
    for object in [untrusted_advice, trusted_advice].into_iter().flatten() {
        auxiliary.push(open_auxiliary::<F, PCS, T, _>(
            &object.plan,
            &object.byte_column,
            &object.setup,
            object.hint.clone(),
            &leaves,
            transcript,
        )?);
    }
    if let Some(program) = program {
        for object in &program.objects {
            auxiliary.push(open_auxiliary::<F, PCS, T, _>(
                &object.plan,
                &object.witness,
                &object.setup,
                object.hint.clone(),
                &leaves,
                transcript,
            )?);
        }
    }

    Ok(AkitaJointOpeningProof::new(one_hot_trace, auxiliary))
}
