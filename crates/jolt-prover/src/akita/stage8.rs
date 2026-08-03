//! Packed stage 8: the final opening — one native same-point Akita batch for
//! the `OneHotTrace` group, plus a joint packed opening for the auxiliary
//! commitment objects (advice byte columns, `ProgramOneHot`).
//!
//! Pure orchestration mirroring `stage8::packed::verify`: the per-column leaf
//! claims come from the verifier's promoted `leaf_claims` resolution over the
//! stage-7 and reconstruction outputs, the shared canonical point from the
//! layout's own `column_point` math, and the auxiliary statements from the
//! promoted `object_statement` assembly. The prover-only work is supplying
//! the group hint (stage 0's commit) and the auxiliary objects' witnesses.

use jolt_claims::protocols::jolt::lattice::packing::advice_bytes_packing;
use jolt_claims::protocols::jolt::lattice::{
    precommitted_packing, OneHotTraceShape, ONE_HOT_TRACE_LAYOUT,
};
use jolt_claims::protocols::jolt::{JoltAdviceKind, JoltCommittedPolynomial, JoltRelationId};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::{
    prove_packed_openings, CommitmentScheme, PackedProverGroup, PackedProverObject,
    PrefixPackedStatement, PrefixPacking,
};
use jolt_poly::MultilinearPoly;
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::proof::AkitaJointOpeningProof;
use jolt_verifier::stages::stage7::outputs::Stage7ClearOutput;
use jolt_verifier::stages::stage8::packed::{leaf_claims, object_statement};
use jolt_verifier::stages::stage8::reconstruction::ReconstructionClearOutput;
use jolt_verifier::{CheckedInputs, VerifierError};

use super::witness::{AdviceOneHot, CommittedOneHotShape, ProgramOneHot};
use crate::{JoltProverPreprocessing, ProverConfig, ProverError};

fn batch_failed<F: Field>(reason: impl ToString) -> ProverError<F> {
    ProverError::Verifier(VerifierError::FinalOpeningBatchFailed {
        reason: reason.to_string(),
    })
}

/// Prove packed stage 8 on `transcript` (positioned at the reconstruction
/// boundary): the native `OneHotTrace` same-point batch from the group hint,
/// then the auxiliary packed openings in canonical object order.
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
) -> Result<AkitaJointOpeningProof<F, PCS::Proof>, ProverError<F>>
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

    // OneHotTrace opens as one native same-point batch: each column's leaf
    // point maps to the shared canonical point through the layout.
    let mut common_point: Option<Vec<F>> = None;
    let mut evaluations = Vec::with_capacity(plan.columns.len());
    for polynomial in &plan.columns {
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
                    "OneHotTrace column {polynomial:?} does not share the canonical opening point"
                )));
            }
        } else {
            common_point = Some(point);
        }
        evaluations.push(claim.value);
    }
    let common_point =
        common_point.ok_or_else(|| batch_failed::<F>("OneHotTrace has no columns"))?;
    let shapes: Vec<CommittedOneHotShape> = plan
        .columns
        .iter()
        .map(|_| CommittedOneHotShape {
            num_vars: plan.column_arity,
        })
        .collect();
    let shape_refs: Vec<&dyn MultilinearPoly<F>> = shapes
        .iter()
        .map(|shape| shape as &dyn MultilinearPoly<F>)
        .collect();
    // The packed sibling of the homomorphic path's stage-8 batch seam: one
    // native same-point opening over the whole group.
    let one_hot_trace =
        tracing::info_span!("CommitmentScheme::open_batch", columns = evaluations.len())
            .in_scope(|| {
                PCS::open_batch(
                    &shape_refs,
                    &common_point,
                    &evaluations,
                    &preprocessing.pcs_setup,
                    one_hot_trace_hint,
                    transcript,
                )
            })
            .map_err(batch_failed::<F>)?;

    // The auxiliary packed objects in canonical order: untrusted advice,
    // trusted advice, ProgramOneHot. Packings and statements are owned here;
    // the borrowed object views assemble below.
    struct Auxiliary<'a, PCS: CommitmentScheme> {
        packing: PrefixPacking<JoltCommittedPolynomial>,
        statement: PrefixPackedStatement<PCS::Field, JoltCommittedPolynomial, PCS::Output>,
        polynomial: &'a dyn MultilinearPoly<PCS::Field>,
        setup: &'a PCS::ProverSetup,
        hint: PCS::OpeningHint,
    }
    let mut auxiliaries: Vec<Auxiliary<'_, PCS>> = Vec::new();
    for (kind, object) in [
        (JoltAdviceKind::Untrusted, untrusted_advice),
        (JoltAdviceKind::Trusted, trusted_advice),
    ] {
        let Some(object) = object else { continue };
        let packing = advice_bytes_packing(kind, object.word_vars).map_err(batch_failed::<F>)?;
        let statement = object_statement(&packing, object.commitment.clone(), &leaves)
            .map_err(ProverError::Verifier)?;
        auxiliaries.push(Auxiliary {
            packing,
            statement,
            polynomial: &object.byte_column,
            setup: &object.setup,
            hint: object.hint.clone(),
        });
    }
    if let Some(object) = program {
        let packing = precommitted_packing(&object.shape).map_err(batch_failed::<F>)?;
        let statement = object_statement(&packing, object.commitment.clone(), &leaves)
            .map_err(ProverError::Verifier)?;
        auxiliaries.push(Auxiliary {
            packing,
            statement,
            polynomial: &object.witness,
            setup: &object.setup,
            hint: object.hint.clone(),
        });
    }

    let auxiliary = if auxiliaries.is_empty() {
        None
    } else {
        let groups = auxiliaries
            .iter()
            .enumerate()
            .map(|(index, auxiliary)| {
                PackedProverGroup::singleton(index, Some(auxiliary.hint.clone()))
            })
            .collect();
        let objects: Vec<PackedProverObject<'_, PCS, JoltCommittedPolynomial>> = auxiliaries
            .iter()
            .map(|auxiliary| PackedProverObject {
                packing: &auxiliary.packing,
                statement: &auxiliary.statement,
                polynomial: auxiliary.polynomial,
                setup: auxiliary.setup,
            })
            .collect();
        Some(prove_packed_openings(objects, groups, transcript).map_err(batch_failed::<F>)?)
    };

    Ok(AkitaJointOpeningProof {
        one_hot_trace,
        auxiliary,
    })
}
