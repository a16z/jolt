//! Stage 8's field-inline seam: the composed final-opening splice. `verify.rs` interacts with the field-inline protocol
//! only through the functions here.

use jolt_claims::protocols::field_inline::geometry::claim_reductions::increments::field_rd_inc_reduced;
use jolt_claims::protocols::jolt::geometry::committed_openings::{
    commitment_embedding_scale, CommitmentEmbedding,
};
use jolt_claims::protocols::jolt::{
    JoltCommittedPolynomial, JoltOpeningId, JoltRelationId, TracePolynomialOrder,
};
use jolt_field::JoltField;

use super::Stage8BatchEntry;
use crate::proof::JoltCommitments;
use crate::VerifierError;
use jolt_claims::protocols::composed::ComposedOpeningId;

/// Splice the reduced `FieldRdInc` final opening into the batch entries at the spec's position
/// — immediately after `RdInc@IncClaimReduction`, before the RA families
/// (`specs/field-inline-protocol.md`, the field-inline final-opening order). Mirrors `RdInc`'s
/// treatment exactly: the commitment comes from the proof's field-inline payload, the claim and point from the stage-6b field-register increment reduction, and
/// the dense embedding scale through the same `commitment_embedding_scale` helper. Public
/// because the prover's stage-8 recipe splices its PCS batch statement identically.
pub fn splice_final_opening<'a, F, C>(
    entries: &mut Vec<Stage8BatchEntry<'a, F, C>>,
    commitments: &'a JoltCommitments<C>,
    trace_order: TracePolynomialOrder,
    opening_point: &[F],
    field_inline_opening_point: &[F],
    opening_claim: Option<F>,
) -> Result<(), VerifierError>
where
    F: JoltField,
{
    let rd_inc_id: ComposedOpeningId = JoltOpeningId::committed(
        JoltCommittedPolynomial::RdInc,
        JoltRelationId::IncClaimReduction,
    )
    .into();
    let splice_position = entries
        .iter()
        .position(|entry| entry.id == rd_inc_id)
        .and_then(|position| position.checked_add(1))
        .ok_or_else(|| VerifierError::FinalOpeningBatchFailed {
            reason: "the final opening batch has no RdInc entry to anchor the FieldRdInc splice"
                .to_string(),
        })?;
    entries.insert(
        splice_position,
        Stage8BatchEntry {
            id: field_rd_inc_reduced().into(),
            commitment: &commitments.field_inline.field_registers.rd_inc,
            opening_claim,
            scale: commitment_embedding_scale(
                opening_point,
                field_inline_opening_point,
                CommitmentEmbedding::Trace {
                    order: trace_order,
                    log_t: field_inline_opening_point.len(),
                },
            )
            .ok_or_else(|| VerifierError::FinalOpeningBatchFailed {
                reason: "the FieldRdInc reduction point is not embedded in the unified \
                             final opening point"
                    .to_string(),
            })?,
        },
    );
    Ok(())
}
