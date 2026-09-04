//! Stage 8's field-inline seam: the composed final-opening splice and the FR
//! commitment-payload presence check. `verify.rs` interacts with the FR
//! protocol only through the functions here.

use jolt_claims::protocols::field_inline::geometry::claim_reductions::increments::field_rd_inc_reduced;
use jolt_claims::protocols::jolt::geometry::committed_openings::commitment_embedding_scale;
use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltOpeningId, JoltRelationId};
use jolt_field::JoltField;

use super::Stage8BatchEntry;
use crate::proof::JoltCommitments;
use crate::stages::ids::VerifierOpeningId;
use crate::VerifierError;

/// The FR commitment payload is part of the expected layout: the composed
/// final opening cannot assemble without the `FieldRdInc` commitment.
pub(super) fn require_commitment<C>(commitments: &JoltCommitments<C>) -> Result<(), VerifierError> {
    if commitments.field_inline.is_none() {
        return Err(VerifierError::MissingProofPayload {
            field: "commitments.field_inline",
        });
    }
    Ok(())
}

/// Splice the reduced `FieldRdInc` final opening into the batch entries at the
/// spec's position — immediately after `RdInc@IncClaimReduction`, before the
/// RA families (`specs/field-inline-protocol.md`, the field-inline
/// final-opening order). Mirrors `RdInc`'s treatment exactly: the commitment
/// comes from the proof's FR payload (present fail-closed), the claim and
/// point from the stage-6b FR increment reduction, and the dense embedding
/// scale through the same `commitment_embedding_scale` helper. Public because
/// the prover's stage-8 recipe splices its PCS batch statement identically.
pub fn splice_final_opening<'a, F, C>(
    entries: &mut Vec<Stage8BatchEntry<'a, F, C>>,
    commitments: &'a JoltCommitments<C>,
    opening_point: &[F],
    field_inline_opening_point: &[F],
    opening_claim: Option<F>,
) -> Result<(), VerifierError>
where
    F: JoltField,
{
    let field_inline =
        commitments
            .field_inline
            .as_ref()
            .ok_or(VerifierError::MissingProofPayload {
                field: "commitments.field_inline",
            })?;
    let rd_inc_id: VerifierOpeningId = JoltOpeningId::committed(
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
            commitment: &field_inline.field_registers.rd_inc,
            opening_claim,
            scale: commitment_embedding_scale(opening_point, field_inline_opening_point)
                .ok_or_else(|| VerifierError::FinalOpeningBatchFailed {
                    reason: "the FieldRdInc reduction point is not embedded in the unified \
                             final opening point"
                        .to_string(),
                })?,
        },
    );
    Ok(())
}
