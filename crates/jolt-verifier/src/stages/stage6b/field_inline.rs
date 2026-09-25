//! Field-inline bytecode composition, preprocessing checks, and transcript ordering
//! for stage 6b.

use jolt_claims::protocols::field_inline::{FieldInlineRelationId, FIELD_REGISTERS_LOG_K};
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_claims::OutputClaims as _;
use jolt_field::JoltField;
use jolt_openings::CommitmentScheme;

use super::outputs::Stage6bOutputClaims;
use crate::preprocessing::ProgramPreprocessing;
use crate::stages::field_inline_bytecode::{
    convert_field_inline_bytecode, field_inline_checked_split, field_inline_stage_gamma_powers,
    required_field_inline_bytecode, FieldInlineBytecodeFold, FieldInlineBytecodeTable,
};
use crate::stages::stage4::Stage4OutputPoints;
use crate::stages::stage5::Stage5OutputPoints;
use crate::stages::stage6a::outputs::Stage6aCarriedChallenges;
use crate::VerifierError;

/// The field-inline extension anchors the field access selectors through the
/// public/preprocessed side table, which committed-program mode cannot supply. Shared with the
/// BlindFold build, which hits the same wall.
pub(crate) fn committed_program_rejection() -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::BytecodeReadRaf,
        reason: "field-inline verification requires the full-program bytecode side \
                 table; committed-program mode is unsupported"
            .to_string(),
    }
}

/// Reject committed-program mode before any member construction (see
/// [`committed_program_rejection`]).
pub(crate) fn require_full_program(committed_program: bool) -> Result<(), VerifierError> {
    if committed_program {
        return Err(committed_program_rejection());
    }
    Ok(())
}

/// The converted field-inline bytecode side table from the verifier preprocessing. A hard
/// preprocessing requirement of stage 6 (spec: "Stage 6 rejects a field-inline proof if the
/// table is missing"); committed-program preprocessing carries no full bytecode, so with
/// field-inline enabled rejects it here too.
pub fn preprocessed_bytecode_table<PCS: CommitmentScheme>(
    program: &ProgramPreprocessing<PCS>,
) -> Result<FieldInlineBytecodeTable, VerifierError> {
    convert_field_inline_bytecode(required_field_inline_bytecode(program)?)
}

/// The field-inline legs of the stage-6b batch build, returned by
/// [`bytecode_fold_and_cycles`]: the bytecode side-table fold inputs, plus the stage-4/5
/// field-inline cycle sub-points (past the field-register address prefix) that feed both the
/// bytecode field-inline public fold and the field-register increment reduction's Eq publics.
pub(super) struct FieldInlineBatchLegs<F> {
    pub fold: FieldInlineBytecodeFold<F>,
    pub read_write_cycle: Vec<F>,
    pub val_evaluation_cycle: Vec<F>,
}

/// Split the stage-4/5 field-inline opening points past the field-register address prefix and
/// assemble the side-table fold legs.
pub(super) fn bytecode_fold_and_cycles<F: JoltField>(
    table: FieldInlineBytecodeTable,
    carried: &Stage6aCarriedChallenges<F>,
    stage4_points: &Stage4OutputPoints<F>,
    stage5_points: &Stage5OutputPoints<F>,
) -> Result<FieldInlineBatchLegs<F>, VerifierError> {
    let (read_write_address, read_write_cycle) = field_inline_checked_split(
        "Stage 6 stage4 field-register read-write opening",
        stage4_points.field_registers_read_write_point(),
        FIELD_REGISTERS_LOG_K,
        FieldInlineRelationId::FieldRegistersReadWriteChecking,
    )?;
    let (val_evaluation_address, val_evaluation_cycle) = field_inline_checked_split(
        "Stage 6 stage5 field-register val-evaluation opening",
        stage5_points.field_registers_val_evaluation_point(),
        FIELD_REGISTERS_LOG_K,
        FieldInlineRelationId::FieldRegistersValEvaluation,
    )?;
    Ok(FieldInlineBatchLegs {
        fold: FieldInlineBytecodeFold {
            table,
            read_write_address: read_write_address.to_vec(),
            read_write_cycle: read_write_cycle.to_vec(),
            val_evaluation_address: val_evaluation_address.to_vec(),
            val_evaluation_cycle: val_evaluation_cycle.to_vec(),
            gammas: field_inline_stage_gamma_powers(&carried.bytecode_read_raf),
        },
        read_write_cycle: read_write_cycle.to_vec(),
        val_evaluation_cycle: val_evaluation_cycle.to_vec(),
    })
}

/// Splice the reduced `FieldRdInc` opening into the stage-6b Fiat-Shamir value
/// order: at its member position, after the ordinary increment reduction and
/// before the optional advice cycle phases (the spec's committed output row
/// order).
pub(super) fn splice_inc_values<F: JoltField>(
    values: &mut Vec<F>,
    claims: &Stage6bOutputClaims<F>,
) {
    values.extend(claims.field_registers_inc_claim_reduction.opening_values());
}
