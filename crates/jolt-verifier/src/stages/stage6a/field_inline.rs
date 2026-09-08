//! Typed extension inputs for the bytecode address phase and geometry
//! attached by the prover for its kernel. Verification evaluates the composed
//! symbolic claim without materializing that geometry.

use jolt_claims::protocols::field_inline::geometry::bytecode::FIELD_INLINE_BYTECODE_STAGE1_FLAGS;
use jolt_claims::protocols::field_inline::geometry::spartan::outer_opening;
use jolt_claims::protocols::field_inline::{
    FieldInlineOpeningId, FieldInlineRelationId, FieldInlineVirtualPolynomial,
};
use jolt_claims::InputClaims;
use jolt_claims::OutputClaims as _;
use jolt_field::JoltField;

use jolt_openings::CommitmentScheme;

use super::outputs::Stage6aSumchecks;
use crate::preprocessing::ProgramPreprocessing;
use crate::stages::field_inline_bytecode::{
    convert_field_inline_bytecode, required_field_inline_bytecode, FieldInlineBytecodeTable,
};
use crate::stages::stage1::Stage1ClearOutput;
use crate::stages::stage4::{Stage4OutputClaims, Stage4OutputPoints};
use crate::stages::stage5::{Stage5OutputClaims, Stage5OutputPoints};
use crate::VerifierError;

/// The converted field-inline bytecode side table from the verifier
/// preprocessing — the stage-6a counterpart of the stage-6b seam's helper
/// (both stages anchor the FR access selectors through the same
/// public/preprocessed table; committed-program preprocessing cannot supply
/// it and rejects here too).
pub fn preprocessed_bytecode_table<PCS: CommitmentScheme>(
    program: &ProgramPreprocessing<PCS>,
) -> Result<FieldInlineBytecodeTable, VerifierError> {
    convert_field_inline_bytecode(required_field_inline_bytecode(program)?)
}

/// The FR geometry the address-phase KERNEL folds over: the converted side
/// table plus the stage-4/5 FR opening points (`FIELD_REGISTERS_LOG_K`-var
/// address prefix ‖ cycle). Prover construction data, attached
/// to the relation via [`compose_bytecode_geometry`]; the verifier itself
/// never evaluates it in this stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldInlineBytecodeReadRafGeometry<F> {
    pub table: FieldInlineBytecodeTable,
    /// The stage-4 FR read-write opening point.
    pub read_write_point: Vec<F>,
    /// The stage-5 FR val-evaluation opening point.
    pub val_evaluation_point: Vec<F>,
}

/// Wire the FR kernel geometry from the preprocessed side table and the
/// stage-4/5 FR opening points, and compose it into the batch's bytecode
/// read-RAF relation. Both fronts compose through this right after the batch
/// build (fail-closed: a kernel prepared without it rejects).
pub fn compose_bytecode_geometry<F: JoltField>(
    sumchecks: Stage6aSumchecks<F>,
    table: FieldInlineBytecodeTable,
    stage4_points: &Stage4OutputPoints<F>,
    stage5_points: &Stage5OutputPoints<F>,
) -> Stage6aSumchecks<F> {
    Stage6aSumchecks {
        bytecode_read_raf: sumchecks.bytecode_read_raf.with_field_inline_geometry(
            FieldInlineBytecodeReadRafGeometry {
                table,
                read_write_point: stage4_points.field_registers_read_write_point().to_vec(),
                val_evaluation_point: stage5_points
                    .field_registers_val_evaluation_point()
                    .to_vec(),
            },
        ),
        ..sumchecks
    }
}

/// The field-inline opening values the extended address-phase input claim
/// folds under the extended stage-1/4/5 gamma powers (spec:
/// `field-inline-protocol.md`, "Stage 6 Composition"). The jolt symbolic input
/// `Expr` cannot name FR openings, so these are composed into the relation
/// (the stage-1/2 pattern) and consumed by the composed `input_claim`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FieldInlineBytecodeReadRafInputs<F> {
    /// The eight `FieldOpFlag` openings from the stage-1 FR Spartan-outer
    /// carrier, in `FIELD_INLINE_BYTECODE_STAGE1_FLAGS` order.
    pub field_op_flags: [F; 8],
    /// `FieldRdWa` / `FieldRs1Ra` / `FieldRs2Ra` from the stage-4 FR
    /// read-write checking.
    pub rd_wa_read_write: F,
    pub rs1_ra: F,
    pub rs2_ra: F,
    /// `FieldRdWa` from the stage-5 FR val evaluation.
    pub rd_wa_val_evaluation: F,
}

/// Wire the FR opening values the extended bytecode read-RAF input claim
/// consumes from the upstream clear outputs. Fail-closed: an FR-on proof
/// whose stage-1 carrier lacks the FR payload cannot feed the extension.
pub fn bytecode_read_raf_inputs<F: JoltField>(
    stage1: &Stage1ClearOutput<F>,
    stage4: &Stage4OutputClaims<F>,
    stage5: &Stage5OutputClaims<F>,
) -> Result<FieldInlineBytecodeReadRafInputs<F>, VerifierError> {
    let outer = &stage1.output_values.outer_remainder.field_inline;
    let mut field_op_flags = [F::zero(); FIELD_INLINE_BYTECODE_STAGE1_FLAGS.len()];
    for (slot, flag) in field_op_flags
        .iter_mut()
        .zip(FIELD_INLINE_BYTECODE_STAGE1_FLAGS)
    {
        let id = outer_opening(FieldInlineVirtualPolynomial::FieldOpFlag(flag));
        *slot = outer
            .resolve_output(&id)
            .ok_or(VerifierError::MissingOpeningClaim { id: id.into() })?;
    }
    let read_write = &stage4.field_registers_read_write;
    Ok(FieldInlineBytecodeReadRafInputs {
        field_op_flags,
        rd_wa_read_write: read_write.rd_wa,
        rs1_ra: read_write.rs1_ra,
        rs2_ra: read_write.rs2_ra,
        rd_wa_val_evaluation: stage5.field_registers_val_evaluation.rd_wa,
    })
}

impl<F> FieldInlineBytecodeReadRafInputs<F> {
    fn opening_ids() -> impl Iterator<Item = FieldInlineOpeningId> {
        FIELD_INLINE_BYTECODE_STAGE1_FLAGS
            .into_iter()
            .map(|flag| outer_opening(FieldInlineVirtualPolynomial::FieldOpFlag(flag)))
            .chain([
                field_access_opening(FieldInlineVirtualPolynomial::FieldRdWa),
                field_access_opening(FieldInlineVirtualPolynomial::FieldRs1Ra),
                field_access_opening(FieldInlineVirtualPolynomial::FieldRs2Ra),
                FieldInlineOpeningId::virtual_polynomial(
                    FieldInlineVirtualPolynomial::FieldRdWa,
                    FieldInlineRelationId::FieldRegistersValEvaluation,
                ),
            ])
    }
}

impl<F: JoltField> InputClaims<F, FieldInlineOpeningId> for FieldInlineBytecodeReadRafInputs<F> {
    fn canonical_order(&self) -> Vec<FieldInlineOpeningId> {
        Self::opening_ids().collect()
    }

    fn resolve_input(&self, id: &FieldInlineOpeningId) -> Option<F> {
        Self::opening_ids()
            .zip(self.field_op_flags.into_iter().chain([
                self.rd_wa_read_write,
                self.rs1_ra,
                self.rs2_ra,
                self.rd_wa_val_evaluation,
            ]))
            .find_map(|(candidate, value)| (candidate == *id).then_some(value))
    }
}

fn field_access_opening(polynomial: FieldInlineVirtualPolynomial) -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        polynomial,
        FieldInlineRelationId::FieldRegistersReadWriteChecking,
    )
}
