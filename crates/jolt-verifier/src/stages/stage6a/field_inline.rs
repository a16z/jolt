//! Typed extension inputs for the bytecode address phase and geometry
//! attached by the prover for its kernel. Verification evaluates the composed
//! symbolic claim without materializing that geometry.

use jolt_claims::protocols::field_inline::{
    FieldInlineOpeningId, FieldInlineRelationId, FieldInlineVirtualPolynomial,
};
use jolt_claims::InputClaims;
use jolt_field::JoltField;

use super::outputs::Stage6aSumchecks;
use crate::stages::stage4::{Stage4OutputClaims, Stage4OutputPoints};
use crate::stages::stage5::{Stage5OutputClaims, Stage5OutputPoints};

/// The field-inline geometry the address-phase KERNEL folds over: the stage-4/5 field-inline opening points (`FIELD_REGISTERS_LOG_K`-var address prefix
/// ‖ cycle). Prover construction data, attached to the relation via
/// [`compose_bytecode_geometry`]; the verifier itself never evaluates it in this stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldInlineBytecodeReadRafGeometry<F> {
    /// The stage-4 field-register read-write opening point.
    pub read_write_point: Vec<F>,
    /// The stage-5 field-register value-evaluation opening point.
    pub val_evaluation_point: Vec<F>,
}

/// Wire the field-inline kernel geometry from the stage-4/5
/// field-inline opening points, and compose it into the batch's bytecode read-RAF relation.
/// Both fronts compose through this right after the batch build (fail-closed: a kernel
/// prepared without it rejects).
pub fn compose_bytecode_geometry<F: JoltField>(
    sumchecks: Stage6aSumchecks<F>,
    stage4_points: &Stage4OutputPoints<F>,
    stage5_points: &Stage5OutputPoints<F>,
) -> Stage6aSumchecks<F> {
    Stage6aSumchecks {
        bytecode_read_raf: sumchecks.bytecode_read_raf.with_field_inline_geometry(
            FieldInlineBytecodeReadRafGeometry {
                read_write_point: stage4_points.field_registers_read_write_point().to_vec(),
                val_evaluation_point: stage5_points
                    .field_registers_val_evaluation_point()
                    .to_vec(),
            },
        ),
        ..sumchecks
    }
}

/// The field-inline opening values the extended address-phase input claim folds under the
/// extended stage-4/5 gamma powers (spec: `field-inline-protocol.md`, "Stage 6
/// Composition"). The jolt symbolic input `Expr` cannot name field-inline openings, so these
/// are composed into the relation (the stage-1/2 pattern) and consumed by the composed
/// `input_claim`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FieldInlineBytecodeReadRafInputs<F> {
    /// `FieldRdWa` / `FieldRs1Ra` / `FieldRs2Ra` from the stage-4 field-inline read-write
    /// checking.
    pub rd_wa_read_write: F,
    pub rs1_ra: F,
    pub rs2_ra: F,
    /// `FieldRdWa` from the stage-5 field-register value evaluation.
    pub rd_wa_val_evaluation: F,
}

/// Wire the field-inline opening values the extended bytecode read-RAF input claim consumes
/// from the upstream clear outputs, from the field-register relations.
pub fn field_inline_bytecode_read_raf_address_phase_input_values_from_upstream<F: JoltField>(
    stage4: &Stage4OutputClaims<F>,
    stage5: &Stage5OutputClaims<F>,
) -> FieldInlineBytecodeReadRafInputs<F> {
    let read_write = &stage4.field_registers_read_write;
    FieldInlineBytecodeReadRafInputs {
        rd_wa_read_write: read_write.rd_wa,
        rs1_ra: read_write.rs1_ra,
        rs2_ra: read_write.rs2_ra,
        rd_wa_val_evaluation: stage5.field_registers_val_evaluation.rd_wa,
    }
}

impl<F> FieldInlineBytecodeReadRafInputs<F> {
    fn opening_ids() -> impl Iterator<Item = FieldInlineOpeningId> {
        [
            field_access_opening(FieldInlineVirtualPolynomial::FieldRdWa),
            field_access_opening(FieldInlineVirtualPolynomial::FieldRs1Ra),
            field_access_opening(FieldInlineVirtualPolynomial::FieldRs2Ra),
            FieldInlineOpeningId::virtual_polynomial(
                FieldInlineVirtualPolynomial::FieldRdWa,
                FieldInlineRelationId::FieldRegistersValEvaluation,
            ),
        ]
        .into_iter()
    }
}

impl<F: JoltField> InputClaims<F, FieldInlineOpeningId> for FieldInlineBytecodeReadRafInputs<F> {
    fn canonical_order(&self) -> Vec<FieldInlineOpeningId> {
        Self::opening_ids().collect()
    }

    fn resolve_input(&self, id: &FieldInlineOpeningId) -> Option<F> {
        Self::opening_ids()
            .zip([
                self.rd_wa_read_write,
                self.rs1_ra,
                self.rs2_ra,
                self.rd_wa_val_evaluation,
            ])
            .find_map(|(candidate, value)| (candidate == *id).then_some(value))
    }
}

fn field_access_opening(polynomial: FieldInlineVirtualPolynomial) -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        polynomial,
        FieldInlineRelationId::FieldRegistersReadWriteChecking,
    )
}
