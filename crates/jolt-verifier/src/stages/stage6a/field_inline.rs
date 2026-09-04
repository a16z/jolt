//! Stage 6a's field-inline seam: every FR-specific divergence of the stage-6a
//! verifier in one place — the FR appendage of the composed bytecode read-RAF
//! input claim (its wiring from the stage-1/4/5 outputs and its gamma-power
//! extension math) and the preprocessed side-table load the batch build
//! carries for the prover's kernel. The `BytecodeReadRafAddressPhase`
//! relation keeps only its appendage carriers (the `OnceLock`s + setters) and
//! the composed `input_claim` shell that adds [`input_claim_extension`] onto
//! the ordinary bind.

use jolt_claims::protocols::field_inline::geometry::bytecode::FIELD_INLINE_BYTECODE_STAGE1_FLAGS;
use jolt_claims::protocols::field_inline::geometry::spartan::outer_opening;
use jolt_claims::protocols::field_inline::FieldInlineVirtualPolynomial;
use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_claims::OutputClaims as _;
use jolt_field::JoltField;
use jolt_riscv::NUM_CIRCUIT_FLAGS;

use jolt_openings::CommitmentScheme;

use super::bytecode_read_raf::BytecodeReadRafAddressPhase;
use crate::preprocessing::ProgramPreprocessing;
use crate::stages::field_inline_bytecode::{
    convert_field_inline_bytecode, field_inline_stage_gamma_powers, required_field_inline_bytecode,
    FieldInlineBytecodeTable,
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
/// address prefix ‖ cycle). Construction-time data both fronts hold, carried
/// on the relation as an appendage (the same OnceLock idiom as the input
/// values below) via [`attach_bytecode_geometry`]; the verifier itself never
/// evaluates it in this stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldInlineBytecodeReadRafGeometry<F> {
    pub table: FieldInlineBytecodeTable,
    /// The stage-4 FR read-write opening point.
    pub read_write_point: Vec<F>,
    /// The stage-5 FR val-evaluation opening point.
    pub val_evaluation_point: Vec<F>,
}

/// Wire the FR kernel geometry from the preprocessed side table and the
/// stage-4/5 FR opening points, and supply it to the composed bytecode
/// read-RAF relation. Both fronts attach through this right after the batch
/// build (fail-closed: a kernel prepared without it rejects).
pub fn attach_bytecode_geometry<F: JoltField>(
    relation: &BytecodeReadRafAddressPhase<F>,
    table: FieldInlineBytecodeTable,
    stage4_points: &Stage4OutputPoints<F>,
    stage5_points: &Stage5OutputPoints<F>,
) -> Result<(), VerifierError> {
    relation.set_field_inline_geometry(FieldInlineBytecodeReadRafGeometry {
        table,
        read_write_point: stage4_points.field_registers_read_write_point().to_vec(),
        val_evaluation_point: stage5_points
            .field_registers_val_evaluation_point()
            .to_vec(),
    })
}

/// The field-inline opening values the extended address-phase input claim
/// folds under the extended stage-1/4/5 gamma powers (spec:
/// `field-inline-protocol.md`, "Stage 6 Composition"). The jolt symbolic input
/// `Expr` cannot name FR openings, so these ride the relation as an appendage
/// (the stage-1/2 OnceLock pattern) consumed by the composed `input_claim`.
#[derive(Clone, Debug, PartialEq, Eq)]
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
    let outer =
        stage1
            .field_inline_output_values
            .as_ref()
            .ok_or(VerifierError::MissingProofPayload {
                field: "stage1.field_inline_output_values",
            })?;
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

/// Wire the FR appendage from the stage-1/4/5 clear outputs and supply it to
/// the composed bytecode read-RAF relation (fail-closed on a missing stage-1
/// FR carrier). Both fronts attach through this before the input claim is
/// computed.
pub fn attach_bytecode_inputs<F: JoltField>(
    relation: &BytecodeReadRafAddressPhase<F>,
    stage1: &Stage1ClearOutput<F>,
    stage4: &Stage4OutputClaims<F>,
    stage5: &Stage5OutputClaims<F>,
) -> Result<(), VerifierError> {
    relation.set_field_inline_inputs(bytecode_read_raf_inputs(stage1, stage4, stage5)?)
}

/// The FR addend of the composed bytecode read-RAF input claim: the appendage
/// openings folded under the EXISTING stage-1/4/5 gamma power sequences — the
/// eight `FieldOpFlag` openings after the ordinary stage-1 powers,
/// `FieldRdWa`/`FieldRs1Ra`/`FieldRs2Ra` after the stage-4 powers, and the
/// stage-5 val-evaluation `FieldRdWa` after the stage-5 powers — each stage's
/// extension riding the same outer gamma power as its ordinary stage claim
/// (γ⁰/γ³/γ⁴). No new challenge draws: the powers extend (see
/// `field_inline_stage_gamma_powers`).
pub(super) fn input_claim_extension<F: JoltField>(
    field_inline: &FieldInlineBytecodeReadRafInputs<F>,
    challenges: &BytecodeReadRafAddressPhaseChallenges<F>,
) -> Result<F, VerifierError> {
    let missing_power = || VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::BytecodeReadRaf,
        reason: "field-inline stage gamma powers do not cover the appended FR terms".to_string(),
    };
    let gammas = field_inline_stage_gamma_powers(challenges);

    let stage1_extension = gammas
        .stage1
        .get(2 + NUM_CIRCUIT_FLAGS..)
        .filter(|powers| powers.len() == FIELD_INLINE_BYTECODE_STAGE1_FLAGS.len())
        .ok_or_else(missing_power)?
        .iter()
        .zip(field_inline.field_op_flags)
        .fold(F::zero(), |acc, (power, flag)| acc + *power * flag);
    let stage4_powers = gammas.stage4.get(3..6).ok_or_else(missing_power)?;
    let stage4_extension = stage4_powers
        .iter()
        .zip([
            field_inline.rd_wa_read_write,
            field_inline.rs1_ra,
            field_inline.rs2_ra,
        ])
        .fold(F::zero(), |acc, (power, opening)| acc + *power * opening);
    let stage5_extension =
        *gammas.stage5.last().ok_or_else(missing_power)? * field_inline.rd_wa_val_evaluation;

    let gamma = challenges.gamma;
    let gamma3 = gamma * gamma * gamma;
    let gamma4 = gamma3 * gamma;
    Ok(stage1_extension + gamma3 * stage4_extension + gamma4 * stage5_extension)
}
