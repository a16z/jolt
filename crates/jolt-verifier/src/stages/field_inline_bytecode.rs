//! Field-inline bytecode side-table plumbing shared by stages 6a and 6b.
//!
//! The verifier preprocessing carries the field-inline bytecode facts as
//! `jolt_program::field_inline::FieldInlineBytecodeMetadata` (op + FR operand
//! slots per row, the S7 program-boundary shape); the bytecode read-RAF
//! formulas consume the jolt-claims geometry shape
//! ([`FieldInlineBytecodeRow`]: per-op flags + operands). This module owns the
//! required-presence check, the conversion between the two shapes, and the
//! FR-extended per-stage gamma power expansion the stage-6 folds share.

use jolt_claims::protocols::field_inline::geometry::bytecode::{
    validate_bytecode_rows, FieldInlineBytecodeFlags, FieldInlineBytecodeOperands,
    FieldInlineBytecodeRow, FIELD_INLINE_BYTECODE_STAGE1_GAMMA_COUNT,
    FIELD_INLINE_BYTECODE_STAGE4_GAMMA_COUNT, FIELD_INLINE_BYTECODE_STAGE5_EXTRA_GAMMAS,
};
use jolt_claims::protocols::field_inline::{FieldInlineRelationId, FIELD_REGISTERS_LOG_K};
use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;
use jolt_field::Field;
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use jolt_openings::CommitmentScheme;
use jolt_program::field_inline::{
    FieldInlineBytecodeMetadata, FieldInlineBytecodeRow as ProgramFieldInlineBytecodeRow,
};
use jolt_riscv::{
    field_inline_operand_shape, FieldInlineOp, FieldInlineXRegisterRole, FieldRegister,
    JoltInstructionRow,
};

use crate::preprocessing::ProgramPreprocessing;
use crate::VerifierError;

/// The field-inline bytecode side table in the jolt-claims geometry shape,
/// converted from the preprocessing metadata. `rows` is index-parallel to the
/// padded ordinary bytecode table (one row per bytecode address).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldInlineBytecodeTable {
    pub rows: Vec<FieldInlineBytecodeRow>,
    pub field_register_log_k: usize,
}

/// The field-inline bytecode side table from the verifier preprocessing,
/// required fail-closed: an FR-on verifier without the metadata (committed
/// program mode, or a full program preprocessed without FR support) cannot
/// anchor the FR access selectors to the bytecode and must reject.
pub fn required_field_inline_bytecode<PCS: CommitmentScheme>(
    program: &ProgramPreprocessing<PCS>,
) -> Result<&FieldInlineBytecodeMetadata, VerifierError> {
    program
        .as_full()
        .and_then(|full| full.bytecode.field_inline.as_ref())
        .ok_or(VerifierError::MissingPreprocessingPayload {
            field: "program.bytecode.field_inline",
        })
}

/// Convert the program-boundary side table (`active`/`op`/operand slots) into
/// the geometry shape the read-RAF formulas consume (per-op flags +
/// operands). The bridge x-register and immediate are S7 payload for the
/// Spartan field constraints, not bytecode read-RAF terms, so they drop here.
/// The converted rows are re-validated through the geometry validator so a
/// cross-crate shape drift fails loudly at conversion.
pub fn convert_field_inline_bytecode(
    metadata: &FieldInlineBytecodeMetadata,
) -> Result<FieldInlineBytecodeTable, VerifierError> {
    let field_register_log_k = usize::from(metadata.field_register_log_k);
    if field_register_log_k != FIELD_REGISTERS_LOG_K {
        return Err(conversion_failed(format!(
            "field-register log_k mismatch: preprocessing carries {field_register_log_k}, the \
             protocol geometry expects {FIELD_REGISTERS_LOG_K}"
        )));
    }
    let rows = metadata
        .rows
        .iter()
        .enumerate()
        .map(|(index, row)| convert_row(index, row))
        .collect::<Result<Vec<_>, _>>()?;
    validate_bytecode_rows(&rows, rows.len(), field_register_log_k).map_err(conversion_failed)?;
    Ok(FieldInlineBytecodeTable {
        rows,
        field_register_log_k,
    })
}

fn convert_row(
    index: usize,
    row: &ProgramFieldInlineBytecodeRow,
) -> Result<FieldInlineBytecodeRow, VerifierError> {
    if !row.active {
        // The program-side validator rejects inactive rows carrying data, so
        // the default (all-false, no operands) row is faithful.
        return Ok(FieldInlineBytecodeRow::default());
    }
    let op = row.op.ok_or_else(|| {
        conversion_failed(format!(
            "field-inline bytecode row {index} is active but carries no op"
        ))
    })?;
    Ok(FieldInlineBytecodeRow {
        operands: FieldInlineBytecodeOperands {
            rd: row.rd.map(FieldRegister::index),
            rs1: row.rs1.map(FieldRegister::index),
            rs2: row.rs2.map(FieldRegister::index),
        },
        flags: flags_for_op(op),
    })
}

fn flags_for_op(op: FieldInlineOp) -> FieldInlineBytecodeFlags {
    let mut flags = FieldInlineBytecodeFlags::default();
    match op {
        FieldInlineOp::Add => flags.add = true,
        FieldInlineOp::Sub => flags.sub = true,
        FieldInlineOp::Mul => flags.mul = true,
        FieldInlineOp::Inv => flags.inv = true,
        FieldInlineOp::AssertEq => flags.assert_eq = true,
        FieldInlineOp::LoadFromX => flags.load_from_x = true,
        FieldInlineOp::StoreToX => flags.store_to_x = true,
        FieldInlineOp::LoadImm => flags.load_imm = true,
    }
    flags
}

fn conversion_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimSumcheckFailed {
        stage: format!("{:?}", FieldInlineRelationId::FieldRegistersSpartanOuter),
        reason: format!(
            "field-inline bytecode side-table conversion failed: {}",
            reason.to_string()
        ),
    }
}

/// The ordinary bytecode rows with each field-op row's FR-operand slots
/// blanked, as the jolt read-RAF register folds must see them: a field-op
/// row's rd/rs1/rs2 carry FR register slots for the side table, and the spec
/// suppresses its ordinary x-register accesses ("Trace Semantics") — only a
/// bridge x-register role keeps its slot (`LoadFromX` reads x-rs1, `StoreToX`
/// writes x-rd). Mirrors
/// `jolt_program::field_inline::FieldInlineBytecodeRow::from_instruction`'s
/// slot classification, so the ordinary and FR folds partition each row's
/// operands exactly. The jolt protocol module cannot express this (protocol
/// modules are import-disjoint), so the composition layer masks the rows
/// before every jolt-side `read_raf_stage_values` fold.
pub fn suppress_field_operand_slots(bytecode: &[JoltInstructionRow]) -> Vec<JoltInstructionRow> {
    bytecode
        .iter()
        .map(|row| {
            let Some(shape) = field_inline_operand_shape(row.instruction_kind) else {
                return *row;
            };
            let mut masked = *row;
            masked.operands.rd = match shape.bridge_x_register_role {
                Some(FieldInlineXRegisterRole::WriteRd) => row.operands.rd,
                _ => None,
            };
            masked.operands.rs1 = match shape.bridge_x_register_role {
                Some(FieldInlineXRegisterRole::ReadRs1) => row.operands.rs1,
                _ => None,
            };
            // No field op reads an ordinary rs2.
            masked.operands.rs2 = None;
            masked
        })
        .collect()
}

/// The FR-extended per-stage gamma power vectors for the bytecode read-RAF
/// folds. Extends the ordinary stage-1/4/5 power sequences (the same drawn
/// scalars, more powers — no new Fiat-Shamir draws) to the field-inline
/// counts; stages 2/3 gain no FR terms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldInlineBytecodeStageGammas<F> {
    pub stage1: Vec<F>,
    pub stage4: Vec<F>,
    pub stage5: Vec<F>,
}

/// Expand the carried stage-1/4/5 scalars into the FR-extended power vectors
/// (`[1, γ, γ², …]`, sized by the field-inline gamma counts).
pub fn field_inline_stage_gamma_powers<F: Field>(
    challenges: &BytecodeReadRafAddressPhaseChallenges<F>,
) -> FieldInlineBytecodeStageGammas<F> {
    FieldInlineBytecodeStageGammas {
        stage1: gamma_powers(
            challenges.stage1_gamma,
            FIELD_INLINE_BYTECODE_STAGE1_GAMMA_COUNT,
        ),
        stage4: gamma_powers(
            challenges.stage4_gamma,
            FIELD_INLINE_BYTECODE_STAGE4_GAMMA_COUNT,
        ),
        stage5: gamma_powers(challenges.stage5_gamma, field_inline_stage5_gamma_count()),
    }
}

/// The FR-extended stage-5 gamma count: the ordinary count plus the appended
/// `FieldRdWa@FieldRegistersValEvaluation` power.
pub const fn field_inline_stage5_gamma_count() -> usize {
    2 + LookupTableKind::<RISCV_XLEN>::COUNT + FIELD_INLINE_BYTECODE_STAGE5_EXTRA_GAMMAS
}

fn gamma_powers<F: Field>(gamma: F, len: usize) -> Vec<F> {
    let mut powers = Vec::with_capacity(len);
    let mut power = F::one();
    for _ in 0..len {
        powers.push(power);
        power *= gamma;
    }
    powers
}

/// The construction-time inputs of the stage-6b full-program bytecode
/// read-RAF field-inline public fold: the converted side table, the FR
/// register address prefixes and cycle sub-points of the stage-4/5 FR
/// openings, and the FR-extended gamma powers. The relation evaluates the FR
/// public stage values from these at `expected_output` time (clear only).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldInlineBytecodeFold<F> {
    pub table: FieldInlineBytecodeTable,
    /// The `FIELD_REGISTERS_LOG_K`-variable address prefix of the stage-4 FR
    /// read-write opening point.
    pub read_write_address: Vec<F>,
    /// The cycle suffix of the stage-4 FR read-write opening point.
    pub read_write_cycle: Vec<F>,
    /// The `FIELD_REGISTERS_LOG_K`-variable address prefix of the stage-5 FR
    /// val-evaluation opening point.
    pub val_evaluation_address: Vec<F>,
    /// The cycle suffix of the stage-5 FR val-evaluation opening point.
    pub val_evaluation_cycle: Vec<F>,
    pub gammas: FieldInlineBytecodeStageGammas<F>,
}

/// [`crate::stages::stage6_checked_split`] for FR opening points, attributing
/// the failure to the field-inline relation consuming the split.
pub(crate) fn field_inline_checked_split<'a, F: Field>(
    label: &'static str,
    point: &'a [F],
    split_at: usize,
    stage: FieldInlineRelationId,
) -> Result<(&'a [F], &'a [F]), VerifierError> {
    if point.len() < split_at {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: format!("{stage:?}"),
            reason: format!(
                "{label} has {} variables, expected at least {split_at}",
                point.len()
            ),
        });
    }
    Ok(point.split_at(split_at))
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_claims::protocols::field_inline::geometry::bytecode::FIELD_INLINE_BYTECODE_STAGE1_FLAGS;
    use jolt_claims::protocols::field_inline::FieldInlineOpFlag;
    use jolt_field::{Fr, Ring};
    use jolt_program::field_inline::{
        FieldEncodedValue, FieldInlineBytecodeRow as ProgramRow, FieldValueEncoding,
    };
    use jolt_riscv::{FIELD_REGISTER_LOG_K, NUM_CIRCUIT_FLAGS};
    use jolt_transcript::{Blake2bTranscript, Transcript};

    fn register(index: u8) -> Option<FieldRegister> {
        Some(FieldRegister::new(index).unwrap())
    }

    /// One active program row per op, operands shaped per the op's operand
    /// layout (the jolt-program metadata validator's shape).
    fn program_row(op: FieldInlineOp) -> ProgramRow {
        let (rd, rs1, rs2) = match op {
            FieldInlineOp::Add | FieldInlineOp::Sub | FieldInlineOp::Mul => {
                (register(1), register(2), register(3))
            }
            FieldInlineOp::Inv => (register(4), register(5), None),
            FieldInlineOp::AssertEq => (None, register(6), register(7)),
            FieldInlineOp::LoadFromX => (register(8), None, None),
            FieldInlineOp::StoreToX => (None, register(9), None),
            FieldInlineOp::LoadImm => (register(10), None, None),
        };
        let bridge_x_register =
            matches!(op, FieldInlineOp::LoadFromX | FieldInlineOp::StoreToX).then_some(11);
        let immediate =
            matches!(op, FieldInlineOp::LoadImm).then(|| FieldEncodedValue::from_u64(42));
        ProgramRow {
            active: true,
            op: Some(op),
            rs1,
            rs2,
            rd,
            bridge_x_register,
            immediate,
        }
    }

    const ALL_OPS: [FieldInlineOp; 8] = [
        FieldInlineOp::Add,
        FieldInlineOp::Sub,
        FieldInlineOp::Mul,
        FieldInlineOp::Inv,
        FieldInlineOp::AssertEq,
        FieldInlineOp::LoadFromX,
        FieldInlineOp::StoreToX,
        FieldInlineOp::LoadImm,
    ];

    fn claims_flag_for_op(op: FieldInlineOp) -> FieldInlineOpFlag {
        match op {
            FieldInlineOp::Add => FieldInlineOpFlag::Add,
            FieldInlineOp::Sub => FieldInlineOpFlag::Sub,
            FieldInlineOp::Mul => FieldInlineOpFlag::Mul,
            FieldInlineOp::Inv => FieldInlineOpFlag::Inv,
            FieldInlineOp::AssertEq => FieldInlineOpFlag::AssertEq,
            FieldInlineOp::LoadFromX => FieldInlineOpFlag::LoadFromX,
            FieldInlineOp::StoreToX => FieldInlineOpFlag::StoreToX,
            FieldInlineOp::LoadImm => FieldInlineOpFlag::LoadImm,
        }
    }

    fn metadata_of(rows: Vec<ProgramRow>) -> FieldInlineBytecodeMetadata {
        FieldInlineBytecodeMetadata {
            rows,
            field_register_log_k: FIELD_REGISTER_LOG_K,
            value_encoding: FieldValueEncoding::BN254_SCALAR_CANONICAL,
            profile_fingerprint: 0,
        }
    }

    /// Every op converts to exactly its geometry flag, and the FR operand
    /// slots carry over index-for-index. The bridge x-register and immediate
    /// are Spartan-constraint payload and drop.
    #[test]
    fn conversion_maps_each_op_to_its_flag_and_carries_operands() {
        let rows: Vec<ProgramRow> = ALL_OPS.into_iter().map(program_row).collect();
        let table = convert_field_inline_bytecode(&metadata_of(rows.clone())).unwrap();

        assert_eq!(table.field_register_log_k, FIELD_REGISTERS_LOG_K);
        assert_eq!(table.rows.len(), rows.len());
        for (converted, (source, op)) in table.rows.iter().zip(rows.iter().zip(ALL_OPS)) {
            let expected_flag = claims_flag_for_op(op);
            for flag in FIELD_INLINE_BYTECODE_STAGE1_FLAGS {
                assert_eq!(
                    converted.flags.get(flag),
                    flag == expected_flag,
                    "op {op:?} must set exactly the {expected_flag:?} flag"
                );
            }
            assert_eq!(converted.operands.rd, source.rd.map(FieldRegister::index));
            assert_eq!(converted.operands.rs1, source.rs1.map(FieldRegister::index));
            assert_eq!(converted.operands.rs2, source.rs2.map(FieldRegister::index));
        }
    }

    /// Inactive program rows convert to the all-false, operand-free geometry
    /// row (the geometry validator rejects anything else for inactive rows).
    #[test]
    fn conversion_maps_inactive_rows_to_all_false() {
        let table =
            convert_field_inline_bytecode(&metadata_of(vec![ProgramRow::default(); 4])).unwrap();
        for row in &table.rows {
            assert_eq!(*row, FieldInlineBytecodeRow::default());
            assert_eq!(row.flags.active_count(), 0);
            assert_eq!(row.operands, FieldInlineBytecodeOperands::default());
        }
    }

    /// A metadata carrying a foreign field-register width is rejected rather
    /// than silently re-shaping the register eq tables.
    #[test]
    fn conversion_rejects_mismatched_field_register_log_k() {
        let mut metadata = metadata_of(Vec::new());
        metadata.field_register_log_k += 1;
        assert!(matches!(
            convert_field_inline_bytecode(&metadata),
            Err(VerifierError::StageClaimSumcheckFailed { .. })
        ));
    }

    /// The side table is a hard requirement: full-program preprocessing
    /// without the FR metadata rejects with the precise payload name.
    #[test]
    fn missing_side_table_is_a_verifier_error() {
        use common::jolt_device::{JoltDevice, MemoryConfig};
        use jolt_dory::DoryScheme;
        use jolt_program::preprocess::{JoltProgramPreprocessing, RAMPreprocessing};
        use std::sync::Arc;

        let program: ProgramPreprocessing<DoryScheme> =
            ProgramPreprocessing::Full(Arc::new(JoltProgramPreprocessing {
                bytecode: Default::default(),
                ram: RAMPreprocessing::preprocess(Vec::new()),
                memory_layout: JoltDevice::new(&MemoryConfig {
                    program_size: Some(1024),
                    ..Default::default()
                })
                .memory_layout,
                max_padded_trace_length: 1 << 10,
            }));
        assert!(matches!(
            required_field_inline_bytecode(&program),
            Err(VerifierError::MissingPreprocessingPayload {
                field: "program.bytecode.field_inline"
            })
        ));
    }

    /// The FR-extended power vectors extend the ordinary draws: each ordinary
    /// stage power sequence is a strict prefix (same squeezed scalar, more
    /// powers — no new Fiat-Shamir draws).
    #[test]
    fn extended_gamma_powers_extend_the_ordinary_power_sequences() {
        let mut transcript = Blake2bTranscript::new(b"fr-gamma-powers");
        let challenges = BytecodeReadRafAddressPhaseChallenges::<Fr> {
            gamma: transcript.challenge_scalar(),
            stage1_gamma: transcript.challenge_scalar(),
            stage2_gamma: transcript.challenge_scalar(),
            stage3_gamma: transcript.challenge_scalar(),
            stage4_gamma: transcript.challenge_scalar(),
            stage5_gamma: transcript.challenge_scalar(),
        };
        let ordinary = challenges.stage_gamma_powers();
        let extended = field_inline_stage_gamma_powers(&challenges);

        assert_eq!(
            extended.stage1.len(),
            2 + NUM_CIRCUIT_FLAGS + FIELD_INLINE_BYTECODE_STAGE1_FLAGS.len()
        );
        assert_eq!(extended.stage4.len(), 6);
        assert_eq!(
            extended.stage5.len(),
            2 + LookupTableKind::<RISCV_XLEN>::COUNT + 1
        );
        for ((extended, ordinary_index), gamma) in [
            (&extended.stage1, 0usize),
            (&extended.stage4, 3),
            (&extended.stage5, 4),
        ]
        .into_iter()
        .zip([
            challenges.stage1_gamma,
            challenges.stage4_gamma,
            challenges.stage5_gamma,
        ]) {
            let ordinary = ordinary.get(ordinary_index).unwrap();
            assert_eq!(extended.get(..ordinary.len()), Some(ordinary.as_slice()));
            // The extension continues the same power recurrence.
            let mut power = Fr::from_u64(1);
            for extended_power in extended {
                assert_eq!(*extended_power, power);
                power *= gamma;
            }
        }
    }
}
