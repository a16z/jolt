use jolt_field::Field;
use jolt_lookup_tables::{LookupTableKind, XLEN};
use jolt_poly::EqPolynomial;
use jolt_riscv::NUM_CIRCUIT_FLAGS;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::super::FieldInlineOpFlag;
use crate::protocols::jolt::formulas::dimensions::JoltFormulaPointError;

pub const FIELD_INLINE_BYTECODE_STAGE1_FLAGS: [FieldInlineOpFlag; 8] = [
    FieldInlineOpFlag::Add,
    FieldInlineOpFlag::Sub,
    FieldInlineOpFlag::Mul,
    FieldInlineOpFlag::Inv,
    FieldInlineOpFlag::AssertEq,
    FieldInlineOpFlag::LoadFromX,
    FieldInlineOpFlag::StoreToX,
    FieldInlineOpFlag::LoadImm,
];

pub const FIELD_INLINE_BYTECODE_STAGE1_GAMMA_COUNT: usize =
    2 + NUM_CIRCUIT_FLAGS + FIELD_INLINE_BYTECODE_STAGE1_FLAGS.len();
pub const FIELD_INLINE_BYTECODE_STAGE4_GAMMA_COUNT: usize = 6;
pub const FIELD_INLINE_BYTECODE_STAGE5_EXTRA_GAMMAS: usize = 1;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldInlineBytecodeRow {
    pub operands: FieldInlineBytecodeOperands,
    pub flags: FieldInlineBytecodeFlags,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldInlineBytecodeOperands {
    pub rd: Option<u8>,
    pub rs1: Option<u8>,
    pub rs2: Option<u8>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldInlineBytecodeFlags {
    pub add: bool,
    pub sub: bool,
    pub mul: bool,
    pub inv: bool,
    pub assert_eq: bool,
    pub load_from_x: bool,
    pub store_to_x: bool,
    pub load_imm: bool,
}

impl FieldInlineBytecodeFlags {
    pub const fn get(self, flag: FieldInlineOpFlag) -> bool {
        match flag {
            FieldInlineOpFlag::Add => self.add,
            FieldInlineOpFlag::Sub => self.sub,
            FieldInlineOpFlag::Mul => self.mul,
            FieldInlineOpFlag::Inv => self.inv,
            FieldInlineOpFlag::AssertEq => self.assert_eq,
            FieldInlineOpFlag::LoadFromX => self.load_from_x,
            FieldInlineOpFlag::StoreToX => self.store_to_x,
            FieldInlineOpFlag::LoadImm => self.load_imm,
        }
    }

    pub fn active_count(self) -> usize {
        FIELD_INLINE_BYTECODE_STAGE1_FLAGS
            .into_iter()
            .filter(|flag| self.get(*flag))
            .count()
    }

    pub fn active_flag(self) -> Option<FieldInlineOpFlag> {
        if self.active_count() != 1 {
            return None;
        }
        FIELD_INLINE_BYTECODE_STAGE1_FLAGS
            .into_iter()
            .find(|flag| self.get(*flag))
    }

    pub fn transcript_bits(self) -> u8 {
        FIELD_INLINE_BYTECODE_STAGE1_FLAGS
            .into_iter()
            .enumerate()
            .fold(0, |bits, (index, flag)| {
                bits | (u8::from(self.get(flag)) << index)
            })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum FieldInlineBytecodeValidationError {
    #[error("field-register log_k {log_k} is too large for this target")]
    InvalidFieldRegisterLogK { log_k: usize },
    #[error("field-inline bytecode length mismatch: expected {expected}, got {got}")]
    LengthMismatch { expected: usize, got: usize },
    #[error("inactive field-inline bytecode row {row} carries operands")]
    InactiveRowHasOperands { row: usize },
    #[error("field-inline bytecode row {row} has {active_count} active flags")]
    InvalidActiveFlagCount { row: usize, active_count: usize },
    #[error("field-inline bytecode row {row} flag {flag:?}: {reason}")]
    InvalidOperands {
        row: usize,
        flag: FieldInlineOpFlag,
        reason: &'static str,
    },
    #[error(
        "field-inline bytecode row {row} {operand} register {register} is outside field register domain {field_register_count}"
    )]
    RegisterOutOfRange {
        row: usize,
        operand: &'static str,
        register: u8,
        field_register_count: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum FieldInlineBytecodeReadRafError {
    #[error("{0}")]
    Point(#[from] JoltFormulaPointError),
    #[error("{0}")]
    Validation(#[from] FieldInlineBytecodeValidationError),
}

pub fn validate_bytecode_rows(
    bytecode: &[FieldInlineBytecodeRow],
    expected_len: usize,
    field_register_log_k: usize,
) -> Result<(), FieldInlineBytecodeValidationError> {
    if bytecode.len() != expected_len {
        return Err(FieldInlineBytecodeValidationError::LengthMismatch {
            expected: expected_len,
            got: bytecode.len(),
        });
    }
    let shift = u32::try_from(field_register_log_k).map_err(|_| {
        FieldInlineBytecodeValidationError::InvalidFieldRegisterLogK {
            log_k: field_register_log_k,
        }
    })?;
    let field_register_count = 1usize.checked_shl(shift).ok_or(
        FieldInlineBytecodeValidationError::InvalidFieldRegisterLogK {
            log_k: field_register_log_k,
        },
    )?;

    for (row, entry) in bytecode.iter().enumerate() {
        validate_bytecode_row(row, entry, field_register_count)?;
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldInlineBytecodeReadRafPublicValues<F: Field> {
    pub stage_values: [F; 5],
}

pub struct FieldInlineBytecodeReadRafEvaluationInputs<'a, F> {
    pub bytecode: &'a [FieldInlineBytecodeRow],
    pub field_register_log_k: usize,
    pub r_address: &'a [F],
    pub r_cycle: &'a [F],
    pub stage1_cycle_point: &'a [F],
    pub field_register_read_write_point: &'a [F],
    pub field_register_read_write_cycle_point: &'a [F],
    pub field_register_val_evaluation_point: &'a [F],
    pub field_register_val_evaluation_cycle_point: &'a [F],
    pub stage1_gammas: &'a [F],
    pub stage4_gammas: &'a [F],
    pub stage5_gammas: &'a [F],
}

pub fn read_raf_public_values<F>(
    inputs: FieldInlineBytecodeReadRafEvaluationInputs<'_, F>,
) -> Result<FieldInlineBytecodeReadRafPublicValues<F>, FieldInlineBytecodeReadRafError>
where
    F: Field,
{
    require_len(
        inputs.stage1_gammas,
        FIELD_INLINE_BYTECODE_STAGE1_GAMMA_COUNT,
    )?;
    require_len(
        inputs.stage4_gammas,
        FIELD_INLINE_BYTECODE_STAGE4_GAMMA_COUNT,
    )?;
    require_len(
        inputs.stage5_gammas,
        2 + LookupTableKind::<XLEN>::COUNT + FIELD_INLINE_BYTECODE_STAGE5_EXTRA_GAMMAS,
    )?;

    let expected_domain = 1usize << inputs.r_address.len();
    if inputs.bytecode.len() != expected_domain {
        return Err(FieldInlineBytecodeReadRafError::Point(
            JoltFormulaPointError::EvaluationDomainLengthMismatch {
                expected: expected_domain,
                got: inputs.bytecode.len(),
            },
        ));
    }
    validate_bytecode_rows(
        inputs.bytecode,
        expected_domain,
        inputs.field_register_log_k,
    )?;

    let address_eq_evals = EqPolynomial::<F>::evals(inputs.r_address, None);
    let row_values = read_raf_stage_values(FieldInlineBytecodeReadRafStageValueInputs {
        bytecode: inputs.bytecode,
        field_register_read_write_point: inputs.field_register_read_write_point,
        field_register_val_evaluation_point: inputs.field_register_val_evaluation_point,
        stage1_gammas: inputs.stage1_gammas,
        stage4_gammas: inputs.stage4_gammas,
        stage5_gammas: inputs.stage5_gammas,
    });

    let mut stage_values = [F::zero(); 5];
    for (row_values, eq_address) in row_values.into_iter().zip(address_eq_evals) {
        for (stage_value, row_value) in stage_values.iter_mut().zip(row_values) {
            *stage_value += row_value * eq_address;
        }
    }

    stage_values[0] *= EqPolynomial::<F>::mle(inputs.stage1_cycle_point, inputs.r_cycle);
    stage_values[3] *=
        EqPolynomial::<F>::mle(inputs.field_register_read_write_cycle_point, inputs.r_cycle);
    stage_values[4] *= EqPolynomial::<F>::mle(
        inputs.field_register_val_evaluation_cycle_point,
        inputs.r_cycle,
    );

    Ok(FieldInlineBytecodeReadRafPublicValues { stage_values })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldInlineBytecodeReadRafRegisterEqEvals<F> {
    pub read_write: Vec<F>,
    pub val_evaluation: Vec<F>,
}

pub struct FieldInlineBytecodeReadRafStageValueInputs<'a, F> {
    pub bytecode: &'a [FieldInlineBytecodeRow],
    pub field_register_read_write_point: &'a [F],
    pub field_register_val_evaluation_point: &'a [F],
    pub stage1_gammas: &'a [F],
    pub stage4_gammas: &'a [F],
    pub stage5_gammas: &'a [F],
}

pub fn read_raf_register_eq_evals<F>(
    field_register_read_write_point: &[F],
    field_register_val_evaluation_point: &[F],
) -> FieldInlineBytecodeReadRafRegisterEqEvals<F>
where
    F: Field,
{
    FieldInlineBytecodeReadRafRegisterEqEvals {
        read_write: EqPolynomial::<F>::evals(field_register_read_write_point, None),
        val_evaluation: EqPolynomial::<F>::evals(field_register_val_evaluation_point, None),
    }
}

pub fn read_raf_stage_values<F>(
    inputs: FieldInlineBytecodeReadRafStageValueInputs<'_, F>,
) -> Vec<[F; 5]>
where
    F: Field,
{
    let field_register_eq = read_raf_register_eq_evals(
        inputs.field_register_read_write_point,
        inputs.field_register_val_evaluation_point,
    );
    inputs
        .bytecode
        .iter()
        .map(|row| {
            read_raf_row_values(
                row,
                &field_register_eq.read_write,
                &field_register_eq.val_evaluation,
                inputs.stage1_gammas,
                inputs.stage4_gammas,
                inputs.stage5_gammas,
            )
        })
        .collect()
}

pub fn read_raf_row_values<F>(
    row: &FieldInlineBytecodeRow,
    field_read_write_eq: &[F],
    field_val_evaluation_eq: &[F],
    stage1_gammas: &[F],
    stage4_gammas: &[F],
    stage5_gammas: &[F],
) -> [F; 5]
where
    F: Field,
{
    let mut stage1 = F::zero();
    for (index, flag) in FIELD_INLINE_BYTECODE_STAGE1_FLAGS.into_iter().enumerate() {
        if row.flags.get(flag) {
            stage1 += stage1_gammas[2 + NUM_CIRCUIT_FLAGS + index];
        }
    }

    let stage4 = register_eq(row.operands.rd, field_read_write_eq) * stage4_gammas[3]
        + register_eq(row.operands.rs1, field_read_write_eq) * stage4_gammas[4]
        + register_eq(row.operands.rs2, field_read_write_eq) * stage4_gammas[5];
    let stage5 = register_eq(row.operands.rd, field_val_evaluation_eq)
        * stage5_gammas[2 + LookupTableKind::<XLEN>::COUNT];

    [stage1, F::zero(), F::zero(), stage4, stage5]
}

fn validate_bytecode_row(
    row: usize,
    entry: &FieldInlineBytecodeRow,
    field_register_count: usize,
) -> Result<(), FieldInlineBytecodeValidationError> {
    let active_count = entry.flags.active_count();
    if active_count == 0 {
        if entry.operands.rd.is_some()
            || entry.operands.rs1.is_some()
            || entry.operands.rs2.is_some()
        {
            return Err(FieldInlineBytecodeValidationError::InactiveRowHasOperands { row });
        }
        return Ok(());
    }
    if active_count != 1 {
        return Err(FieldInlineBytecodeValidationError::InvalidActiveFlagCount {
            row,
            active_count,
        });
    }

    let Some(flag) = entry.flags.active_flag() else {
        return Err(FieldInlineBytecodeValidationError::InvalidActiveFlagCount {
            row,
            active_count,
        });
    };
    validate_operand_layout(row, flag, entry.operands)?;
    validate_register(row, "rd", entry.operands.rd, field_register_count)?;
    validate_register(row, "rs1", entry.operands.rs1, field_register_count)?;
    validate_register(row, "rs2", entry.operands.rs2, field_register_count)?;
    Ok(())
}

fn validate_operand_layout(
    row: usize,
    flag: FieldInlineOpFlag,
    operands: FieldInlineBytecodeOperands,
) -> Result<(), FieldInlineBytecodeValidationError> {
    let valid = match flag {
        FieldInlineOpFlag::Add | FieldInlineOpFlag::Sub | FieldInlineOpFlag::Mul => {
            operands.rd.is_some() && operands.rs1.is_some() && operands.rs2.is_some()
        }
        FieldInlineOpFlag::Inv => {
            operands.rd.is_some() && operands.rs1.is_some() && operands.rs2.is_none()
        }
        FieldInlineOpFlag::AssertEq => {
            operands.rd.is_none() && operands.rs1.is_some() && operands.rs2.is_some()
        }
        FieldInlineOpFlag::LoadFromX | FieldInlineOpFlag::LoadImm => {
            operands.rd.is_some() && operands.rs1.is_none() && operands.rs2.is_none()
        }
        FieldInlineOpFlag::StoreToX => {
            operands.rd.is_none() && operands.rs1.is_some() && operands.rs2.is_none()
        }
    };
    if valid {
        Ok(())
    } else {
        Err(FieldInlineBytecodeValidationError::InvalidOperands {
            row,
            flag,
            reason: "operand presence does not match the active field-inline instruction",
        })
    }
}

fn validate_register(
    row: usize,
    operand: &'static str,
    register: Option<u8>,
    field_register_count: usize,
) -> Result<(), FieldInlineBytecodeValidationError> {
    if let Some(register) = register {
        if usize::from(register) >= field_register_count {
            return Err(FieldInlineBytecodeValidationError::RegisterOutOfRange {
                row,
                operand,
                register,
                field_register_count,
            });
        }
    }
    Ok(())
}

fn register_eq<F: Field>(register: Option<u8>, eq: &[F]) -> F {
    register
        .and_then(|register| eq.get(register as usize))
        .copied()
        .unwrap_or_else(F::zero)
}

fn require_len<F>(values: &[F], expected: usize) -> Result<(), FieldInlineBytecodeReadRafError> {
    if values.len() < expected {
        return Err(FieldInlineBytecodeReadRafError::Point(
            JoltFormulaPointError::ChallengeLengthMismatch {
                expected,
                got: values.len(),
            },
        ));
    }
    Ok(())
}

#[cfg(test)]
#[expect(clippy::panic)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::EqPolynomial;

    use super::*;

    #[test]
    fn read_raf_register_eq_evals_builds_field_register_address_tables() {
        let read_write = vec![Fr::from_u64(2), Fr::from_u64(3)];
        let val_evaluation = vec![Fr::from_u64(5), Fr::from_u64(7)];
        let eq = read_raf_register_eq_evals(&read_write, &val_evaluation);

        assert_eq!(
            eq,
            FieldInlineBytecodeReadRafRegisterEqEvals {
                read_write: EqPolynomial::<Fr>::evals(&read_write, None),
                val_evaluation: EqPolynomial::<Fr>::evals(&val_evaluation, None),
            }
        );
    }

    #[test]
    fn read_raf_stage_values_match_field_row_formula() {
        let rows = vec![
            FieldInlineBytecodeRow {
                operands: FieldInlineBytecodeOperands {
                    rd: Some(1),
                    rs1: Some(2),
                    rs2: Some(3),
                },
                flags: FieldInlineBytecodeFlags {
                    mul: true,
                    ..FieldInlineBytecodeFlags::default()
                },
            },
            FieldInlineBytecodeRow::default(),
        ];
        let read_write_point = vec![Fr::from_u64(2); 4];
        let val_evaluation_point = vec![Fr::from_u64(3); 4];
        let stage1_gammas = (0..FIELD_INLINE_BYTECODE_STAGE1_GAMMA_COUNT)
            .map(|value| Fr::from_u64(value as u64 + 1))
            .collect::<Vec<_>>();
        let stage4_gammas = (0..FIELD_INLINE_BYTECODE_STAGE4_GAMMA_COUNT)
            .map(|value| Fr::from_u64(value as u64 + 11))
            .collect::<Vec<_>>();
        let stage5_gammas = (0..=2 + LookupTableKind::<XLEN>::COUNT)
            .map(|value| Fr::from_u64(value as u64 + 17))
            .collect::<Vec<_>>();
        let register_eq = read_raf_register_eq_evals(&read_write_point, &val_evaluation_point);

        let stage_values = read_raf_stage_values(FieldInlineBytecodeReadRafStageValueInputs {
            bytecode: &rows,
            field_register_read_write_point: &read_write_point,
            field_register_val_evaluation_point: &val_evaluation_point,
            stage1_gammas: &stage1_gammas,
            stage4_gammas: &stage4_gammas,
            stage5_gammas: &stage5_gammas,
        });
        let expected = rows
            .iter()
            .map(|row| {
                read_raf_row_values(
                    row,
                    &register_eq.read_write,
                    &register_eq.val_evaluation,
                    &stage1_gammas,
                    &stage4_gammas,
                    &stage5_gammas,
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(stage_values, expected);
    }

    #[test]
    fn public_values_evaluate_field_rows() {
        let rows = vec![
            FieldInlineBytecodeRow {
                operands: FieldInlineBytecodeOperands {
                    rd: Some(1),
                    rs1: Some(2),
                    rs2: Some(3),
                },
                flags: FieldInlineBytecodeFlags {
                    mul: true,
                    ..FieldInlineBytecodeFlags::default()
                },
            },
            FieldInlineBytecodeRow::default(),
        ];
        let stage1_gammas = (0..FIELD_INLINE_BYTECODE_STAGE1_GAMMA_COUNT)
            .map(|value| Fr::from_u64(value as u64))
            .collect::<Vec<_>>();
        let stage4_gammas = (0..FIELD_INLINE_BYTECODE_STAGE4_GAMMA_COUNT)
            .map(|value| Fr::from_u64(value as u64))
            .collect::<Vec<_>>();
        let stage5_gammas = (0..=2 + LookupTableKind::<XLEN>::COUNT)
            .map(|value| Fr::from_u64(value as u64))
            .collect::<Vec<_>>();
        let zero_scalar = Fr::from_u64(0);
        let zero = vec![zero_scalar];
        let zero_point = [zero_scalar; 4];

        let values = read_raf_public_values(FieldInlineBytecodeReadRafEvaluationInputs {
            bytecode: &rows,
            field_register_log_k: 4,
            r_address: &[zero_scalar],
            r_cycle: &[zero_scalar],
            stage1_cycle_point: &zero,
            field_register_read_write_point: &zero_point,
            field_register_read_write_cycle_point: &zero,
            field_register_val_evaluation_point: &zero_point,
            field_register_val_evaluation_cycle_point: &zero,
            stage1_gammas: &stage1_gammas,
            stage4_gammas: &stage4_gammas,
            stage5_gammas: &stage5_gammas,
        })
        .unwrap_or_else(|error| panic!("field bytecode public values should evaluate: {error}"));

        assert_eq!(
            values.stage_values[0],
            stage1_gammas[2 + NUM_CIRCUIT_FLAGS + 2]
        );
        assert_eq!(values.stage_values[3], zero_scalar);
        assert_eq!(values.stage_values[4], zero_scalar);
    }

    #[test]
    fn rejects_missing_operands_for_active_field_row() {
        let rows = [FieldInlineBytecodeRow {
            flags: FieldInlineBytecodeFlags {
                mul: true,
                ..FieldInlineBytecodeFlags::default()
            },
            ..FieldInlineBytecodeRow::default()
        }];

        assert!(matches!(
            validate_bytecode_rows(&rows, 1, 4),
            Err(FieldInlineBytecodeValidationError::InvalidOperands {
                row: 0,
                flag: FieldInlineOpFlag::Mul,
                ..
            })
        ));
    }

    #[test]
    fn rejects_out_of_range_field_register_operand() {
        let rows = [FieldInlineBytecodeRow {
            operands: FieldInlineBytecodeOperands {
                rd: Some(16),
                ..FieldInlineBytecodeOperands::default()
            },
            flags: FieldInlineBytecodeFlags {
                load_imm: true,
                ..FieldInlineBytecodeFlags::default()
            },
        }];

        assert!(matches!(
            validate_bytecode_rows(&rows, 1, 4),
            Err(FieldInlineBytecodeValidationError::RegisterOutOfRange {
                row: 0,
                operand: "rd",
                register: 16,
                field_register_count: 16
            })
        ));
    }
}
