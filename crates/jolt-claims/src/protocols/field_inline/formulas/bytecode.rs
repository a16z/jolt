use jolt_field::{Field, RingCore};
use jolt_lookup_tables::{LookupTableKind, XLEN};
use jolt_poly::EqPolynomial;
use jolt_riscv::NUM_CIRCUIT_FLAGS;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{challenge, opening, Expr};

use super::super::{
    FieldInlineOpFlag, FieldInlineOpeningId, FieldInlineRelationId, FieldInlineVirtualPolynomial,
};
use crate::protocols::jolt::formulas::dimensions::JoltFormulaPointError;
use crate::protocols::jolt::{BytecodeReadRafChallenge, JoltChallengeId};

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

pub type FieldInlineBytecodeExpr<F> = Expr<F, FieldInlineOpeningId, (), JoltChallengeId>;

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
    let field_register_count = 1usize
        .checked_shl(field_register_log_k as u32)
        .ok_or(FieldInlineBytecodeValidationError::InvalidFieldRegisterLogK {
            log_k: field_register_log_k,
        })?;

    for (row, entry) in bytecode.iter().enumerate() {
        validate_bytecode_row(row, entry, field_register_count)?;
    }
    Ok(())
}

pub fn bytecode_transcript_bytes(
    bytecode: &[FieldInlineBytecodeRow],
    field_register_log_k: usize,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16 + bytecode.len() * 7);
    bytes.extend_from_slice(&(field_register_log_k as u64).to_le_bytes());
    bytes.extend_from_slice(&(bytecode.len() as u64).to_le_bytes());
    for row in bytecode {
        encode_operand(row.operands.rd, &mut bytes);
        encode_operand(row.operands.rs1, &mut bytes);
        encode_operand(row.operands.rs2, &mut bytes);
        bytes.push(row.flags.transcript_bits());
    }
    bytes
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

pub fn read_raf_input_extension<F>() -> FieldInlineBytecodeExpr<F>
where
    F: RingCore,
{
    let gamma = bytecode_challenge(BytecodeReadRafChallenge::Gamma);

    stage1_claim() + gamma.clone().pow(3) * stage4_claim() + gamma.pow(4) * stage5_claim::<F>()
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
    let field_read_write_eq =
        EqPolynomial::<F>::evals(inputs.field_register_read_write_point, None);
    let field_val_evaluation_eq =
        EqPolynomial::<F>::evals(inputs.field_register_val_evaluation_point, None);

    let mut stage_values = [F::zero(); 5];
    for (row, eq_address) in inputs.bytecode.iter().zip(address_eq_evals) {
        let row_values = read_raf_row_values(
            row,
            &field_read_write_eq,
            &field_val_evaluation_eq,
            inputs.stage1_gammas,
            inputs.stage4_gammas,
            inputs.stage5_gammas,
        );
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

pub fn read_raf_input_openings() -> [FieldInlineOpeningId; 12] {
    [
        field_op_flag_spartan_outer(FieldInlineOpFlag::Add),
        field_op_flag_spartan_outer(FieldInlineOpFlag::Sub),
        field_op_flag_spartan_outer(FieldInlineOpFlag::Mul),
        field_op_flag_spartan_outer(FieldInlineOpFlag::Inv),
        field_op_flag_spartan_outer(FieldInlineOpFlag::AssertEq),
        field_op_flag_spartan_outer(FieldInlineOpFlag::LoadFromX),
        field_op_flag_spartan_outer(FieldInlineOpFlag::StoreToX),
        field_op_flag_spartan_outer(FieldInlineOpFlag::LoadImm),
        field_rd_wa_read_write(),
        field_rs1_ra_read_write(),
        field_rs2_ra_read_write(),
        field_rd_wa_val_evaluation(),
    ]
}

fn stage1_claim<F>() -> FieldInlineBytecodeExpr<F>
where
    F: RingCore,
{
    let beta = bytecode_challenge(BytecodeReadRafChallenge::Stage1Gamma);
    let mut claim = FieldInlineBytecodeExpr::zero();
    for (index, flag) in FIELD_INLINE_BYTECODE_STAGE1_FLAGS.into_iter().enumerate() {
        claim = claim
            + beta.clone().pow(2 + NUM_CIRCUIT_FLAGS + index)
                * opening(field_op_flag_spartan_outer(flag));
    }
    claim
}

fn stage4_claim<F>() -> FieldInlineBytecodeExpr<F>
where
    F: RingCore,
{
    let beta = bytecode_challenge(BytecodeReadRafChallenge::Stage4Gamma);
    beta.clone().pow(3) * opening(field_rd_wa_read_write())
        + beta.clone().pow(4) * opening(field_rs1_ra_read_write())
        + beta.pow(5) * opening(field_rs2_ra_read_write())
}

fn stage5_claim<F>() -> FieldInlineBytecodeExpr<F>
where
    F: RingCore,
{
    let beta = bytecode_challenge(BytecodeReadRafChallenge::Stage5Gamma);
    beta.pow(2 + LookupTableKind::<XLEN>::COUNT) * opening(field_rd_wa_val_evaluation())
}

fn read_raf_row_values<F>(
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
        if entry.operands.rd.is_some() || entry.operands.rs1.is_some() || entry.operands.rs2.is_some()
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

fn require_len<F>(
    values: &[F],
    expected: usize,
) -> Result<(), FieldInlineBytecodeReadRafError> {
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

fn encode_operand(register: Option<u8>, bytes: &mut Vec<u8>) {
    match register {
        Some(register) => {
            bytes.push(1);
            bytes.push(register);
        }
        None => {
            bytes.push(0);
            bytes.push(0);
        }
    }
}

fn bytecode_challenge<F>(id: BytecodeReadRafChallenge) -> FieldInlineBytecodeExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

fn field_op_flag_spartan_outer(flag: FieldInlineOpFlag) -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldOpFlag(flag),
        FieldInlineRelationId::FieldRegistersSpartanOuter,
    )
}

fn field_rs1_ra_read_write() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRs1Ra,
        FieldInlineRelationId::FieldRegistersReadWriteChecking,
    )
}

fn field_rs2_ra_read_write() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRs2Ra,
        FieldInlineRelationId::FieldRegistersReadWriteChecking,
    )
}

fn field_rd_wa_read_write() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRdWa,
        FieldInlineRelationId::FieldRegistersReadWriteChecking,
    )
}

fn field_rd_wa_val_evaluation() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRdWa,
        FieldInlineRelationId::FieldRegistersValEvaluation,
    )
}

#[cfg(test)]
#[expect(clippy::panic)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};

    use super::*;

    #[test]
    fn input_extension_uses_field_inline_openings_and_bytecode_challenges() {
        let expr = read_raf_input_extension::<Fr>();
        assert_eq!(expr.required_openings(), read_raf_input_openings().to_vec());
        assert_eq!(
            expr.required_challenges(),
            vec![
                JoltChallengeId::from(BytecodeReadRafChallenge::Stage1Gamma),
                JoltChallengeId::from(BytecodeReadRafChallenge::Gamma),
                JoltChallengeId::from(BytecodeReadRafChallenge::Stage4Gamma),
                JoltChallengeId::from(BytecodeReadRafChallenge::Stage5Gamma),
            ]
        );
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

    #[test]
    fn transcript_bytes_bind_field_bytecode_payload() {
        let mut rows = vec![FieldInlineBytecodeRow {
            operands: FieldInlineBytecodeOperands {
                rd: Some(1),
                ..FieldInlineBytecodeOperands::default()
            },
            flags: FieldInlineBytecodeFlags {
                load_imm: true,
                ..FieldInlineBytecodeFlags::default()
            },
        }];
        let original = bytecode_transcript_bytes(&rows, 4);
        rows[0].operands.rd = Some(2);

        assert_ne!(original, bytecode_transcript_bytes(&rows, 4));
    }
}
