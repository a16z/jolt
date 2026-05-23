use jolt_field::{Field, RingCore};
use jolt_lookup_tables::{LookupTableKind, XLEN};
use jolt_poly::EqPolynomial;
use jolt_riscv::NUM_CIRCUIT_FLAGS;
use serde::{Deserialize, Serialize};

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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldInlineBytecodeReadRafPublicValues<F: Field> {
    pub stage_values: [F; 5],
}

pub struct FieldInlineBytecodeReadRafEvaluationInputs<'a, F> {
    pub bytecode: &'a [FieldInlineBytecodeRow],
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
) -> Result<FieldInlineBytecodeReadRafPublicValues<F>, JoltFormulaPointError>
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
        return Err(JoltFormulaPointError::EvaluationDomainLengthMismatch {
            expected: expected_domain,
            got: inputs.bytecode.len(),
        });
    }

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

fn register_eq<F: Field>(register: Option<u8>, eq: &[F]) -> F {
    register
        .and_then(|register| eq.get(register as usize))
        .copied()
        .unwrap_or_else(F::zero)
}

fn require_len<F>(values: &[F], expected: usize) -> Result<(), JoltFormulaPointError> {
    if values.len() < expected {
        return Err(JoltFormulaPointError::ChallengeLengthMismatch {
            expected,
            got: values.len(),
        });
    }
    Ok(())
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
}
