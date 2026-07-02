use jolt_field::{Field, RingCore};
use jolt_lookup_tables::{LookupTableKind, XLEN};
use jolt_poly::EqPolynomial;
use jolt_riscv::NUM_CIRCUIT_FLAGS;
use serde::{Deserialize, Serialize};

use crate::{challenge, opening, Expr};

use super::super::{
    FieldInlineOpFlag, FieldInlineOpeningId, FieldInlineRelationId, FieldInlineVirtualPolynomial,
};
use crate::protocols::jolt::formulas::error::{require_len, JoltFormulaPointError};
use crate::protocols::jolt::{BytecodeReadRafChallenge, JoltChallengeId};

pub const FIELD_INLINE_OP_FLAGS: [FieldInlineOpFlag; 8] = [
    FieldInlineOpFlag::Add,
    FieldInlineOpFlag::Sub,
    FieldInlineOpFlag::Mul,
    FieldInlineOpFlag::Inv,
    FieldInlineOpFlag::AssertEq,
    FieldInlineOpFlag::LoadFromX,
    FieldInlineOpFlag::StoreToX,
    FieldInlineOpFlag::LoadImm,
];
pub const FIELD_INLINE_STAGE1_GAMMA_OFFSET: usize = 2 + NUM_CIRCUIT_FLAGS;
pub const FIELD_INLINE_STAGE4_GAMMA_OFFSET: usize = 3;
pub const FIELD_INLINE_STAGE5_GAMMA_OFFSET: usize = 2 + LookupTableKind::<XLEN>::COUNT;

pub type FieldInlineBytecodeExpr<F> = Expr<F, FieldInlineOpeningId, (), JoltChallengeId>;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldInlineBytecodeRow {
    pub flags: FieldInlineBytecodeFlags,
    pub rd: Option<usize>,
    pub rs1: Option<usize>,
    pub rs2: Option<usize>,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
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
    pub const fn flag(self, flag: FieldInlineOpFlag) -> bool {
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

pub fn read_raf_input_extension<F>() -> FieldInlineBytecodeExpr<F>
where
    F: RingCore,
{
    let gamma = bytecode_challenge(BytecodeReadRafChallenge::Gamma);

    field_stage1_claim()
        + gamma.clone().pow(3) * field_stage4_claim()
        + gamma.pow(4) * field_stage5_claim()
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldInlineBytecodePublicValues<F: Field> {
    pub stage_values: [F; 5],
}

pub fn read_raf_public_values<F>(
    inputs: FieldInlineBytecodeReadRafEvaluationInputs<'_, F>,
) -> Result<FieldInlineBytecodePublicValues<F>, JoltFormulaPointError>
where
    F: Field,
{
    require_len(
        inputs.stage1_gammas,
        FIELD_INLINE_STAGE1_GAMMA_OFFSET + FIELD_INLINE_OP_FLAGS.len(),
    )?;
    require_len(inputs.stage4_gammas, FIELD_INLINE_STAGE4_GAMMA_OFFSET + 3)?;
    require_len(inputs.stage5_gammas, FIELD_INLINE_STAGE5_GAMMA_OFFSET + 1)?;
    require_len(
        inputs.field_register_read_write_point,
        inputs.field_register_log_k,
    )?;
    require_len(
        inputs.field_register_val_evaluation_point,
        inputs.field_register_log_k,
    )?;

    let expected_domain = 1usize << inputs.r_address.len();
    if inputs.bytecode.len() != expected_domain {
        return Err(JoltFormulaPointError::EvaluationDomainLengthMismatch {
            expected: expected_domain,
            got: inputs.bytecode.len(),
        });
    }

    let address_eq_evals = EqPolynomial::<F>::evals(inputs.r_address, None);
    let read_write_eq = EqPolynomial::<F>::evals(inputs.field_register_read_write_point, None);
    let val_evaluation_eq =
        EqPolynomial::<F>::evals(inputs.field_register_val_evaluation_point, None);

    let mut stage_values = [F::zero(); 5];
    for (row, eq_address) in inputs.bytecode.iter().zip(address_eq_evals) {
        let row_values = read_raf_row_values(
            row,
            &read_write_eq,
            &val_evaluation_eq,
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

    Ok(FieldInlineBytecodePublicValues { stage_values })
}

fn read_raf_row_values<F>(
    row: &FieldInlineBytecodeRow,
    read_write_eq: &[F],
    val_evaluation_eq: &[F],
    stage1_gammas: &[F],
    stage4_gammas: &[F],
    stage5_gammas: &[F],
) -> [F; 5]
where
    F: Field,
{
    let mut stage1 = F::zero();
    for (index, flag) in FIELD_INLINE_OP_FLAGS.into_iter().enumerate() {
        if row.flags.flag(flag) {
            stage1 += stage1_gammas[FIELD_INLINE_STAGE1_GAMMA_OFFSET + index];
        }
    }

    let stage4 = register_eq(row.rd, read_write_eq)
        * stage4_gammas[FIELD_INLINE_STAGE4_GAMMA_OFFSET]
        + register_eq(row.rs1, read_write_eq) * stage4_gammas[FIELD_INLINE_STAGE4_GAMMA_OFFSET + 1]
        + register_eq(row.rs2, read_write_eq) * stage4_gammas[FIELD_INLINE_STAGE4_GAMMA_OFFSET + 2];
    let stage5 =
        register_eq(row.rd, val_evaluation_eq) * stage5_gammas[FIELD_INLINE_STAGE5_GAMMA_OFFSET];

    [stage1, F::zero(), F::zero(), stage4, stage5]
}

fn register_eq<F: Field>(register: Option<usize>, eq: &[F]) -> F {
    register
        .and_then(|register| eq.get(register))
        .copied()
        .unwrap_or_else(F::zero)
}

fn field_stage1_claim<F>() -> FieldInlineBytecodeExpr<F>
where
    F: RingCore,
{
    let beta = bytecode_challenge(BytecodeReadRafChallenge::Stage1Gamma);
    let mut claim = FieldInlineBytecodeExpr::zero();
    for (index, flag) in FIELD_INLINE_OP_FLAGS.into_iter().enumerate() {
        claim = claim
            + beta.clone().pow(FIELD_INLINE_STAGE1_GAMMA_OFFSET + index)
                * opening(field_op_flag_spartan_outer(flag));
    }
    claim
}

fn field_stage4_claim<F>() -> FieldInlineBytecodeExpr<F>
where
    F: RingCore,
{
    let beta = bytecode_challenge(BytecodeReadRafChallenge::Stage4Gamma);

    beta.clone().pow(FIELD_INLINE_STAGE4_GAMMA_OFFSET) * opening(field_rd_wa_read_write())
        + beta.clone().pow(FIELD_INLINE_STAGE4_GAMMA_OFFSET + 1)
            * opening(field_rs1_ra_read_write())
        + beta.pow(FIELD_INLINE_STAGE4_GAMMA_OFFSET + 2) * opening(field_rs2_ra_read_write())
}

fn field_stage5_claim<F>() -> FieldInlineBytecodeExpr<F>
where
    F: RingCore,
{
    let beta = bytecode_challenge(BytecodeReadRafChallenge::Stage5Gamma);

    beta.pow(FIELD_INLINE_STAGE5_GAMMA_OFFSET) * opening(field_rd_wa_val_evaluation())
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

fn field_rd_wa_read_write() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRdWa,
        FieldInlineRelationId::FieldRegistersReadWriteChecking,
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

fn field_rd_wa_val_evaluation() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRdWa,
        FieldInlineRelationId::FieldRegistersValEvaluation,
    )
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "tests should fail loudly")]

    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn read_raf_extension_uses_field_stage_openings() {
        let expr = read_raf_input_extension::<Fr>();

        assert!(expr
            .required_openings()
            .contains(&field_op_flag_spartan_outer(FieldInlineOpFlag::Add)));
        assert!(expr.required_openings().contains(&field_rd_wa_read_write()));
        assert!(expr
            .required_openings()
            .contains(&field_rd_wa_val_evaluation()));
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
    fn read_raf_public_values_evaluate_field_side_table_rows() {
        let row = FieldInlineBytecodeRow {
            flags: FieldInlineBytecodeFlags {
                add: true,
                store_to_x: true,
                ..FieldInlineBytecodeFlags::default()
            },
            rd: Some(1),
            rs1: Some(1),
            rs2: None,
        };
        let zero = Fr::from_u64(0);
        let one = Fr::from_u64(1);
        let stage1_gammas =
            vec![one; FIELD_INLINE_STAGE1_GAMMA_OFFSET + FIELD_INLINE_OP_FLAGS.len()];
        let stage4_gammas = vec![one; FIELD_INLINE_STAGE4_GAMMA_OFFSET + 3];
        let stage5_gammas = vec![one; FIELD_INLINE_STAGE5_GAMMA_OFFSET + 1];

        let values = read_raf_public_values(FieldInlineBytecodeReadRafEvaluationInputs {
            bytecode: &[row],
            field_register_log_k: 2,
            r_address: &[],
            r_cycle: &[],
            stage1_cycle_point: &[],
            field_register_read_write_point: &[zero, one],
            field_register_read_write_cycle_point: &[],
            field_register_val_evaluation_point: &[zero, one],
            field_register_val_evaluation_cycle_point: &[],
            stage1_gammas: &stage1_gammas,
            stage4_gammas: &stage4_gammas,
            stage5_gammas: &stage5_gammas,
        })
        .expect("field bytecode publics evaluate");

        assert_eq!(values.stage_values[0], Fr::from_u64(2));
        assert_eq!(values.stage_values[3], Fr::from_u64(2));
        assert_eq!(values.stage_values[4], Fr::from_u64(1));
    }
}
