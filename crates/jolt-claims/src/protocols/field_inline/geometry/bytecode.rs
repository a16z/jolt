use jolt_field::JoltField;
use jolt_lookup_tables::{LookupTableKind, XLEN};
use jolt_poly::EqPolynomial;
use jolt_riscv::JoltInstructionRow;

use crate::formula_error::PointGeometryError;

pub const FIELD_INLINE_BYTECODE_STAGE4_GAMMA_COUNT: usize = 6;
pub const FIELD_INLINE_BYTECODE_STAGE5_EXTRA_GAMMAS: usize = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldInlineBytecodeReadRafPublicValues<F: JoltField> {
    pub stage_values: [F; 5],
}

pub struct FieldInlineBytecodeReadRafEvaluationInputs<'a, F> {
    pub bytecode: &'a [JoltInstructionRow],
    pub r_address: &'a [F],
    pub r_cycle: &'a [F],
    pub field_register_read_write_point: &'a [F],
    pub field_register_read_write_cycle_point: &'a [F],
    pub field_register_val_evaluation_point: &'a [F],
    pub field_register_val_evaluation_cycle_point: &'a [F],
    pub stage4_gammas: &'a [F],
    pub stage5_gammas: &'a [F],
}

pub fn read_raf_public_values<F>(
    inputs: FieldInlineBytecodeReadRafEvaluationInputs<'_, F>,
) -> Result<FieldInlineBytecodeReadRafPublicValues<F>, PointGeometryError>
where
    F: JoltField,
{
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
        return Err(PointGeometryError::EvaluationDomainLengthMismatch {
            expected: expected_domain,
            got: inputs.bytecode.len(),
        });
    }

    let address_eq_evals = EqPolynomial::<F>::evals(inputs.r_address, None);
    let mut stage_values = read_raf_folded_stage_values(
        FieldInlineBytecodeReadRafStageValueInputs {
            bytecode: inputs.bytecode,
            field_register_read_write_point: inputs.field_register_read_write_point,
            field_register_val_evaluation_point: inputs.field_register_val_evaluation_point,
            stage4_gammas: inputs.stage4_gammas,
            stage5_gammas: inputs.stage5_gammas,
        },
        &address_eq_evals,
    );

    stage_values[3] *=
        EqPolynomial::<F>::mle(inputs.field_register_read_write_cycle_point, inputs.r_cycle);
    stage_values[4] *= EqPolynomial::<F>::mle(
        inputs.field_register_val_evaluation_cycle_point,
        inputs.r_cycle,
    );

    Ok(FieldInlineBytecodeReadRafPublicValues { stage_values })
}

pub struct FieldInlineBytecodeReadRafStageValueInputs<'a, F> {
    pub bytecode: &'a [JoltInstructionRow],
    pub field_register_read_write_point: &'a [F],
    pub field_register_val_evaluation_point: &'a [F],
    pub stage4_gammas: &'a [F],
    pub stage5_gammas: &'a [F],
}

pub fn read_raf_stage_values<F>(
    inputs: FieldInlineBytecodeReadRafStageValueInputs<'_, F>,
) -> Vec<[F; 5]>
where
    F: JoltField,
{
    let read_write_eq = EqPolynomial::<F>::evals(inputs.field_register_read_write_point, None);
    let val_evaluation_eq =
        EqPolynomial::<F>::evals(inputs.field_register_val_evaluation_point, None);
    inputs
        .bytecode
        .iter()
        .map(|row| {
            read_raf_row_values(
                row,
                &read_write_eq,
                &val_evaluation_eq,
                inputs.stage4_gammas,
                inputs.stage5_gammas,
            )
        })
        .collect()
}

pub fn read_raf_row_values<F>(
    row: &JoltInstructionRow,
    field_read_write_eq: &[F],
    field_val_evaluation_eq: &[F],
    stage4_gammas: &[F],
    stage5_gammas: &[F],
) -> [F; 5]
where
    F: JoltField,
{
    let operands = row.field_operands();
    let stage4 = register_eq(operands.rd, field_read_write_eq) * stage4_gammas[3]
        + register_eq(operands.rs1, field_read_write_eq) * stage4_gammas[4]
        + register_eq(operands.rs2, field_read_write_eq) * stage4_gammas[5];
    let stage5 = register_eq(operands.rd, field_val_evaluation_eq)
        * stage5_gammas[2 + LookupTableKind::<XLEN>::COUNT];

    [F::zero(), F::zero(), F::zero(), stage4, stage5]
}

/// `Σ_r address_eq[r] · read_raf_row_values(r)`, folded column-first: each
/// row adds its eq weight into a bucket per field-register operand, and the
/// buckets meet the register eq tables and gammas once.
pub fn read_raf_folded_stage_values<F>(
    inputs: FieldInlineBytecodeReadRafStageValueInputs<'_, F>,
    address_eq: &[F],
) -> [F; 5]
where
    F: JoltField,
{
    let read_write_eq = EqPolynomial::<F>::evals(inputs.field_register_read_write_point, None);
    let val_evaluation_eq =
        EqPolynomial::<F>::evals(inputs.field_register_val_evaluation_point, None);
    let mut buckets: [Vec<F>; 3] =
        std::array::from_fn(|_| vec![F::zero(); read_write_eq.len().max(val_evaluation_eq.len())]);
    for (row, &eq) in inputs.bytecode.iter().zip(address_eq) {
        let operands = row.field_operands();
        for (bucket, register) in buckets
            .iter_mut()
            .zip([operands.rd, operands.rs1, operands.rs2])
        {
            if let Some(slot) = register.and_then(|register| bucket.get_mut(usize::from(register)))
            {
                *slot += eq;
            }
        }
    }
    let read_write = |bucket: &[F]| F::dot_product(&read_write_eq, &bucket[..read_write_eq.len()]);
    let stage4 = read_write(&buckets[0]) * inputs.stage4_gammas[3]
        + read_write(&buckets[1]) * inputs.stage4_gammas[4]
        + read_write(&buckets[2]) * inputs.stage4_gammas[5];
    let stage5 = F::dot_product(&val_evaluation_eq, &buckets[0][..val_evaluation_eq.len()])
        * inputs.stage5_gammas[2 + LookupTableKind::<XLEN>::COUNT];
    [F::zero(), F::zero(), F::zero(), stage4, stage5]
}

fn register_eq<F: JoltField>(register: Option<u8>, eq: &[F]) -> F {
    register
        .and_then(|register| eq.get(usize::from(register)))
        .copied()
        .unwrap_or_else(F::zero)
}

fn require_len<F>(values: &[F], expected: usize) -> Result<(), PointGeometryError> {
    if values.len() < expected {
        return Err(PointGeometryError::ChallengeLengthMismatch {
            expected,
            got: values.len(),
        });
    }
    Ok(())
}
