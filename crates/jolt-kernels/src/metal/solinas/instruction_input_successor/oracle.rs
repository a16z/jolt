//! Scalar correctness oracle with no Metal/runtime dependency.

use jolt_field::Field;

use super::abi::{
    InstructionInputSuccessorError, InstructionInputSuccessorRow, InstructionInputSuccessorTable,
    FLAG_IMM_POSITIVE, FLAG_LEFT_OPERAND_IS_PC, FLAG_LEFT_OPERAND_IS_RS1,
    FLAG_RIGHT_OPERAND_IS_IMM, FLAG_RIGHT_OPERAND_IS_RS2, INSTRUCTION_INPUT_SUCCESSOR_TABLES,
    ROW_EFFECTIVE_RS2, ROW_RS1, ROW_UNEXPANDED_PC,
};
use super::model::{checked_dense_message_shape, checked_materialize_shape};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuadraticDescriptors<F> {
    values: [F; 3],
}

impl<F: Field> QuadraticDescriptors<F> {
    pub const fn from_values(values: [F; 3]) -> Self {
        Self { values }
    }

    pub const fn values(self) -> [F; 3] {
        self.values
    }

    /// Reconstructs `q(0)..=q(3)` from `[q(0), q(1), t^2 coefficient]`.
    pub fn evals_0_to_3(self) -> [F; 4] {
        let [q_at_0, q_at_1, quadratic] = self.values;
        let twice_quadratic = quadratic + quadratic;
        let q_at_2 = q_at_1 + q_at_1 - q_at_0 + twice_quadratic;
        let q_at_3 = q_at_2 + q_at_1 - q_at_0 + twice_quadratic + twice_quadratic;
        [q_at_0, q_at_1, q_at_2, q_at_3]
    }
}

pub fn row_fields<F: Field>(
    row: InstructionInputSuccessorRow,
) -> Result<[F; INSTRUCTION_INPUT_SUCCESSOR_TABLES], InstructionInputSuccessorError> {
    row.validate()?;
    let magnitude = F::from_u128(row.imm_magnitude());
    let imm = if row.flag(FLAG_IMM_POSITIVE) {
        magnitude
    } else {
        -magnitude
    };
    Ok([
        F::from_bool(row.flag(FLAG_LEFT_OPERAND_IS_RS1)),
        F::from_u64(row.word(ROW_RS1)),
        F::from_bool(row.flag(FLAG_LEFT_OPERAND_IS_PC)),
        F::from_u64(row.word(ROW_UNEXPANDED_PC)),
        F::from_bool(row.flag(FLAG_RIGHT_OPERAND_IS_RS2)),
        F::from_u64(row.word(ROW_EFFECTIVE_RS2)),
        F::from_bool(row.flag(FLAG_RIGHT_OPERAND_IS_IMM)),
        imm,
    ])
}

/// Materializes eight table-major `N/2` tables with low-to-high binding.
pub fn materialize_first_bind<F: Field>(
    rows: &[InstructionInputSuccessorRow],
    challenge: F,
) -> Result<Vec<F>, InstructionInputSuccessorError> {
    let shape = checked_materialize_shape(rows.len(), u64::MAX)?;
    let bound_elements = shape.grid_threads();
    let mut dense = vec![F::zero(); INSTRUCTION_INPUT_SUCCESSOR_TABLES * bound_elements];

    for y in 0..bound_elements {
        let low = row_fields::<F>(rows[2 * y])?;
        let high = row_fields::<F>(rows[2 * y + 1])?;
        for table in InstructionInputSuccessorTable::ALL {
            dense[table.index() * bound_elements + y] =
                bind_low_to_high(low[table.index()], high[table.index()], challenge);
        }
    }
    Ok(dense)
}

/// Computes the three device descriptors from already-bound dense tables.
pub fn dense_message<F: Field>(
    tables: &[F],
    table_elements: usize,
    e_in: &[F],
    e_out: &[F],
    gamma: F,
) -> Result<QuadraticDescriptors<F>, InstructionInputSuccessorError> {
    let _shape =
        checked_dense_message_shape(table_elements, e_in.len(), e_out.len(), 128, u64::MAX)?;
    let expected = INSTRUCTION_INPUT_SUCCESSOR_TABLES
        .checked_mul(table_elements)
        .ok_or(InstructionInputSuccessorError::GeometryOverflow)?;
    if tables.len() != expected {
        return Err(InstructionInputSuccessorError::InvalidTableStorage {
            expected,
            got: tables.len(),
        });
    }

    let mut output = [F::zero(); 3];
    for (x_out, &outer_weight) in e_out.iter().enumerate() {
        let mut lanes = [F::zero(); 3];
        for (x_in, &inner_weight) in e_in.iter().enumerate() {
            let pair = x_out * e_in.len() + x_in;
            let source = 2 * pair;
            let left = add_descriptors(
                table_factor(tables, table_elements, source, 0, 1),
                table_factor(tables, table_elements, source, 2, 3),
            );
            let right = add_descriptors(
                table_factor(tables, table_elements, source, 4, 5),
                table_factor(tables, table_elements, source, 6, 7),
            );
            for descriptor in 0..3 {
                let q = right[descriptor] + gamma * left[descriptor];
                lanes[descriptor] += inner_weight * q;
            }
        }
        for descriptor in 0..3 {
            output[descriptor] += outer_weight * lanes[descriptor];
        }
    }
    Ok(QuadraticDescriptors::from_values(output))
}

pub fn split_first_bind_message<F: Field>(
    rows: &[InstructionInputSuccessorRow],
    first_challenge: F,
    e_in: &[F],
    e_out: &[F],
    gamma: F,
) -> Result<QuadraticDescriptors<F>, InstructionInputSuccessorError> {
    let dense = materialize_first_bind(rows, first_challenge)?;
    dense_message(&dense, rows.len() / 2, e_in, e_out, gamma)
}

/// Independent direct walk used to test table layout and pair orientation.
///
/// It binds original rows `(4y, 4y+1)` and `(4y+2, 4y+3)` at the first
/// challenge, then extends those two results at `t = 0..=3` without building
/// table-major storage or quadratic descriptors.
pub fn direct_after_first_bind_evals<F: Field>(
    rows: &[InstructionInputSuccessorRow],
    first_challenge: F,
    e_in: &[F],
    e_out: &[F],
    gamma: F,
) -> Result<[F; 4], InstructionInputSuccessorError> {
    let table_elements = rows.len() / 2;
    let _shape =
        checked_dense_message_shape(table_elements, e_in.len(), e_out.len(), 128, u64::MAX)?;
    let points = [F::zero(), F::one(), F::from_u64(2), F::from_u64(3)];
    let mut output = [F::zero(); 4];

    for (x_out, &outer_weight) in e_out.iter().enumerate() {
        for (x_in, &inner_weight) in e_in.iter().enumerate() {
            let pair = x_out * e_in.len() + x_in;
            let base = 4 * pair;
            let row_0 = row_fields::<F>(rows[base])?;
            let row_1 = row_fields::<F>(rows[base + 1])?;
            let row_2 = row_fields::<F>(rows[base + 2])?;
            let row_3 = row_fields::<F>(rows[base + 3])?;

            for (slot, &point) in output.iter_mut().zip(&points) {
                let fields = core::array::from_fn(|table| {
                    let low = bind_low_to_high(row_0[table], row_1[table], first_challenge);
                    let high = bind_low_to_high(row_2[table], row_3[table], first_challenge);
                    bind_low_to_high(low, high, point)
                });
                *slot += outer_weight * inner_weight * relation(fields, gamma);
            }
        }
    }
    Ok(output)
}

fn bind_low_to_high<F: Field>(low: F, high: F, challenge: F) -> F {
    low + challenge * (high - low)
}

fn table_factor<F: Field>(
    tables: &[F],
    table_elements: usize,
    source: usize,
    flag_table: usize,
    value_table: usize,
) -> [F; 3] {
    let flag_0 = tables[flag_table * table_elements + source];
    let flag_1 = tables[flag_table * table_elements + source + 1];
    let value_0 = tables[value_table * table_elements + source];
    let value_1 = tables[value_table * table_elements + source + 1];
    [
        flag_0 * value_0,
        flag_1 * value_1,
        (flag_1 - flag_0) * (value_1 - value_0),
    ]
}

fn add_descriptors<F: Field>(lhs: [F; 3], rhs: [F; 3]) -> [F; 3] {
    [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]]
}

fn relation<F: Field>(fields: [F; INSTRUCTION_INPUT_SUCCESSOR_TABLES], gamma: F) -> F {
    let left = fields[0] * fields[1] + fields[2] * fields[3];
    let right = fields[4] * fields[5] + fields[6] * fields[7];
    right + gamma * left
}
