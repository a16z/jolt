//! Compile-time selected Jolt R1CS composition.

use jolt_field::Field;
use jolt_poly::lagrange::{centered_lagrange_evals, CenteredIntegerDomainError};

use crate::ConstraintMatrices;
#[cfg(feature = "field-inline")]
use crate::SparseRow;

use super::rv64;

#[cfg(feature = "field-inline")]
use super::field_constraints;

#[cfg(feature = "field-inline")]
pub const FIELD_INLINE_COLUMN_BASE: usize = rv64::NUM_VARS_PER_CYCLE;

#[cfg(feature = "field-inline")]
pub const FIELD_INLINE_REUSED_NONCONST_COLUMNS: usize = 3;

#[cfg(feature = "field-inline")]
pub const FIELD_INLINE_APPENDED_COLUMNS: usize =
    field_constraints::NUM_VARS_PER_CYCLE - 1 - FIELD_INLINE_REUSED_NONCONST_COLUMNS;

#[cfg(feature = "field-inline")]
pub const NUM_VARS_PER_CYCLE: usize = rv64::NUM_VARS_PER_CYCLE + FIELD_INLINE_APPENDED_COLUMNS;

#[cfg(not(feature = "field-inline"))]
pub const NUM_VARS_PER_CYCLE: usize = rv64::NUM_VARS_PER_CYCLE;

#[cfg(feature = "field-inline")]
pub const NUM_CONSTRAINTS_PER_CYCLE: usize =
    rv64::NUM_CONSTRAINTS_PER_CYCLE + field_constraints::NUM_CONSTRAINTS_PER_CYCLE;

#[cfg(not(feature = "field-inline"))]
pub const NUM_CONSTRAINTS_PER_CYCLE: usize = rv64::NUM_CONSTRAINTS_PER_CYCLE;

#[cfg(feature = "field-inline")]
pub const OUTER_EQ_ROW_COUNT: usize =
    rv64::NUM_EQ_CONSTRAINTS + field_constraints::NUM_EQ_CONSTRAINTS;

#[cfg(not(feature = "field-inline"))]
pub const OUTER_EQ_ROW_COUNT: usize = rv64::NUM_EQ_CONSTRAINTS;

pub const OUTER_UNISKIP_DOMAIN_SIZE: usize = OUTER_EQ_ROW_COUNT.div_ceil(2);
pub const OUTER_UNISKIP_FIRST_ROUND_DEGREE: usize = 3 * OUTER_UNISKIP_DOMAIN_SIZE - 3;
pub const OUTER_REMAINDER_DEGREE: usize = 3;
pub const OUTER_SECOND_GROUP_ROW_COUNT: usize = OUTER_EQ_ROW_COUNT - OUTER_UNISKIP_DOMAIN_SIZE;

#[cfg(not(feature = "field-inline"))]
pub const OUTER_FIRST_GROUP_ROWS: [usize; OUTER_UNISKIP_DOMAIN_SIZE] =
    [1, 2, 3, 4, 5, 6, 11, 14, 17, 18];

#[cfg(feature = "field-inline")]
pub const OUTER_FIRST_GROUP_ROWS: [usize; OUTER_UNISKIP_DOMAIN_SIZE] = [
    1,
    2,
    3,
    4,
    5,
    6,
    11,
    14,
    17,
    18,
    rv64::NUM_EQ_CONSTRAINTS + field_constraints::ROW_FADD,
    rv64::NUM_EQ_CONSTRAINTS + field_constraints::ROW_FSUB,
    rv64::NUM_EQ_CONSTRAINTS + field_constraints::ROW_FMUL,
    rv64::NUM_EQ_CONSTRAINTS + field_constraints::ROW_FINV,
];

#[cfg(not(feature = "field-inline"))]
pub const OUTER_SECOND_GROUP_ROWS: [usize; OUTER_SECOND_GROUP_ROW_COUNT] =
    [0, 7, 8, 9, 10, 12, 13, 15, 16];

#[cfg(feature = "field-inline")]
pub const OUTER_SECOND_GROUP_ROWS: [usize; OUTER_SECOND_GROUP_ROW_COUNT] = [
    0,
    7,
    8,
    9,
    10,
    12,
    13,
    15,
    16,
    rv64::NUM_EQ_CONSTRAINTS + field_constraints::ROW_ASSERT_EQ,
    rv64::NUM_EQ_CONSTRAINTS + field_constraints::ROW_LOAD_FROM_X,
    rv64::NUM_EQ_CONSTRAINTS + field_constraints::ROW_STORE_TO_X,
    rv64::NUM_EQ_CONSTRAINTS + field_constraints::ROW_LOAD_IMM,
];

pub fn jolt_eq_constraints<F: Field>() -> ConstraintMatrices<F> {
    let constraints = rv64::rv64_eq_constraints();
    #[cfg(feature = "field-inline")]
    {
        append_field_inline_columns(constraints, field_constraints::field_eq_constraints())
    }
    #[cfg(not(feature = "field-inline"))]
    {
        constraints
    }
}

pub fn jolt_constraints<F: Field>() -> ConstraintMatrices<F> {
    let constraints = rv64::rv64_constraints();
    #[cfg(feature = "field-inline")]
    {
        append_field_inline_columns(constraints, field_constraints::field_constraints())
    }
    #[cfg(not(feature = "field-inline"))]
    {
        constraints
    }
}

pub fn outer_row_weights<F: Field>(
    uniskip: F,
    stream: F,
) -> Result<Vec<F>, CenteredIntegerDomainError> {
    let lagrange_weights = centered_lagrange_evals(OUTER_UNISKIP_DOMAIN_SIZE, uniskip)?;
    let mut weights = vec![F::zero(); OUTER_EQ_ROW_COUNT];

    for (index, &row) in OUTER_FIRST_GROUP_ROWS.iter().enumerate() {
        weights[row] += (F::one() - stream) * lagrange_weights[index];
    }
    for (index, &row) in OUTER_SECOND_GROUP_ROWS.iter().enumerate() {
        weights[row] += stream * lagrange_weights[index];
    }

    Ok(weights)
}

#[cfg(feature = "field-inline")]
pub const fn field_inline_column(local_column: usize) -> Option<usize> {
    match local_column {
        field_constraints::V_CONST => Some(rv64::V_CONST),
        field_constraints::V_FIELD_RS1_VALUE => Some(FIELD_INLINE_COLUMN_BASE),
        field_constraints::V_FIELD_RS2_VALUE => Some(FIELD_INLINE_COLUMN_BASE + 1),
        field_constraints::V_FIELD_RD_VALUE => Some(FIELD_INLINE_COLUMN_BASE + 2),
        field_constraints::V_FIELD_PRODUCT => Some(FIELD_INLINE_COLUMN_BASE + 3),
        field_constraints::V_FIELD_INV_PRODUCT => Some(FIELD_INLINE_COLUMN_BASE + 4),
        field_constraints::V_X_RS1_VALUE => Some(rv64::V_RS1_VALUE),
        field_constraints::V_X_RD_WRITE_VALUE => Some(rv64::V_RD_WRITE_VALUE),
        field_constraints::V_IMM => Some(rv64::V_IMM),
        field_constraints::V_IS_FIELD_ADD => Some(FIELD_INLINE_COLUMN_BASE + 5),
        field_constraints::V_IS_FIELD_SUB => Some(FIELD_INLINE_COLUMN_BASE + 6),
        field_constraints::V_IS_FIELD_MUL => Some(FIELD_INLINE_COLUMN_BASE + 7),
        field_constraints::V_IS_FIELD_INV => Some(FIELD_INLINE_COLUMN_BASE + 8),
        field_constraints::V_IS_FIELD_ASSERT_EQ => Some(FIELD_INLINE_COLUMN_BASE + 9),
        field_constraints::V_IS_FIELD_LOAD_FROM_X => Some(FIELD_INLINE_COLUMN_BASE + 10),
        field_constraints::V_IS_FIELD_STORE_TO_X => Some(FIELD_INLINE_COLUMN_BASE + 11),
        field_constraints::V_IS_FIELD_LOAD_IMM => Some(FIELD_INLINE_COLUMN_BASE + 12),
        _ => None,
    }
}

#[cfg(feature = "field-inline")]
pub const fn field_inline_input_column(input_index: usize) -> Option<usize> {
    match field_constraints::input_column(input_index) {
        Some(local_column) => field_inline_column(local_column),
        None => None,
    }
}

#[cfg(feature = "field-inline")]
fn append_field_inline_columns<F: Field>(
    base: ConstraintMatrices<F>,
    extension: ConstraintMatrices<F>,
) -> ConstraintMatrices<F> {
    let num_constraints = base.num_constraints + extension.num_constraints;
    let num_vars = base.num_vars + FIELD_INLINE_APPENDED_COLUMNS;

    let mut a = base.a;
    let mut b = base.b;
    let mut c = base.c;
    a.extend(remap_rows(extension.a));
    b.extend(remap_rows(extension.b));
    c.extend(remap_rows(extension.c));

    ConstraintMatrices::new(num_constraints, num_vars, a, b, c)
}

#[cfg(feature = "field-inline")]
fn remap_rows<F: Field>(rows: Vec<SparseRow<F>>) -> Vec<SparseRow<F>> {
    rows.into_iter()
        .map(|row| {
            row.into_iter()
                .map(|(column, coefficient)| {
                    let column = remap_field_inline_column(column);
                    (column, coefficient)
                })
                .collect()
        })
        .collect()
}

#[cfg(feature = "field-inline")]
fn remap_field_inline_column(column: usize) -> usize {
    let Some(column) = field_inline_column(column) else {
        unreachable!("field-inline constraint row referenced an unknown local column")
    };
    column
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "field-inline")]
    use jolt_claims::protocols::field_inline::{
        formulas::spartan::{
            outer_output_openings, FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS,
            FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUT_COUNT,
        },
        FieldInlineOpFlag, FieldInlineVirtualPolynomial,
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    #[cfg(feature = "field-inline")]
    use num_traits::Zero;

    #[cfg(not(feature = "field-inline"))]
    #[test]
    fn default_selected_constraints_match_rv64_shape() {
        let selected = jolt_constraints::<Fr>();
        let rv64 = rv64::rv64_constraints::<Fr>();

        assert_eq!(selected.num_constraints, rv64.num_constraints);
        assert_eq!(selected.num_vars, rv64.num_vars);
        assert_eq!(selected.a, rv64.a);
        assert_eq!(selected.b, rv64.b);
        assert_eq!(selected.c, rv64.c);
    }

    #[cfg(not(feature = "field-inline"))]
    #[test]
    fn default_selected_outer_geometry_matches_rv64() {
        assert_eq!(OUTER_EQ_ROW_COUNT, rv64::NUM_EQ_CONSTRAINTS);
        assert_eq!(OUTER_UNISKIP_DOMAIN_SIZE, 10);
        assert_eq!(OUTER_UNISKIP_FIRST_ROUND_DEGREE, 27);
        assert_eq!(OUTER_REMAINDER_DEGREE, 3);
        assert_eq!(OUTER_FIRST_GROUP_ROWS, [1, 2, 3, 4, 5, 6, 11, 14, 17, 18]);
        assert_eq!(OUTER_SECOND_GROUP_ROWS, [0, 7, 8, 9, 10, 12, 13, 15, 16]);
        assert_eq!(
            outer_row_weights(Fr::from_u64(2), Fr::from_u64(3)).map(|weights| weights.len()),
            Ok(rv64::NUM_EQ_CONSTRAINTS)
        );
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn field_inline_selected_constraints_append_field_shape() {
        let selected = jolt_constraints::<Fr>();

        assert_eq!(selected.num_constraints, NUM_CONSTRAINTS_PER_CYCLE);
        assert_eq!(selected.num_vars, NUM_VARS_PER_CYCLE);
        assert_eq!(field_inline_input_column(0), Some(FIELD_INLINE_COLUMN_BASE));
        assert_eq!(
            field_inline_column(field_constraints::V_CONST),
            Some(rv64::V_CONST)
        );
        assert_eq!(
            field_inline_column(field_constraints::V_X_RS1_VALUE),
            Some(rv64::V_RS1_VALUE)
        );
        assert_eq!(
            field_inline_column(field_constraints::V_X_RD_WRITE_VALUE),
            Some(rv64::V_RD_WRITE_VALUE)
        );
        assert_eq!(
            field_inline_column(field_constraints::V_IMM),
            Some(rv64::V_IMM)
        );
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn field_inline_selected_outer_geometry_includes_field_rows() {
        assert_eq!(
            OUTER_EQ_ROW_COUNT,
            rv64::NUM_EQ_CONSTRAINTS + field_constraints::NUM_EQ_CONSTRAINTS
        );
        assert_eq!(OUTER_UNISKIP_DOMAIN_SIZE, 14);
        assert_eq!(OUTER_UNISKIP_FIRST_ROUND_DEGREE, 39);
        assert_eq!(OUTER_REMAINDER_DEGREE, 3);
        assert_eq!(
            &OUTER_FIRST_GROUP_ROWS[10..],
            &[
                rv64::NUM_EQ_CONSTRAINTS + field_constraints::ROW_FADD,
                rv64::NUM_EQ_CONSTRAINTS + field_constraints::ROW_FSUB,
                rv64::NUM_EQ_CONSTRAINTS + field_constraints::ROW_FMUL,
                rv64::NUM_EQ_CONSTRAINTS + field_constraints::ROW_FINV,
            ]
        );
        assert_eq!(
            &OUTER_SECOND_GROUP_ROWS[9..],
            &[
                rv64::NUM_EQ_CONSTRAINTS + field_constraints::ROW_ASSERT_EQ,
                rv64::NUM_EQ_CONSTRAINTS + field_constraints::ROW_LOAD_FROM_X,
                rv64::NUM_EQ_CONSTRAINTS + field_constraints::ROW_STORE_TO_X,
                rv64::NUM_EQ_CONSTRAINTS + field_constraints::ROW_LOAD_IMM,
            ]
        );
        assert_eq!(
            outer_row_weights(Fr::from_u64(2), Fr::from_u64(3)).map(|weights| weights.len()),
            Ok(OUTER_EQ_ROW_COUNT)
        );
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn field_inline_spartan_openings_match_appended_column_order() {
        assert_eq!(
            FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUT_COUNT,
            FIELD_INLINE_APPENDED_COLUMNS
        );
        assert_eq!(outer_output_openings().len(), FIELD_INLINE_APPENDED_COLUMNS);

        let expected_inputs = [
            FieldInlineVirtualPolynomial::FieldRs1Value,
            FieldInlineVirtualPolynomial::FieldRs2Value,
            FieldInlineVirtualPolynomial::FieldRdValue,
            FieldInlineVirtualPolynomial::FieldProduct,
            FieldInlineVirtualPolynomial::FieldInvProduct,
            FieldInlineVirtualPolynomial::FieldOpFlag(FieldInlineOpFlag::Add),
            FieldInlineVirtualPolynomial::FieldOpFlag(FieldInlineOpFlag::Sub),
            FieldInlineVirtualPolynomial::FieldOpFlag(FieldInlineOpFlag::Mul),
            FieldInlineVirtualPolynomial::FieldOpFlag(FieldInlineOpFlag::Inv),
            FieldInlineVirtualPolynomial::FieldOpFlag(FieldInlineOpFlag::AssertEq),
            FieldInlineVirtualPolynomial::FieldOpFlag(FieldInlineOpFlag::LoadFromX),
            FieldInlineVirtualPolynomial::FieldOpFlag(FieldInlineOpFlag::StoreToX),
            FieldInlineVirtualPolynomial::FieldOpFlag(FieldInlineOpFlag::LoadImm),
        ];
        assert_eq!(FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS, expected_inputs);

        let local_columns = [
            field_constraints::V_FIELD_RS1_VALUE,
            field_constraints::V_FIELD_RS2_VALUE,
            field_constraints::V_FIELD_RD_VALUE,
            field_constraints::V_FIELD_PRODUCT,
            field_constraints::V_FIELD_INV_PRODUCT,
            field_constraints::V_IS_FIELD_ADD,
            field_constraints::V_IS_FIELD_SUB,
            field_constraints::V_IS_FIELD_MUL,
            field_constraints::V_IS_FIELD_INV,
            field_constraints::V_IS_FIELD_ASSERT_EQ,
            field_constraints::V_IS_FIELD_LOAD_FROM_X,
            field_constraints::V_IS_FIELD_STORE_TO_X,
            field_constraints::V_IS_FIELD_LOAD_IMM,
        ];
        for (index, local_column) in local_columns.into_iter().enumerate() {
            assert_eq!(
                field_inline_column(local_column),
                Some(FIELD_INLINE_COLUMN_BASE + index)
            );
        }
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn field_inline_composed_constraints_share_constant_column() {
        let selected = jolt_constraints::<Fr>();
        let mut witness = vec![Fr::zero(); selected.num_vars];

        witness[rv64::V_CONST] = Fr::from_u64(1);
        witness[rv64::V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] = Fr::from_u64(1);
        witness[remap_field_inline_column(field_constraints::V_FIELD_RS1_VALUE)] = Fr::from_u64(5);
        witness[remap_field_inline_column(field_constraints::V_FIELD_RS2_VALUE)] = Fr::from_u64(7);
        witness[remap_field_inline_column(field_constraints::V_FIELD_RD_VALUE)] = Fr::from_u64(12);
        witness[remap_field_inline_column(field_constraints::V_FIELD_PRODUCT)] = Fr::from_u64(35);
        witness[remap_field_inline_column(field_constraints::V_FIELD_INV_PRODUCT)] =
            Fr::from_u64(60);
        witness[rv64::V_RS1_VALUE] = Fr::from_u64(12);
        witness[rv64::V_RD_WRITE_VALUE] = Fr::from_u64(5);
        witness[rv64::V_IMM] = Fr::from_u64(12);
        witness[remap_field_inline_column(field_constraints::V_IS_FIELD_ADD)] = Fr::from_u64(1);

        assert_eq!(selected.check_witness(&witness), Ok(()));
    }
}
