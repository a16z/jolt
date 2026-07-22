//! Native field-inline R1CS variable layout and constraints.
//!
//! This module defines the per-cycle local constraints for field-inline
//! instruction semantics. The bridge constraints here use a native single-field
//! representation for ordinary values; multi-limb bridge packing should be
//! layered on once those bridge payloads are explicit in the trace.

use crate::constraint::SparseRow;
use jolt_field::Field;

type ConstraintRows<F> = (Vec<SparseRow<F>>, Vec<SparseRow<F>>, Vec<SparseRow<F>>);

pub const V_CONST: usize = 0;

pub const V_FIELD_RS1_VALUE: usize = 1;
pub const V_FIELD_RS2_VALUE: usize = 2;
pub const V_FIELD_RD_VALUE: usize = 3;
pub const V_FIELD_PRODUCT: usize = 4;
pub const V_FIELD_INV_PRODUCT: usize = 5;

pub const V_X_RS1_VALUE: usize = 6;
pub const V_X_RD_WRITE_VALUE: usize = 7;
pub const V_IMM: usize = 8;

pub const V_IS_FIELD_ADD: usize = 9;
pub const V_IS_FIELD_SUB: usize = 10;
pub const V_IS_FIELD_MUL: usize = 11;
pub const V_IS_FIELD_INV: usize = 12;
pub const V_IS_FIELD_ASSERT_EQ: usize = 13;
pub const V_IS_FIELD_LOAD_FROM_X: usize = 14;
pub const V_IS_FIELD_STORE_TO_X: usize = 15;
pub const V_IS_FIELD_LOAD_IMM: usize = 16;

pub const NUM_R1CS_INPUTS: usize = NUM_VARS_PER_CYCLE - 1;
pub const NUM_VARS_PER_CYCLE: usize = 17;

pub const ROW_FADD: usize = 0;
pub const ROW_FSUB: usize = 1;
pub const ROW_FMUL: usize = 2;
pub const ROW_FINV: usize = 3;
pub const ROW_ASSERT_EQ: usize = 4;
pub const ROW_LOAD_FROM_X: usize = 5;
pub const ROW_STORE_TO_X: usize = 6;
pub const ROW_LOAD_IMM: usize = 7;
pub const NUM_EQ_CONSTRAINTS: usize = 8;

pub const ROW_FIELD_PRODUCT: usize = NUM_EQ_CONSTRAINTS;
pub const ROW_FIELD_INV_PRODUCT: usize = NUM_EQ_CONSTRAINTS + 1;
pub const NUM_PRODUCT_CONSTRAINTS: usize = 2;
pub const NUM_CONSTRAINTS_PER_CYCLE: usize = NUM_EQ_CONSTRAINTS + NUM_PRODUCT_CONSTRAINTS;

pub const fn const_column() -> usize {
    V_CONST
}

pub const fn input_column(input_index: usize) -> Option<usize> {
    if input_index < NUM_R1CS_INPUTS {
        Some(1 + input_index)
    } else {
        None
    }
}

fn row<F: Field>(entries: &[(usize, i64)]) -> SparseRow<F> {
    entries
        .iter()
        .filter(|(_, coefficient)| *coefficient != 0)
        .map(|&(index, coefficient)| (index, F::from_i64(coefficient)))
        .collect()
}

fn field_eq_constraint_rows<F: Field>() -> ConstraintRows<F> {
    let mut a_rows = Vec::with_capacity(NUM_EQ_CONSTRAINTS);
    let mut b_rows = Vec::with_capacity(NUM_EQ_CONSTRAINTS);
    let mut c_rows = Vec::with_capacity(NUM_EQ_CONSTRAINTS);

    let empty = || Vec::new();

    a_rows.push(row::<F>(&[(V_IS_FIELD_ADD, 1)]));
    b_rows.push(row::<F>(&[
        (V_FIELD_RS1_VALUE, 1),
        (V_FIELD_RS2_VALUE, 1),
        (V_FIELD_RD_VALUE, -1),
    ]));
    c_rows.push(empty());

    a_rows.push(row::<F>(&[(V_IS_FIELD_SUB, 1)]));
    b_rows.push(row::<F>(&[
        (V_FIELD_RS1_VALUE, 1),
        (V_FIELD_RS2_VALUE, -1),
        (V_FIELD_RD_VALUE, -1),
    ]));
    c_rows.push(empty());

    a_rows.push(row::<F>(&[(V_IS_FIELD_MUL, 1)]));
    b_rows.push(row::<F>(&[(V_FIELD_PRODUCT, 1), (V_FIELD_RD_VALUE, -1)]));
    c_rows.push(empty());

    a_rows.push(row::<F>(&[(V_IS_FIELD_INV, 1)]));
    b_rows.push(row::<F>(&[(V_FIELD_INV_PRODUCT, 1), (V_CONST, -1)]));
    c_rows.push(empty());

    a_rows.push(row::<F>(&[(V_IS_FIELD_ASSERT_EQ, 1)]));
    b_rows.push(row::<F>(&[(V_FIELD_RS1_VALUE, 1), (V_FIELD_RS2_VALUE, -1)]));
    c_rows.push(empty());

    a_rows.push(row::<F>(&[(V_IS_FIELD_LOAD_FROM_X, 1)]));
    b_rows.push(row::<F>(&[(V_FIELD_RD_VALUE, 1), (V_X_RS1_VALUE, -1)]));
    c_rows.push(empty());

    a_rows.push(row::<F>(&[(V_IS_FIELD_STORE_TO_X, 1)]));
    b_rows.push(row::<F>(&[
        (V_X_RD_WRITE_VALUE, 1),
        (V_FIELD_RS1_VALUE, -1),
    ]));
    c_rows.push(empty());

    a_rows.push(row::<F>(&[(V_IS_FIELD_LOAD_IMM, 1)]));
    b_rows.push(row::<F>(&[(V_FIELD_RD_VALUE, 1), (V_IMM, -1)]));
    c_rows.push(empty());

    (a_rows, b_rows, c_rows)
}

fn append_product_constraints<F: Field>(
    a_rows: &mut Vec<SparseRow<F>>,
    b_rows: &mut Vec<SparseRow<F>>,
    c_rows: &mut Vec<SparseRow<F>>,
) {
    a_rows.push(row::<F>(&[(V_FIELD_RS1_VALUE, 1)]));
    b_rows.push(row::<F>(&[(V_FIELD_RS2_VALUE, 1)]));
    c_rows.push(row::<F>(&[(V_FIELD_PRODUCT, 1)]));

    a_rows.push(row::<F>(&[(V_FIELD_RS1_VALUE, 1)]));
    b_rows.push(row::<F>(&[(V_FIELD_RD_VALUE, 1)]));
    c_rows.push(row::<F>(&[(V_FIELD_INV_PRODUCT, 1)]));
}

/// Build only field-inline guarded equality constraints.
///
/// Product constraints are intentionally excluded for consumers that handle the
/// field multiplication checks in a separate protocol step.
pub fn field_inline_spartan_outer_constraints<F: Field>() -> crate::ConstraintMatrices<F> {
    let (a_rows, b_rows, c_rows) = field_eq_constraint_rows();
    crate::ConstraintMatrices::new(
        NUM_EQ_CONSTRAINTS,
        NUM_VARS_PER_CYCLE,
        a_rows,
        b_rows,
        c_rows,
    )
}

/// Build the full native field-inline R1CS constraint matrices.
///
/// Returns 10 constraints over 17 variables per cycle:
/// - 8 equality-conditional rows: `guard * (left - right) = 0`
/// - 2 product rows for `FieldProduct` and `FieldInvProduct`
pub fn field_inline_trace_constraints<F: Field>() -> crate::ConstraintMatrices<F> {
    let (mut a_rows, mut b_rows, mut c_rows) = field_eq_constraint_rows();
    a_rows.reserve(NUM_PRODUCT_CONSTRAINTS);
    b_rows.reserve(NUM_PRODUCT_CONSTRAINTS);
    c_rows.reserve(NUM_PRODUCT_CONSTRAINTS);
    append_product_constraints(&mut a_rows, &mut b_rows, &mut c_rows);

    crate::ConstraintMatrices::new(
        NUM_CONSTRAINTS_PER_CYCLE,
        NUM_VARS_PER_CYCLE,
        a_rows,
        b_rows,
        c_rows,
    )
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests may unwind via panic")]
mod tests {
    use super::*;
    use jolt_field::{FieldCore, Fr, FromPrimitiveInt};
    use num_traits::Zero;

    fn witness(field_rs1: Fr, field_rs2: Fr, field_rd: Fr, flags: &[(usize, Fr)]) -> Vec<Fr> {
        let mut witness = vec![Fr::zero(); NUM_VARS_PER_CYCLE];
        witness[V_CONST] = Fr::from_u64(1);
        witness[V_FIELD_RS1_VALUE] = field_rs1;
        witness[V_FIELD_RS2_VALUE] = field_rs2;
        witness[V_FIELD_RD_VALUE] = field_rd;
        witness[V_FIELD_PRODUCT] = field_rs1 * field_rs2;
        witness[V_FIELD_INV_PRODUCT] = field_rs1 * field_rd;
        witness[V_X_RS1_VALUE] = field_rd;
        witness[V_X_RD_WRITE_VALUE] = field_rs1;
        witness[V_IMM] = field_rd;
        for &(index, value) in flags {
            witness[index] = value;
        }
        witness
    }

    fn one() -> Fr {
        Fr::from_u64(1)
    }

    #[test]
    fn field_add_satisfies_constraints() {
        let witness = witness(
            Fr::from_u64(5),
            Fr::from_u64(7),
            Fr::from_u64(12),
            &[(V_IS_FIELD_ADD, one())],
        );

        field_inline_trace_constraints::<Fr>()
            .check_witness(&witness)
            .expect("FADD witness satisfies constraints");
    }

    #[test]
    fn field_sub_satisfies_constraints() {
        let witness = witness(
            Fr::from_u64(13),
            Fr::from_u64(5),
            Fr::from_u64(8),
            &[(V_IS_FIELD_SUB, one())],
        );

        field_inline_trace_constraints::<Fr>()
            .check_witness(&witness)
            .expect("FSUB witness satisfies constraints");
    }

    #[test]
    fn field_mul_checks_product_is_destination() {
        let witness = witness(
            Fr::from_u64(5),
            Fr::from_u64(7),
            Fr::from_u64(35),
            &[(V_IS_FIELD_MUL, one())],
        );

        field_inline_trace_constraints::<Fr>()
            .check_witness(&witness)
            .expect("FMUL witness satisfies constraints");
    }

    #[test]
    fn field_mul_rejects_bad_destination() {
        let witness = witness(
            Fr::from_u64(5),
            Fr::from_u64(7),
            Fr::from_u64(36),
            &[(V_IS_FIELD_MUL, one())],
        );

        assert_eq!(
            field_inline_trace_constraints::<Fr>().check_witness(&witness),
            Err(ROW_FMUL)
        );
    }

    #[test]
    fn product_row_rejects_bad_field_product() {
        let mut witness = witness(Fr::from_u64(5), Fr::from_u64(7), Fr::from_u64(35), &[]);
        witness[V_FIELD_PRODUCT] = Fr::from_u64(34);

        assert_eq!(
            field_inline_trace_constraints::<Fr>().check_witness(&witness),
            Err(ROW_FIELD_PRODUCT)
        );
    }

    #[test]
    fn inactive_field_mul_does_not_pin_destination_to_product() {
        let witness = witness(Fr::from_u64(5), Fr::from_u64(7), Fr::from_u64(99), &[]);

        field_inline_trace_constraints::<Fr>()
            .check_witness(&witness)
            .expect("inactive FMUL guard leaves destination unconstrained");
    }

    #[test]
    fn field_inverse_uses_intermediate_product() {
        let field_rs1 = Fr::from_u64(5);
        let field_rd = field_rs1
            .inverse()
            .expect("nonzero test element has inverse");
        let witness = witness(
            field_rs1,
            Fr::from_u64(9),
            field_rd,
            &[(V_IS_FIELD_INV, one())],
        );

        field_inline_trace_constraints::<Fr>()
            .check_witness(&witness)
            .expect("FINV witness satisfies constraints");
    }

    #[test]
    fn field_inverse_rejects_bad_inverse() {
        let witness = witness(
            Fr::from_u64(5),
            Fr::from_u64(9),
            Fr::from_u64(8),
            &[(V_IS_FIELD_INV, one())],
        );

        assert_eq!(
            field_inline_trace_constraints::<Fr>().check_witness(&witness),
            Err(ROW_FINV)
        );
    }

    #[test]
    fn field_assert_eq_checks_inputs_match() {
        let witness = witness(
            Fr::from_u64(11),
            Fr::from_u64(11),
            Fr::from_u64(4),
            &[(V_IS_FIELD_ASSERT_EQ, one())],
        );

        field_inline_trace_constraints::<Fr>()
            .check_witness(&witness)
            .expect("ASSERT_EQ witness satisfies constraints");
    }

    #[test]
    fn native_identity_bridge_constraints_are_gated() {
        let mut witness = witness(Fr::from_u64(5), Fr::from_u64(7), Fr::from_u64(42), &[]);
        witness[V_X_RS1_VALUE] = Fr::from_u64(42);
        witness[V_X_RD_WRITE_VALUE] = Fr::from_u64(5);
        witness[V_IMM] = Fr::from_u64(42);

        for selector in [
            V_IS_FIELD_LOAD_FROM_X,
            V_IS_FIELD_STORE_TO_X,
            V_IS_FIELD_LOAD_IMM,
        ] {
            let mut active = witness.clone();
            active[selector] = one();
            field_inline_trace_constraints::<Fr>()
                .check_witness(&active)
                .expect("native identity bridge witness satisfies constraints");
        }
    }

    #[test]
    fn input_columns_follow_const_then_inputs_layout() {
        assert_eq!(const_column(), V_CONST);
        assert_eq!(input_column(0), Some(V_FIELD_RS1_VALUE));
        assert_eq!(input_column(NUM_R1CS_INPUTS - 1), Some(V_IS_FIELD_LOAD_IMM));
        assert_eq!(input_column(NUM_R1CS_INPUTS), None);
    }
}
