//! Native field-inline R1CS variable layout and constraints.
//!
//! This module defines the per-cycle local constraints for field-inline
//! instruction semantics. Ingress folds 64-bit limbs into a native field
//! accumulator; egress constrains an advised limb, and `AssertZero` pins a
//! field-register value to zero.
//!
//! Bridge rows (spec `field-inline-protocol.md`, "Conversion Rows"): the
//! x-register side of every bridge is a 64-bit RV64 column, so each row must
//! be range-sound, not just an identity.
//!
//! - `FIELD_LOAD_ACCUMULATE_FROM_REGISTER` folds the range-bound RV64 `Rs1Value`
//!   under `2^64 * FieldRs1Value`, where field-register checking binds
//!   `FieldRs1Value` to the destination's old value. `FIELD_LOAD_IMM` sets
//!   `FieldRdValue = Imm`, using the bytecode-bound immediate.
//! - `FIELD_ADVICE_LIMB` constrains `FieldRs1Value = RdWriteValue +
//!   2^64 * FieldRdValue`. RV64 row 12 and the `RangeCheck` lookup bound
//!   the advice limb below 2^64; canonicality belongs to the guest readout.

use crate::constraint::SparseRow;
use jolt_field::JoltField;

type ConstraintRows<F> = (Vec<SparseRow<F>>, Vec<SparseRow<F>>, Vec<SparseRow<F>>);

use super::rv64::{
    flag_column, NUM_VARS_PER_CYCLE as RV64_NUM_VARS_PER_CYCLE, V_CONST, V_IMM, V_RD_WRITE_VALUE,
    V_RS1_VALUE,
};
use jolt_riscv::CircuitFlags;

pub const V_FIELD_RS1_VALUE: usize = RV64_NUM_VARS_PER_CYCLE;
pub const V_FIELD_RS2_VALUE: usize = V_FIELD_RS1_VALUE + 1;
pub const V_FIELD_RD_VALUE: usize = V_FIELD_RS1_VALUE + 2;
pub const V_FIELD_PRODUCT: usize = V_FIELD_RS1_VALUE + 3;
pub const V_FIELD_INV_PRODUCT: usize = V_FIELD_RS1_VALUE + 4;
pub const NUM_FIELD_COLUMNS: usize = 5;
pub const NUM_VARS_PER_CYCLE: usize = RV64_NUM_VARS_PER_CYCLE + NUM_FIELD_COLUMNS;

pub const ROW_FADD: usize = 0;
pub const ROW_FSUB: usize = 1;
pub const ROW_FMUL: usize = 2;
pub const ROW_FINV: usize = 3;
pub const ROW_ASSERT_EQ: usize = 4;
pub const ROW_LOAD_ACCUMULATE_FROM_REGISTER: usize = 5;
pub const ROW_ASSERT_ZERO: usize = 6;
pub const ROW_LOAD_IMM: usize = 7;
/// `IsFieldLoadAccumulateFromMemory ·
/// (FieldRdValue − 2^64·FieldRs1Value − RdWriteValue) = 0`: RV64 load rows
/// bind the word in the integer destination, and field-register checking
/// binds `FieldRs1Value` to the destination's old value.
pub const ROW_LOAD_ACCUMULATE_FROM_MEMORY: usize = 8;
/// `IsFieldAdviceLimb · (FieldRs1Value − RdWriteValue − 2^64·FieldRdValue) = 0`:
/// the x-register write is a 64-bit advice limb (RV64 row 12 plus the `RangeCheck`
/// lookup bound it below 2^64) and the field destination the quotient.
pub const ROW_ADVICE_LIMB: usize = 9;
pub const NUM_EQ_CONSTRAINTS: usize = 10;

pub const ROW_FIELD_PRODUCT: usize = NUM_EQ_CONSTRAINTS;
pub const ROW_FIELD_INV_PRODUCT: usize = NUM_EQ_CONSTRAINTS + 1;
pub const NUM_PRODUCT_CONSTRAINTS: usize = 2;
pub const NUM_CONSTRAINTS_PER_CYCLE: usize = NUM_EQ_CONSTRAINTS + NUM_PRODUCT_CONSTRAINTS;

fn row<F: JoltField>(entries: &[(usize, i64)]) -> SparseRow<F> {
    entries
        .iter()
        .filter(|(_, coefficient)| *coefficient != 0)
        .map(|&(index, coefficient)| (index, F::from_i64(coefficient)))
        .collect()
}

/// Radix for folding a 64-bit limb into a field accumulator.
pub fn limb_radix<F: JoltField>() -> F {
    F::from_u128(1u128 << 64)
}

fn field_eq_constraint_rows<F: JoltField>() -> ConstraintRows<F> {
    let mut a_rows = Vec::with_capacity(NUM_EQ_CONSTRAINTS);
    let mut b_rows = Vec::with_capacity(NUM_EQ_CONSTRAINTS);
    let mut c_rows = Vec::with_capacity(NUM_EQ_CONSTRAINTS);

    let empty = || Vec::new();

    // Eq-conditional constraints (0-9), with arithmetic in the proof field.
    // Form: guard · (left − right) = 0  →  A=guard, B=left−right, C=0

    // 0: FieldAdd
    //    guard = IsFieldAdd
    //    left  = FieldRs1Value + FieldRs2Value
    //    right = FieldRdValue
    a_rows.push(row::<F>(&[(flag_column(CircuitFlags::FieldAdd), 1)]));
    b_rows.push(row::<F>(&[
        (V_FIELD_RS1_VALUE, 1),
        (V_FIELD_RS2_VALUE, 1),
        (V_FIELD_RD_VALUE, -1),
    ]));
    c_rows.push(empty());

    // 1: FieldSub
    //    guard = IsFieldSub
    //    left  = FieldRs1Value − FieldRs2Value
    //    right = FieldRdValue
    a_rows.push(row::<F>(&[(flag_column(CircuitFlags::FieldSub), 1)]));
    b_rows.push(row::<F>(&[
        (V_FIELD_RS1_VALUE, 1),
        (V_FIELD_RS2_VALUE, -1),
        (V_FIELD_RD_VALUE, -1),
    ]));
    c_rows.push(empty());

    // 2: FieldMulDestination
    //    guard = IsFieldMul
    //    left  = FieldProduct
    //    right = FieldRdValue
    // FieldProduct = FieldRs1Value · FieldRs2Value is checked separately.
    a_rows.push(row::<F>(&[(flag_column(CircuitFlags::FieldMul), 1)]));
    b_rows.push(row::<F>(&[(V_FIELD_PRODUCT, 1), (V_FIELD_RD_VALUE, -1)]));
    c_rows.push(empty());

    // 3: FieldInverseProduct
    //    guard = IsFieldInv
    //    left  = FieldInvProduct
    //    right = 1
    // FieldInvProduct = FieldRs1Value · FieldRdValue is checked separately.
    a_rows.push(row::<F>(&[(flag_column(CircuitFlags::FieldInv), 1)]));
    b_rows.push(row::<F>(&[(V_FIELD_INV_PRODUCT, 1), (V_CONST, -1)]));
    c_rows.push(empty());

    // 4: FieldAssertEq
    //    guard = IsFieldAssertEq
    //    left  = FieldRs1Value
    //    right = FieldRs2Value
    a_rows.push(row::<F>(&[(flag_column(CircuitFlags::FieldAssertEq), 1)]));
    b_rows.push(row::<F>(&[(V_FIELD_RS1_VALUE, 1), (V_FIELD_RS2_VALUE, -1)]));
    c_rows.push(empty());

    // 5: FieldLoadAccumulateFromRegister
    //    guard = IsFieldLoadAccumulateFromRegister
    //    left  = FieldRdValue
    //    right = 2^64 · FieldRs1Value + Rs1Value
    // Field-register checking binds FieldRs1Value to the destination's old value.
    a_rows.push(row::<F>(&[(
        flag_column(CircuitFlags::FieldLoadAccumulateFromRegister),
        1,
    )]));
    b_rows.push(vec![
        (V_FIELD_RD_VALUE, F::one()),
        (V_FIELD_RS1_VALUE, -limb_radix::<F>()),
        (V_RS1_VALUE, -F::one()),
    ]);
    c_rows.push(empty());

    // 6: FieldAssertZero
    //    guard = IsFieldAssertZero
    //    left  = FieldRs1Value
    //    right = 0
    a_rows.push(row::<F>(&[(flag_column(CircuitFlags::FieldAssertZero), 1)]));
    b_rows.push(row::<F>(&[(V_FIELD_RS1_VALUE, 1)]));
    c_rows.push(empty());

    // 7: FieldLoadImm
    //    guard = IsFieldLoadImm
    //    left  = FieldRdValue
    //    right = Imm
    a_rows.push(row::<F>(&[(flag_column(CircuitFlags::FieldLoadImm), 1)]));
    b_rows.push(row::<F>(&[(V_FIELD_RD_VALUE, 1), (V_IMM, -1)]));
    c_rows.push(empty());

    // 8: FieldLoadAccumulateFromMemory
    //    guard = IsFieldLoadAccumulateFromMemory
    //    left  = FieldRdValue
    //    right = 2^64 · FieldRs1Value + RdWriteValue
    // RV64 load rows bind RdWriteValue to the loaded word; field-register
    // checking binds FieldRs1Value to the destination's old value.
    a_rows.push(row::<F>(&[(
        flag_column(CircuitFlags::FieldLoadAccumulateFromMemory),
        1,
    )]));
    b_rows.push(vec![
        (V_FIELD_RD_VALUE, F::one()),
        (V_FIELD_RS1_VALUE, -limb_radix::<F>()),
        (V_RD_WRITE_VALUE, -F::one()),
    ]);
    c_rows.push(empty());

    // 9: FieldAdviceLimb
    //    guard = IsFieldAdviceLimb
    //    left  = FieldRs1Value
    //    right = RdWriteValue + 2^64 · FieldRdValue
    // FieldRdValue is a field quotient. RV64/lookup constraints range-check
    // the limb; canonical integer readout requires checks in the guest.
    a_rows.push(row::<F>(&[(flag_column(CircuitFlags::FieldAdviceLimb), 1)]));
    b_rows.push(vec![
        (V_FIELD_RS1_VALUE, F::one()),
        (V_RD_WRITE_VALUE, -F::one()),
        (V_FIELD_RD_VALUE, -limb_radix::<F>()),
    ]);
    c_rows.push(empty());

    (a_rows, b_rows, c_rows)
}

fn append_product_constraints<F: JoltField>(
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
pub fn field_inline_spartan_outer_constraints<F: JoltField>() -> crate::ConstraintMatrices<F> {
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
/// Returns 12 constraints using the shared RV64 columns and five field columns:
/// - 10 equality-conditional rows: `guard * (left - right) = 0`
/// - 2 product rows for `FieldProduct` and `FieldInvProduct`
pub fn field_inline_trace_constraints<F: JoltField>() -> crate::ConstraintMatrices<F> {
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
#[expect(clippy::indexing_slicing, reason = "tests index fixture data")]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr, Ring};
    use num_traits::Zero;

    fn witness(field_rs1: Fr, field_rs2: Fr, field_rd: Fr, flags: &[(usize, Fr)]) -> Vec<Fr> {
        let mut witness = vec![Fr::zero(); NUM_VARS_PER_CYCLE];
        witness[V_CONST] = Fr::from_u64(1);
        witness[V_FIELD_RS1_VALUE] = field_rs1;
        witness[V_FIELD_RS2_VALUE] = field_rs2;
        witness[V_FIELD_RD_VALUE] = field_rd;
        witness[V_FIELD_PRODUCT] = field_rs1 * field_rs2;
        witness[V_FIELD_INV_PRODUCT] = field_rs1 * field_rd;
        witness[V_RS1_VALUE] = field_rd;
        witness[V_RD_WRITE_VALUE] = field_rs1;
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
            &[(flag_column(CircuitFlags::FieldAdd), one())],
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
            &[(flag_column(CircuitFlags::FieldSub), one())],
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
            &[(flag_column(CircuitFlags::FieldMul), one())],
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
            &[(flag_column(CircuitFlags::FieldMul), one())],
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
            &[(flag_column(CircuitFlags::FieldInv), one())],
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
            &[(flag_column(CircuitFlags::FieldInv), one())],
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
            &[(flag_column(CircuitFlags::FieldAssertEq), one())],
        );

        field_inline_trace_constraints::<Fr>()
            .check_witness(&witness)
            .expect("ASSERT_EQ witness satisfies constraints");
    }

    #[test]
    fn assert_zero_accepts_only_zero_when_selected() {
        let constraints = field_inline_trace_constraints::<Fr>();
        for value in [Fr::zero(), one(), -one(), limb_radix::<Fr>()] {
            let mut row = witness(value, Fr::from_u64(7), Fr::from_u64(42), &[]);
            constraints
                .check_witness(&row)
                .expect("an inactive zero assertion leaves the value unconstrained");
            row[flag_column(CircuitFlags::FieldAssertZero)] = one();
            assert_eq!(
                constraints.check_witness(&row),
                if value.is_zero() {
                    Ok(())
                } else {
                    Err(ROW_ASSERT_ZERO)
                }
            );
        }
    }

    #[test]
    fn load_accumulate_rows_bind_the_word_and_old_destination() {
        let word = Fr::from_u64(0xdead_beef);
        let constraints = field_inline_trace_constraints::<Fr>();
        for (selector, source, row) in [
            (
                flag_column(CircuitFlags::FieldLoadAccumulateFromRegister),
                V_RS1_VALUE,
                ROW_LOAD_ACCUMULATE_FROM_REGISTER,
            ),
            (
                flag_column(CircuitFlags::FieldLoadAccumulateFromMemory),
                V_RD_WRITE_VALUE,
                ROW_LOAD_ACCUMULATE_FROM_MEMORY,
            ),
        ] {
            for high in [Fr::zero(), Fr::from_u64(7), -one()] {
                let mut load = witness(
                    high,
                    Fr::zero(),
                    high * limb_radix::<Fr>() + word,
                    &[(selector, one())],
                );
                load[source] = word;
                constraints
                    .check_witness(&load)
                    .expect("accumulation is valid");
                for column in [V_FIELD_RS1_VALUE, source, V_FIELD_RD_VALUE] {
                    let mut tampered = load.clone();
                    tampered[column] += one();
                    assert_eq!(constraints.check_witness(&tampered), Err(row));
                }
            }
        }
    }

    /// Advice fixes a residue relation, not a canonical integer decomposition.
    #[test]
    fn advice_limb_row_binds_the_low_limb_and_quotient() {
        let low = Fr::from_u64(0x1234_5678);
        let quotient = Fr::from_u64(9);
        let mut split = witness(
            low + quotient * limb_radix::<Fr>(),
            Fr::from_u64(0),
            quotient,
            &[(flag_column(CircuitFlags::FieldAdviceLimb), one())],
        );
        split[V_RD_WRITE_VALUE] = low;
        field_inline_trace_constraints::<Fr>()
            .check_witness(&split)
            .expect("a split writes the low limb and keeps the quotient");
        split[V_FIELD_RD_VALUE] = quotient + one();
        assert_eq!(
            field_inline_trace_constraints::<Fr>().check_witness(&split),
            Err(ROW_ADVICE_LIMB)
        );
    }

    #[test]
    fn assert_zero_rejects_residual_from_unchecked_advice_limb() {
        let quotient = -limb_radix::<Fr>().inverse().expect("nonzero limb radix");
        let mut row = witness(
            Fr::from_u64(0),
            Fr::from_u64(0),
            quotient,
            &[(flag_column(CircuitFlags::FieldAdviceLimb), one())],
        );
        row[V_RD_WRITE_VALUE] = one();
        field_inline_trace_constraints::<Fr>()
            .check_witness(&row)
            .expect("canonicality belongs to the complete guest readout");
        let residual = witness(
            quotient,
            Fr::zero(),
            Fr::zero(),
            &[(flag_column(CircuitFlags::FieldAssertZero), one())],
        );
        assert_eq!(
            field_inline_trace_constraints::<Fr>().check_witness(&residual),
            Err(ROW_ASSERT_ZERO)
        );
    }

    #[test]
    fn load_imm_binds_the_destination_to_the_bytecode_value() {
        let mut row = witness(
            Fr::from_u64(5),
            Fr::from_u64(7),
            Fr::from_u64(42),
            &[(flag_column(CircuitFlags::FieldLoadImm), one())],
        );
        let constraints = field_inline_trace_constraints::<Fr>();
        constraints.check_witness(&row).expect("matching immediate");
        row[V_IMM] += one();
        assert_eq!(constraints.check_witness(&row), Err(ROW_LOAD_IMM));
    }
}
