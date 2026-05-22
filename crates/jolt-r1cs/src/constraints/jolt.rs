//! Compile-time Jolt R1CS composition.

use jolt_field::Field;
use jolt_poly::{
    lagrange::{centered_lagrange_evals, centered_lagrange_kernel, CenteredIntegerDomainError},
    EqPolynomial,
};
use thiserror::Error as ThisError;

#[cfg(feature = "field-inline")]
use crate::SparseRow;
use crate::{ConstraintMatrices, ConstraintMatrixEvalError};

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
pub const SPARTAN_OUTER_ROW_COUNT: usize =
    rv64::NUM_EQ_CONSTRAINTS + field_constraints::NUM_EQ_CONSTRAINTS;

#[cfg(not(feature = "field-inline"))]
pub const SPARTAN_OUTER_ROW_COUNT: usize = rv64::NUM_EQ_CONSTRAINTS;

pub const SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE: usize = SPARTAN_OUTER_ROW_COUNT.div_ceil(2);
pub const SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE: usize =
    3 * SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE - 3;
pub const SPARTAN_OUTER_REMAINDER_DEGREE: usize = 3;
pub const SPARTAN_OUTER_SECOND_GROUP_ROW_COUNT: usize =
    SPARTAN_OUTER_ROW_COUNT - SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE;
pub const SPARTAN_PRODUCT_BASE_LANES: usize = 3;

#[cfg(feature = "field-inline")]
pub const SPARTAN_PRODUCT_FIELD_INLINE_LANES: usize = field_constraints::NUM_PRODUCT_CONSTRAINTS;

#[cfg(not(feature = "field-inline"))]
pub const SPARTAN_PRODUCT_FIELD_INLINE_LANES: usize = 0;

pub const SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE: usize =
    SPARTAN_PRODUCT_BASE_LANES + SPARTAN_PRODUCT_FIELD_INLINE_LANES;
pub const SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE: usize =
    3 * (SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE - 1);

#[cfg(not(feature = "field-inline"))]
pub const SPARTAN_OUTER_FIRST_GROUP_ROWS: [usize; SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE] =
    [1, 2, 3, 4, 5, 6, 11, 14, 17, 18];

#[cfg(feature = "field-inline")]
pub const SPARTAN_OUTER_FIRST_GROUP_ROWS: [usize; SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE] = [
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
pub const SPARTAN_OUTER_SECOND_GROUP_ROWS: [usize; SPARTAN_OUTER_SECOND_GROUP_ROW_COUNT] =
    [0, 7, 8, 9, 10, 12, 13, 15, 16];

#[cfg(feature = "field-inline")]
pub const SPARTAN_OUTER_SECOND_GROUP_ROWS: [usize; SPARTAN_OUTER_SECOND_GROUP_ROW_COUNT] = [
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

pub fn spartan_outer_constraints<F: Field>() -> ConstraintMatrices<F> {
    let constraints = rv64::rv64_spartan_outer_constraints();
    #[cfg(feature = "field-inline")]
    {
        append_field_inline_columns(
            constraints,
            field_constraints::field_inline_spartan_outer_constraints(),
        )
    }
    #[cfg(not(feature = "field-inline"))]
    {
        constraints
    }
}

pub fn trace_constraints<F: Field>() -> ConstraintMatrices<F> {
    let constraints = rv64::rv64_trace_constraints();
    #[cfg(feature = "field-inline")]
    {
        append_field_inline_columns(
            constraints,
            field_constraints::field_inline_trace_constraints(),
        )
    }
    #[cfg(not(feature = "field-inline"))]
    {
        constraints
    }
}

pub fn spartan_outer_row_weights<F: Field>(
    uniskip: F,
    stream: F,
) -> Result<Vec<F>, CenteredIntegerDomainError> {
    let lagrange_weights = centered_lagrange_evals(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, uniskip)?;
    let mut weights = vec![F::zero(); SPARTAN_OUTER_ROW_COUNT];

    for (index, &row) in SPARTAN_OUTER_FIRST_GROUP_ROWS.iter().enumerate() {
        weights[row] += (F::one() - stream) * lagrange_weights[index];
    }
    for (index, &row) in SPARTAN_OUTER_SECOND_GROUP_ROWS.iter().enumerate() {
        weights[row] += stream * lagrange_weights[index];
    }

    Ok(weights)
}

pub fn spartan_outer_opening_columns() -> Vec<usize> {
    let columns = (0..rv64::NUM_R1CS_INPUTS)
        .map(|index| rv64::V_LEFT_INSTRUCTION_INPUT + index)
        .collect::<Vec<_>>();

    #[cfg(feature = "field-inline")]
    {
        let mut columns = columns;
        columns.extend(
            FIELD_INLINE_COLUMN_BASE..FIELD_INLINE_COLUMN_BASE + FIELD_INLINE_APPENDED_COLUMNS,
        );
        columns
    }

    #[cfg(not(feature = "field-inline"))]
    columns
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoltSpartanOuterPublic {
    QuadraticCoefficient { left: usize, right: usize },
    LinearCoefficient(usize),
    ConstantCoefficient,
}

#[derive(Clone, Debug, ThisError, PartialEq, Eq)]
pub enum JoltSpartanOuterRemainderError {
    #[error("missing Spartan outer remainder stream challenge")]
    MissingStreamChallenge,
    #[error("{0}")]
    InvalidUniskipDomain(#[from] CenteredIntegerDomainError),
    #[error("challenge length mismatch: expected {expected}, got {got}")]
    ChallengeLengthMismatch { expected: usize, got: usize },
    #[error("{0}")]
    Matrix(#[from] ConstraintMatrixEvalError),
    #[error("opening length mismatch: expected {expected}, got {got}")]
    OpeningLengthMismatch { expected: usize, got: usize },
    #[error("Spartan outer rows unexpectedly contribute to the C linear form")]
    UnexpectedCContribution,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JoltSpartanOuterRemainder<F: Field> {
    tau_kernel: F,
    az_coefficients: Vec<F>,
    bz_coefficients: Vec<F>,
    az_constant: F,
    bz_constant: F,
}

#[derive(Clone, Copy, Debug)]
pub struct JoltSpartanOuterRemainderChallenges<'a, F> {
    pub tau: &'a [F],
    pub uniskip: F,
    pub remainder: &'a [F],
}

impl<F: Field> JoltSpartanOuterRemainder<F> {
    pub fn new(
        challenges: JoltSpartanOuterRemainderChallenges<'_, F>,
    ) -> Result<Self, JoltSpartanOuterRemainderError> {
        let Some((&r_stream, _)) = challenges.remainder.split_first() else {
            return Err(JoltSpartanOuterRemainderError::MissingStreamChallenge);
        };

        let row_weights = spartan_outer_row_weights(challenges.uniskip, r_stream)?;
        let columns = spartan_outer_opening_columns();
        let matrices = spartan_outer_constraints::<F>();
        let weighted = matrices.weighted_columns(&row_weights, &columns)?;
        if weighted.c.iter().any(|coefficient| !coefficient.is_zero()) {
            return Err(JoltSpartanOuterRemainderError::UnexpectedCContribution);
        }

        let constant_contributions =
            matrices.public_column_contributions(&row_weights, rv64::const_column(), F::one())?;
        if !constant_contributions.c.is_zero() {
            return Err(JoltSpartanOuterRemainderError::UnexpectedCContribution);
        }

        Ok(Self {
            tau_kernel: spartan_outer_tau_kernel(
                challenges.tau,
                challenges.uniskip,
                challenges.remainder,
            )?,
            az_coefficients: weighted.a,
            bz_coefficients: weighted.b,
            az_constant: constant_contributions.a,
            bz_constant: constant_contributions.b,
        })
    }

    pub fn expected_output_claim(
        &self,
        openings: &[F],
    ) -> Result<F, JoltSpartanOuterRemainderError> {
        let expected = self.az_coefficients.len();
        if openings.len() != expected {
            return Err(JoltSpartanOuterRemainderError::OpeningLengthMismatch {
                expected,
                got: openings.len(),
            });
        }

        Ok(self.tau_kernel
            * eval_linear_form(&self.az_coefficients, self.az_constant, openings)
            * eval_linear_form(&self.bz_coefficients, self.bz_constant, openings))
    }

    pub fn public_coefficients(&self) -> Vec<(JoltSpartanOuterPublic, F)> {
        let count = self.az_coefficients.len();
        let mut coefficients = Vec::with_capacity(count * count + count + 1);
        for left in 0..count {
            for right in 0..count {
                coefficients.push((
                    JoltSpartanOuterPublic::QuadraticCoefficient { left, right },
                    self.tau_kernel * self.az_coefficients[left] * self.bz_coefficients[right],
                ));
            }
        }
        for index in 0..count {
            let coefficient = self.az_coefficients[index] * self.bz_constant
                + self.az_constant * self.bz_coefficients[index];
            coefficients.push((
                JoltSpartanOuterPublic::LinearCoefficient(index),
                self.tau_kernel * coefficient,
            ));
        }
        coefficients.push((
            JoltSpartanOuterPublic::ConstantCoefficient,
            self.tau_kernel * self.az_constant * self.bz_constant,
        ));
        coefficients
    }
}

fn spartan_outer_tau_kernel<F: Field>(
    tau: &[F],
    uniskip: F,
    remainder_challenges: &[F],
) -> Result<F, JoltSpartanOuterRemainderError> {
    let expected = remainder_challenges.len() + 1;
    if tau.len() != expected {
        return Err(JoltSpartanOuterRemainderError::ChallengeLengthMismatch {
            expected,
            got: tau.len(),
        });
    }

    let tau_high = tau[tau.len() - 1];
    let tau_high_bound_r0 =
        centered_lagrange_kernel(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, tau_high, uniskip)?;
    let mut reversed_challenges = remainder_challenges.to_vec();
    reversed_challenges.reverse();
    Ok(tau_high_bound_r0 * EqPolynomial::<F>::mle(&tau[..tau.len() - 1], &reversed_challenges))
}

fn eval_linear_form<F: Field>(coefficients: &[F], constant: F, inputs: &[F]) -> F {
    coefficients
        .iter()
        .zip(inputs)
        .fold(constant, |acc, (&coefficient, &input)| {
            acc + coefficient * input
        })
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
#[expect(clippy::expect_used, reason = "tests may unwind via panic")]
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
    #[cfg(not(feature = "field-inline"))]
    use jolt_claims::protocols::jolt::{
        formulas::spartan::SpartanOuterDimensions, SpartanOuterPublic,
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    #[cfg(feature = "field-inline")]
    use num_traits::Zero;

    #[cfg(not(feature = "field-inline"))]
    #[test]
    fn default_composed_constraints_match_rv64_shape() {
        let composed = trace_constraints::<Fr>();
        let rv64 = rv64::rv64_trace_constraints::<Fr>();

        assert_eq!(composed.num_constraints, rv64.num_constraints);
        assert_eq!(composed.num_vars, rv64.num_vars);
        assert_eq!(composed.a, rv64.a);
        assert_eq!(composed.b, rv64.b);
        assert_eq!(composed.c, rv64.c);
    }

    #[cfg(not(feature = "field-inline"))]
    #[test]
    fn default_spartan_outer_geometry_matches_rv64() {
        assert_eq!(SPARTAN_OUTER_ROW_COUNT, rv64::NUM_EQ_CONSTRAINTS);
        assert_eq!(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, 10);
        assert_eq!(SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE, 27);
        assert_eq!(SPARTAN_OUTER_REMAINDER_DEGREE, 3);
        assert_eq!(
            SPARTAN_OUTER_FIRST_GROUP_ROWS,
            [1, 2, 3, 4, 5, 6, 11, 14, 17, 18]
        );
        assert_eq!(
            SPARTAN_OUTER_SECOND_GROUP_ROWS,
            [0, 7, 8, 9, 10, 12, 13, 15, 16]
        );
        assert_eq!(
            spartan_outer_row_weights(Fr::from_u64(2), Fr::from_u64(3))
                .map(|weights| weights.len()),
            Ok(rv64::NUM_EQ_CONSTRAINTS)
        );
        assert_eq!(
            spartan_outer_opening_columns(),
            (rv64::V_LEFT_INSTRUCTION_INPUT..=rv64::NUM_R1CS_INPUTS).collect::<Vec<_>>()
        );
    }

    #[cfg(not(feature = "field-inline"))]
    #[test]
    fn default_spartan_outer_remainder_matches_rv64() {
        let dimensions = SpartanOuterDimensions::rv64(3);
        let tau = [
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(4),
            Fr::from_u64(5),
            Fr::from_u64(6),
        ];
        let remainder = [
            Fr::from_u64(7),
            Fr::from_u64(8),
            Fr::from_u64(9),
            Fr::from_u64(10),
        ];
        let openings = (1..=rv64::NUM_R1CS_INPUTS as u64)
            .map(Fr::from_u64)
            .collect::<Vec<_>>();

        let rv64_remainder = rv64::Rv64SpartanOuterRemainder::new(
            &dimensions,
            rv64::Rv64SpartanOuterRemainderChallenges {
                tau: &tau,
                uniskip: Fr::from_u64(11),
                remainder: &remainder,
            },
        )
        .expect("RV64 remainder derives");
        let composed_remainder =
            JoltSpartanOuterRemainder::new(JoltSpartanOuterRemainderChallenges {
                tau: &tau,
                uniskip: Fr::from_u64(11),
                remainder: &remainder,
            })
            .expect("composed remainder derives");

        assert_eq!(
            composed_remainder
                .expected_output_claim(&openings)
                .expect("composed output claim evaluates"),
            rv64_remainder
                .expected_output_claim(&openings)
                .expect("RV64 output claim evaluates")
        );

        let composed_public = composed_remainder.public_coefficients();
        let rv64_public = rv64_remainder
            .public_claims(&dimensions)
            .expect("RV64 public coefficients derive");
        assert_eq!(composed_public.len(), rv64_public.len());
        for ((composed_id, composed_value), (rv64_id, rv64_value)) in
            composed_public.into_iter().zip(rv64_public)
        {
            let ids_match = match (composed_id, rv64_id) {
                (
                    JoltSpartanOuterPublic::QuadraticCoefficient { left, right },
                    SpartanOuterPublic::QuadraticCoefficient {
                        left: rv64_left,
                        right: rv64_right,
                    },
                ) => {
                    assert_eq!(left, rv64_left);
                    assert_eq!(right, rv64_right);
                    true
                }
                (
                    JoltSpartanOuterPublic::LinearCoefficient(index),
                    SpartanOuterPublic::LinearCoefficient(rv64_index),
                ) => {
                    assert_eq!(index, rv64_index);
                    true
                }
                (
                    JoltSpartanOuterPublic::ConstantCoefficient,
                    SpartanOuterPublic::ConstantCoefficient,
                ) => true,
                _ => false,
            };
            assert!(ids_match, "public coefficient kind mismatch");
            assert_eq!(composed_value, rv64_value);
        }
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn field_inline_composed_constraints_append_field_shape() {
        let composed = trace_constraints::<Fr>();

        assert_eq!(composed.num_constraints, NUM_CONSTRAINTS_PER_CYCLE);
        assert_eq!(composed.num_vars, NUM_VARS_PER_CYCLE);
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
    fn field_inline_spartan_outer_geometry_includes_field_rows() {
        assert_eq!(
            SPARTAN_OUTER_ROW_COUNT,
            rv64::NUM_EQ_CONSTRAINTS + field_constraints::NUM_EQ_CONSTRAINTS
        );
        assert_eq!(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, 14);
        assert_eq!(SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE, 39);
        assert_eq!(SPARTAN_OUTER_REMAINDER_DEGREE, 3);
        assert_eq!(
            &SPARTAN_OUTER_FIRST_GROUP_ROWS[10..],
            &[
                rv64::NUM_EQ_CONSTRAINTS + field_constraints::ROW_FADD,
                rv64::NUM_EQ_CONSTRAINTS + field_constraints::ROW_FSUB,
                rv64::NUM_EQ_CONSTRAINTS + field_constraints::ROW_FMUL,
                rv64::NUM_EQ_CONSTRAINTS + field_constraints::ROW_FINV,
            ]
        );
        assert_eq!(
            &SPARTAN_OUTER_SECOND_GROUP_ROWS[9..],
            &[
                rv64::NUM_EQ_CONSTRAINTS + field_constraints::ROW_ASSERT_EQ,
                rv64::NUM_EQ_CONSTRAINTS + field_constraints::ROW_LOAD_FROM_X,
                rv64::NUM_EQ_CONSTRAINTS + field_constraints::ROW_STORE_TO_X,
                rv64::NUM_EQ_CONSTRAINTS + field_constraints::ROW_LOAD_IMM,
            ]
        );
        assert_eq!(
            spartan_outer_row_weights(Fr::from_u64(2), Fr::from_u64(3))
                .map(|weights| weights.len()),
            Ok(SPARTAN_OUTER_ROW_COUNT)
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
        assert_eq!(
            spartan_outer_opening_columns()[rv64::NUM_R1CS_INPUTS..],
            (FIELD_INLINE_COLUMN_BASE..FIELD_INLINE_COLUMN_BASE + FIELD_INLINE_APPENDED_COLUMNS)
                .collect::<Vec<_>>()
        );
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn field_inline_spartan_outer_remainder_uses_appended_openings() {
        let tau = [
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(4),
            Fr::from_u64(5),
            Fr::from_u64(6),
        ];
        let remainder = [
            Fr::from_u64(7),
            Fr::from_u64(8),
            Fr::from_u64(9),
            Fr::from_u64(10),
        ];
        let formula = JoltSpartanOuterRemainder::new(JoltSpartanOuterRemainderChallenges {
            tau: &tau,
            uniskip: Fr::from_u64(11),
            remainder: &remainder,
        })
        .expect("composed field-inline remainder derives");
        let opening_count = spartan_outer_opening_columns().len();
        let openings = (1..=opening_count as u64)
            .map(Fr::from_u64)
            .collect::<Vec<_>>();

        let _output_claim = formula
            .expected_output_claim(&openings)
            .expect("field-inline output claim evaluates");
        assert_eq!(
            opening_count,
            rv64::NUM_R1CS_INPUTS + FIELD_INLINE_APPENDED_COLUMNS
        );
        assert_eq!(
            formula.public_coefficients().len(),
            opening_count * opening_count + opening_count + 1
        );
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn field_inline_composed_constraints_share_constant_column() {
        let composed = trace_constraints::<Fr>();
        let mut witness = vec![Fr::zero(); composed.num_vars];

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

        assert_eq!(composed.check_witness(&witness), Ok(()));
    }
}
