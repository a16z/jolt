//! Compile-time Jolt R1CS composition.

use jolt_claims::protocols::jolt::geometry::{dimensions, spartan::SPARTAN_OUTER_R1CS_INPUTS};
use jolt_claims::protocols::jolt::JoltVirtualPolynomial;
use jolt_field::JoltField;
use jolt_poly::{
    lagrange::{centered_lagrange_evals, centered_lagrange_kernel, CenteredIntegerDomainError},
    EqPolynomial,
};
use jolt_riscv::CircuitFlags;
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

#[cfg(feature = "field-inline")]
pub const SPARTAN_PRODUCT_LANES: usize =
    rv64::NUM_PRODUCT_CONSTRAINTS + field_constraints::NUM_PRODUCT_CONSTRAINTS;

#[cfg(not(feature = "field-inline"))]
pub const SPARTAN_PRODUCT_LANES: usize = rv64::NUM_PRODUCT_CONSTRAINTS;

// The uni-skip geometry is owned by `jolt-claims` (it cannot depend on this
// crate). These assertions pin the constraint tables built here to the row and
// lane counts that geometry was derived from.
const _: () = assert!(
    SPARTAN_OUTER_ROW_COUNT == dimensions::SPARTAN_OUTER_ROW_COUNT,
    "Spartan outer eq-constraint row count diverges from jolt-claims geometry"
);
const _: () = assert!(
    SPARTAN_PRODUCT_LANES == dimensions::SPARTAN_PRODUCT_LANES,
    "Spartan product lane count diverges from jolt-claims geometry"
);

pub use dimensions::{
    OUTER_UNISKIP_DOMAIN_SIZE as SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE,
    OUTER_UNISKIP_FIRST_ROUND_DEGREE as SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
    PRODUCT_UNISKIP_DOMAIN_SIZE as SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
    PRODUCT_UNISKIP_FIRST_ROUND_DEGREE as SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
};

pub const SPARTAN_OUTER_REMAINDER_DEGREE: usize = 3;
pub const SPARTAN_OUTER_SECOND_GROUP_ROW_COUNT: usize =
    SPARTAN_OUTER_ROW_COUNT - SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE;

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

/// Bitmask of `rows`, panicking at compile time on a repeated row.
#[expect(
    clippy::indexing_slicing,
    reason = "const evaluation: `index < rows.len()` is the loop bound, and a failure would be a compile error"
)]
const fn row_group_mask(rows: &[usize]) -> u64 {
    let mut mask = 0u64;
    let mut index = 0;
    while index < rows.len() {
        let bit = 1u64 << rows[index];
        assert!(mask & bit == 0, "Spartan outer row group repeats a row");
        mask |= bit;
        index += 1;
    }
    mask
}

// The two row groups must partition `0..SPARTAN_OUTER_ROW_COUNT`: a dropped
// row silently unweights its constraint and a duplicated one double-weights
// it, and the prover shares `spartan_outer_row_weights`, so such proofs would
// still verify.
const _: () = {
    assert!(SPARTAN_OUTER_ROW_COUNT <= u64::BITS as usize);
    let first = row_group_mask(&SPARTAN_OUTER_FIRST_GROUP_ROWS);
    let second = row_group_mask(&SPARTAN_OUTER_SECOND_GROUP_ROWS);
    assert!(first & second == 0, "Spartan outer row groups overlap");
    assert!(
        first | second == (1u64 << SPARTAN_OUTER_ROW_COUNT) - 1,
        "Spartan outer row groups do not cover every constraint row"
    );
};

/// The RV64 witness column each Spartan outer R1CS input is opened as, by
/// name. `None` for virtual polynomials that are not R1CS inputs.
const fn rv64_input_column(input: JoltVirtualPolynomial) -> Option<usize> {
    match input {
        JoltVirtualPolynomial::LeftInstructionInput => Some(rv64::V_LEFT_INSTRUCTION_INPUT),
        JoltVirtualPolynomial::RightInstructionInput => Some(rv64::V_RIGHT_INSTRUCTION_INPUT),
        JoltVirtualPolynomial::Product => Some(rv64::V_PRODUCT),
        JoltVirtualPolynomial::ShouldBranch => Some(rv64::V_SHOULD_BRANCH),
        JoltVirtualPolynomial::PC => Some(rv64::V_PC),
        JoltVirtualPolynomial::UnexpandedPC => Some(rv64::V_UNEXPANDED_PC),
        JoltVirtualPolynomial::Imm => Some(rv64::V_IMM),
        JoltVirtualPolynomial::RamAddress => Some(rv64::V_RAM_ADDRESS),
        JoltVirtualPolynomial::Rs1Value => Some(rv64::V_RS1_VALUE),
        JoltVirtualPolynomial::Rs2Value => Some(rv64::V_RS2_VALUE),
        JoltVirtualPolynomial::RdWriteValue => Some(rv64::V_RD_WRITE_VALUE),
        JoltVirtualPolynomial::RamReadValue => Some(rv64::V_RAM_READ_VALUE),
        JoltVirtualPolynomial::RamWriteValue => Some(rv64::V_RAM_WRITE_VALUE),
        JoltVirtualPolynomial::LeftLookupOperand => Some(rv64::V_LEFT_LOOKUP_OPERAND),
        JoltVirtualPolynomial::RightLookupOperand => Some(rv64::V_RIGHT_LOOKUP_OPERAND),
        JoltVirtualPolynomial::NextUnexpandedPC => Some(rv64::V_NEXT_UNEXPANDED_PC),
        JoltVirtualPolynomial::NextPC => Some(rv64::V_NEXT_PC),
        JoltVirtualPolynomial::NextIsVirtual => Some(rv64::V_NEXT_IS_VIRTUAL),
        JoltVirtualPolynomial::NextIsFirstInSequence => Some(rv64::V_NEXT_IS_FIRST_IN_SEQUENCE),
        JoltVirtualPolynomial::LookupOutput => Some(rv64::V_LOOKUP_OUTPUT),
        JoltVirtualPolynomial::ShouldJump => Some(rv64::V_SHOULD_JUMP),
        JoltVirtualPolynomial::OpFlags(flag) => Some(match flag {
            CircuitFlags::AddOperands => rv64::V_FLAG_ADD_OPERANDS,
            CircuitFlags::SubtractOperands => rv64::V_FLAG_SUBTRACT_OPERANDS,
            CircuitFlags::MultiplyOperands => rv64::V_FLAG_MULTIPLY_OPERANDS,
            CircuitFlags::Load => rv64::V_FLAG_LOAD,
            CircuitFlags::Store => rv64::V_FLAG_STORE,
            CircuitFlags::Jump => rv64::V_FLAG_JUMP,
            CircuitFlags::WriteLookupOutputToRD => rv64::V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD,
            CircuitFlags::VirtualInstruction => rv64::V_FLAG_VIRTUAL_INSTRUCTION,
            CircuitFlags::Assert => rv64::V_FLAG_ASSERT,
            CircuitFlags::DoNotUpdateUnexpandedPC => rv64::V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC,
            CircuitFlags::Advice => rv64::V_FLAG_ADVICE,
            CircuitFlags::IsCompressed => rv64::V_FLAG_IS_COMPRESSED,
            CircuitFlags::IsFirstInSequence => rv64::V_FLAG_IS_FIRST_IN_SEQUENCE,
            CircuitFlags::IsLastInSequence => rv64::V_FLAG_IS_LAST_IN_SEQUENCE,
        }),
        JoltVirtualPolynomial::NextIsNoop
        | JoltVirtualPolynomial::Rd
        | JoltVirtualPolynomial::Rs1Ra
        | JoltVirtualPolynomial::Rs2Ra
        | JoltVirtualPolynomial::RdWa
        | JoltVirtualPolynomial::InstructionRaf
        | JoltVirtualPolynomial::InstructionRafFlag
        | JoltVirtualPolynomial::InstructionRa(_)
        | JoltVirtualPolynomial::RegistersVal
        | JoltVirtualPolynomial::RamRa
        | JoltVirtualPolynomial::RamVal
        | JoltVirtualPolynomial::RamValInit
        | JoltVirtualPolynomial::RamValFinal
        | JoltVirtualPolynomial::RamHammingWeight
        | JoltVirtualPolynomial::UnivariateSkip
        | JoltVirtualPolynomial::InstructionFlags(_)
        | JoltVirtualPolynomial::LookupTableFlag(_)
        | JoltVirtualPolynomial::BytecodeValClaim(_)
        | JoltVirtualPolynomial::BytecodeReadRafAddrClaim
        | JoltVirtualPolynomial::BooleanityAddrClaim
        | JoltVirtualPolynomial::BytecodeClaimReductionIntermediate
        | JoltVirtualPolynomial::ProgramImageInitContributionRw
        | JoltVirtualPolynomial::FusedInc => None,
    }
}

// `SPARTAN_OUTER_R1CS_INPUTS` (the opening order) and the `rv64::V_*` columns
// (the constraint tables) are otherwise tied only by position: a reorder of
// either relabels the constraint system identically for prover and verifier.
#[expect(
    clippy::indexing_slicing,
    reason = "const evaluation: `index < NUM_R1CS_INPUTS` is the loop bound, and a failure would be a compile error"
)]
const _: () = {
    assert!(SPARTAN_OUTER_R1CS_INPUTS.len() == rv64::NUM_R1CS_INPUTS);
    let mut index = 0;
    while index < rv64::NUM_R1CS_INPUTS {
        match rv64_input_column(SPARTAN_OUTER_R1CS_INPUTS[index]) {
            Some(column) => assert!(
                column == rv64::V_LEFT_INSTRUCTION_INPUT + index,
                "SPARTAN_OUTER_R1CS_INPUTS order disagrees with the rv64 column layout"
            ),
            None => panic!("SPARTAN_OUTER_R1CS_INPUTS names a polynomial without an rv64 column"),
        }
        index += 1;
    }
};

pub fn spartan_outer_constraints<F: JoltField>() -> ConstraintMatrices<F> {
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

pub fn trace_constraints<F: JoltField>() -> ConstraintMatrices<F> {
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

pub fn spartan_outer_row_weights<F: JoltField>(
    uniskip: F,
    stream: F,
) -> Result<Vec<F>, CenteredIntegerDomainError> {
    let lagrange_weights = centered_lagrange_evals(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, uniskip)?;
    // The row-group arrays are typed to the domain size, so only a short
    // weight vector could make the zips below drop rows silently.
    debug_assert_eq!(lagrange_weights.len(), SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE);
    let mut weights = vec![F::zero(); SPARTAN_OUTER_ROW_COUNT];

    #[expect(
        clippy::indexing_slicing,
        reason = "SPARTAN_OUTER_FIRST_GROUP_ROWS entries are compile-time constants below SPARTAN_OUTER_ROW_COUNT"
    )]
    for (&row, &lagrange_weight) in SPARTAN_OUTER_FIRST_GROUP_ROWS.iter().zip(&lagrange_weights) {
        weights[row] += (F::one() - stream) * lagrange_weight;
    }
    #[expect(
        clippy::indexing_slicing,
        reason = "SPARTAN_OUTER_SECOND_GROUP_ROWS entries are compile-time constants below SPARTAN_OUTER_ROW_COUNT"
    )]
    for (&row, &lagrange_weight) in SPARTAN_OUTER_SECOND_GROUP_ROWS
        .iter()
        .zip(&lagrange_weights)
    {
        weights[row] += stream * lagrange_weight;
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
    TauKernel,
    AzWeight(usize),
    BzWeight(usize),
    AzConstant,
    BzConstant,
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
pub struct JoltSpartanOuterRemainder<F: JoltField> {
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

impl<F: JoltField> JoltSpartanOuterRemainder<F> {
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
        let mut coefficients = Vec::with_capacity(2 * count + 3);
        coefficients.push((JoltSpartanOuterPublic::TauKernel, self.tau_kernel));
        for (index, &weight) in self.az_coefficients.iter().enumerate() {
            coefficients.push((JoltSpartanOuterPublic::AzWeight(index), weight));
        }
        for (index, &weight) in self.bz_coefficients.iter().enumerate() {
            coefficients.push((JoltSpartanOuterPublic::BzWeight(index), weight));
        }
        coefficients.push((JoltSpartanOuterPublic::AzConstant, self.az_constant));
        coefficients.push((JoltSpartanOuterPublic::BzConstant, self.bz_constant));
        coefficients
    }
}

fn spartan_outer_tau_kernel<F: JoltField>(
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

    // `tau` is non-empty: `tau.len() == expected >= 1` is checked above.
    let Some((&tau_high, tau_low)) = tau.split_last() else {
        return Err(JoltSpartanOuterRemainderError::ChallengeLengthMismatch { expected, got: 0 });
    };
    let tau_high_bound_r0 =
        centered_lagrange_kernel(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, tau_high, uniskip)?;
    let mut reversed_challenges = remainder_challenges.to_vec();
    reversed_challenges.reverse();
    Ok(tau_high_bound_r0 * EqPolynomial::<F>::mle(tau_low, &reversed_challenges))
}

fn eval_linear_form<F: JoltField>(coefficients: &[F], constant: F, inputs: &[F]) -> F {
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
fn append_field_inline_columns<F: JoltField>(
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
fn remap_rows<F: JoltField>(rows: Vec<SparseRow<F>>) -> Vec<SparseRow<F>> {
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
#[cfg_attr(
    feature = "field-inline",
    expect(clippy::expect_used, reason = "tests may unwind via panic")
)]
mod tests {
    use super::*;
    #[cfg(feature = "field-inline")]
    use jolt_claims::protocols::field_inline::{
        geometry::spartan::{
            outer_output_openings, FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS,
            FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUT_COUNT,
        },
        FieldInlineOpFlag, FieldInlineVirtualPolynomial,
    };
    use jolt_field::{Fr, Ring};
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
    #[expect(clippy::indexing_slicing, reason = "tests index fixture data")]
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
        // The factored publics: the tau kernel, one Az and one Bz weight per
        // opening (appended field-inline columns included), and the two
        // affine constants.
        assert_eq!(formula.public_coefficients().len(), 2 * opening_count + 3);
    }

    #[cfg(feature = "field-inline")]
    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index fixture data")]
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
