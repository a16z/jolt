//! Jolt RV64 R1CS variable layout.
//!
//! Defines the per-cycle witness variable indices and constraint counts
//! for the Jolt RV64IMAC R1CS constraint system.
//!
//! # Variable layout
//!
//! Each cycle has [`NUM_VARS_PER_CYCLE`] witness variables:
//!
//! | Range | Description |
//! |-------|-------------|
//! | `[0]` | Constant 1 |
//! | `[1..=35]` | R1CS inputs (registers, flags, PC, lookups) |
//! | `[36..=37]` | Product factor variables (`Branch`, `NextIsNoop`) |
//!
//! # Constraint forms
//!
//! - **Eq-conditional** (rows 0–18): `guard · (left − right) = 0`
//! - **Product** (rows 19–21): `left · right = output`

/// Constant-1 wire.
pub const V_CONST: usize = 0;

pub const V_LEFT_INSTRUCTION_INPUT: usize = 1;
pub const V_RIGHT_INSTRUCTION_INPUT: usize = 2;
pub const V_PRODUCT: usize = 3;
pub const V_SHOULD_BRANCH: usize = 4;
pub const V_PC: usize = 5;
pub const V_UNEXPANDED_PC: usize = 6;
pub const V_IMM: usize = 7;
pub const V_RAM_ADDRESS: usize = 8;
pub const V_RS1_VALUE: usize = 9;
pub const V_RS2_VALUE: usize = 10;
pub const V_RD_WRITE_VALUE: usize = 11;
pub const V_RAM_READ_VALUE: usize = 12;
pub const V_RAM_WRITE_VALUE: usize = 13;
pub const V_LEFT_LOOKUP_OPERAND: usize = 14;
pub const V_RIGHT_LOOKUP_OPERAND: usize = 15;
pub const V_NEXT_UNEXPANDED_PC: usize = 16;
pub const V_NEXT_PC: usize = 17;
pub const V_NEXT_IS_VIRTUAL: usize = 18;
pub const V_NEXT_IS_FIRST_IN_SEQUENCE: usize = 19;
pub const V_LOOKUP_OUTPUT: usize = 20;
pub const V_SHOULD_JUMP: usize = 21;

pub const V_FLAG_ADD_OPERANDS: usize = 22;
pub const V_FLAG_SUBTRACT_OPERANDS: usize = 23;
pub const V_FLAG_MULTIPLY_OPERANDS: usize = 24;
pub const V_FLAG_LOAD: usize = 25;
pub const V_FLAG_STORE: usize = 26;
pub const V_FLAG_JUMP: usize = 27;
pub const V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD: usize = 28;
pub const V_FLAG_VIRTUAL_INSTRUCTION: usize = 29;
pub const V_FLAG_ASSERT: usize = 30;
pub const V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC: usize = 31;
pub const V_FLAG_ADVICE: usize = 32;
pub const V_FLAG_IS_COMPRESSED: usize = 33;
pub const V_FLAG_IS_FIRST_IN_SEQUENCE: usize = 34;
pub const V_FLAG_IS_LAST_IN_SEQUENCE: usize = 35;

#[cfg(feature = "implicit-carry")]
pub const V_FLAG_USES_CARRY: usize = 36;
#[cfg(feature = "implicit-carry")]
pub const V_FLAG_PRODUCES_CARRY: usize = 37;
#[cfg(feature = "implicit-carry")]
pub const V_CARRY_USED: usize = 38;
#[cfg(feature = "implicit-carry")]
pub const V_NEXT_CARRY: usize = 39;

/// Number of implicit-carry extension inputs appended after the base 35.
#[cfg(not(feature = "implicit-carry"))]
pub const IMPLICIT_CARRY_EXTRA_INPUTS: usize = 0;
#[cfg(feature = "implicit-carry")]
pub const IMPLICIT_CARRY_EXTRA_INPUTS: usize = 4;

pub const V_BRANCH: usize = 36 + IMPLICIT_CARRY_EXTRA_INPUTS;
pub const V_NEXT_IS_NOOP: usize = 37 + IMPLICIT_CARRY_EXTRA_INPUTS;
/// Committed Carry as the right factor of the CarryUsed product row.
#[cfg(feature = "implicit-carry")]
pub const V_CARRY: usize = 38 + IMPLICIT_CARRY_EXTRA_INPUTS;

pub const NUM_R1CS_INPUTS: usize = 35 + IMPLICIT_CARRY_EXTRA_INPUTS;
pub const NUM_PRODUCT_FACTORS: usize = 2 + cfg!(feature = "implicit-carry") as usize;
pub const NUM_VARS_PER_CYCLE: usize = 1 + NUM_R1CS_INPUTS + NUM_PRODUCT_FACTORS;
pub const NUM_EQ_CONSTRAINTS: usize = 19 + 2 * cfg!(feature = "implicit-carry") as usize;
pub const NUM_PRODUCT_CONSTRAINTS: usize = 3 + cfg!(feature = "implicit-carry") as usize;
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

/// Two's complement bias for subtraction: 2^64.
const TWOS_COMPLEMENT_BIAS: i128 = 0x1_0000_0000_0000_0000;

use crate::constraint::SparseRow;
use jolt_field::Field;

type ConstraintRows<F> = (Vec<SparseRow<F>>, Vec<SparseRow<F>>, Vec<SparseRow<F>>);

/// Helper: sparse row from `[(variable_index, coefficient)]` pairs.
///
/// Panics at compile-time constant initialization if any coefficient does not
/// fit in `i64`; callers with wider constants (e.g. `2^64`) must use
/// [`row_wide`].
#[expect(
    clippy::expect_used,
    reason = "compile-time constant table; silent i128→i64 truncation would be a correctness bug"
)]
fn row<F: Field>(entries: &[(usize, i128)]) -> SparseRow<F> {
    entries
        .iter()
        .filter(|(_, c)| *c != 0)
        .map(|&(idx, c)| {
            let narrow = i64::try_from(c).expect("coefficient out of i64 range; use row_wide");
            (idx, F::from_i64(narrow))
        })
        .collect()
}

/// Helper: sparse row entry from i128 coefficient, handling large constants
/// that don't fit in i64 (e.g. 2^64 bias).
fn row_wide<F: Field>(entries: &[(usize, i128)]) -> SparseRow<F> {
    entries
        .iter()
        .filter(|(_, c)| *c != 0)
        .map(|&(idx, c)| (idx, F::from_i128(c)))
        .collect()
}

fn rv64_eq_constraint_rows<F: Field>() -> ConstraintRows<F> {
    let mut a_rows: Vec<SparseRow<F>> = Vec::with_capacity(NUM_EQ_CONSTRAINTS);
    let mut b_rows: Vec<SparseRow<F>> = Vec::with_capacity(NUM_EQ_CONSTRAINTS);
    let mut c_rows: Vec<SparseRow<F>> = Vec::with_capacity(NUM_EQ_CONSTRAINTS);

    let empty = || Vec::new();

    // Eq-conditional constraints (0-18)
    // Form: guard · (left − right) = 0  →  A=guard, B=left−right, C=0

    // 0: RamAddrEqRs1PlusImmIfLoadStore
    //    guard = Load + Store
    //    left  = RamAddress
    //    right = Rs1Value + Imm
    a_rows.push(row::<F>(&[(V_FLAG_LOAD, 1), (V_FLAG_STORE, 1)]));
    b_rows.push(row::<F>(&[
        (V_RAM_ADDRESS, 1),
        (V_RS1_VALUE, -1),
        (V_IMM, -1),
    ]));
    c_rows.push(empty());

    // 1: RamAddrEqZeroIfNotLoadStore
    //    guard = 1 − Load − Store
    //    left  = RamAddress
    //    right = 0
    a_rows.push(row::<F>(&[
        (V_CONST, 1),
        (V_FLAG_LOAD, -1),
        (V_FLAG_STORE, -1),
    ]));
    b_rows.push(row::<F>(&[(V_RAM_ADDRESS, 1)]));
    c_rows.push(empty());

    // 2: RamReadEqRamWriteIfLoad
    //    guard = Load
    //    left  = RamReadValue
    //    right = RamWriteValue
    a_rows.push(row::<F>(&[(V_FLAG_LOAD, 1)]));
    b_rows.push(row::<F>(&[(V_RAM_READ_VALUE, 1), (V_RAM_WRITE_VALUE, -1)]));
    c_rows.push(empty());

    // 3: RamReadEqRdWriteIfLoad
    //    guard = Load
    //    left  = RamReadValue
    //    right = RdWriteValue
    a_rows.push(row::<F>(&[(V_FLAG_LOAD, 1)]));
    b_rows.push(row::<F>(&[(V_RAM_READ_VALUE, 1), (V_RD_WRITE_VALUE, -1)]));
    c_rows.push(empty());

    // 4: Rs2EqRamWriteIfStore
    //    guard = Store
    //    left  = Rs2Value
    //    right = RamWriteValue
    a_rows.push(row::<F>(&[(V_FLAG_STORE, 1)]));
    b_rows.push(row::<F>(&[(V_RS2_VALUE, 1), (V_RAM_WRITE_VALUE, -1)]));
    c_rows.push(empty());

    // 5: LeftLookupZeroUnlessAddSubMul
    //    guard = Add + Sub + Mul
    //    left  = LeftLookupOperand
    //    right = 0
    a_rows.push(row::<F>(&[
        (V_FLAG_ADD_OPERANDS, 1),
        (V_FLAG_SUBTRACT_OPERANDS, 1),
        (V_FLAG_MULTIPLY_OPERANDS, 1),
    ]));
    b_rows.push(row::<F>(&[(V_LEFT_LOOKUP_OPERAND, 1)]));
    c_rows.push(empty());

    // 6: LeftLookupEqLeftInputOtherwise
    //    guard = 1 − Add − Sub − Mul
    //    left  = LeftLookupOperand
    //    right = LeftInstructionInput
    a_rows.push(row::<F>(&[
        (V_CONST, 1),
        (V_FLAG_ADD_OPERANDS, -1),
        (V_FLAG_SUBTRACT_OPERANDS, -1),
        (V_FLAG_MULTIPLY_OPERANDS, -1),
    ]));
    b_rows.push(row::<F>(&[
        (V_LEFT_LOOKUP_OPERAND, 1),
        (V_LEFT_INSTRUCTION_INPUT, -1),
    ]));
    c_rows.push(empty());

    // 7: RightLookupAdd
    //    guard = Add
    //    left  = RightLookupOperand
    //    right = LeftInstructionInput + RightInstructionInput
    a_rows.push(row::<F>(&[(V_FLAG_ADD_OPERANDS, 1)]));
    #[cfg(not(feature = "implicit-carry"))]
    b_rows.push(row::<F>(&[
        (V_RIGHT_LOOKUP_OPERAND, 1),
        (V_LEFT_INSTRUCTION_INPUT, -1),
        (V_RIGHT_INSTRUCTION_INPUT, -1),
    ]));
    #[cfg(feature = "implicit-carry")]
    b_rows.push(row::<F>(&[
        (V_RIGHT_LOOKUP_OPERAND, 1),
        (V_LEFT_INSTRUCTION_INPUT, -1),
        (V_RIGHT_INSTRUCTION_INPUT, -1),
        (V_CARRY_USED, -1),
    ]));
    c_rows.push(empty());

    // 8: RightLookupSub
    //    guard = Sub
    //    left  = RightLookupOperand
    //    right = LeftInstructionInput − RightInstructionInput + 2^64
    a_rows.push(row::<F>(&[(V_FLAG_SUBTRACT_OPERANDS, 1)]));
    b_rows.push(row_wide::<F>(&[
        (V_RIGHT_LOOKUP_OPERAND, 1),
        (V_LEFT_INSTRUCTION_INPUT, -1),
        (V_RIGHT_INSTRUCTION_INPUT, 1),
        (V_CONST, -TWOS_COMPLEMENT_BIAS),
    ]));
    c_rows.push(empty());

    // 9: RightLookupEqProductIfMul
    //    guard = Mul
    //    left  = RightLookupOperand
    //    right = Product
    a_rows.push(row::<F>(&[(V_FLAG_MULTIPLY_OPERANDS, 1)]));
    #[cfg(not(feature = "implicit-carry"))]
    b_rows.push(row::<F>(&[(V_RIGHT_LOOKUP_OPERAND, 1), (V_PRODUCT, -1)]));
    #[cfg(feature = "implicit-carry")]
    b_rows.push(row::<F>(&[
        (V_RIGHT_LOOKUP_OPERAND, 1),
        (V_PRODUCT, -1),
        (V_CARRY_USED, -1),
    ]));
    c_rows.push(empty());

    // 10: RightLookupEqRightInputOtherwise
    //     guard = 1 − Add − Sub − Mul − Advice
    //     left  = RightLookupOperand
    //     right = RightInstructionInput
    a_rows.push(row::<F>(&[
        (V_CONST, 1),
        (V_FLAG_ADD_OPERANDS, -1),
        (V_FLAG_SUBTRACT_OPERANDS, -1),
        (V_FLAG_MULTIPLY_OPERANDS, -1),
        (V_FLAG_ADVICE, -1),
    ]));
    b_rows.push(row::<F>(&[
        (V_RIGHT_LOOKUP_OPERAND, 1),
        (V_RIGHT_INSTRUCTION_INPUT, -1),
    ]));
    c_rows.push(empty());

    // 11: AssertLookupOne
    //     guard = Assert
    //     left  = LookupOutput
    //     right = 1
    a_rows.push(row::<F>(&[(V_FLAG_ASSERT, 1)]));
    b_rows.push(row::<F>(&[(V_LOOKUP_OUTPUT, 1), (V_CONST, -1)]));
    c_rows.push(empty());

    // 12: RdWriteEqLookupIfWriteLookupToRd
    //     guard = WriteLookupOutputToRD
    //     left  = RdWriteValue
    //     right = LookupOutput
    a_rows.push(row::<F>(&[(V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD, 1)]));
    b_rows.push(row::<F>(&[(V_RD_WRITE_VALUE, 1), (V_LOOKUP_OUTPUT, -1)]));
    c_rows.push(empty());

    // 13: RdWriteEqPCPlusConstIfWritePCtoRD
    //     guard = Jump
    //     left  = RdWriteValue
    //     right = UnexpandedPC + 4 − 2·IsCompressed
    a_rows.push(row::<F>(&[(V_FLAG_JUMP, 1)]));
    b_rows.push(row::<F>(&[
        (V_RD_WRITE_VALUE, 1),
        (V_UNEXPANDED_PC, -1),
        (V_CONST, -4),
        (V_FLAG_IS_COMPRESSED, 2),
    ]));
    c_rows.push(empty());

    // 14: NextUnexpPCEqLookupIfShouldJump
    //     guard = ShouldJump
    //     left  = NextUnexpandedPC
    //     right = LookupOutput
    a_rows.push(row::<F>(&[(V_SHOULD_JUMP, 1)]));
    b_rows.push(row::<F>(&[
        (V_NEXT_UNEXPANDED_PC, 1),
        (V_LOOKUP_OUTPUT, -1),
    ]));
    c_rows.push(empty());

    // 15: NextUnexpPCEqPCPlusImmIfShouldBranch
    //     guard = ShouldBranch
    //     left  = NextUnexpandedPC
    //     right = UnexpandedPC + Imm
    a_rows.push(row::<F>(&[(V_SHOULD_BRANCH, 1)]));
    b_rows.push(row::<F>(&[
        (V_NEXT_UNEXPANDED_PC, 1),
        (V_UNEXPANDED_PC, -1),
        (V_IMM, -1),
    ]));
    c_rows.push(empty());

    // 16: NextUnexpPCUpdateOtherwise
    //     guard = 1 − ShouldBranch − Jump
    //     left  = NextUnexpandedPC
    //     right = UnexpandedPC + 4 − 4·DoNotUpdate − 2·IsCompressed
    a_rows.push(row::<F>(&[
        (V_CONST, 1),
        (V_SHOULD_BRANCH, -1),
        (V_FLAG_JUMP, -1),
    ]));
    b_rows.push(row::<F>(&[
        (V_NEXT_UNEXPANDED_PC, 1),
        (V_UNEXPANDED_PC, -1),
        (V_CONST, -4),
        (V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC, 4),
        (V_FLAG_IS_COMPRESSED, 2),
    ]));
    c_rows.push(empty());

    // 17: NextPCEqPCPlusOneIfInline
    //     guard = VirtualInstruction − IsLastInSequence
    //     left  = NextPC
    //     right = PC + 1
    //
    // NOTE: `IsLastInSequence` fires for every cycle whose
    // `virtual_sequence_remaining == Some(0)`, not just `JALR`. That
    // looks lax — at a non-`JALR` terminal step the guard zeros out and
    // `NextPC` isn't pinned to `PC + 1` here — but `NextPC` is still
    // uniquely determined by the rest of the system: #14
    // (`NextUnexpPCEqLookupIfShouldJump`) / #16
    // (`NextUnexpPCUpdateOtherwise`) constrain `NextUnexpandedPC`, #18
    // (`MustStartSequenceFromBeginning`) forces the next row to be
    // non-virtual or the first step of a new sequence, and the
    // bytecode-row commitment ties `NextPC` to a unique row matching both
    // properties. If any of those are ever removed or weakened, revisit the
    // terminal-sequence flag semantics to avoid an unconstrained-`NextPC`
    // exploit.
    a_rows.push(row::<F>(&[
        (V_FLAG_VIRTUAL_INSTRUCTION, 1),
        (V_FLAG_IS_LAST_IN_SEQUENCE, -1),
    ]));
    b_rows.push(row::<F>(&[(V_NEXT_PC, 1), (V_PC, -1), (V_CONST, -1)]));
    c_rows.push(empty());

    // 18: MustStartSequenceFromBeginning
    //     guard = NextIsVirtual − NextIsFirstInSequence
    //     left  = 1
    //     right = DoNotUpdateUnexpandedPC
    a_rows.push(row::<F>(&[
        (V_NEXT_IS_VIRTUAL, 1),
        (V_NEXT_IS_FIRST_IN_SEQUENCE, -1),
    ]));
    b_rows.push(row::<F>(&[
        (V_CONST, 1),
        (V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC, -1),
    ]));
    c_rows.push(empty());

    // Implicit-carry rows (19-20), appended to match the legacy enum order.
    #[cfg(feature = "implicit-carry")]
    {
        // 19: LookupSplitsIntoOutputAndNextCarry
        //     guard = ProducesCarry
        //     left  = RightLookupOperand
        //     right = LookupOutput + 2^64 * NextCarry
        a_rows.push(row::<F>(&[(V_FLAG_PRODUCES_CARRY, 1)]));
        b_rows.push(row_wide::<F>(&[
            (V_RIGHT_LOOKUP_OPERAND, 1),
            (V_LOOKUP_OUTPUT, -1),
            (V_NEXT_CARRY, -TWOS_COMPLEMENT_BIAS),
        ]));
        c_rows.push(empty());

        // 20: NextCarryZeroIfNotProducesCarry
        //     guard = 1 − ProducesCarry
        //     left  = NextCarry
        //     right = 0
        a_rows.push(row::<F>(&[(V_CONST, 1), (V_FLAG_PRODUCES_CARRY, -1)]));
        b_rows.push(row::<F>(&[(V_NEXT_CARRY, 1)]));
        c_rows.push(empty());
    }

    (a_rows, b_rows, c_rows)
}

fn append_product_constraints<F: Field>(
    a_rows: &mut Vec<SparseRow<F>>,
    b_rows: &mut Vec<SparseRow<F>>,
    c_rows: &mut Vec<SparseRow<F>>,
) {
    // Product constraints (19-21)
    // Form: left · right = output  →  A=left, B=right, C=output

    // 19: Product = LeftInstructionInput × RightInstructionInput
    a_rows.push(row::<F>(&[(V_LEFT_INSTRUCTION_INPUT, 1)]));
    b_rows.push(row::<F>(&[(V_RIGHT_INSTRUCTION_INPUT, 1)]));
    c_rows.push(row::<F>(&[(V_PRODUCT, 1)]));

    // 20: ShouldBranch = LookupOutput × Branch
    a_rows.push(row::<F>(&[(V_LOOKUP_OUTPUT, 1)]));
    b_rows.push(row::<F>(&[(V_BRANCH, 1)]));
    c_rows.push(row::<F>(&[(V_SHOULD_BRANCH, 1)]));

    // 21: ShouldJump = Jump × (1 − NextIsNoop)
    a_rows.push(row::<F>(&[(V_FLAG_JUMP, 1)]));
    b_rows.push(row::<F>(&[(V_CONST, 1), (V_NEXT_IS_NOOP, -1)]));
    c_rows.push(row::<F>(&[(V_SHOULD_JUMP, 1)]));

    // CarryUsed = UsesCarry × Carry (committed)
    #[cfg(feature = "implicit-carry")]
    {
        a_rows.push(row::<F>(&[(V_FLAG_USES_CARRY, 1)]));
        b_rows.push(row::<F>(&[(V_CARRY, 1)]));
        c_rows.push(row::<F>(&[(V_CARRY_USED, 1)]));
    }
}

/// Build only the Jolt RV64 equality-conditional constraints.
///
/// Returns the 19 rows with form `guard · (left - right) = 0`, over the
/// standard 38-variable per-cycle witness layout. Product constraints are
/// intentionally excluded for consumers that handle multiplication checks in
/// a separate protocol step.
pub fn rv64_spartan_outer_constraints<F: Field>() -> crate::ConstraintMatrices<F> {
    let (a_rows, b_rows, c_rows) = rv64_eq_constraint_rows();
    crate::ConstraintMatrices::new(
        NUM_EQ_CONSTRAINTS,
        NUM_VARS_PER_CYCLE,
        a_rows,
        b_rows,
        c_rows,
    )
}

/// Build the full Jolt RV64 R1CS constraint matrices.
///
/// Returns 22 constraints over 38 variables per cycle:
/// - 19 equality-conditional: `guard · (left − right) = 0` → A=guard, B=left−right, C=0
/// - 3 product: `left · right = output` → A=left, B=right, C=output
///
/// Variable layout matches the constants in this module (V_CONST=0, inputs at 1–35,
/// product factors at 36–37).
pub fn rv64_trace_constraints<F: Field>() -> crate::ConstraintMatrices<F> {
    let (mut a_rows, mut b_rows, mut c_rows) = rv64_eq_constraint_rows();
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
    use jolt_field::{Fr, Ring};
    use num_traits::Zero;

    /// A no-op cycle: const=1, all else zero. All eq-conditional guards
    /// evaluate to 0 (Load=0, Store=0, etc.) except constraint 16
    /// (NextUnexpPCUpdateOtherwise) whose guard = 1−0−0 = 1.
    /// Constraint 16 requires: NextUnexpPC = UnexpPC + 4 − 4·DoNotUpdate − 2·IsCompressed.
    /// For the no-op (DoNotUpdate=1): NextUnexpPC = UnexpPC + 4 − 4 = UnexpPC.
    /// With both at 0 this holds.
    fn noop_witness() -> Vec<Fr> {
        let mut w = vec![Fr::zero(); NUM_VARS_PER_CYCLE];
        w[V_CONST] = Fr::from_u64(1);
        // DoNotUpdateUnexpandedPC = 1 for no-ops
        w[V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] = Fr::from_u64(1);
        w
    }

    #[test]
    fn noop_satisfies_constraints() {
        let matrices = rv64_trace_constraints::<Fr>();
        assert_eq!(matrices.num_constraints, NUM_CONSTRAINTS_PER_CYCLE);
        assert_eq!(matrices.num_vars, NUM_VARS_PER_CYCLE);
        matrices
            .check_witness(&noop_witness())
            .expect("noop should satisfy all constraints");
    }

    #[test]
    fn constraint_count() {
        // 19 eq + 3 product rows; implicit-carry adds 2 eq + 1 product row.
        let expected = 22 + 3 * cfg!(feature = "implicit-carry") as usize;
        let matrices = rv64_trace_constraints::<Fr>();
        assert_eq!(matrices.a.len(), expected);
        assert_eq!(matrices.b.len(), expected);
        assert_eq!(matrices.c.len(), expected);
    }

    #[test]
    fn spartan_outer_constraints_plus_product_constraints_match_full_constraints() {
        let (mut a_rows, mut b_rows, mut c_rows) = rv64_eq_constraint_rows::<Fr>();
        append_product_constraints(&mut a_rows, &mut b_rows, &mut c_rows);
        let matrices = rv64_trace_constraints::<Fr>();

        assert_eq!(matrices.a, a_rows);
        assert_eq!(matrices.b, b_rows);
        assert_eq!(matrices.c, c_rows);
    }

    #[test]
    fn eq_constraint_public_column_has_no_c_contribution() {
        let matrices = rv64_spartan_outer_constraints::<Fr>();
        let row_weights = vec![Fr::from_u64(1); NUM_EQ_CONSTRAINTS];
        let contributions = matrices
            .public_column_contributions(&row_weights, const_column(), Fr::from_u64(1))
            .expect("const column evaluates");

        assert!(contributions.c.is_zero());
    }

    #[test]
    fn input_columns_follow_const_then_inputs_layout() {
        assert_eq!(const_column(), V_CONST);
        assert_eq!(input_column(0), Some(V_LEFT_INSTRUCTION_INPUT));
        #[cfg(not(feature = "implicit-carry"))]
        assert_eq!(
            input_column(NUM_R1CS_INPUTS - 1),
            Some(V_FLAG_IS_LAST_IN_SEQUENCE)
        );
        #[cfg(feature = "implicit-carry")]
        assert_eq!(input_column(NUM_R1CS_INPUTS - 1), Some(V_NEXT_CARRY));
        assert_eq!(input_column(NUM_R1CS_INPUTS), None);
    }
}
