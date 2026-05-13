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
//! | `[1..=47]` | R1CS inputs (registers, flags, PC, lookups, FR operands) |
//! | `[48..=49]` | Product factor variables (`Branch`, `NextIsNoop`) |
//!
//! # Constraint forms
//!
//! - **Eq-conditional** (rows 0–31): `guard · (left − right) = 0`
//! - **Product** (rows 32–34): `left · right = output`

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

// BN254 Fr coprocessor circuit flags (CircuitFlags variants 14..=22).
pub const V_FLAG_IS_FIELD_MUL: usize = 36;
pub const V_FLAG_IS_FIELD_ADD: usize = 37;
pub const V_FLAG_IS_FIELD_SUB: usize = 38;
pub const V_FLAG_IS_FIELD_INV: usize = 39;
pub const V_FLAG_IS_FIELD_ASSERT_EQ: usize = 40;
pub const V_FLAG_IS_FIELD_MOV: usize = 41;
pub const V_FLAG_IS_FIELD_SLL64: usize = 42;
pub const V_FLAG_IS_FIELD_SLL128: usize = 43;
pub const V_FLAG_IS_FIELD_SLL192: usize = 44;

// Virtual FR operand openings. These columns appear as coefficients in the
// FR R1CS rows but their per-cycle values are NOT populated by the normal
// witness generator — they are bound by the FR Twist sumcheck. R1CS unit
// tests below populate them directly to verify row shape.
pub const V_FIELD_RS1_VALUE: usize = 45;
pub const V_FIELD_RS2_VALUE: usize = 46;
pub const V_FIELD_RD_VALUE: usize = 47;

pub const V_BRANCH: usize = 48;
pub const V_NEXT_IS_NOOP: usize = 49;

pub const NUM_R1CS_INPUTS: usize = 47;
pub const NUM_PRODUCT_FACTORS: usize = 2;
pub const NUM_VARS_PER_CYCLE: usize = 1 + NUM_R1CS_INPUTS + NUM_PRODUCT_FACTORS; // 50
pub const NUM_EQ_CONSTRAINTS: usize = 32; // 19 RV base + 13 FR
pub const NUM_PRODUCT_CONSTRAINTS: usize = 3;
pub const NUM_CONSTRAINTS_PER_CYCLE: usize = NUM_EQ_CONSTRAINTS + NUM_PRODUCT_CONSTRAINTS; // 35

/// Two's complement bias for subtraction: 2^64.
const TWOS_COMPLEMENT_BIAS: i128 = 0x1_0000_0000_0000_0000;

use crate::constraint::SparseRow;
use jolt_field::Field;

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

/// Helper: sparse row from `[(variable_index, F coefficient)]` pairs. Used
/// when coefficients exceed the i128 range (e.g. `2^128`, `2^192` in the
/// FieldSLL bridge rows).
fn row_bigcoeff<F: Field>(entries: &[(usize, F)]) -> SparseRow<F> {
    entries
        .iter()
        .filter(|(_, c)| !c.is_zero())
        .map(|&(idx, c)| (idx, c))
        .collect()
}

/// Build the Jolt RV64 R1CS constraint matrices.
///
/// Returns 35 constraints over 50 variables per cycle:
/// - 32 equality-conditional: `guard · (left − right) = 0` → A=guard, B=left−right, C=0
///   - 19 base RV64 (rows 0-18) + 13 BN254 Fr coprocessor (rows 19-31)
/// - 3 product: `left · right = output` → A=left, B=right, C=output (rows 32-34)
///
/// Variable layout matches the constants in this module (V_CONST=0, inputs at
/// 1-47 including FR flags 36-44 and FR virtual operand columns 45-47,
/// product factors at 48-49).
pub fn rv64_constraints<F: Field>() -> crate::ConstraintMatrices<F> {
    let mut a_rows: Vec<SparseRow<F>> = Vec::with_capacity(NUM_CONSTRAINTS_PER_CYCLE);
    let mut b_rows: Vec<SparseRow<F>> = Vec::with_capacity(NUM_CONSTRAINTS_PER_CYCLE);
    let mut c_rows: Vec<SparseRow<F>> = Vec::with_capacity(NUM_CONSTRAINTS_PER_CYCLE);

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
    b_rows.push(row::<F>(&[
        (V_RIGHT_LOOKUP_OPERAND, 1),
        (V_LEFT_INSTRUCTION_INPUT, -1),
        (V_RIGHT_INSTRUCTION_INPUT, -1),
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
    b_rows.push(row::<F>(&[(V_RIGHT_LOOKUP_OPERAND, 1), (V_PRODUCT, -1)]));
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
    // properties. If any of those are ever removed or weakened, tighten
    // `IsLastInSequence` back to `JALR`-only (see jolt-core's
    // `instruction/jalr.rs`) to avoid an unconstrained-`NextPC` exploit.
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

    // BN254 Fr coprocessor eq-conditional constraints (19-31). Each gates on
    // a FR circuit flag. The virtual V_FIELD_RS1_VALUE / V_FIELD_RS2_VALUE /
    // V_FIELD_RD_VALUE columns are bound by the FR Twist sumcheck (port steps
    // 8-10). For now, R1CS unit tests populate them directly to verify row
    // shape.

    // 19: FieldAdd — IsFieldAdd · (FieldRs1Value + FieldRs2Value − FieldRdValue) = 0
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_ADD, 1)]));
    b_rows.push(row::<F>(&[
        (V_FIELD_RS1_VALUE, 1),
        (V_FIELD_RS2_VALUE, 1),
        (V_FIELD_RD_VALUE, -1),
    ]));
    c_rows.push(empty());

    // 20: FieldSub — IsFieldSub · (FieldRs1Value − FieldRs2Value − FieldRdValue) = 0
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_SUB, 1)]));
    b_rows.push(row::<F>(&[
        (V_FIELD_RS1_VALUE, 1),
        (V_FIELD_RS2_VALUE, -1),
        (V_FIELD_RD_VALUE, -1),
    ]));
    c_rows.push(empty());

    // 21-23: FieldMul via V_PRODUCT reuse. The V_PRODUCT column is materialized
    // by product constraint 32 as LeftInstructionInput × RightInstructionInput;
    // on FMUL cycles this equals FieldRs1Value × FieldRs2Value, which must
    // equal FieldRdValue.

    // 21: IsFieldMul · (LeftInstructionInput − FieldRs1Value) = 0
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_MUL, 1)]));
    b_rows.push(row::<F>(&[
        (V_LEFT_INSTRUCTION_INPUT, 1),
        (V_FIELD_RS1_VALUE, -1),
    ]));
    c_rows.push(empty());

    // 22: IsFieldMul · (RightInstructionInput − FieldRs2Value) = 0
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_MUL, 1)]));
    b_rows.push(row::<F>(&[
        (V_RIGHT_INSTRUCTION_INPUT, 1),
        (V_FIELD_RS2_VALUE, -1),
    ]));
    c_rows.push(empty());

    // 23: IsFieldMul · (Product − FieldRdValue) = 0
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_MUL, 1)]));
    b_rows.push(row::<F>(&[(V_PRODUCT, 1), (V_FIELD_RD_VALUE, -1)]));
    c_rows.push(empty());

    // 24-26: FieldInv via V_PRODUCT reuse. Same pattern as FMUL but B is the
    // inverse and the product is forced to 1. The constraint `rs1 · rd = 1`
    // is unsatisfiable for rs1 = 0 — see audit C8 (SDK contract guarantees
    // tracer never emits FINV(0)).

    // 24: IsFieldInv · (LeftInstructionInput − FieldRs1Value) = 0
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_INV, 1)]));
    b_rows.push(row::<F>(&[
        (V_LEFT_INSTRUCTION_INPUT, 1),
        (V_FIELD_RS1_VALUE, -1),
    ]));
    c_rows.push(empty());

    // 25: IsFieldInv · (RightInstructionInput − FieldRdValue) = 0
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_INV, 1)]));
    b_rows.push(row::<F>(&[
        (V_RIGHT_INSTRUCTION_INPUT, 1),
        (V_FIELD_RD_VALUE, -1),
    ]));
    c_rows.push(empty());

    // 26: IsFieldInv · (Product − 1) = 0
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_INV, 1)]));
    b_rows.push(row::<F>(&[(V_PRODUCT, 1), (V_CONST, -1)]));
    c_rows.push(empty());

    // 27: FieldAssertEq — IsFieldAssertEq · (FieldRs1Value − FieldRs2Value) = 0
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_ASSERT_EQ, 1)]));
    b_rows.push(row::<F>(&[(V_FIELD_RS1_VALUE, 1), (V_FIELD_RS2_VALUE, -1)]));
    c_rows.push(empty());

    // 28: FieldMov — IsFieldMov · (Rs1Value − FieldRdValue) = 0
    // Cross-domain bridge: V_RS1_VALUE bound by Registers Twist, V_FIELD_RD_VALUE
    // by FR Twist.
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_MOV, 1)]));
    b_rows.push(row::<F>(&[(V_RS1_VALUE, 1), (V_FIELD_RD_VALUE, -1)]));
    c_rows.push(empty());

    // 29: FieldSLL64 — IsFieldSLL64 · (2^64 · Rs1Value − FieldRdValue) = 0
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_SLL64, 1)]));
    b_rows.push(row_wide::<F>(&[
        (V_RS1_VALUE, 1i128 << 64),
        (V_FIELD_RD_VALUE, -1),
    ]));
    c_rows.push(empty());

    // 30: FieldSLL128 — IsFieldSLL128 · (2^128 · Rs1Value − FieldRdValue) = 0
    // Coefficient exceeds i128 range, so we compute it as a field element.
    let two_pow_128: F = F::one().mul_pow_2(128);
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_SLL128, 1)]));
    b_rows.push(row_bigcoeff::<F>(&[
        (V_RS1_VALUE, two_pow_128),
        (V_FIELD_RD_VALUE, -F::one()),
    ]));
    c_rows.push(empty());

    // 31: FieldSLL192 — IsFieldSLL192 · (2^192 · Rs1Value − FieldRdValue) = 0
    let two_pow_192: F = F::one().mul_pow_2(192);
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_SLL192, 1)]));
    b_rows.push(row_bigcoeff::<F>(&[
        (V_RS1_VALUE, two_pow_192),
        (V_FIELD_RD_VALUE, -F::one()),
    ]));
    c_rows.push(empty());

    // Product constraints (32-34)
    // Form: left · right = output  →  A=left, B=right, C=output

    // 32: Product = LeftInstructionInput × RightInstructionInput
    a_rows.push(row::<F>(&[(V_LEFT_INSTRUCTION_INPUT, 1)]));
    b_rows.push(row::<F>(&[(V_RIGHT_INSTRUCTION_INPUT, 1)]));
    c_rows.push(row::<F>(&[(V_PRODUCT, 1)]));

    // 33: ShouldBranch = LookupOutput × Branch
    a_rows.push(row::<F>(&[(V_LOOKUP_OUTPUT, 1)]));
    b_rows.push(row::<F>(&[(V_BRANCH, 1)]));
    c_rows.push(row::<F>(&[(V_SHOULD_BRANCH, 1)]));

    // 34: ShouldJump = Jump × (1 − NextIsNoop)
    a_rows.push(row::<F>(&[(V_FLAG_JUMP, 1)]));
    b_rows.push(row::<F>(&[(V_CONST, 1), (V_NEXT_IS_NOOP, -1)]));
    c_rows.push(row::<F>(&[(V_SHOULD_JUMP, 1)]));

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
    use jolt_field::Fr;
    use num_traits::{One, Zero};

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
        let matrices = rv64_constraints::<Fr>();
        assert_eq!(matrices.num_constraints, NUM_CONSTRAINTS_PER_CYCLE);
        assert_eq!(matrices.num_vars, NUM_VARS_PER_CYCLE);
        matrices
            .check_witness(&noop_witness())
            .expect("noop should satisfy all constraints");
    }

    #[test]
    fn constraint_count() {
        let matrices = rv64_constraints::<Fr>();
        assert_eq!(matrices.a.len(), NUM_CONSTRAINTS_PER_CYCLE);
        assert_eq!(matrices.b.len(), NUM_CONSTRAINTS_PER_CYCLE);
        assert_eq!(matrices.c.len(), NUM_CONSTRAINTS_PER_CYCLE);
    }

    /// Every FR row is guarded by a FR flag. With no FR flag set, the FR
    /// rows are 0·(…) = 0 — so a no-op cycle satisfies them trivially.
    #[test]
    fn fr_rows_noop_satisfied() {
        let matrices = rv64_constraints::<Fr>();
        matrices
            .check_witness(&noop_witness())
            .expect("noop satisfies FR rows (FR flags are 0)");
    }

    /// FieldAdd row (19): IsFieldAdd · (Rs1V + Rs2V − RdV) = 0. R1CS unit
    /// tests populate FR virtual slots directly; in a real proof these are
    /// bound by the FR Twist sumcheck.
    #[test]
    fn field_add_row_accepts_honest_sum() {
        let matrices = rv64_constraints::<Fr>();
        let mut w = noop_witness();
        w[V_FLAG_IS_FIELD_ADD] = Fr::from_u64(1);
        w[V_FIELD_RS1_VALUE] = Fr::from_u64(10);
        w[V_FIELD_RS2_VALUE] = Fr::from_u64(32);
        w[V_FIELD_RD_VALUE] = Fr::from_u64(42);
        matrices
            .check_witness(&w)
            .expect("FieldAdd 10+32=42 satisfies row 19");
    }

    #[test]
    fn field_add_row_rejects_mismatch() {
        let matrices = rv64_constraints::<Fr>();
        let mut w = noop_witness();
        w[V_FLAG_IS_FIELD_ADD] = Fr::from_u64(1);
        w[V_FIELD_RS1_VALUE] = Fr::from_u64(10);
        w[V_FIELD_RS2_VALUE] = Fr::from_u64(32);
        w[V_FIELD_RD_VALUE] = Fr::from_u64(43);
        assert!(
            matrices.check_witness(&w).is_err(),
            "FieldAdd row must reject 10+32 ≠ 43"
        );
    }

    /// FieldSub row (20): IsFieldSub · (Rs1V − Rs2V − RdV) = 0.
    #[test]
    fn field_sub_row_accepts_honest_diff() {
        let matrices = rv64_constraints::<Fr>();
        let mut w = noop_witness();
        w[V_FLAG_IS_FIELD_SUB] = Fr::from_u64(1);
        w[V_FIELD_RS1_VALUE] = Fr::from_u64(50);
        w[V_FIELD_RS2_VALUE] = Fr::from_u64(8);
        w[V_FIELD_RD_VALUE] = Fr::from_u64(42);
        matrices
            .check_witness(&w)
            .expect("FieldSub 50-8=42 satisfies row 20");
    }

    /// FieldMul rows 21-23 + product constraint 32 chain together. Rows 6
    /// and 10 (LeftLookup/RightLookup match inputs unless RV Add/Sub/Mul/Advice
    /// are set) fire for the FieldMul cycle since FR-MUL isn't an RV lookup
    /// flag — set lookup operands to match instruction inputs.
    #[test]
    fn field_mul_rows_accept_honest_product() {
        let matrices = rv64_constraints::<Fr>();
        let mut w = noop_witness();
        w[V_FLAG_IS_FIELD_MUL] = Fr::from_u64(1);
        w[V_LEFT_INSTRUCTION_INPUT] = Fr::from_u64(7);
        w[V_RIGHT_INSTRUCTION_INPUT] = Fr::from_u64(9);
        w[V_LEFT_LOOKUP_OPERAND] = Fr::from_u64(7);
        w[V_RIGHT_LOOKUP_OPERAND] = Fr::from_u64(9);
        w[V_PRODUCT] = Fr::from_u64(63);
        w[V_FIELD_RS1_VALUE] = Fr::from_u64(7);
        w[V_FIELD_RS2_VALUE] = Fr::from_u64(9);
        w[V_FIELD_RD_VALUE] = Fr::from_u64(63);
        matrices
            .check_witness(&w)
            .expect("FieldMul 7*9=63 satisfies rows 21-23 + product");
    }

    /// FieldMov row (28): cross-domain bridge — V_RS1_VALUE (integer side)
    /// must equal V_FIELD_RD_VALUE (FR side).
    #[test]
    fn field_mov_row_bridges_int_to_fr() {
        let matrices = rv64_constraints::<Fr>();
        let mut w = noop_witness();
        w[V_FLAG_IS_FIELD_MOV] = Fr::from_u64(1);
        w[V_RS1_VALUE] = Fr::from_u64(0xdead_beef);
        w[V_FIELD_RD_VALUE] = Fr::from_u64(0xdead_beef);
        matrices
            .check_witness(&w)
            .expect("FieldMov bridge satisfies row 28");
    }

    /// FieldSLL64 row (29): 2^64 · Rs1V = RdV. Uses `row_wide` (i128 fits 2^64).
    #[test]
    fn field_sll64_row_shifts_by_64() {
        let matrices = rv64_constraints::<Fr>();
        let mut w = noop_witness();
        w[V_FLAG_IS_FIELD_SLL64] = Fr::from_u64(1);
        w[V_RS1_VALUE] = Fr::from_u64(3);
        w[V_FIELD_RD_VALUE] = Fr::one().mul_pow_2(64) * Fr::from_u64(3);
        matrices
            .check_witness(&w)
            .expect("FieldSLL64 3·2^64 satisfies row 29");
    }

    /// FieldSLL128 row (30): uses `row_bigcoeff` because 2^128 doesn't fit in i128.
    #[test]
    fn field_sll128_row_shifts_by_128() {
        let matrices = rv64_constraints::<Fr>();
        let mut w = noop_witness();
        w[V_FLAG_IS_FIELD_SLL128] = Fr::from_u64(1);
        w[V_RS1_VALUE] = Fr::from_u64(5);
        w[V_FIELD_RD_VALUE] = Fr::one().mul_pow_2(128) * Fr::from_u64(5);
        matrices
            .check_witness(&w)
            .expect("FieldSLL128 5·2^128 satisfies row 30");
    }

    /// FieldSLL192 row (31): also uses `row_bigcoeff`.
    #[test]
    fn field_sll192_row_shifts_by_192() {
        let matrices = rv64_constraints::<Fr>();
        let mut w = noop_witness();
        w[V_FLAG_IS_FIELD_SLL192] = Fr::from_u64(1);
        w[V_RS1_VALUE] = Fr::from_u64(2);
        w[V_FIELD_RD_VALUE] = Fr::one().mul_pow_2(192) * Fr::from_u64(2);
        matrices
            .check_witness(&w)
            .expect("FieldSLL192 2·2^192 satisfies row 31");
    }
}
