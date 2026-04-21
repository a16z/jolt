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
//! | `[1..=21]` | R1CS core inputs (registers, PC, lookups) |
//! | `[22..=35]` | RV base circuit flags (14 flags, see [`CircuitFlags`] indices 0..14) |
//! | `[36..=39]` | BN254 Fr coprocessor flags (IsFieldMul/Add/Sub/Inv) |
//! | `[40..=42]` | BN254 Fr coprocessor operand columns (FieldOpA/B/Result) |
//! | `[43..=44]` | Product factor variables (`Branch`, `NextIsNoop`) |
//!
//! # Constraint forms
//!
//! - **Eq-conditional** (rows 0–18 base + 19 FieldAdd + 20 FieldSub):
//!   `guard · (left − right) = 0`
//! - **Product** (rows 21–23): `left · right = output`

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

// --- BN254 Fr coprocessor slots (Phase 2b task #59) ---

/// Per-cycle FMUL flag (1 on FMUL cycles, 0 otherwise).
pub const V_FLAG_IS_FIELD_MUL: usize = 36;
/// Per-cycle FADD flag.
pub const V_FLAG_IS_FIELD_ADD: usize = 37;
/// Per-cycle FSUB flag.
pub const V_FLAG_IS_FIELD_SUB: usize = 38;
/// Per-cycle FINV flag.
pub const V_FLAG_IS_FIELD_INV: usize = 39;
/// FieldOp read source A (`field_regs[frs1]` as Fr scalar).
pub const V_FIELD_OP_A: usize = 40;
/// FieldOp read source B (`field_regs[frs2]` as Fr scalar; 0 on FINV cycles).
pub const V_FIELD_OP_B: usize = 41;
/// FieldOp write destination (`field_regs[frd]` post-value as Fr scalar).
pub const V_FIELD_OP_RESULT: usize = 42;

// --- Product factors (shifted by +7 to make room for field-op slots) ---

pub const V_BRANCH: usize = 43;
pub const V_NEXT_IS_NOOP: usize = 44;

pub const NUM_R1CS_INPUTS: usize = 42;
pub const NUM_PRODUCT_FACTORS: usize = 2;
pub const NUM_VARS_PER_CYCLE: usize = 1 + NUM_R1CS_INPUTS + NUM_PRODUCT_FACTORS; // 45
pub const NUM_EQ_CONSTRAINTS: usize = 27;
pub const NUM_PRODUCT_CONSTRAINTS: usize = 3;
pub const NUM_CONSTRAINTS_PER_CYCLE: usize = NUM_EQ_CONSTRAINTS + NUM_PRODUCT_CONSTRAINTS; // 30

/// Two's complement bias for subtraction: 2^64.
const TWOS_COMPLEMENT_BIAS: i128 = 0x1_0000_0000_0000_0000;

use crate::constraint::SparseRow;
use jolt_field::Field;

/// Helper: sparse row from `[(variable_index, coefficient)]` pairs.
fn row<F: Field>(entries: &[(usize, i128)]) -> SparseRow<F> {
    entries
        .iter()
        .filter(|(_, c)| *c != 0)
        .map(|&(idx, c)| (idx, F::from_i64(c as i64)))
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

/// Build the Jolt RV64 R1CS constraint matrices.
///
/// Returns [`NUM_CONSTRAINTS_PER_CYCLE`] constraints over [`NUM_VARS_PER_CYCLE`]
/// variables per cycle:
/// - [`NUM_EQ_CONSTRAINTS`] equality-conditional: `guard · (left − right) = 0`
///   → A=guard, B=left−right, C=0
/// - [`NUM_PRODUCT_CONSTRAINTS`] product: `left · right = output`
///   → A=left, B=right, C=output
///
/// Variable layout matches the constants in this module. The FMUL/FINV gates
/// reuse the first product constraint (`V_PRODUCT = V_LEFT·V_RIGHT`) so no new
/// product rows are introduced — see rows 21–26 for the routing.
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

    // 19: FieldAdd gate
    //     IsFieldAdd · (FieldOpA + FieldOpB − FieldOpResult) = 0
    //     Enforces BN254 Fr addition semantics on cycles where funct3 = FADD.
    //     In Fr, a + b wraps mod p automatically, so the equality holds iff
    //     the guest-provided result matches a + b mod p.
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_ADD, 1)]));
    b_rows.push(row::<F>(&[
        (V_FIELD_OP_A, 1),
        (V_FIELD_OP_B, 1),
        (V_FIELD_OP_RESULT, -1),
    ]));
    c_rows.push(empty());

    // 20: FieldSub gate
    //     IsFieldSub · (FieldOpA − FieldOpB − FieldOpResult) = 0
    //     Enforces BN254 Fr subtraction semantics on FSUB cycles.
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_SUB, 1)]));
    b_rows.push(row::<F>(&[
        (V_FIELD_OP_A, 1),
        (V_FIELD_OP_B, -1),
        (V_FIELD_OP_RESULT, -1),
    ]));
    c_rows.push(empty());

    // FMUL and FINV need a multiplication in R1CS (A·B = Result for FMUL;
    // A·Result = 1 for FINV). Rather than adding a new product constraint
    // (which would bump NUM_PRODUCT_CONSTRAINTS and ripple into the product
    // virtual sumcheck dimensions and cross-verify compatibility), we REUSE
    // the existing product constraint 21 (`V_PRODUCT = V_LEFT · V_RIGHT`) by
    // binding the FieldOp operands onto V_LEFT / V_RIGHT on FMUL/FINV cycles.
    //
    // Contract enforced by rows 21-26:
    //   FMUL:  V_LEFT = A, V_RIGHT = B, V_PRODUCT = FieldOpResult
    //   FINV:  V_LEFT = A, V_RIGHT = FieldOpResult, V_PRODUCT = 1
    //
    // On a genuine FMUL/FINV cycle, all RV circuit flags (AddOperands,
    // SubtractOperands, MultiplyOperands, Load, Store, Jump, ...) are 0, so
    // the other eq constraints touching V_LEFT/V_RIGHT either have vacuous
    // guards or require V_LOOKUP_OPERAND mirroring V_LEFT/V_RIGHT — both of
    // which the witness builder handles via `apply_field_op_events_to_r1cs`.
    // FINV(0)=0 is NOT covered by this gate (would require an is-zero helper
    // variable); guests must avoid taking the inverse of zero.

    // 21: FieldMul operand A binding
    //     IsFieldMul · (V_LEFT_INSTRUCTION_INPUT − FieldOpA) = 0
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_MUL, 1)]));
    b_rows.push(row::<F>(&[
        (V_LEFT_INSTRUCTION_INPUT, 1),
        (V_FIELD_OP_A, -1),
    ]));
    c_rows.push(empty());

    // 22: FieldMul operand B binding
    //     IsFieldMul · (V_RIGHT_INSTRUCTION_INPUT − FieldOpB) = 0
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_MUL, 1)]));
    b_rows.push(row::<F>(&[
        (V_RIGHT_INSTRUCTION_INPUT, 1),
        (V_FIELD_OP_B, -1),
    ]));
    c_rows.push(empty());

    // 23: FieldMul product output
    //     IsFieldMul · (V_PRODUCT − FieldOpResult) = 0
    //     Combined with product-constraint 27 (V_PRODUCT = V_LEFT · V_RIGHT),
    //     this forces FieldOpResult = A · B on FMUL cycles.
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_MUL, 1)]));
    b_rows.push(row::<F>(&[(V_PRODUCT, 1), (V_FIELD_OP_RESULT, -1)]));
    c_rows.push(empty());

    // 24: FieldInv operand A binding
    //     IsFieldInv · (V_LEFT_INSTRUCTION_INPUT − FieldOpA) = 0
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_INV, 1)]));
    b_rows.push(row::<F>(&[
        (V_LEFT_INSTRUCTION_INPUT, 1),
        (V_FIELD_OP_A, -1),
    ]));
    c_rows.push(empty());

    // 25: FieldInv result-as-right-operand binding
    //     IsFieldInv · (V_RIGHT_INSTRUCTION_INPUT − FieldOpResult) = 0
    //     On FINV cycles, V_RIGHT carries the result so V_PRODUCT = A·Result.
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_INV, 1)]));
    b_rows.push(row::<F>(&[
        (V_RIGHT_INSTRUCTION_INPUT, 1),
        (V_FIELD_OP_RESULT, -1),
    ]));
    c_rows.push(empty());

    // 26: FieldInv unit product
    //     IsFieldInv · (V_PRODUCT − 1) = 0
    //     With rows 24/25 and product-constraint 27, this forces A·Result = 1.
    //     Does NOT cover FINV(0)=0 — guests must avoid inverting zero.
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_INV, 1)]));
    b_rows.push(row::<F>(&[(V_PRODUCT, 1), (V_CONST, -1)]));
    c_rows.push(empty());

    // Product constraints (27-29)
    // Form: left · right = output  →  A=left, B=right, C=output

    // 27: Product = LeftInstructionInput × RightInstructionInput
    //     Also supplies A·B (or A·Result for FINV) to the FMUL/FINV gates.
    a_rows.push(row::<F>(&[(V_LEFT_INSTRUCTION_INPUT, 1)]));
    b_rows.push(row::<F>(&[(V_RIGHT_INSTRUCTION_INPUT, 1)]));
    c_rows.push(row::<F>(&[(V_PRODUCT, 1)]));

    // 28: ShouldBranch = LookupOutput × Branch
    a_rows.push(row::<F>(&[(V_LOOKUP_OUTPUT, 1)]));
    b_rows.push(row::<F>(&[(V_BRANCH, 1)]));
    c_rows.push(row::<F>(&[(V_SHOULD_BRANCH, 1)]));

    // 29: ShouldJump = Jump × (1 − NextIsNoop)
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
mod tests {
    use super::*;
    use jolt_field::Fr;
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

    /// A FieldOp-like witness mirroring the no-op skeleton but with the
    /// FieldOp slots populated. Used by both the FADD/FSUB positive and
    /// negative tests so the non-FieldOp constraints stay vacuously satisfied.
    fn field_op_witness(
        flag_idx: usize,
        a: Fr,
        b: Fr,
        result: Fr,
    ) -> Vec<Fr> {
        let mut w = noop_witness();
        w[flag_idx] = Fr::from_u64(1);
        w[V_FIELD_OP_A] = a;
        w[V_FIELD_OP_B] = b;
        w[V_FIELD_OP_RESULT] = result;
        w
    }

    #[test]
    fn fadd_gate_accepts_correct_result() {
        let matrices = rv64_constraints::<Fr>();
        let a = Fr::from_u64(1234);
        let b = Fr::from_u64(5678);
        let w = field_op_witness(V_FLAG_IS_FIELD_ADD, a, b, a + b);
        matrices
            .check_witness(&w)
            .expect("FADD with a+b=result should satisfy all constraints");
    }

    #[test]
    fn fadd_gate_rejects_wrong_result() {
        let matrices = rv64_constraints::<Fr>();
        let a = Fr::from_u64(1234);
        let b = Fr::from_u64(5678);
        let w = field_op_witness(V_FLAG_IS_FIELD_ADD, a, b, a + b + Fr::from_u64(1));
        assert!(
            matrices.check_witness(&w).is_err(),
            "FADD with tampered result must violate constraint 19",
        );
    }

    #[test]
    fn fadd_gate_ignored_when_flag_zero() {
        let matrices = rv64_constraints::<Fr>();
        // Flag = 0, so the guard nullifies the check; any A/B/RESULT is fine.
        let mut w = noop_witness();
        w[V_FIELD_OP_A] = Fr::from_u64(7);
        w[V_FIELD_OP_B] = Fr::from_u64(9);
        w[V_FIELD_OP_RESULT] = Fr::from_u64(42);
        matrices
            .check_witness(&w)
            .expect("FADD must be vacuous when flag=0");
    }

    #[test]
    fn fsub_gate_accepts_correct_result() {
        let matrices = rv64_constraints::<Fr>();
        let a = Fr::from_u64(9999);
        let b = Fr::from_u64(1234);
        let w = field_op_witness(V_FLAG_IS_FIELD_SUB, a, b, a - b);
        matrices
            .check_witness(&w)
            .expect("FSUB with a-b=result should satisfy all constraints");
    }

    #[test]
    fn fsub_gate_rejects_wrong_result() {
        let matrices = rv64_constraints::<Fr>();
        let a = Fr::from_u64(9999);
        let b = Fr::from_u64(1234);
        let w = field_op_witness(V_FLAG_IS_FIELD_SUB, a, b, a - b - Fr::from_u64(1));
        assert!(
            matrices.check_witness(&w).is_err(),
            "FSUB with tampered result must violate constraint 20",
        );
    }

    /// Sanity: FADD gate is NOT activated when FSUB flag is set. Used to verify
    /// the flag→gate mapping isn't scrambled.
    #[test]
    fn field_flags_are_independent() {
        let matrices = rv64_constraints::<Fr>();
        let a = Fr::from_u64(100);
        let b = Fr::from_u64(30);
        // Set FSUB flag, supply correct sub result — FADD gate must not mis-fire.
        let w = field_op_witness(V_FLAG_IS_FIELD_SUB, a, b, a - b);
        matrices
            .check_witness(&w)
            .expect("FSUB flag must not trigger FADD gate");
    }

    /// FMUL and FINV reuse V_PRODUCT, so their witnesses must populate
    /// V_LEFT/V_RIGHT (= A / B-or-Result) in addition to the FieldOp columns,
    /// and mirror those values into V_LEFT_LOOKUP_OPERAND / V_RIGHT_LOOKUP_OPERAND
    /// to satisfy rows 6 and 10 whose guards fire on any cycle with RV
    /// Add/Sub/Mul all zero.
    fn field_op_mul_witness(a: Fr, b: Fr, result: Fr) -> Vec<Fr> {
        let mut w = noop_witness();
        w[V_FLAG_IS_FIELD_MUL] = Fr::from_u64(1);
        w[V_FIELD_OP_A] = a;
        w[V_FIELD_OP_B] = b;
        w[V_FIELD_OP_RESULT] = result;
        // Bind via V_PRODUCT: V_LEFT·V_RIGHT = V_PRODUCT (product constraint 27)
        w[V_LEFT_INSTRUCTION_INPUT] = a;
        w[V_RIGHT_INSTRUCTION_INPUT] = b;
        w[V_PRODUCT] = a * b;
        // Rows 6/10 fire (guard = 1-Add-Sub-Mul[-Advice]) → mirror into lookup slots.
        w[V_LEFT_LOOKUP_OPERAND] = a;
        w[V_RIGHT_LOOKUP_OPERAND] = b;
        w
    }

    fn field_op_inv_witness(a: Fr, result: Fr) -> Vec<Fr> {
        let mut w = noop_witness();
        w[V_FLAG_IS_FIELD_INV] = Fr::from_u64(1);
        w[V_FIELD_OP_A] = a;
        // FINV: FieldOpB unused. Witness B=0; row 10 will force V_RIGHT_LOOKUP=V_RIGHT=Result.
        w[V_FIELD_OP_B] = Fr::zero();
        w[V_FIELD_OP_RESULT] = result;
        w[V_LEFT_INSTRUCTION_INPUT] = a;
        w[V_RIGHT_INSTRUCTION_INPUT] = result;
        w[V_PRODUCT] = a * result;
        w[V_LEFT_LOOKUP_OPERAND] = a;
        w[V_RIGHT_LOOKUP_OPERAND] = result;
        w
    }

    #[test]
    fn fmul_gate_accepts_correct_result() {
        let matrices = rv64_constraints::<Fr>();
        let a = Fr::from_u64(1234);
        let b = Fr::from_u64(5678);
        let w = field_op_mul_witness(a, b, a * b);
        matrices
            .check_witness(&w)
            .expect("FMUL with a·b=result should satisfy all constraints");
    }

    #[test]
    fn fmul_gate_rejects_wrong_result() {
        let matrices = rv64_constraints::<Fr>();
        let a = Fr::from_u64(1234);
        let b = Fr::from_u64(5678);
        // Tamper: claim product is a·b + 1. V_PRODUCT stays a·b per the routing,
        // so gate 23 fires: 1·(a·b − (a·b+1)) = -1 ≠ 0.
        let w = field_op_mul_witness(a, b, a * b + Fr::from_u64(1));
        assert!(
            matrices.check_witness(&w).is_err(),
            "FMUL with tampered result must violate gate 23",
        );
    }

    #[test]
    fn fmul_gate_ignored_when_flag_zero() {
        let matrices = rv64_constraints::<Fr>();
        // Noop skeleton with random FieldOp columns + flag=0 — all FMUL rows vacuous.
        let mut w = noop_witness();
        w[V_FIELD_OP_A] = Fr::from_u64(7);
        w[V_FIELD_OP_B] = Fr::from_u64(9);
        w[V_FIELD_OP_RESULT] = Fr::from_u64(42);
        matrices
            .check_witness(&w)
            .expect("FMUL must be vacuous when flag=0");
    }

    #[test]
    fn finv_gate_accepts_correct_result() {
        let matrices = rv64_constraints::<Fr>();
        let a = Fr::from_u64(7);
        let inv = a.inverse().expect("7 ≠ 0 in BN254 Fr");
        let w = field_op_inv_witness(a, inv);
        matrices
            .check_witness(&w)
            .expect("FINV with a·inv=1 should satisfy all constraints");
    }

    #[test]
    fn finv_gate_rejects_wrong_result() {
        let matrices = rv64_constraints::<Fr>();
        let a = Fr::from_u64(7);
        // Claim result=2 so a·2 = 14 ≠ 1 — gate 26 fires: 1·(14 − 1) ≠ 0.
        let w = field_op_inv_witness(a, Fr::from_u64(2));
        assert!(
            matrices.check_witness(&w).is_err(),
            "FINV with tampered result must violate gate 26",
        );
    }

    #[test]
    fn fmul_and_finv_flags_are_exclusive() {
        let matrices = rv64_constraints::<Fr>();
        let a = Fr::from_u64(100);
        let b = Fr::from_u64(30);
        // Set IsFieldMul, populate the FMUL witness — FINV gate 26 must not
        // fire (IsFieldInv=0 keeps its guard at zero).
        let w = field_op_mul_witness(a, b, a * b);
        matrices
            .check_witness(&w)
            .expect("IsFieldMul must not activate FINV's unit-product gate");
    }
}
