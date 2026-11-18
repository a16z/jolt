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
//! | `[1..=NUM_R1CS_INPUTS]` | R1CS inputs (registers, flags, PC, lookups) |
//! | Next three columns | Product factor variables (`Branch`, `NextIsNoop`, `IsNoop`) |
//!
//! # Constraint forms
//!
//! - **Eq-conditional** (rows 0–18): `guard · (left − right) = 0`
//! - **Product** (rows 19–21): `left · right = output`

use jolt_riscv::{CircuitFlags, NUM_CIRCUIT_FLAGS};

use crate::protocols::jolt::geometry::spartan::PRODUCT_HALT_LANE_OUTPUT;

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

pub const V_FLAG_ADD_OPERANDS: usize = 21;
pub const V_FLAG_SUBTRACT_OPERANDS: usize = flag_column(CircuitFlags::SubtractOperands);
pub const V_FLAG_MULTIPLY_OPERANDS: usize = flag_column(CircuitFlags::MultiplyOperands);
pub const V_FLAG_LOAD: usize = flag_column(CircuitFlags::Load);
pub const V_FLAG_STORE: usize = flag_column(CircuitFlags::Store);
pub const V_FLAG_JUMP: usize = flag_column(CircuitFlags::Jump);
pub const V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD: usize =
    flag_column(CircuitFlags::WriteLookupOutputToRD);
pub const V_FLAG_VIRTUAL_INSTRUCTION: usize = flag_column(CircuitFlags::VirtualInstruction);
pub const V_FLAG_ASSERT: usize = flag_column(CircuitFlags::Assert);
pub const V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC: usize =
    flag_column(CircuitFlags::DoNotUpdateUnexpandedPC);
pub const V_FLAG_ADVICE: usize = flag_column(CircuitFlags::Advice);
pub const V_FLAG_IS_COMPRESSED: usize = flag_column(CircuitFlags::IsCompressed);
pub const V_FLAG_IS_FIRST_IN_SEQUENCE: usize = flag_column(CircuitFlags::IsFirstInSequence);
pub const V_FLAG_IS_LAST_IN_SEQUENCE: usize = flag_column(CircuitFlags::IsLastInSequence);
pub const V_FLAG_HALT: usize = flag_column(CircuitFlags::Halt);
pub const V_FLAG_HALT_PANIC: usize = flag_column(CircuitFlags::HaltPanic);

const _: () = assert!(V_FLAG_ADD_OPERANDS == V_LOOKUP_OUTPUT + 1);

pub const V_BRANCH: usize = 1 + NUM_R1CS_INPUTS;
pub const V_NEXT_IS_NOOP: usize = V_BRANCH + 1;
pub const V_IS_NOOP: usize = V_NEXT_IS_NOOP + 1;

pub const NUM_R1CS_INPUTS: usize = 20 + NUM_CIRCUIT_FLAGS;
pub const NUM_PRODUCT_FACTORS: usize = 3;
pub const NUM_VARS_PER_CYCLE: usize = 1 + NUM_R1CS_INPUTS + NUM_PRODUCT_FACTORS;
pub const NUM_EQ_CONSTRAINTS: usize = 19;
pub const NUM_PRODUCT_CONSTRAINTS: usize = 3;
pub const NUM_CONSTRAINTS_PER_CYCLE: usize = NUM_EQ_CONSTRAINTS + NUM_PRODUCT_CONSTRAINTS;

pub const fn flag_column(flag: CircuitFlags) -> usize {
    V_FLAG_ADD_OPERANDS + flag as usize
}

pub const RV64_VARIABLE_NAMES: [&str; NUM_VARS_PER_CYCLE] = {
    let mut names = [""; NUM_VARS_PER_CYCLE];
    names[V_CONST] = "One";
    names[V_LEFT_INSTRUCTION_INPUT] = "LeftInstructionInput";
    names[V_RIGHT_INSTRUCTION_INPUT] = "RightInstructionInput";
    names[V_PRODUCT] = "Product";
    names[V_SHOULD_BRANCH] = "ShouldBranch";
    names[V_PC] = "PC";
    names[V_UNEXPANDED_PC] = "UnexpandedPC";
    names[V_IMM] = "Imm";
    names[V_RAM_ADDRESS] = "RamAddress";
    names[V_RS1_VALUE] = "Rs1Value";
    names[V_RS2_VALUE] = "Rs2Value";
    names[V_RD_WRITE_VALUE] = "RdWriteValue";
    names[V_RAM_READ_VALUE] = "RamReadValue";
    names[V_RAM_WRITE_VALUE] = "RamWriteValue";
    names[V_LEFT_LOOKUP_OPERAND] = "LeftLookupOperand";
    names[V_RIGHT_LOOKUP_OPERAND] = "RightLookupOperand";
    names[V_NEXT_UNEXPANDED_PC] = "NextUnexpandedPC";
    names[V_NEXT_PC] = "NextPC";
    names[V_NEXT_IS_VIRTUAL] = "NextIsVirtual";
    names[V_NEXT_IS_FIRST_IN_SEQUENCE] = "NextIsFirstInSequence";
    names[V_LOOKUP_OUTPUT] = "LookupOutput";
    names[V_FLAG_ADD_OPERANDS] = "OpFlags_AddOperands";
    names[V_FLAG_SUBTRACT_OPERANDS] = "OpFlags_SubtractOperands";
    names[V_FLAG_MULTIPLY_OPERANDS] = "OpFlags_MultiplyOperands";
    names[V_FLAG_LOAD] = "OpFlags_Load";
    names[V_FLAG_STORE] = "OpFlags_Store";
    names[V_FLAG_JUMP] = "OpFlags_Jump";
    names[V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD] = "OpFlags_WriteLookupOutputToRD";
    names[V_FLAG_VIRTUAL_INSTRUCTION] = "OpFlags_VirtualInstruction";
    names[V_FLAG_ASSERT] = "OpFlags_Assert";
    names[V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] = "OpFlags_DoNotUpdateUnexpandedPC";
    names[V_FLAG_ADVICE] = "OpFlags_Advice";
    names[V_FLAG_IS_COMPRESSED] = "OpFlags_IsCompressed";
    names[V_FLAG_IS_FIRST_IN_SEQUENCE] = "OpFlags_IsFirstInSequence";
    names[V_FLAG_IS_LAST_IN_SEQUENCE] = "OpFlags_IsLastInSequence";
    names[V_FLAG_HALT] = "OpFlags_Halt";
    names[V_FLAG_HALT_PANIC] = "OpFlags_HaltPanic";
    #[cfg(feature = "field-inline")]
    #[expect(
        clippy::indexing_slicing,
        reason = "const evaluation checks these circuit-flag indices against the variable layout"
    )]
    {
        names[flag_column(CircuitFlags::FieldAdd)] = "OpFlags_FieldAdd";
        names[flag_column(CircuitFlags::FieldSub)] = "OpFlags_FieldSub";
        names[flag_column(CircuitFlags::FieldMul)] = "OpFlags_FieldMul";
        names[flag_column(CircuitFlags::FieldInv)] = "OpFlags_FieldInv";
        names[flag_column(CircuitFlags::FieldAssertEq)] = "OpFlags_FieldAssertEq";
        names[flag_column(CircuitFlags::FieldLoadAccumulateFromRegister)] =
            "OpFlags_FieldLoadAccumulateFromRegister";
        names[flag_column(CircuitFlags::FieldAssertZero)] = "OpFlags_FieldAssertZero";
        names[flag_column(CircuitFlags::FieldLoadImm)] = "OpFlags_FieldLoadImm";
        names[flag_column(CircuitFlags::FieldLoadAccumulateFromMemory)] =
            "OpFlags_FieldLoadAccumulateFromMemory";
        names[flag_column(CircuitFlags::FieldAdviceLimb)] = "OpFlags_FieldAdviceLimb";
    }
    names[V_BRANCH] = "Branch";
    names[V_NEXT_IS_NOOP] = "NextIsNoop";
    names[V_IS_NOOP] = "IsNoop";
    names
};

pub const RV64_CONSTRAINT_NAMES: [&str; NUM_CONSTRAINTS_PER_CYCLE] = [
    "RamAddrEqRs1PlusImmMinusLookupIfLoadStore",
    "RamAddrEqZeroIfNotLoadStore",
    "RamReadEqRamWriteIfLoad",
    "RamReadEqRdWriteIfLoad",
    "Rs2EqRamWriteIfStore",
    "LeftLookupZeroUnlessAddSubMul",
    "LeftLookupEqLeftInputOtherwise",
    "RightLookupAdd",
    "RightLookupSub",
    "RightLookupEqProductIfMul",
    "RightLookupEqRightInputOtherwise",
    "AssertLookupOne",
    "RdWriteEqLookupIfWriteLookupToRd",
    "RdWriteEqPCPlusConstIfWritePCtoRD",
    "NextUnexpPCEqLookupIfJump",
    "NextUnexpPCEqPCPlusImmIfShouldBranch",
    "NextUnexpPCUpdateOtherwise",
    "NextPCEqPCPlusOneIfInline",
    "MustStartSequenceFromBeginning",
    "Product",
    "ShouldBranch",
    "NextIsNoopEqIsNoopUnlessHalt",
];

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

const TWOS_COMPLEMENT_BIAS: i128 = 0x1_0000_0000_0000_0000;

use jolt_field::Field;
use jolt_r1cs::{ConstraintMatrices, SparseRow};

type ConstraintRows<F> = (Vec<SparseRow<F>>, Vec<SparseRow<F>>, Vec<SparseRow<F>>);

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

    // 0: RamAddrEqRs1PlusImmMinusLookupIfLoadStore
    //    guard = Load + Store
    //    left  = RamAddress
    //    right = Rs1Value + Imm − LookupOutput
    //
    // NOTE: a memory row's lookup output is an intra-doubleword offset.
    // `LD`/`SD` look up no table, so instruction read-RAF forces it to 0 and
    // `RamAddress` is the effective address. The floor-addressed
    // `VirtualLoad*`/`VirtualStore*` rows look up `AddressOffset{B,H,W}`,
    // i.e. `ea & 7`, `ea & 6` or `ea & 4`, on the `AddOperands` path, whose
    // index rows 5 and 7 pin to the integer `Rs1Value + Imm` (on Akita, whose
    // modulus is below 2^128, instruction read-RAF also pins it to its
    // canonical representative). `RamAddress` is then `ea` with the offset
    // bits its access width allows cleared. RAM RAF evaluation admits only
    // `RamAddress = Σ_k ra(k)·(8k + lowest_address)` with boolean `ra`, a
    // multiple of 8 because the verifier rejects a layout whose
    // `lowest_address` is not (`validate_ram_remap_base`). Every quantity is
    // far below either field modulus, so that identity holds over the
    // integers, and an access misaligned for its width (any misaligned
    // `LD`/`SD`; `LH`/`LW`/`SH`/`SW` whose `ea` is not a multiple of the width)
    // has no satisfying RAM address. jolt-lookup-tables pins that a memory row
    // looks up no table or an `AddressOffset` table; any other table would
    // shift its RAM address.
    //
    // `Rs1Value + Imm` is the unwrapped sum, while the tracer wraps `ea` mod
    // 2^64, so an access whose sum carries has no satisfying RAM address on
    // any layout the tracer can allocate. Load and store immediates are signed
    // 12-bit (decoded rows by encoding, inline rows by jolt-program's
    // `materialize_inline`), so such an access lands within 2 KiB of 0 or
    // 2^64: past that RAM, or below `lowest_address`, which is at least 2^30
    // because `MemoryLayout::try_new` pads I/O below `RAM_START_ADDRESS` = 2^31
    // to a power of two and rejects a region reaching 0. The tracer rejects
    // accesses below `lowest_address` and past the allocated RAM.
    a_rows.push(row::<F>(&[(V_FLAG_LOAD, 1), (V_FLAG_STORE, 1)]));
    b_rows.push(row::<F>(&[
        (V_RAM_ADDRESS, 1),
        (V_RS1_VALUE, -1),
        (V_IMM, -1),
        (V_LOOKUP_OUTPUT, 1),
    ]));
    c_rows.push(empty());

    a_rows.push(row::<F>(&[
        (V_CONST, 1),
        (V_FLAG_LOAD, -1),
        (V_FLAG_STORE, -1),
    ]));
    b_rows.push(row::<F>(&[(V_RAM_ADDRESS, 1)]));
    c_rows.push(empty());

    a_rows.push(row::<F>(&[(V_FLAG_LOAD, 1)]));
    b_rows.push(row::<F>(&[(V_RAM_READ_VALUE, 1), (V_RAM_WRITE_VALUE, -1)]));
    c_rows.push(empty());

    a_rows.push(row::<F>(&[(V_FLAG_LOAD, 1)]));
    b_rows.push(row::<F>(&[(V_RAM_READ_VALUE, 1), (V_RD_WRITE_VALUE, -1)]));
    c_rows.push(empty());

    a_rows.push(row::<F>(&[(V_FLAG_STORE, 1)]));
    b_rows.push(row::<F>(&[(V_RS2_VALUE, 1), (V_RAM_WRITE_VALUE, -1)]));
    c_rows.push(empty());

    a_rows.push(row::<F>(&[
        (V_FLAG_ADD_OPERANDS, 1),
        (V_FLAG_SUBTRACT_OPERANDS, 1),
        (V_FLAG_MULTIPLY_OPERANDS, 1),
    ]));
    b_rows.push(row::<F>(&[(V_LEFT_LOOKUP_OPERAND, 1)]));
    c_rows.push(empty());

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

    a_rows.push(row::<F>(&[(V_FLAG_ADD_OPERANDS, 1)]));
    b_rows.push(row::<F>(&[
        (V_RIGHT_LOOKUP_OPERAND, 1),
        (V_LEFT_INSTRUCTION_INPUT, -1),
        (V_RIGHT_INSTRUCTION_INPUT, -1),
    ]));
    c_rows.push(empty());

    a_rows.push(row::<F>(&[(V_FLAG_SUBTRACT_OPERANDS, 1)]));
    b_rows.push(row_wide::<F>(&[
        (V_RIGHT_LOOKUP_OPERAND, 1),
        (V_LEFT_INSTRUCTION_INPUT, -1),
        (V_RIGHT_INSTRUCTION_INPUT, 1),
        (V_CONST, -TWOS_COMPLEMENT_BIAS),
    ]));
    c_rows.push(empty());

    a_rows.push(row::<F>(&[(V_FLAG_MULTIPLY_OPERANDS, 1)]));
    b_rows.push(row::<F>(&[(V_RIGHT_LOOKUP_OPERAND, 1), (V_PRODUCT, -1)]));
    c_rows.push(empty());

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

    a_rows.push(row::<F>(&[(V_FLAG_ASSERT, 1)]));
    b_rows.push(row::<F>(&[(V_LOOKUP_OUTPUT, 1), (V_CONST, -1)]));
    c_rows.push(empty());

    a_rows.push(row::<F>(&[(V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD, 1)]));
    b_rows.push(row::<F>(&[(V_RD_WRITE_VALUE, 1), (V_LOOKUP_OUTPUT, -1)]));
    c_rows.push(empty());

    a_rows.push(row::<F>(&[(V_FLAG_JUMP, 1)]));
    b_rows.push(row::<F>(&[
        (V_RD_WRITE_VALUE, 1),
        (V_UNEXPANDED_PC, -1),
        (V_CONST, -4),
        (V_FLAG_IS_COMPRESSED, 2),
    ]));
    c_rows.push(empty());

    // A halt's target is its lookup output 0 only because halts have no lookup
    // table; attaching one to a halt kind would need the guard `Jump − Halt`.
    a_rows.push(row::<F>(&[(V_FLAG_JUMP, 1)]));
    b_rows.push(row::<F>(&[
        (V_NEXT_UNEXPANDED_PC, 1),
        (V_LOOKUP_OUTPUT, -1),
    ]));
    c_rows.push(empty());

    a_rows.push(row::<F>(&[(V_SHOULD_BRANCH, 1)]));
    b_rows.push(row::<F>(&[
        (V_NEXT_UNEXPANDED_PC, 1),
        (V_UNEXPANDED_PC, -1),
        (V_IMM, -1),
    ]));
    c_rows.push(empty());

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
    // (`NextUnexpPCEqLookupIfJump`) / #16
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

    a_rows.push(row::<F>(&[
        (V_NEXT_IS_VIRTUAL, 1),
        (V_NEXT_IS_FIRST_IN_SEQUENCE, -1),
    ]));
    b_rows.push(row::<F>(&[
        (V_CONST, 1),
        (V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC, -1),
    ]));
    c_rows.push(empty());

    (a_rows, b_rows, c_rows)
}

fn append_product_constraints<F: Field>(
    a_rows: &mut Vec<SparseRow<F>>,
    b_rows: &mut Vec<SparseRow<F>>,
    c_rows: &mut Vec<SparseRow<F>>,
    claimed_panic: bool,
) {
    a_rows.push(row::<F>(&[(V_LEFT_INSTRUCTION_INPUT, 1)]));
    b_rows.push(row::<F>(&[(V_RIGHT_INSTRUCTION_INPUT, 1)]));
    c_rows.push(row::<F>(&[(V_PRODUCT, 1)]));

    a_rows.push(row::<F>(&[(V_LOOKUP_OUTPUT, 1)]));
    b_rows.push(row::<F>(&[(V_BRANCH, 1)]));
    c_rows.push(row::<F>(&[(V_SHOULD_BRANCH, 1)]));

    a_rows.push(row::<F>(&[(V_CONST, 1)]));
    b_rows.push(row::<F>(&[(V_NEXT_IS_NOOP, 1), (V_IS_NOOP, -1)]));
    let halt_lane_output: Vec<(usize, i128)> = PRODUCT_HALT_LANE_OUTPUT
        .iter()
        .map(|term| {
            (
                flag_column(term.flag),
                i128::from(term.constant) + i128::from(term.per_panic) * i128::from(claimed_panic),
            )
        })
        .collect();
    c_rows.push(row::<F>(&halt_lane_output));
}

/// Build only the Jolt RV64 equality-conditional constraints.
///
/// Returns the 19 rows with form `guard · (left - right) = 0`, over the
/// [`NUM_VARS_PER_CYCLE`] per-cycle witness variables. Product constraints are
/// intentionally excluded for consumers that handle multiplication checks in
/// a separate protocol step.
pub fn rv64_spartan_outer_constraints<F: Field>() -> ConstraintMatrices<F> {
    let (a_rows, b_rows, c_rows) = rv64_eq_constraint_rows();
    ConstraintMatrices::new(
        NUM_EQ_CONSTRAINTS,
        NUM_VARS_PER_CYCLE,
        a_rows,
        b_rows,
        c_rows,
    )
}

/// Build the full Jolt RV64 R1CS constraint matrices.
///
/// Returns 22 constraints over [`NUM_VARS_PER_CYCLE`] variables per cycle:
/// - 19 equality-conditional: `guard · (left − right) = 0` → A=guard, B=left−right, C=0
/// - 3 product: `left · right = output` → A=left, B=right, C=output
///
/// Row 21's output is product lane 3's (`PRODUCT_HALT_LANE_OUTPUT`), which
/// depends on the statement's claimed panic bit. The proof system checks these
/// rows through the Spartan outer and product relations, never through this
/// full-trace matrix.
///
/// Variable layout matches the constants in this module: the constant, all R1CS
/// inputs, then the three product factors.
pub fn rv64_trace_constraints<F: Field>(claimed_panic: bool) -> ConstraintMatrices<F> {
    let (mut a_rows, mut b_rows, mut c_rows) = rv64_eq_constraint_rows();
    a_rows.reserve(NUM_PRODUCT_CONSTRAINTS);
    b_rows.reserve(NUM_PRODUCT_CONSTRAINTS);
    c_rows.reserve(NUM_PRODUCT_CONSTRAINTS);
    append_product_constraints(&mut a_rows, &mut b_rows, &mut c_rows, claimed_panic);

    ConstraintMatrices::new(
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
    use std::collections::BTreeSet;

    use super::*;
    use jolt_field::{Fr, Ring};
    use jolt_riscv::CIRCUIT_FLAGS;
    use num_traits::Zero;

    /// A no-op cycle followed by another: const, DoNotUpdate, IsNoop and
    /// NextIsNoop are 1, all else zero. Constraint 16
    /// (NextUnexpPCUpdateOtherwise), whose guard is 1−0−0 = 1, requires
    /// NextUnexpPC = UnexpPC + 4 − 4·DoNotUpdate − 2·IsCompressed.
    /// For the no-op (DoNotUpdate=1): NextUnexpPC = UnexpPC + 4 − 4 = UnexpPC.
    /// With both at 0 this holds.
    fn noop_witness() -> Vec<Fr> {
        let mut w = vec![Fr::zero(); NUM_VARS_PER_CYCLE];
        w[V_CONST] = Fr::from_u64(1);
        w[V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] = Fr::from_u64(1);
        w[V_IS_NOOP] = Fr::from_u64(1);
        w[V_NEXT_IS_NOOP] = Fr::from_u64(1);
        w
    }

    #[test]
    fn noop_satisfies_constraints() {
        for claimed_panic in [false, true] {
            let matrices = rv64_trace_constraints::<Fr>(claimed_panic);
            assert_eq!(matrices.num_constraints, NUM_CONSTRAINTS_PER_CYCLE);
            assert_eq!(matrices.num_vars, NUM_VARS_PER_CYCLE);
            matrices
                .check_witness(&noop_witness())
                .expect("noop should satisfy all constraints");
        }
    }

    #[test]
    fn exported_layout_names_are_complete_and_unique() {
        assert!(RV64_VARIABLE_NAMES.iter().all(|name| !name.is_empty()));
        assert!(RV64_CONSTRAINT_NAMES.iter().all(|name| !name.is_empty()));
        assert_eq!(
            RV64_VARIABLE_NAMES
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len(),
            NUM_VARS_PER_CYCLE
        );
        assert_eq!(
            RV64_CONSTRAINT_NAMES
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len(),
            NUM_CONSTRAINTS_PER_CYCLE
        );
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
        assert_eq!(
            input_column(NUM_R1CS_INPUTS - 1),
            CIRCUIT_FLAGS.last().copied().map(flag_column)
        );
        assert_eq!(input_column(NUM_R1CS_INPUTS), None);
    }
}

/// Satisfaction tests with realistic per-instruction execution witnesses.
///
/// Each witness models one cycle of a concrete RV64 instruction with
/// hand-computed values from the RISC-V spec. Cell semantics follow the
/// witness generation pipeline:
/// - circuit flags per instruction: `jolt-riscv/src/instructions/` (the
///   `jolt_instruction!` macro in `jolt-riscv/src/lib.rs` adds the row-level
///   `VirtualInstruction`/`DoNotUpdateUnexpandedPC`/`IsLastInSequence`/
///   `IsCompressed` flags);
/// - instruction inputs and lookup operands: the per-instruction
///   `LookupQuery` impls in `jolt-lookup-tables/src/instructions/riscv/`;
/// - remaining trace columns: `trace_virtual_value` in
///   `jolt-witness/src/protocols/jolt_vm/trace.rs`.
///
/// Falsification tests perturb one semantically meaningful cell and assert
/// the exact first-violated constraint row reported by `check_witness`.
#[cfg(test)]
#[expect(clippy::indexing_slicing, reason = "tests index fixed witness vectors")]
mod execution_witness_tests {
    use super::*;
    use jolt_field::{Fr, Ring};
    use jolt_riscv::{
        Flags, JoltInstruction, JoltInstructionKind, JoltInstructionRow, NormalizedOperands,
        CIRCUIT_FLAGS,
    };
    use num_traits::Zero;

    // Constraint row indices, matching the push order in
    // `rv64_eq_constraint_rows` and `append_product_constraints`.
    const RAM_ADDR_EQ_RS1_PLUS_IMM_MINUS_LOOKUP_IF_LOAD_STORE: usize = 0;
    const RAM_READ_EQ_RAM_WRITE_IF_LOAD: usize = 2;
    const RAM_READ_EQ_RD_WRITE_IF_LOAD: usize = 3;
    const RS2_EQ_RAM_WRITE_IF_STORE: usize = 4;
    const LEFT_LOOKUP_EQ_LEFT_INPUT_OTHERWISE: usize = 6;
    const RIGHT_LOOKUP_ADD: usize = 7;
    const RIGHT_LOOKUP_EQ_PRODUCT_IF_MUL: usize = 9;
    const RIGHT_LOOKUP_EQ_RIGHT_INPUT_OTHERWISE: usize = 10;
    const RD_WRITE_EQ_LOOKUP_IF_WRITE_LOOKUP_TO_RD: usize = 12;
    const RD_WRITE_EQ_PC_PLUS_CONST_IF_JUMP: usize = 13;
    const NEXT_UNEXP_PC_EQ_LOOKUP_IF_JUMP: usize = 14;
    const NEXT_UNEXP_PC_EQ_PC_PLUS_IMM_IF_SHOULD_BRANCH: usize = 15;
    const NEXT_UNEXP_PC_UPDATE_OTHERWISE: usize = 16;
    const NEXT_PC_EQ_PC_PLUS_ONE_IF_INLINE: usize = 17;
    const MUST_START_SEQUENCE_FROM_BEGINNING: usize = 18;
    const PRODUCT_EQ_LEFT_TIMES_RIGHT: usize = 19;
    const SHOULD_BRANCH_EQ_LOOKUP_TIMES_BRANCH: usize = 20;
    const HALT_LANE: usize = 21;

    fn check(witness: &[Fr], claimed_panic: bool) -> Result<(), usize> {
        rv64_trace_constraints::<Fr>(claimed_panic).check_witness(witness)
    }

    fn with_cell(witness: &[Fr], cell: usize, value: Fr) -> Vec<Fr> {
        let mut witness = witness.to_vec();
        witness[cell] = value;
        witness
    }

    fn cycle_witness() -> Vec<Fr> {
        let mut w = vec![Fr::zero(); NUM_VARS_PER_CYCLE];
        w[V_CONST] = Fr::from_u64(1);
        w
    }

    /// `ADD x3, x1, x2` with rs1 = 2^64 − 2, rs2 = 5.
    ///
    /// Circuit flags [AddOperands, WriteLookupOutputToRD] (i/add.rs). The
    /// right lookup operand carries the unwrapped 65-bit sum
    /// (2^64 − 2) + 5 = 2^64 + 3, while rd receives the RV64 wrapped
    /// result 3.
    fn add_witness() -> Vec<Fr> {
        const RS1: u64 = u64::MAX - 1;
        const RS2: u64 = 5;
        const UNWRAPPED_SUM: u128 = (1u128 << 64) + 3;
        const WRAPPED_SUM: u64 = 3;
        const UNEXPANDED_PC: u64 = 0x8000_0010;

        let mut w = cycle_witness();
        w[V_LEFT_INSTRUCTION_INPUT] = Fr::from_u64(RS1);
        w[V_RIGHT_INSTRUCTION_INPUT] = Fr::from_u64(RS2);
        // (2^64 − 2)·5 = 5·2^64 − 10, committed unconditionally (row 19).
        w[V_PRODUCT] = Fr::from_u128(5 * ((1u128 << 64) - 2));
        w[V_PC] = Fr::from_u64(7);
        w[V_NEXT_PC] = Fr::from_u64(8);
        w[V_UNEXPANDED_PC] = Fr::from_u64(UNEXPANDED_PC);
        w[V_NEXT_UNEXPANDED_PC] = Fr::from_u64(UNEXPANDED_PC + 4);
        w[V_RS1_VALUE] = Fr::from_u64(RS1);
        w[V_RS2_VALUE] = Fr::from_u64(RS2);
        w[V_RD_WRITE_VALUE] = Fr::from_u64(WRAPPED_SUM);
        w[V_RIGHT_LOOKUP_OPERAND] = Fr::from_u128(UNWRAPPED_SUM);
        w[V_LOOKUP_OUTPUT] = Fr::from_u64(WRAPPED_SUM);
        w[V_FLAG_ADD_OPERANDS] = Fr::from_u64(1);
        w[V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD] = Fr::from_u64(1);
        w
    }

    /// `SLTU x5, x1, x2` with rs1 = 7, rs2 = 9, so rd = 1.
    ///
    /// Circuit flags [WriteLookupOutputToRD] (i/sltu.rs). No operand
    /// combination flag is set, so the lookup operands pass through the
    /// instruction inputs unchanged (interleaved-operand default).
    fn sltu_witness() -> Vec<Fr> {
        const UNEXPANDED_PC: u64 = 0x8000_0020;

        let mut w = cycle_witness();
        w[V_LEFT_INSTRUCTION_INPUT] = Fr::from_u64(7);
        w[V_RIGHT_INSTRUCTION_INPUT] = Fr::from_u64(9);
        w[V_PRODUCT] = Fr::from_u64(63);
        w[V_PC] = Fr::from_u64(9);
        w[V_NEXT_PC] = Fr::from_u64(10);
        w[V_UNEXPANDED_PC] = Fr::from_u64(UNEXPANDED_PC);
        w[V_NEXT_UNEXPANDED_PC] = Fr::from_u64(UNEXPANDED_PC + 4);
        w[V_RS1_VALUE] = Fr::from_u64(7);
        w[V_RS2_VALUE] = Fr::from_u64(9);
        w[V_RD_WRITE_VALUE] = Fr::from_u64(1);
        w[V_LEFT_LOOKUP_OPERAND] = Fr::from_u64(7);
        w[V_RIGHT_LOOKUP_OPERAND] = Fr::from_u64(9);
        w[V_LOOKUP_OUTPUT] = Fr::from_u64(1);
        w[V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD] = Fr::from_u64(1);
        w
    }

    /// `LD x11, 8(x10)` with base 0x8000_1000 loading 0xDEAD_BEEF_CAFE_F00D.
    ///
    /// Circuit flags [Load] (i/ld.rs). Loads route the loaded value into
    /// RamReadValue, RamWriteValue (read-write identity), and RdWriteValue;
    /// the right instruction/lookup operand carries that value so instruction
    /// read-RAF proves it is a `u64`.
    fn ld_witness() -> Vec<Fr> {
        const BASE: u64 = 0x8000_1000;
        const LOADED: u64 = 0xDEAD_BEEF_CAFE_F00D;
        const UNEXPANDED_PC: u64 = 0x8000_0030;

        let mut w = cycle_witness();
        w[V_PC] = Fr::from_u64(11);
        w[V_NEXT_PC] = Fr::from_u64(12);
        w[V_UNEXPANDED_PC] = Fr::from_u64(UNEXPANDED_PC);
        w[V_NEXT_UNEXPANDED_PC] = Fr::from_u64(UNEXPANDED_PC + 4);
        w[V_IMM] = Fr::from_u64(8);
        w[V_RAM_ADDRESS] = Fr::from_u64(BASE + 8);
        w[V_RS1_VALUE] = Fr::from_u64(BASE);
        w[V_RD_WRITE_VALUE] = Fr::from_u64(LOADED);
        w[V_RAM_READ_VALUE] = Fr::from_u64(LOADED);
        w[V_RAM_WRITE_VALUE] = Fr::from_u64(LOADED);
        w[V_RIGHT_INSTRUCTION_INPUT] = Fr::from_u64(LOADED);
        w[V_RIGHT_LOOKUP_OPERAND] = Fr::from_u64(LOADED);
        w[V_FLAG_LOAD] = Fr::from_u64(1);
        w
    }

    /// `SD x12, -8(x10)` with base 0x8000_2000 storing 0x1122_3344_5566_7788
    /// over old memory value 0x0F0F_0F0F_0F0F_0F0F.
    ///
    /// Circuit flags [Store] (i/sd.rs). Stores write rs2 to memory
    /// (RamWriteValue = Rs2Value) and write no register (RdWriteValue = 0).
    /// The negative offset exercises signed immediate handling in the
    /// address constraint.
    fn sd_witness() -> Vec<Fr> {
        const BASE: u64 = 0x8000_2000;
        const STORED: u64 = 0x1122_3344_5566_7788;
        const OLD_VALUE: u64 = 0x0F0F_0F0F_0F0F_0F0F;
        const UNEXPANDED_PC: u64 = 0x8000_0034;

        let mut w = cycle_witness();
        w[V_PC] = Fr::from_u64(12);
        w[V_NEXT_PC] = Fr::from_u64(13);
        w[V_UNEXPANDED_PC] = Fr::from_u64(UNEXPANDED_PC);
        w[V_NEXT_UNEXPANDED_PC] = Fr::from_u64(UNEXPANDED_PC + 4);
        w[V_IMM] = Fr::from_i64(-8);
        w[V_RAM_ADDRESS] = Fr::from_u64(BASE - 8);
        w[V_RS1_VALUE] = Fr::from_u64(BASE);
        w[V_RS2_VALUE] = Fr::from_u64(STORED);
        w[V_RAM_READ_VALUE] = Fr::from_u64(OLD_VALUE);
        w[V_RAM_WRITE_VALUE] = Fr::from_u64(STORED);
        w[V_FLAG_STORE] = Fr::from_u64(1);
        w
    }

    /// `VirtualLoadH` opening the `LH` sequence for `ea = 0x8000_1010 − 6`,
    /// loading the doubleword at 0x8000_1008.
    ///
    /// Circuit flags [Load, AddOperands] plus the sequence-head flags; inputs
    /// rs1 and the signed immediate (virt/load_h.rs). The `AddressOffsetH`
    /// lookup over `ea = 0x8000_100A` returns `ea & 6 = 2`, which constraint 0
    /// subtracts.
    fn load_h_witness() -> Vec<Fr> {
        const BASE: u64 = 0x8000_1010;
        const IMM: i64 = -6;
        const EA: u64 = 0x8000_100A;
        const LOADED: u64 = 0x0123_4567_89AB_CDEF;
        const UNEXPANDED_PC: u64 = 0x8000_0038;

        let mut w = cycle_witness();
        w[V_LEFT_INSTRUCTION_INPUT] = Fr::from_u64(BASE);
        w[V_RIGHT_INSTRUCTION_INPUT] = Fr::from_i64(IMM);
        w[V_PRODUCT] = Fr::from_u64(BASE) * Fr::from_i64(IMM);
        w[V_PC] = Fr::from_u64(13);
        w[V_NEXT_PC] = Fr::from_u64(14);
        w[V_UNEXPANDED_PC] = Fr::from_u64(UNEXPANDED_PC);
        w[V_NEXT_UNEXPANDED_PC] = Fr::from_u64(UNEXPANDED_PC);
        w[V_IMM] = Fr::from_i64(IMM);
        w[V_RAM_ADDRESS] = Fr::from_u64(0x8000_1008);
        w[V_RS1_VALUE] = Fr::from_u64(BASE);
        w[V_RD_WRITE_VALUE] = Fr::from_u64(LOADED);
        w[V_RAM_READ_VALUE] = Fr::from_u64(LOADED);
        w[V_RAM_WRITE_VALUE] = Fr::from_u64(LOADED);
        w[V_RIGHT_LOOKUP_OPERAND] = Fr::from_u64(EA);
        w[V_LOOKUP_OUTPUT] = Fr::from_u64(2);
        w[V_NEXT_IS_VIRTUAL] = Fr::from_u64(1);
        w[V_FLAG_ADD_OPERANDS] = Fr::from_u64(1);
        w[V_FLAG_LOAD] = Fr::from_u64(1);
        w[V_FLAG_VIRTUAL_INSTRUCTION] = Fr::from_u64(1);
        w[V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] = Fr::from_u64(1);
        w[V_FLAG_IS_FIRST_IN_SEQUENCE] = Fr::from_u64(1);
        w
    }

    /// `BEQ x1, x2, -16` at 0x8000_0040 with rs1 = rs2 = 42: branch taken.
    ///
    /// BEQ sets no circuit flags (i/beq.rs); Branch is an instruction flag
    /// surfacing as the committed product factor `V_BRANCH`. The Equal
    /// lookup returns 1, so ShouldBranch = 1·1 and the next unexpanded PC
    /// is the backward target 0x8000_0040 − 16.
    fn beq_taken_witness() -> Vec<Fr> {
        const UNEXPANDED_PC: u64 = 0x8000_0040;

        let mut w = cycle_witness();
        w[V_LEFT_INSTRUCTION_INPUT] = Fr::from_u64(42);
        w[V_RIGHT_INSTRUCTION_INPUT] = Fr::from_u64(42);
        w[V_PRODUCT] = Fr::from_u64(1764);
        w[V_SHOULD_BRANCH] = Fr::from_u64(1);
        w[V_PC] = Fr::from_u64(15);
        w[V_NEXT_PC] = Fr::from_u64(11);
        w[V_UNEXPANDED_PC] = Fr::from_u64(UNEXPANDED_PC);
        w[V_NEXT_UNEXPANDED_PC] = Fr::from_u64(UNEXPANDED_PC - 16);
        w[V_IMM] = Fr::from_i64(-16);
        w[V_RS1_VALUE] = Fr::from_u64(42);
        w[V_RS2_VALUE] = Fr::from_u64(42);
        w[V_LEFT_LOOKUP_OPERAND] = Fr::from_u64(42);
        w[V_RIGHT_LOOKUP_OPERAND] = Fr::from_u64(42);
        w[V_LOOKUP_OUTPUT] = Fr::from_u64(1);
        w[V_BRANCH] = Fr::from_u64(1);
        w
    }

    fn beq_not_taken_witness() -> Vec<Fr> {
        const UNEXPANDED_PC: u64 = 0x8000_0040;

        let mut w = cycle_witness();
        w[V_LEFT_INSTRUCTION_INPUT] = Fr::from_u64(7);
        w[V_RIGHT_INSTRUCTION_INPUT] = Fr::from_u64(9);
        w[V_PRODUCT] = Fr::from_u64(63);
        w[V_PC] = Fr::from_u64(15);
        w[V_NEXT_PC] = Fr::from_u64(16);
        w[V_UNEXPANDED_PC] = Fr::from_u64(UNEXPANDED_PC);
        w[V_NEXT_UNEXPANDED_PC] = Fr::from_u64(UNEXPANDED_PC + 4);
        w[V_IMM] = Fr::from_i64(-16);
        w[V_RS1_VALUE] = Fr::from_u64(7);
        w[V_RS2_VALUE] = Fr::from_u64(9);
        w[V_LEFT_LOOKUP_OPERAND] = Fr::from_u64(7);
        w[V_RIGHT_LOOKUP_OPERAND] = Fr::from_u64(9);
        w[V_BRANCH] = Fr::from_u64(1);
        w
    }

    /// `JAL x1, +0x100` at 0x8000_0100: jump to 0x8000_0200, link 0x8000_0104.
    ///
    /// Circuit flags [AddOperands, Jump] (i/jal.rs); the left instruction
    /// input is the unexpanded PC and the right is the immediate, so the
    /// RangeCheck lookup computes the jump target PC + imm. The next row is
    /// a real instruction.
    fn jal_witness() -> Vec<Fr> {
        const UNEXPANDED_PC: u64 = 0x8000_0100;
        const IMM: u64 = 0x100;
        const TARGET: u64 = 0x8000_0200;
        const LINK: u64 = 0x8000_0104;

        let mut w = cycle_witness();
        w[V_LEFT_INSTRUCTION_INPUT] = Fr::from_u64(UNEXPANDED_PC);
        w[V_RIGHT_INSTRUCTION_INPUT] = Fr::from_u64(IMM);
        w[V_PRODUCT] = Fr::from_u128((UNEXPANDED_PC as u128) * (IMM as u128));
        w[V_PC] = Fr::from_u64(12);
        w[V_NEXT_PC] = Fr::from_u64(37);
        w[V_UNEXPANDED_PC] = Fr::from_u64(UNEXPANDED_PC);
        w[V_NEXT_UNEXPANDED_PC] = Fr::from_u64(TARGET);
        w[V_IMM] = Fr::from_u64(IMM);
        w[V_RD_WRITE_VALUE] = Fr::from_u64(LINK);
        w[V_RIGHT_LOOKUP_OPERAND] = Fr::from_u64(TARGET);
        w[V_LOOKUP_OUTPUT] = Fr::from_u64(TARGET);
        w[V_FLAG_ADD_OPERANDS] = Fr::from_u64(1);
        w[V_FLAG_JUMP] = Fr::from_u64(1);
        w
    }

    /// `MUL x5, x1, x2` with rs1 = 2^63 + 1, rs2 = 3.
    ///
    /// Circuit flags [MultiplyOperands, WriteLookupOutputToRD] (m/mul.rs).
    /// The full product (2^63 + 1)·3 = 2^64 + 2^63 + 3 exceeds 64 bits: the
    /// right lookup operand and the committed Product carry the full value,
    /// while rd receives the RV64 truncated low 64 bits 2^63 + 3.
    fn mul_witness() -> Vec<Fr> {
        const RS1: u64 = 0x8000_0000_0000_0001;
        const RS2: u64 = 3;
        const FULL_PRODUCT: u128 = 0x1_8000_0000_0000_0003;
        const TRUNCATED: u64 = 0x8000_0000_0000_0003;
        const UNEXPANDED_PC: u64 = 0x8000_0060;

        let mut w = cycle_witness();
        w[V_LEFT_INSTRUCTION_INPUT] = Fr::from_u64(RS1);
        w[V_RIGHT_INSTRUCTION_INPUT] = Fr::from_u64(RS2);
        w[V_PRODUCT] = Fr::from_u128(FULL_PRODUCT);
        w[V_PC] = Fr::from_u64(22);
        w[V_NEXT_PC] = Fr::from_u64(23);
        w[V_UNEXPANDED_PC] = Fr::from_u64(UNEXPANDED_PC);
        w[V_NEXT_UNEXPANDED_PC] = Fr::from_u64(UNEXPANDED_PC + 4);
        w[V_RS1_VALUE] = Fr::from_u64(RS1);
        w[V_RS2_VALUE] = Fr::from_u64(RS2);
        w[V_RD_WRITE_VALUE] = Fr::from_u64(TRUNCATED);
        w[V_RIGHT_LOOKUP_OPERAND] = Fr::from_u128(FULL_PRODUCT);
        w[V_LOOKUP_OUTPUT] = Fr::from_u64(TRUNCATED);
        w[V_FLAG_MULTIPLY_OPERANDS] = Fr::from_u64(1);
        w[V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD] = Fr::from_u64(1);
        w
    }

    /// A mid-sequence inline step of a virtual expansion (e.g. Virtual
    /// MOVSIGN inside a DIV expansion) with rs1 = 2^63 (negative), so the
    /// sign-mask lookup output is all-ones.
    ///
    /// Per the `jolt_instruction!` macro, `virtual_sequence_remaining =
    /// Some(k > 0)` sets [VirtualInstruction, DoNotUpdateUnexpandedPC] and
    /// not IsLastInSequence, so the expanded PC advances by 1 while the
    /// unexpanded PC stays put. The next row is the following (non-first)
    /// step of the same sequence.
    fn virtual_inline_step_witness() -> Vec<Fr> {
        const RS1: u64 = 0x8000_0000_0000_0000;
        const SIGN_MASK: u64 = u64::MAX;
        const UNEXPANDED_PC: u64 = 0x8000_0050;

        let mut w = cycle_witness();
        w[V_LEFT_INSTRUCTION_INPUT] = Fr::from_u64(RS1);
        w[V_PC] = Fr::from_u64(20);
        w[V_NEXT_PC] = Fr::from_u64(21);
        w[V_UNEXPANDED_PC] = Fr::from_u64(UNEXPANDED_PC);
        w[V_NEXT_UNEXPANDED_PC] = Fr::from_u64(UNEXPANDED_PC);
        w[V_RS1_VALUE] = Fr::from_u64(RS1);
        w[V_RD_WRITE_VALUE] = Fr::from_u64(SIGN_MASK);
        w[V_LEFT_LOOKUP_OPERAND] = Fr::from_u64(RS1);
        w[V_LOOKUP_OUTPUT] = Fr::from_u64(SIGN_MASK);
        w[V_NEXT_IS_VIRTUAL] = Fr::from_u64(1);
        w[V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD] = Fr::from_u64(1);
        w[V_FLAG_VIRTUAL_INSTRUCTION] = Fr::from_u64(1);
        w[V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] = Fr::from_u64(1);
        w
    }

    #[test]
    fn add_with_carry_execution_witness_satisfies_constraints() {
        assert_eq!(check(&add_witness(), false), Ok(()));
    }

    #[test]
    fn add_rejects_wrong_rd_write_value() {
        let w = with_cell(&add_witness(), V_RD_WRITE_VALUE, Fr::from_u64(4));
        assert_eq!(
            check(&w, false),
            Err(RD_WRITE_EQ_LOOKUP_IF_WRITE_LOOKUP_TO_RD)
        );
    }

    #[test]
    fn add_rejects_wrapped_right_lookup_operand() {
        let w = with_cell(&add_witness(), V_RIGHT_LOOKUP_OPERAND, Fr::from_u64(3));
        assert_eq!(check(&w, false), Err(RIGHT_LOOKUP_ADD));
    }

    #[test]
    fn compressed_add_execution_witness_satisfies_constraints() {
        // C.ADD occupies 2 bytes, so the unexpanded PC advances by 2.
        let mut w = add_witness();
        w[V_FLAG_IS_COMPRESSED] = Fr::from_u64(1);
        w[V_NEXT_UNEXPANDED_PC] = Fr::from_u64(0x8000_0010 + 2);
        assert_eq!(check(&w, false), Ok(()));
    }

    #[test]
    fn compressed_add_rejects_full_width_pc_increment() {
        let mut w = add_witness();
        w[V_FLAG_IS_COMPRESSED] = Fr::from_u64(1);
        // +4 is only correct for uncompressed instructions.
        assert_eq!(check(&w, false), Err(NEXT_UNEXP_PC_UPDATE_OTHERWISE));
    }

    #[test]
    fn sltu_write_back_execution_witness_satisfies_constraints() {
        assert_eq!(check(&sltu_witness(), false), Ok(()));
    }

    #[test]
    fn sltu_rejects_left_lookup_operand_mismatch() {
        let w = with_cell(&sltu_witness(), V_LEFT_LOOKUP_OPERAND, Fr::from_u64(8));
        assert_eq!(check(&w, false), Err(LEFT_LOOKUP_EQ_LEFT_INPUT_OTHERWISE));
    }

    #[test]
    fn ld_execution_witness_satisfies_constraints() {
        assert_eq!(check(&ld_witness(), false), Ok(()));
    }

    #[test]
    fn ld_rejects_wrong_rd_write_value() {
        let w = with_cell(
            &ld_witness(),
            V_RD_WRITE_VALUE,
            Fr::from_u64(0xDEAD_BEEF_CAFE_F00D + 1),
        );
        assert_eq!(check(&w, false), Err(RAM_READ_EQ_RD_WRITE_IF_LOAD));
    }

    #[test]
    fn ld_rejects_ram_write_differing_from_read() {
        let w = with_cell(&ld_witness(), V_RAM_WRITE_VALUE, Fr::from_u64(0));
        assert_eq!(check(&w, false), Err(RAM_READ_EQ_RAM_WRITE_IF_LOAD));
    }

    #[test]
    fn ld_rejects_wrong_ram_address() {
        let w = with_cell(&ld_witness(), V_RAM_ADDRESS, Fr::from_u64(0x8000_1000));
        assert_eq!(
            check(&w, false),
            Err(RAM_ADDR_EQ_RS1_PLUS_IMM_MINUS_LOOKUP_IF_LOAD_STORE)
        );
    }

    #[test]
    fn ld_rejects_right_lookup_not_carried_from_rd_write_input() {
        let w = with_cell(&ld_witness(), V_RIGHT_LOOKUP_OPERAND, Fr::from_u64(0));
        assert_eq!(check(&w, false), Err(RIGHT_LOOKUP_EQ_RIGHT_INPUT_OTHERWISE));
    }

    #[test]
    fn sd_negative_offset_execution_witness_satisfies_constraints() {
        assert_eq!(check(&sd_witness(), false), Ok(()));
    }

    #[test]
    fn sd_rejects_dropped_store_value() {
        let w = with_cell(
            &sd_witness(),
            V_RAM_WRITE_VALUE,
            Fr::from_u64(0x0F0F_0F0F_0F0F_0F0F),
        );
        assert_eq!(check(&w, false), Err(RS2_EQ_RAM_WRITE_IF_STORE));
    }

    #[test]
    fn sd_rejects_sign_error_in_address() {
        let w = with_cell(&sd_witness(), V_RAM_ADDRESS, Fr::from_u64(0x8000_2008));
        assert_eq!(
            check(&w, false),
            Err(RAM_ADDR_EQ_RS1_PLUS_IMM_MINUS_LOOKUP_IF_LOAD_STORE)
        );
    }

    #[test]
    fn load_h_execution_witness_satisfies_constraints() {
        assert_eq!(check(&load_h_witness(), false), Ok(()));
    }

    #[test]
    fn load_h_rejects_unfloored_ram_address() {
        let w = with_cell(&load_h_witness(), V_RAM_ADDRESS, Fr::from_u64(0x8000_100A));
        assert_eq!(
            check(&w, false),
            Err(RAM_ADDR_EQ_RS1_PLUS_IMM_MINUS_LOOKUP_IF_LOAD_STORE)
        );
    }

    #[test]
    fn load_h_rejects_dropped_offset() {
        let w = with_cell(&load_h_witness(), V_LOOKUP_OUTPUT, Fr::from_u64(0));
        assert_eq!(
            check(&w, false),
            Err(RAM_ADDR_EQ_RS1_PLUS_IMM_MINUS_LOOKUP_IF_LOAD_STORE)
        );
    }

    #[test]
    fn beq_taken_execution_witness_satisfies_constraints() {
        assert_eq!(check(&beq_taken_witness(), false), Ok(()));
    }

    #[test]
    fn beq_taken_rejects_fall_through_next_pc() {
        let w = with_cell(
            &beq_taken_witness(),
            V_NEXT_UNEXPANDED_PC,
            Fr::from_u64(0x8000_0044),
        );
        assert_eq!(
            check(&w, false),
            Err(NEXT_UNEXP_PC_EQ_PC_PLUS_IMM_IF_SHOULD_BRANCH)
        );
    }

    #[test]
    fn beq_taken_rejects_denied_should_branch() {
        // Zeroing ShouldBranch shifts the PC obligation to the fall-through
        // constraint, which the branch-target next PC then violates.
        let w = with_cell(&beq_taken_witness(), V_SHOULD_BRANCH, Fr::from_u64(0));
        assert_eq!(check(&w, false), Err(NEXT_UNEXP_PC_UPDATE_OTHERWISE));
    }

    #[test]
    fn beq_not_taken_execution_witness_satisfies_constraints() {
        assert_eq!(check(&beq_not_taken_witness(), false), Ok(()));
    }

    #[test]
    fn beq_not_taken_rejects_forced_branch() {
        // Forcing ShouldBranch = 1 with a consistent branch-target next PC
        // still fails: the lookup output is 0, so 0·Branch ≠ ShouldBranch.
        let mut w = beq_not_taken_witness();
        w[V_SHOULD_BRANCH] = Fr::from_u64(1);
        w[V_NEXT_UNEXPANDED_PC] = Fr::from_u64(0x8000_0030);
        assert_eq!(check(&w, false), Err(SHOULD_BRANCH_EQ_LOOKUP_TIMES_BRANCH));
    }

    #[test]
    fn jal_execution_witness_satisfies_constraints() {
        assert_eq!(check(&jal_witness(), false), Ok(()));
    }

    #[test]
    fn jal_rejects_wrong_link_value() {
        let w = with_cell(&jal_witness(), V_RD_WRITE_VALUE, Fr::from_u64(0x8000_0100));
        assert_eq!(check(&w, false), Err(RD_WRITE_EQ_PC_PLUS_CONST_IF_JUMP));
    }

    #[test]
    fn jal_rejects_fall_through_next_pc() {
        let w = with_cell(
            &jal_witness(),
            V_NEXT_UNEXPANDED_PC,
            Fr::from_u64(0x8000_0104),
        );
        assert_eq!(check(&w, false), Err(NEXT_UNEXP_PC_EQ_LOOKUP_IF_JUMP));
    }

    /// A jump into padding (`NextIsNoop = 1`) is no end: only a halt may precede padding.
    #[test]
    fn jal_rejects_a_padding_successor() {
        let w = with_cell(&jal_witness(), V_NEXT_IS_NOOP, Fr::from_u64(1));
        for claimed_panic in [false, true] {
            assert_eq!(check(&w, claimed_panic), Err(HALT_LANE));
        }
    }

    #[test]
    fn mul_high_bit_execution_witness_satisfies_constraints() {
        assert_eq!(check(&mul_witness(), false), Ok(()));
    }

    #[test]
    fn mul_rejects_truncated_right_lookup_operand() {
        // The lookup operand must carry the full 65-bit product, not the
        // RV64-truncated low 64 bits.
        let w = with_cell(
            &mul_witness(),
            V_RIGHT_LOOKUP_OPERAND,
            Fr::from_u64(0x8000_0000_0000_0003),
        );
        assert_eq!(check(&w, false), Err(RIGHT_LOOKUP_EQ_PRODUCT_IF_MUL));
    }

    #[test]
    fn mul_rejects_truncated_product() {
        // Truncating both Product and the lookup operand keeps them mutually
        // consistent but breaks Product = Left·Right over the field.
        let mut w = mul_witness();
        w[V_PRODUCT] = Fr::from_u64(0x8000_0000_0000_0003);
        w[V_RIGHT_LOOKUP_OPERAND] = Fr::from_u64(0x8000_0000_0000_0003);
        assert_eq!(check(&w, false), Err(PRODUCT_EQ_LEFT_TIMES_RIGHT));
    }

    #[test]
    fn virtual_inline_step_execution_witness_satisfies_constraints() {
        assert_eq!(check(&virtual_inline_step_witness(), false), Ok(()));
    }

    #[test]
    fn virtual_inline_step_rejects_skipped_expanded_pc() {
        let w = with_cell(&virtual_inline_step_witness(), V_NEXT_PC, Fr::from_u64(22));
        assert_eq!(check(&w, false), Err(NEXT_PC_EQ_PC_PLUS_ONE_IF_INLINE));
    }

    #[test]
    fn virtual_inline_step_rejects_entering_sequence_mid_way() {
        // Pretending the expansion ended (normal +4 advance) while the next
        // row is a non-first virtual step means control would enter the
        // middle of a virtual sequence.
        let mut w = virtual_inline_step_witness();
        w[V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] = Fr::from_u64(0);
        w[V_NEXT_UNEXPANDED_PC] = Fr::from_u64(0x8000_0050 + 4);
        assert_eq!(check(&w, false), Err(MUST_START_SEQUENCE_FROM_BEGINNING));
    }

    /// Places `witness` on the last row of a trace that fills its padded
    /// domain. EqPlusOne does not wrap around, so every shifted `Next*`
    /// column reads 0 there and `NextIsNoop` reads 1, as if a no-op followed.
    fn at_last_row(witness: &[Fr]) -> Vec<Fr> {
        let mut w = witness.to_vec();
        w[V_NEXT_PC] = Fr::zero();
        w[V_NEXT_UNEXPANDED_PC] = Fr::zero();
        w[V_NEXT_IS_VIRTUAL] = Fr::zero();
        w[V_NEXT_IS_FIRST_IN_SEQUENCE] = Fr::zero();
        w[V_NEXT_IS_NOOP] = Fr::from_u64(1);
        w
    }

    /// A `j .` self-loop: `JAL x0, 0` at 0x8000_0100, whose target is its own
    /// address.
    fn jal_self_loop_witness() -> Vec<Fr> {
        const UNEXPANDED_PC: u64 = 0x8000_0100;

        let mut w = jal_witness();
        w[V_RIGHT_INSTRUCTION_INPUT] = Fr::zero();
        w[V_PRODUCT] = Fr::zero();
        w[V_NEXT_PC] = w[V_PC];
        w[V_NEXT_UNEXPANDED_PC] = Fr::from_u64(UNEXPANDED_PC);
        w[V_IMM] = Fr::zero();
        w[V_RD_WRITE_VALUE] = Fr::from_u64(UNEXPANDED_PC + 4);
        w[V_RIGHT_LOOKUP_OPERAND] = Fr::from_u64(UNEXPANDED_PC);
        w[V_LOOKUP_OUTPUT] = Fr::from_u64(UNEXPANDED_PC);
        w
    }

    /// `j .` can loop but cannot end a trace: its target is its own address, not
    /// the padding row's 0.
    #[test]
    fn last_row_rejects_final_self_jump() {
        assert_eq!(check(&jal_self_loop_witness(), false), Ok(()));
        assert_eq!(
            check(&at_last_row(&jal_self_loop_witness()), false),
            Err(NEXT_UNEXP_PC_EQ_LOOKUP_IF_JUMP)
        );
    }

    /// `JALR x1, 0(x6)` with `x6 = 0` at 0x8000_0300: a jump whose target is 0.
    ///
    /// Circuit flags [AddOperands, Jump] (i/jalr.rs); the lookup output, the
    /// target, is `rs1 + imm` with bit 0 cleared.
    fn jalr_to_zero_witness() -> Vec<Fr> {
        const UNEXPANDED_PC: u64 = 0x8000_0300;

        let mut w = cycle_witness();
        w[V_PC] = Fr::from_u64(30);
        w[V_UNEXPANDED_PC] = Fr::from_u64(UNEXPANDED_PC);
        w[V_RD_WRITE_VALUE] = Fr::from_u64(UNEXPANDED_PC + 4);
        w[V_FLAG_ADD_OPERANDS] = Fr::from_u64(1);
        w[V_FLAG_JUMP] = Fr::from_u64(1);
        w
    }

    /// A jump to address 0 meets row 14 before padding, whose next address is 0,
    /// but is not a halt, so the halt lane rejects it.
    #[test]
    fn last_row_rejects_a_jump_to_zero() {
        let w = at_last_row(&jalr_to_zero_witness());
        for claimed_panic in [false, true] {
            assert_eq!(check(&w, claimed_panic), Err(HALT_LANE));
        }
    }

    #[test]
    fn last_row_rejects_sequential_instruction() {
        // A non-jump row must advance UnexpandedPC to a nonzero address,
        // but the shifted NextUnexpandedPC is 0.
        assert_eq!(
            check(&at_last_row(&add_witness()), false),
            Err(NEXT_UNEXP_PC_UPDATE_OTHERWISE)
        );
        assert_eq!(
            check(&at_last_row(&ld_witness()), false),
            Err(NEXT_UNEXP_PC_UPDATE_OTHERWISE)
        );
    }

    #[test]
    fn last_row_rejects_branch() {
        assert_eq!(
            check(&at_last_row(&beq_taken_witness()), false),
            Err(NEXT_UNEXP_PC_EQ_PC_PLUS_IMM_IF_SHOULD_BRANCH)
        );
        assert_eq!(
            check(&at_last_row(&beq_not_taken_witness()), false),
            Err(NEXT_UNEXP_PC_UPDATE_OTHERWISE)
        );
    }

    #[test]
    fn last_row_rejects_unfinished_virtual_sequence() {
        let w = at_last_row(&virtual_inline_step_witness());
        // Row 16 is the first to fail: with DoNotUpdateUnexpandedPC it still
        // requires NextUnexpandedPC = UnexpandedPC, which is nonzero.
        assert_eq!(check(&w, false), Err(NEXT_UNEXP_PC_UPDATE_OTHERWISE));

        // Row 17 rejects the row on its own as well: zeroing every term of
        // row 16 leaves NextPC = 0 ≠ PC + 1.
        let mut w = w;
        w[V_UNEXPANDED_PC] = Fr::zero();
        assert_eq!(check(&w, false), Err(NEXT_PC_EQ_PC_PLUS_ONE_IF_INLINE));
    }

    /// A halt row as tracers emit it: `jolt.success` (`vsr = None`) or EBREAK's one-row `Panic`
    /// lowering (`vsr = Some(0)`, first, optionally compressed). Its circuit flags come from the
    /// production flag table; it has no lookup table (output 0), no instruction flags (inputs 0),
    /// and writes its link to a temporary.
    fn halt_witness(
        instruction_kind: JoltInstructionKind,
        virtual_sequence_remaining: Option<u16>,
        is_compressed: bool,
    ) -> Vec<Fr> {
        const UNEXPANDED_PC: u64 = 0x8000_0200;

        let row = JoltInstructionRow {
            instruction_kind,
            address: UNEXPANDED_PC as usize,
            operands: NormalizedOperands {
                rd: Some(40),
                rs1: Some(0),
                rs2: None,
                imm: 0,
            },
            virtual_sequence_remaining,
            is_first_in_sequence: virtual_sequence_remaining.is_some(),
            is_compressed,
        };
        let flags = JoltInstruction::from(row).circuit_flags();
        let mut w = cycle_witness();
        for flag in CIRCUIT_FLAGS {
            if flags[flag] {
                w[flag_column(flag)] = Fr::from_u64(1);
            }
        }
        w[V_PC] = Fr::from_u64(12);
        w[V_UNEXPANDED_PC] = Fr::from_u64(UNEXPANDED_PC);
        w[V_RD_WRITE_VALUE] = Fr::from_u64(UNEXPANDED_PC + if is_compressed { 2 } else { 4 });
        w
    }

    const HALTS: [(JoltInstructionKind, Option<u16>, bool); 3] = [
        (JoltInstructionKind::JoltSuccess, None, false),
        (JoltInstructionKind::JoltPanic, Some(0), false),
        (JoltInstructionKind::JoltPanic, Some(0), true),
    ];

    /// A halt ends the trace under its own claim: `jolt.success` with
    /// `panic = false`, EBREAK's `Panic` row with `panic = true`.
    #[test]
    fn last_row_accepts_final_halt_under_its_claim() {
        for (kind, virtual_sequence_remaining, is_compressed) in HALTS {
            let w = halt_witness(kind, virtual_sequence_remaining, is_compressed);
            let claimed_panic = kind == JoltInstructionKind::JoltPanic;
            assert_eq!(check(&at_last_row(&w), claimed_panic), Ok(()), "{kind:?}");
        }
    }

    #[test]
    fn last_row_rejects_final_halt_under_the_opposite_claim() {
        for (kind, virtual_sequence_remaining, is_compressed) in HALTS {
            let w = halt_witness(kind, virtual_sequence_remaining, is_compressed);
            let claimed_panic = kind != JoltInstructionKind::JoltPanic;
            assert_eq!(
                check(&at_last_row(&w), claimed_panic),
                Err(HALT_LANE),
                "{kind:?}"
            );
        }
    }

    /// A halt's target is 0, so a real successor fails row 14.
    #[test]
    fn halt_rejects_a_real_successor() {
        for (kind, virtual_sequence_remaining, is_compressed) in HALTS {
            let mut w = halt_witness(kind, virtual_sequence_remaining, is_compressed);
            w[V_NEXT_PC] = Fr::from_u64(13);
            w[V_NEXT_UNEXPANDED_PC] = Fr::from_u64(0x8000_0204);
            let claimed_panic = kind == JoltInstructionKind::JoltPanic;
            assert_eq!(
                check(&w, claimed_panic),
                Err(NEXT_UNEXP_PC_EQ_LOOKUP_IF_JUMP),
                "{kind:?}"
            );
        }
    }

    /// Row 21 over every boolean `(IsNoop, Halt, HaltPanic, NextIsNoop)` and both
    /// claims, on a row whose equality rows hold whatever those cells are. A real
    /// row keeps the run real, padding stays padding, and only a halt of the
    /// claimed kind starts it. The lane admits one more assignment, which no
    /// instruction kind carries: a no-op that is a success halt, with a real
    /// successor, under `panic = true`.
    #[test]
    fn halt_lane_truth_table() {
        for claimed_panic in [false, true] {
            for bits in 0..16u8 {
                let [is_noop, halt, halt_panic, next_is_noop] =
                    [0, 1, 2, 3].map(|bit| (bits >> bit) & 1 == 1);
                let mut w = cycle_witness();
                w[V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] = Fr::from_u64(1);
                w[V_IS_NOOP] = Fr::from_bool(is_noop);
                w[V_FLAG_HALT] = Fr::from_bool(halt);
                w[V_FLAG_HALT_PANIC] = Fr::from_bool(halt_panic);
                w[V_NEXT_IS_NOOP] = Fr::from_bool(next_is_noop);
                let admitted = match (is_noop, halt, halt_panic, next_is_noop) {
                    (false, false, false, false) | (true, false, false, true) => true,
                    (false, true, kind, true) => kind == claimed_panic,
                    (true, true, false, false) => claimed_panic,
                    _ => false,
                };
                assert_eq!(
                    check(&w, claimed_panic),
                    if admitted { Ok(()) } else { Err(HALT_LANE) },
                    "IsNoop {is_noop}, Halt {halt}, HaltPanic {halt_panic}, \
                     NextIsNoop {next_is_noop}, panic {claimed_panic}"
                );
            }
        }
    }
}
