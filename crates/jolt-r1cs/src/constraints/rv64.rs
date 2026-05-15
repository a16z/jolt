//! Jolt RV64 R1CS variable layout.
//!
//! Defines the per-cycle witness variable indices and constraint counts
//! for the Jolt RV64IMAC R1CS constraint system, including the BN254 Fr
//! coprocessor row block.
//!
//! # Variable layout
//!
//! Each cycle has [`NUM_VARS_PER_CYCLE`] witness variables:
//!
//! | Range | Description |
//! |-------|-------------|
//! | `[0]` | Constant 1 |
//! | `[1..=35]` | RV64 inputs (registers, flags, PC, lookups) |
//! | `[36..=44]` | BN254 Fr coprocessor flag slots (9) |
//! | `[45..=47]` | BN254 Fr virtual operand slots (rs1/rs2/rd) |
//! | `[48..=49]` | Product factor variables (`Branch`, `NextIsNoop`) |
//!
//! # Constraint forms
//!
//! - **Eq-conditional** (rows 0–31): `guard · (left − right) = 0`
//! - **Product** (rows 32–35): `left · right = output`
//!
//! The Fr eq-conditional rows (19–31) are gated by the per-op Fr flags
//! (V_FLAG_IS_FIELD_*). Until Phase 3 wires the FieldReg witness, those
//! flag slots stay zero across every cycle and the new rows trivially
//! satisfy.

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

// --- BN254 Fr coprocessor witness slots ---------------------------------
// Flag slots (9) — one per BN254 Fr instruction kind. Mirrors `CircuitFlags`
// indices ≥ 14 (declared in `crates/jolt-riscv/src/flags.rs`).
pub const V_FLAG_IS_FIELD_MUL: usize = 36;
pub const V_FLAG_IS_FIELD_ADD: usize = 37;
pub const V_FLAG_IS_FIELD_SUB: usize = 38;
pub const V_FLAG_IS_FIELD_INV: usize = 39;
pub const V_FLAG_IS_FIELD_ASSERT_EQ: usize = 40;
pub const V_FLAG_IS_FIELD_MOV: usize = 41;
pub const V_FLAG_IS_FIELD_SLL64: usize = 42;
pub const V_FLAG_IS_FIELD_SLL128: usize = 43;
pub const V_FLAG_IS_FIELD_SLL192: usize = 44;

// Virtual Fr operand slots — Fr-valued witnesses bound to FieldReg sumcheck
// claims in Phase 4. Zero on non-FR cycles.
pub const V_FIELD_RS1_VALUE: usize = 45;
pub const V_FIELD_RS2_VALUE: usize = 46;
pub const V_FIELD_RD_WRITE_VALUE: usize = 47;

pub const V_BRANCH: usize = 48;
pub const V_NEXT_IS_NOOP: usize = 49;

pub const NUM_R1CS_INPUTS: usize = 47;
pub const NUM_PRODUCT_FACTORS: usize = 2;
pub const NUM_VARS_PER_CYCLE: usize = 1 + NUM_R1CS_INPUTS + NUM_PRODUCT_FACTORS; // 50
pub const NUM_EQ_CONSTRAINTS: usize = 32;
pub const NUM_PRODUCT_CONSTRAINTS: usize = 3;
pub const NUM_CONSTRAINTS_PER_CYCLE: usize = NUM_EQ_CONSTRAINTS + NUM_PRODUCT_CONSTRAINTS; // 35

/// Two's complement bias for subtraction: 2^64.
const TWOS_COMPLEMENT_BIAS: i128 = 0x1_0000_0000_0000_0000;

/// 2^64 as i128 — used by Fr SLL64 row coefficient.
const TWO_POW_64: i128 = 1i128 << 64;

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

/// Mixed-magnitude row coefficient. `Small` covers anything that fits in
/// i128 (handled identically to [`row_wide`]); `Pow2 { exp, sign }` denotes
/// `sign * 2^exp` for exponents that overflow i128 — currently 2^128 and
/// 2^192 used by the Fr integer→field on-ramp rows.
#[derive(Clone, Copy)]
pub enum Coef {
    Small(i128),
    Pow2 { exp: u32, sign: i8 },
}

impl From<i128> for Coef {
    fn from(v: i128) -> Self {
        Coef::Small(v)
    }
}

fn pow2_to_field<F: Field>(exp: u32) -> F {
    // 2^exp = (2^64)^chunks · 2^remainder. Multiplies are cheap and only
    // happen at constraint-matrix build time (not per-cycle).
    let chunks = exp / 64;
    let remainder = exp % 64;
    let mut result = F::from_u64(1);
    if chunks > 0 {
        // 2^64 doesn't fit in u64; build it as 2 * 2^63 via F::from_u128.
        let two_pow_64 = F::from_u128(1u128 << 63) * F::from_u64(2);
        for _ in 0..chunks {
            result *= two_pow_64;
        }
    }
    if remainder > 0 {
        result *= F::from_u64(1u64 << remainder);
    }
    result
}

/// Sparse row with mixed-magnitude coefficients. Pow2 entries with
/// `exp >= 128` go through field multiplication of 2^64 chunks since
/// neither i128 nor u128 can hold them.
fn row_bigcoeff<F: Field>(entries: &[(usize, Coef)]) -> SparseRow<F> {
    entries
        .iter()
        .filter_map(|&(idx, c)| match c {
            Coef::Small(0) => None,
            Coef::Small(v) => Some((idx, F::from_i128(v))),
            Coef::Pow2 { exp, sign } => {
                if sign == 0 {
                    None
                } else {
                    let mut value = pow2_to_field::<F>(exp);
                    if sign < 0 {
                        value = -value;
                    }
                    Some((idx, value))
                }
            }
        })
        .collect()
}

/// Build the Jolt RV64 R1CS constraint matrices.
///
/// Returns 22 constraints over 38 variables per cycle:
/// - 19 equality-conditional: `guard · (left − right) = 0` → A=guard, B=left−right, C=0
/// - 3 product: `left · right = output` → A=left, B=right, C=output
///
/// Variable layout matches the constants in this module (V_CONST=0, inputs at 1–35,
/// product factors at 36–37).
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

    // --- BN254 Fr coprocessor eq-conditional rows (19-31) ---------------
    // Until Phase 3 wires FieldReg witness population, all Fr flag slots
    // (V_FLAG_IS_FIELD_*) stay zero, so every guard collapses to 0 and these
    // rows trivially satisfy. Phase 4 will tighten the FMUL/FINV placeholder
    // rows once the FR Twist sumchecks bind the operand witnesses.

    // 19: IsFieldAdd · (V_FIELD_RD − V_FIELD_RS1 − V_FIELD_RS2) = 0
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_ADD, 1)]));
    b_rows.push(row::<F>(&[
        (V_FIELD_RD_WRITE_VALUE, 1),
        (V_FIELD_RS1_VALUE, -1),
        (V_FIELD_RS2_VALUE, -1),
    ]));
    c_rows.push(empty());

    // 20: IsFieldSub · (V_FIELD_RD − V_FIELD_RS1 + V_FIELD_RS2) = 0
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_SUB, 1)]));
    b_rows.push(row::<F>(&[
        (V_FIELD_RD_WRITE_VALUE, 1),
        (V_FIELD_RS1_VALUE, -1),
        (V_FIELD_RS2_VALUE, 1),
    ]));
    c_rows.push(empty());

    // 21: IsFieldAssertEq · (V_FIELD_RS1 − V_FIELD_RS2) = 0
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_ASSERT_EQ, 1)]));
    b_rows.push(row::<F>(&[
        (V_FIELD_RS1_VALUE, 1),
        (V_FIELD_RS2_VALUE, -1),
    ]));
    c_rows.push(empty());

    // 22: IsFieldMov · (V_FIELD_RD − V_RS1_VALUE) = 0
    //     Low-limb load: V_FIELD_RD = XReg[rs1] as Fr (no shift).
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_MOV, 1)]));
    b_rows.push(row::<F>(&[
        (V_FIELD_RD_WRITE_VALUE, 1),
        (V_RS1_VALUE, -1),
    ]));
    c_rows.push(empty());

    // 23: IsFieldSLL64 · (V_FIELD_RD − V_RS1_VALUE · 2^64) = 0
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_SLL64, 1)]));
    b_rows.push(row_wide::<F>(&[
        (V_FIELD_RD_WRITE_VALUE, 1),
        (V_RS1_VALUE, -TWO_POW_64),
    ]));
    c_rows.push(empty());

    // 24: IsFieldSLL128 · (V_FIELD_RD − V_RS1_VALUE · 2^128) = 0
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_SLL128, 1)]));
    b_rows.push(row_bigcoeff::<F>(&[
        (V_FIELD_RD_WRITE_VALUE, Coef::Small(1)),
        (
            V_RS1_VALUE,
            Coef::Pow2 {
                exp: 128,
                sign: -1,
            },
        ),
    ]));
    c_rows.push(empty());

    // 25: IsFieldSLL192 · (V_FIELD_RD − V_RS1_VALUE · 2^192) = 0
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_SLL192, 1)]));
    b_rows.push(row_bigcoeff::<F>(&[
        (V_FIELD_RD_WRITE_VALUE, Coef::Small(1)),
        (
            V_RS1_VALUE,
            Coef::Pow2 {
                exp: 192,
                sign: -1,
            },
        ),
    ]));
    c_rows.push(empty());

    // 26: IsFieldMul · (V_FIELD_RD − V_FIELD_RS1 − V_FIELD_RS2) = 0
    //     PLACEHOLDER — only enforces "RD depends on operands", not the
    //     actual product. FR Twist (Phase 4) binds RD to RS1·RS2 via the
    //     FieldRegRW sumcheck. Trivially satisfied while flag is inert.
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_MUL, 1)]));
    b_rows.push(row::<F>(&[
        (V_FIELD_RD_WRITE_VALUE, 1),
        (V_FIELD_RS1_VALUE, -1),
        (V_FIELD_RS2_VALUE, -1),
    ]));
    c_rows.push(empty());

    // 27: IsFieldInv · (V_FIELD_RD − V_FIELD_RS1) = 0
    //     PLACEHOLDER — FR Twist (Phase 4) proves the actual inverse with
    //     the 0 → 0 carve-out. Trivially satisfied while flag is inert.
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_INV, 1)]));
    b_rows.push(row::<F>(&[
        (V_FIELD_RD_WRITE_VALUE, 1),
        (V_FIELD_RS1_VALUE, -1),
    ]));
    c_rows.push(empty());

    // 28: (1 − ΣFR_flags) · V_FIELD_RS1 = 0
    //     V_FIELD_RS1 is zero on every non-Fr cycle. The guard sums all
    //     9 Fr flags; an active Fr cycle has exactly one flag set, so the
    //     guard is 0 and RS1 is unconstrained.
    a_rows.push(row::<F>(&[
        (V_CONST, 1),
        (V_FLAG_IS_FIELD_MUL, -1),
        (V_FLAG_IS_FIELD_ADD, -1),
        (V_FLAG_IS_FIELD_SUB, -1),
        (V_FLAG_IS_FIELD_INV, -1),
        (V_FLAG_IS_FIELD_ASSERT_EQ, -1),
        (V_FLAG_IS_FIELD_MOV, -1),
        (V_FLAG_IS_FIELD_SLL64, -1),
        (V_FLAG_IS_FIELD_SLL128, -1),
        (V_FLAG_IS_FIELD_SLL192, -1),
    ]));
    b_rows.push(row::<F>(&[(V_FIELD_RS1_VALUE, 1)]));
    c_rows.push(empty());

    // 29: (1 − Σtwo_input_flags) · V_FIELD_RS2 = 0
    //     RS2 is zero unless the cycle is Fmul / Fadd / Fsub / FassertEq.
    a_rows.push(row::<F>(&[
        (V_CONST, 1),
        (V_FLAG_IS_FIELD_MUL, -1),
        (V_FLAG_IS_FIELD_ADD, -1),
        (V_FLAG_IS_FIELD_SUB, -1),
        (V_FLAG_IS_FIELD_ASSERT_EQ, -1),
    ]));
    b_rows.push(row::<F>(&[(V_FIELD_RS2_VALUE, 1)]));
    c_rows.push(empty());

    // 30: (1 − Σwrite_flags) · V_FIELD_RD = 0
    //     RD is zero unless the cycle writes a field register (every Fr
    //     op except FieldAssertEq).
    a_rows.push(row::<F>(&[
        (V_CONST, 1),
        (V_FLAG_IS_FIELD_MUL, -1),
        (V_FLAG_IS_FIELD_ADD, -1),
        (V_FLAG_IS_FIELD_SUB, -1),
        (V_FLAG_IS_FIELD_INV, -1),
        (V_FLAG_IS_FIELD_MOV, -1),
        (V_FLAG_IS_FIELD_SLL64, -1),
        (V_FLAG_IS_FIELD_SLL128, -1),
        (V_FLAG_IS_FIELD_SLL192, -1),
    ]));
    b_rows.push(row::<F>(&[(V_FIELD_RD_WRITE_VALUE, 1)]));
    c_rows.push(empty());

    // 31: IsFieldMov · V_FIELD_RS2 = 0
    //     FieldMov + FieldSLL* are 1-input ops over the integer register
    //     file; they MUST NOT read a field RS2. Pinning V_FIELD_RS2 to 0
    //     on those rows prevents the prover from smuggling extra data.
    //     One representative row (FieldMov) is enforced here; the SLL64/
    //     128/192 cases are subsumed by row 29 (their flags are excluded
    //     from the two-input mask).
    a_rows.push(row::<F>(&[(V_FLAG_IS_FIELD_MOV, 1)]));
    b_rows.push(row::<F>(&[(V_FIELD_RS2_VALUE, 1)]));
    c_rows.push(empty());

    // --- Product constraints (32-35) ------------------------------------
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

    #[test]
    fn shape_invariants() {
        assert_eq!(NUM_R1CS_INPUTS, 47);
        assert_eq!(NUM_VARS_PER_CYCLE, 50);
        assert_eq!(NUM_EQ_CONSTRAINTS, 32);
        assert_eq!(NUM_CONSTRAINTS_PER_CYCLE, 35);
        // 50 vars round up to 64 — keeps prior `num_vars_padded`.
        assert!(NUM_VARS_PER_CYCLE.next_power_of_two() == 64);
    }

    #[test]
    fn pow2_to_field_matches_repeated_doubling() {
        // 2^128 via the Pow2 path = 2^64 * 2^64.
        let p128: Fr = super::pow2_to_field(128);
        let p64: Fr = super::pow2_to_field(64);
        assert_eq!(p128, p64 * p64);
        // 2^192 = 2^128 * 2^64.
        let p192: Fr = super::pow2_to_field(192);
        assert_eq!(p192, p128 * p64);
    }
}
