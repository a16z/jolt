//! Jolt R1CS constraint definitions and [`UniformSpartanKey`] construction.
//!
//! Translates the 24 uniform per-cycle constraints (19 eq-conditional + 5 product)
//! into the sparse matrix format expected by [`UniformSpartanKey`].
//!
//! # Variable layout
//!
//! Each cycle has [`NUM_VARS_PER_CYCLE`] witness variables:
//!
//! | Range | Description |
//! |-------|-------------|
//! | `[0]` | Constant 1 |
//! | `[1..=37]` | R1CS inputs (canonical `JoltR1CSInputs` order) |
//! | `[38..=40]` | Product factor variables (`IsRdNotZero`, `Branch`, `NextIsNoop`) |
//!
//! # Constraint forms
//!
//! - **Eq-conditional** (rows 0–18): $\text{guard} \cdot (\text{left} - \text{right}) = 0$.
//! - **Product** (rows 19–23): $\text{left} \cdot \text{right} = \text{output}$.

use jolt_field::Field;
use jolt_spartan::UniformSpartanKey;

// ── Dimensions ──────────────────────────────────────────────────────────────

/// Number of per-cycle R1CS inputs from the outer sumcheck.
pub const NUM_R1CS_INPUTS: usize = 37;

/// Number of additional variables for product constraint factors.
pub const NUM_PRODUCT_FACTORS: usize = 3;

/// Total variables per cycle: constant-1 + R1CS inputs + product factors.
pub const NUM_VARS_PER_CYCLE: usize = 1 + NUM_R1CS_INPUTS + NUM_PRODUCT_FACTORS; // 41

/// Number of eq-conditional constraints.
pub const NUM_EQ_CONSTRAINTS: usize = 19;

/// Number of product constraints.
pub const NUM_PRODUCT_CONSTRAINTS: usize = 5;

/// Total constraints per cycle.
pub const NUM_CONSTRAINTS_PER_CYCLE: usize = NUM_EQ_CONSTRAINTS + NUM_PRODUCT_CONSTRAINTS; // 24

// ── Variable indices ────────────────────────────────────────────────────────

/// Constant 1 wire.
pub const V_CONST: usize = 0;

// R1CS inputs [1..=37], matching canonical `JoltR1CSInputs` order.
pub const V_LEFT_INSTRUCTION_INPUT: usize = 1;
pub const V_RIGHT_INSTRUCTION_INPUT: usize = 2;
pub const V_PRODUCT: usize = 3;
pub const V_WRITE_LOOKUP_OUTPUT_TO_RD: usize = 4;
pub const V_WRITE_PC_TO_RD: usize = 5;
pub const V_SHOULD_BRANCH: usize = 6;
pub const V_PC: usize = 7;
pub const V_UNEXPANDED_PC: usize = 8;
pub const V_IMM: usize = 9;
pub const V_RAM_ADDRESS: usize = 10;
pub const V_RS1_VALUE: usize = 11;
pub const V_RS2_VALUE: usize = 12;
pub const V_RD_WRITE_VALUE: usize = 13;
pub const V_RAM_READ_VALUE: usize = 14;
pub const V_RAM_WRITE_VALUE: usize = 15;
pub const V_LEFT_LOOKUP_OPERAND: usize = 16;
pub const V_RIGHT_LOOKUP_OPERAND: usize = 17;
pub const V_NEXT_UNEXPANDED_PC: usize = 18;
pub const V_NEXT_PC: usize = 19;
pub const V_NEXT_IS_VIRTUAL: usize = 20;
pub const V_NEXT_IS_FIRST_IN_SEQUENCE: usize = 21;
pub const V_LOOKUP_OUTPUT: usize = 22;
pub const V_SHOULD_JUMP: usize = 23;
pub const V_FLAG_ADD_OPERANDS: usize = 24;
pub const V_FLAG_SUBTRACT_OPERANDS: usize = 25;
pub const V_FLAG_MULTIPLY_OPERANDS: usize = 26;
pub const V_FLAG_LOAD: usize = 27;
pub const V_FLAG_STORE: usize = 28;
pub const V_FLAG_JUMP: usize = 29;
pub const V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD: usize = 30;
pub const V_FLAG_VIRTUAL_INSTRUCTION: usize = 31;
pub const V_FLAG_ASSERT: usize = 32;
pub const V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC: usize = 33;
pub const V_FLAG_ADVICE: usize = 34;
pub const V_FLAG_IS_COMPRESSED: usize = 35;
pub const V_FLAG_IS_FIRST_IN_SEQUENCE: usize = 36;
pub const V_FLAG_IS_LAST_IN_SEQUENCE: usize = 37;

// Product factor variables [38..=40].
pub const V_IS_RD_NOT_ZERO: usize = 38;
pub const V_BRANCH: usize = 39;
pub const V_NEXT_IS_NOOP: usize = 40;

// ── Sparse matrix builder ───────────────────────────────────────────────────

type Sparse<F> = Vec<Vec<(usize, F)>>;

/// Helper to convert an i128 coefficient to a field element.
#[inline]
fn coeff<F: Field>(c: i128) -> F {
    F::from_i128(c)
}

/// Builds the Jolt R1CS as a [`UniformSpartanKey`].
///
/// The key encodes 24 constraints × [`NUM_VARS_PER_CYCLE`] variables per cycle.
/// Row order: 19 eq-conditional (matching `R1CS_CONSTRAINTS` order in jolt-core),
/// then 5 product constraints (matching `PRODUCT_CONSTRAINTS` order).
pub fn build_jolt_spartan_key<F: Field>(num_cycles: usize) -> UniformSpartanKey<F> {
    let one: F = F::from_u64(1);
    let neg_one: F = -one;

    let mut a_sparse: Sparse<F> = Vec::with_capacity(NUM_CONSTRAINTS_PER_CYCLE);
    let mut b_sparse: Sparse<F> = Vec::with_capacity(NUM_CONSTRAINTS_PER_CYCLE);
    let mut c_sparse: Sparse<F> = Vec::with_capacity(NUM_CONSTRAINTS_PER_CYCLE);

    // ── Eq-conditional constraints: guard · (left − right) = 0 ──────────

    // 0: RamAddrEqRs1PlusImmIfLoadStore
    //    guard = Load + Store
    //    left − right = RamAddress − Rs1Value − Imm
    a_sparse.push(vec![(V_FLAG_LOAD, one), (V_FLAG_STORE, one)]);
    b_sparse.push(vec![
        (V_RAM_ADDRESS, one),
        (V_RS1_VALUE, neg_one),
        (V_IMM, neg_one),
    ]);
    c_sparse.push(vec![]);

    // 1: RamAddrEqZeroIfNotLoadStore
    //    guard = 1 − Load − Store
    //    left − right = RamAddress
    a_sparse.push(vec![
        (V_CONST, one),
        (V_FLAG_LOAD, neg_one),
        (V_FLAG_STORE, neg_one),
    ]);
    b_sparse.push(vec![(V_RAM_ADDRESS, one)]);
    c_sparse.push(vec![]);

    // 2: RamReadEqRamWriteIfLoad
    //    guard = Load
    //    left − right = RamReadValue − RamWriteValue
    a_sparse.push(vec![(V_FLAG_LOAD, one)]);
    b_sparse.push(vec![(V_RAM_READ_VALUE, one), (V_RAM_WRITE_VALUE, neg_one)]);
    c_sparse.push(vec![]);

    // 3: RamReadEqRdWriteIfLoad
    //    guard = Load
    //    left − right = RamReadValue − RdWriteValue
    a_sparse.push(vec![(V_FLAG_LOAD, one)]);
    b_sparse.push(vec![(V_RAM_READ_VALUE, one), (V_RD_WRITE_VALUE, neg_one)]);
    c_sparse.push(vec![]);

    // 4: Rs2EqRamWriteIfStore
    //    guard = Store
    //    left − right = Rs2Value − RamWriteValue
    a_sparse.push(vec![(V_FLAG_STORE, one)]);
    b_sparse.push(vec![(V_RS2_VALUE, one), (V_RAM_WRITE_VALUE, neg_one)]);
    c_sparse.push(vec![]);

    // 5: LeftLookupZeroUnlessAddSubMul
    //    guard = AddOperands + SubtractOperands + MultiplyOperands
    //    left − right = LeftLookupOperand
    a_sparse.push(vec![
        (V_FLAG_ADD_OPERANDS, one),
        (V_FLAG_SUBTRACT_OPERANDS, one),
        (V_FLAG_MULTIPLY_OPERANDS, one),
    ]);
    b_sparse.push(vec![(V_LEFT_LOOKUP_OPERAND, one)]);
    c_sparse.push(vec![]);

    // 6: LeftLookupEqLeftInputOtherwise
    //    guard = 1 − AddOperands − SubtractOperands − MultiplyOperands
    //    left − right = LeftLookupOperand − LeftInstructionInput
    a_sparse.push(vec![
        (V_CONST, one),
        (V_FLAG_ADD_OPERANDS, neg_one),
        (V_FLAG_SUBTRACT_OPERANDS, neg_one),
        (V_FLAG_MULTIPLY_OPERANDS, neg_one),
    ]);
    b_sparse.push(vec![
        (V_LEFT_LOOKUP_OPERAND, one),
        (V_LEFT_INSTRUCTION_INPUT, neg_one),
    ]);
    c_sparse.push(vec![]);

    // 7: RightLookupAdd
    //    guard = AddOperands
    //    left − right = RightLookupOperand − LeftInstructionInput − RightInstructionInput
    a_sparse.push(vec![(V_FLAG_ADD_OPERANDS, one)]);
    b_sparse.push(vec![
        (V_RIGHT_LOOKUP_OPERAND, one),
        (V_LEFT_INSTRUCTION_INPUT, neg_one),
        (V_RIGHT_INSTRUCTION_INPUT, neg_one),
    ]);
    c_sparse.push(vec![]);

    // 8: RightLookupSub
    //    guard = SubtractOperands
    //    left − right = RightLookupOperand − LeftInstructionInput + RightInstructionInput − 2^64
    //    The +2^64 in `right` converts unsigned subtraction to two's complement.
    a_sparse.push(vec![(V_FLAG_SUBTRACT_OPERANDS, one)]);
    b_sparse.push(vec![
        (V_RIGHT_LOOKUP_OPERAND, one),
        (V_LEFT_INSTRUCTION_INPUT, neg_one),
        (V_RIGHT_INSTRUCTION_INPUT, one),
        (V_CONST, coeff::<F>(-0x1_0000_0000_0000_0000)),
    ]);
    c_sparse.push(vec![]);

    // 9: RightLookupEqProductIfMul
    //    guard = MultiplyOperands
    //    left − right = RightLookupOperand − Product
    a_sparse.push(vec![(V_FLAG_MULTIPLY_OPERANDS, one)]);
    b_sparse.push(vec![(V_RIGHT_LOOKUP_OPERAND, one), (V_PRODUCT, neg_one)]);
    c_sparse.push(vec![]);

    // 10: RightLookupEqRightInputOtherwise
    //     guard = 1 − AddOperands − SubtractOperands − MultiplyOperands − Advice
    //     left − right = RightLookupOperand − RightInstructionInput
    a_sparse.push(vec![
        (V_CONST, one),
        (V_FLAG_ADD_OPERANDS, neg_one),
        (V_FLAG_SUBTRACT_OPERANDS, neg_one),
        (V_FLAG_MULTIPLY_OPERANDS, neg_one),
        (V_FLAG_ADVICE, neg_one),
    ]);
    b_sparse.push(vec![
        (V_RIGHT_LOOKUP_OPERAND, one),
        (V_RIGHT_INSTRUCTION_INPUT, neg_one),
    ]);
    c_sparse.push(vec![]);

    // 11: AssertLookupOne
    //     guard = Assert
    //     left − right = LookupOutput − 1
    a_sparse.push(vec![(V_FLAG_ASSERT, one)]);
    b_sparse.push(vec![(V_LOOKUP_OUTPUT, one), (V_CONST, neg_one)]);
    c_sparse.push(vec![]);

    // 12: RdWriteEqLookupIfWriteLookupToRd
    //     guard = WriteLookupOutputToRD (product-derived boolean)
    //     left − right = RdWriteValue − LookupOutput
    a_sparse.push(vec![(V_WRITE_LOOKUP_OUTPUT_TO_RD, one)]);
    b_sparse.push(vec![(V_RD_WRITE_VALUE, one), (V_LOOKUP_OUTPUT, neg_one)]);
    c_sparse.push(vec![]);

    // 13: RdWriteEqPCPlusConstIfWritePCtoRD
    //     guard = WritePCtoRD (product-derived boolean)
    //     left − right = RdWriteValue − UnexpandedPC − 4 + 2·IsCompressed
    a_sparse.push(vec![(V_WRITE_PC_TO_RD, one)]);
    b_sparse.push(vec![
        (V_RD_WRITE_VALUE, one),
        (V_UNEXPANDED_PC, neg_one),
        (V_CONST, coeff::<F>(-4)),
        (V_FLAG_IS_COMPRESSED, coeff::<F>(2)),
    ]);
    c_sparse.push(vec![]);

    // 14: NextUnexpPCEqLookupIfShouldJump
    //     guard = ShouldJump (product-derived boolean)
    //     left − right = NextUnexpandedPC − LookupOutput
    a_sparse.push(vec![(V_SHOULD_JUMP, one)]);
    b_sparse.push(vec![
        (V_NEXT_UNEXPANDED_PC, one),
        (V_LOOKUP_OUTPUT, neg_one),
    ]);
    c_sparse.push(vec![]);

    // 15: NextUnexpPCEqPCPlusImmIfShouldBranch
    //     guard = ShouldBranch (product-derived boolean)
    //     left − right = NextUnexpandedPC − UnexpandedPC − Imm
    a_sparse.push(vec![(V_SHOULD_BRANCH, one)]);
    b_sparse.push(vec![
        (V_NEXT_UNEXPANDED_PC, one),
        (V_UNEXPANDED_PC, neg_one),
        (V_IMM, neg_one),
    ]);
    c_sparse.push(vec![]);

    // 16: NextUnexpPCUpdateOtherwise
    //     guard = 1 − ShouldBranch − OpFlags(Jump)
    //     left − right = NextUnexpandedPC − UnexpandedPC − 4
    //                   + 4·DoNotUpdateUnexpandedPC + 2·IsCompressed
    a_sparse.push(vec![
        (V_CONST, one),
        (V_SHOULD_BRANCH, neg_one),
        (V_FLAG_JUMP, neg_one),
    ]);
    b_sparse.push(vec![
        (V_NEXT_UNEXPANDED_PC, one),
        (V_UNEXPANDED_PC, neg_one),
        (V_CONST, coeff::<F>(-4)),
        (V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC, coeff::<F>(4)),
        (V_FLAG_IS_COMPRESSED, coeff::<F>(2)),
    ]);
    c_sparse.push(vec![]);

    // 17: NextPCEqPCPlusOneIfInline
    //     guard = VirtualInstruction − IsLastInSequence
    //     left − right = NextPC − PC − 1
    a_sparse.push(vec![
        (V_FLAG_VIRTUAL_INSTRUCTION, one),
        (V_FLAG_IS_LAST_IN_SEQUENCE, neg_one),
    ]);
    b_sparse.push(vec![(V_NEXT_PC, one), (V_PC, neg_one), (V_CONST, neg_one)]);
    c_sparse.push(vec![]);

    // 18: MustStartSequenceFromBeginning
    //     guard = NextIsVirtual − NextIsFirstInSequence
    //     left − right = 1 − DoNotUpdateUnexpandedPC
    a_sparse.push(vec![
        (V_NEXT_IS_VIRTUAL, one),
        (V_NEXT_IS_FIRST_IN_SEQUENCE, neg_one),
    ]);
    b_sparse.push(vec![
        (V_CONST, one),
        (V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC, neg_one),
    ]);
    c_sparse.push(vec![]);

    // ── Product constraints: left · right = output ──────────────────────

    // 19: Product = LeftInstructionInput · RightInstructionInput
    a_sparse.push(vec![(V_LEFT_INSTRUCTION_INPUT, one)]);
    b_sparse.push(vec![(V_RIGHT_INSTRUCTION_INPUT, one)]);
    c_sparse.push(vec![(V_PRODUCT, one)]);

    // 20: WriteLookupOutputToRD = IsRdNotZero · OpFlags(WriteLookupOutputToRD)
    a_sparse.push(vec![(V_IS_RD_NOT_ZERO, one)]);
    b_sparse.push(vec![(V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD, one)]);
    c_sparse.push(vec![(V_WRITE_LOOKUP_OUTPUT_TO_RD, one)]);

    // 21: WritePCtoRD = IsRdNotZero · OpFlags(Jump)
    a_sparse.push(vec![(V_IS_RD_NOT_ZERO, one)]);
    b_sparse.push(vec![(V_FLAG_JUMP, one)]);
    c_sparse.push(vec![(V_WRITE_PC_TO_RD, one)]);

    // 22: ShouldBranch = LookupOutput · Branch
    a_sparse.push(vec![(V_LOOKUP_OUTPUT, one)]);
    b_sparse.push(vec![(V_BRANCH, one)]);
    c_sparse.push(vec![(V_SHOULD_BRANCH, one)]);

    // 23: ShouldJump = OpFlags(Jump) · (1 − NextIsNoop)
    a_sparse.push(vec![(V_FLAG_JUMP, one)]);
    b_sparse.push(vec![(V_CONST, one), (V_NEXT_IS_NOOP, neg_one)]);
    c_sparse.push(vec![(V_SHOULD_JUMP, one)]);

    debug_assert_eq!(a_sparse.len(), NUM_CONSTRAINTS_PER_CYCLE);
    debug_assert_eq!(b_sparse.len(), NUM_CONSTRAINTS_PER_CYCLE);
    debug_assert_eq!(c_sparse.len(), NUM_CONSTRAINTS_PER_CYCLE);

    UniformSpartanKey::new(
        num_cycles,
        NUM_CONSTRAINTS_PER_CYCLE,
        NUM_VARS_PER_CYCLE,
        a_sparse,
        b_sparse,
        c_sparse,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;

    fn f(val: u64) -> Fr {
        Fr::from_u64(val)
    }

    fn fi(val: i128) -> Fr {
        Fr::from_i128(val)
    }

    /// NOP witness: all zeros except constant=1.
    /// Every eq-conditional constraint has guard=0 or body=0 → Az·Bz = 0 = Cz.
    /// Every product output is 0 and both factors include a zero → Az·Bz = 0 = Cz.
    fn nop_witness() -> Vec<Fr> {
        let mut w = vec![Fr::from_u64(0); NUM_VARS_PER_CYCLE];
        w[V_CONST] = f(1);
        // Default PC update: NextUnexpandedPC = UnexpandedPC + 4
        w[V_UNEXPANDED_PC] = f(0);
        w[V_NEXT_UNEXPANDED_PC] = f(4);
        w
    }

    /// LOAD witness: Rs1=100, Imm=20, RamAddr=120, Ram/Rd read=42.
    fn load_witness() -> Vec<Fr> {
        let mut w = vec![Fr::from_u64(0); NUM_VARS_PER_CYCLE];
        w[V_CONST] = f(1);

        // Instruction: simple LOAD (no lookup, no branch/jump)
        w[V_FLAG_LOAD] = f(1);
        w[V_RS1_VALUE] = f(100);
        w[V_IMM] = fi(20);
        w[V_RAM_ADDRESS] = f(120); // Rs1 + Imm
        w[V_RAM_READ_VALUE] = f(42);
        w[V_RAM_WRITE_VALUE] = f(42); // Load → read == write
        w[V_RD_WRITE_VALUE] = f(42); // Load → read == rd_write

        // PC update: straight-line (no branch/jump, not compressed, not doNotUpdate)
        w[V_UNEXPANDED_PC] = f(1000);
        w[V_NEXT_UNEXPANDED_PC] = f(1004); // +4
        w[V_PC] = f(50);
        w[V_NEXT_PC] = f(51);

        w
    }

    /// ADD witness: Rs1=7 + Imm=3 → LeftInput=7, RightInput=3, Product=21
    /// LeftLookup=0 (AddOperands), RightLookup=10 (LeftInput+RightInput)
    /// LookupOutput=10, WriteLookupOutputToRD → RdWrite=10
    fn add_witness() -> Vec<Fr> {
        let mut w = vec![Fr::from_u64(0); NUM_VARS_PER_CYCLE];
        w[V_CONST] = f(1);

        // Flags
        w[V_FLAG_ADD_OPERANDS] = f(1);
        w[V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD] = f(1);
        w[V_IS_RD_NOT_ZERO] = f(1);

        // Instruction inputs
        w[V_LEFT_INSTRUCTION_INPUT] = f(7);
        w[V_RIGHT_INSTRUCTION_INPUT] = f(3);
        w[V_PRODUCT] = f(21); // 7 * 3

        // Lookup operands (add mode: left_lookup=0, right_lookup=left+right)
        w[V_LEFT_LOOKUP_OPERAND] = f(0);
        w[V_RIGHT_LOOKUP_OPERAND] = f(10); // 7 + 3
        w[V_LOOKUP_OUTPUT] = f(10);

        // Product-derived booleans
        w[V_WRITE_LOOKUP_OUTPUT_TO_RD] = f(1); // IsRdNotZero * WriteLookupFlag = 1*1
        w[V_WRITE_PC_TO_RD] = f(0); // IsRdNotZero * Jump = 1*0

        // Register writes
        w[V_RD_WRITE_VALUE] = f(10); // == LookupOutput

        // PC update: straight-line
        w[V_UNEXPANDED_PC] = f(500);
        w[V_NEXT_UNEXPANDED_PC] = f(504);
        w[V_PC] = f(25);
        w[V_NEXT_PC] = f(26);

        w
    }

    /// Evaluate Az, Bz, Cz for each constraint and check Az*Bz == Cz.
    fn assert_witness_satisfies(witness: &[Fr]) {
        let key = build_jolt_spartan_key::<Fr>(1);
        assert_eq!(witness.len(), NUM_VARS_PER_CYCLE);

        for k in 0..NUM_CONSTRAINTS_PER_CYCLE {
            let a_val = dot_sparse(&key.a_sparse[k], witness);
            let b_val = dot_sparse(&key.b_sparse[k], witness);
            let c_val = dot_sparse(&key.c_sparse[k], witness);

            assert_eq!(
                a_val * b_val,
                c_val,
                "constraint {k} violated: Az={a_val:?}, Bz={b_val:?}, Cz={c_val:?}, Az*Bz={:?}",
                a_val * b_val,
            );
        }
    }

    fn dot_sparse(entries: &[(usize, Fr)], witness: &[Fr]) -> Fr {
        entries
            .iter()
            .map(|&(idx, coeff)| coeff * witness[idx])
            .sum()
    }

    #[test]
    fn key_dimensions() {
        let key = build_jolt_spartan_key::<Fr>(4);
        assert_eq!(key.num_constraints, NUM_CONSTRAINTS_PER_CYCLE);
        assert_eq!(key.num_vars, NUM_VARS_PER_CYCLE);
        assert_eq!(key.num_cycles, 4);
        assert_eq!(key.a_sparse.len(), NUM_CONSTRAINTS_PER_CYCLE);
    }

    #[test]
    fn nop_satisfies() {
        assert_witness_satisfies(&nop_witness());
    }

    #[test]
    fn load_satisfies() {
        assert_witness_satisfies(&load_witness());
    }

    #[test]
    fn add_satisfies() {
        assert_witness_satisfies(&add_witness());
    }

    #[test]
    fn nop_bad_pc_update_fails() {
        let mut w = nop_witness();
        w[V_NEXT_UNEXPANDED_PC] = f(999); // wrong: should be UnexpandedPC + 4
        let key = build_jolt_spartan_key::<Fr>(1);

        // Constraint 16 (NextUnexpPCUpdateOtherwise) should fail:
        // guard = 1 - 0 - 0 = 1, B = 999 - 0 - 4 = 995 ≠ 0
        let a16 = dot_sparse(&key.a_sparse[16], &w);
        let b16 = dot_sparse(&key.b_sparse[16], &w);
        let c16 = dot_sparse(&key.c_sparse[16], &w);
        assert_ne!(a16 * b16, c16);
    }

    #[test]
    fn load_bad_ram_addr_fails() {
        let mut w = load_witness();
        w[V_RAM_ADDRESS] = f(999); // wrong: should be Rs1+Imm = 120
        let key = build_jolt_spartan_key::<Fr>(1);

        // Constraint 0 (RamAddrEqRs1PlusImmIfLoadStore) should fail:
        // guard = Load = 1, B = 999 - 100 - 20 = 879 ≠ 0
        let a0 = dot_sparse(&key.a_sparse[0], &w);
        let b0 = dot_sparse(&key.b_sparse[0], &w);
        let c0 = dot_sparse(&key.c_sparse[0], &w);
        assert_ne!(a0 * b0, c0);
    }

    #[test]
    fn product_constraint_fails_on_mismatch() {
        let mut w = add_witness();
        w[V_PRODUCT] = f(999); // wrong: should be 7*3 = 21
        let key = build_jolt_spartan_key::<Fr>(1);

        // Constraint 19 (Product = LeftInput * RightInput) should fail
        let a19 = dot_sparse(&key.a_sparse[19], &w);
        let b19 = dot_sparse(&key.b_sparse[19], &w);
        let c19 = dot_sparse(&key.c_sparse[19], &w);
        assert_ne!(a19 * b19, c19);
    }

    #[test]
    fn prove_and_verify_nop_cycle() {
        use jolt_openings::mock::MockCommitmentScheme;
        use jolt_spartan::{UniformSpartanProver, UniformSpartanVerifier};
        use jolt_transcript::{Blake2bTranscript, Transcript};

        type MockPCS = MockCommitmentScheme<Fr>;

        let key = build_jolt_spartan_key::<Fr>(1);
        let witnesses = vec![nop_witness()];

        let mut pt = Blake2bTranscript::new(b"jolt-r1cs-nop");
        let proof = UniformSpartanProver::prove_dense::<MockPCS, _>(&key, &witnesses, &(), &mut pt)
            .expect("NOP cycle should prove");

        let mut vt = Blake2bTranscript::new(b"jolt-r1cs-nop");
        UniformSpartanVerifier::verify::<MockPCS, _>(&key, &proof, &(), &mut vt)
            .expect("NOP cycle should verify");
    }

    #[test]
    fn prove_and_verify_load_cycle() {
        use jolt_openings::mock::MockCommitmentScheme;
        use jolt_spartan::{UniformSpartanProver, UniformSpartanVerifier};
        use jolt_transcript::{Blake2bTranscript, Transcript};

        type MockPCS = MockCommitmentScheme<Fr>;

        let key = build_jolt_spartan_key::<Fr>(1);
        let witnesses = vec![load_witness()];

        let mut pt = Blake2bTranscript::new(b"jolt-r1cs-load");
        let proof = UniformSpartanProver::prove_dense::<MockPCS, _>(&key, &witnesses, &(), &mut pt)
            .expect("LOAD cycle should prove");

        let mut vt = Blake2bTranscript::new(b"jolt-r1cs-load");
        UniformSpartanVerifier::verify::<MockPCS, _>(&key, &proof, &(), &mut vt)
            .expect("LOAD cycle should verify");
    }

    #[test]
    fn prove_and_verify_mixed_cycles() {
        use jolt_openings::mock::MockCommitmentScheme;
        use jolt_spartan::{UniformSpartanProver, UniformSpartanVerifier};
        use jolt_transcript::{Blake2bTranscript, Transcript};

        type MockPCS = MockCommitmentScheme<Fr>;

        let key = build_jolt_spartan_key::<Fr>(4);
        let witnesses = vec![nop_witness(), load_witness(), add_witness(), nop_witness()];

        let mut pt = Blake2bTranscript::new(b"jolt-r1cs-mixed");
        let proof = UniformSpartanProver::prove_dense::<MockPCS, _>(&key, &witnesses, &(), &mut pt)
            .expect("mixed cycles should prove");

        let mut vt = Blake2bTranscript::new(b"jolt-r1cs-mixed");
        UniformSpartanVerifier::verify::<MockPCS, _>(&key, &proof, &(), &mut vt)
            .expect("mixed cycles should verify");
    }

    #[test]
    fn bad_witness_rejected_by_prover() {
        use jolt_openings::mock::MockCommitmentScheme;
        use jolt_spartan::{SpartanError, UniformSpartanProver};
        use jolt_transcript::{Blake2bTranscript, Transcript};

        type MockPCS = MockCommitmentScheme<Fr>;

        let key = build_jolt_spartan_key::<Fr>(1);
        let mut w = load_witness();
        w[V_RAM_ADDRESS] = f(999); // Violates constraint 0
        let witnesses = vec![w];

        let mut pt = Blake2bTranscript::new(b"jolt-r1cs-bad");
        let result =
            UniformSpartanProver::prove_dense::<MockPCS, _>(&key, &witnesses, &(), &mut pt);
        assert!(matches!(result, Err(SpartanError::ConstraintViolation(_))));
    }
}
