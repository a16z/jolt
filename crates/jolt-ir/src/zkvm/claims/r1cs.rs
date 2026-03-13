//! R1CS constraint definitions as symbolic expressions.
//!
//! Translates the 24 Jolt per-cycle R1CS constraints into [`Expr`] trees.
//! Each expression evaluates to zero when the constraint is satisfied.
//!
//! # Variable layout
//!
//! [`Opening`](crate::Var::Opening) indices correspond to witness vector
//! positions defined by the `V_*` constants. Index 0 (`V_CONST`) is the
//! constant-1 wire; it never appears as an opening variable — constant
//! terms are encoded as [`ExprNode::Constant`](crate::expr::ExprNode::Constant).
//!
//! # Constraint forms
//!
//! - **Eq-conditional** (indices 0–18): `guard · (left − right)` = 0
//! - **Product** (indices 19–23): `left · right − output` = 0

use crate::builder::ExprBuilder;
use crate::expr::Expr;

/// Constant-1 wire. Not used as an `Opening` in expressions; constant
/// terms are inlined via [`ExprNode::Constant`](crate::expr::ExprNode::Constant).
pub const V_CONST: usize = 0;

/// R1CS input indices (1–35), matching old `JoltR1CSInputs` ordering.
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

/// Product factor indices (36–37), not R1CS inputs but needed by product constraints.
pub const V_BRANCH: usize = 36;
pub const V_NEXT_IS_NOOP: usize = 37;

pub const NUM_R1CS_INPUTS: usize = 35;
pub const NUM_PRODUCT_FACTORS: usize = 2;
pub const NUM_VARS_PER_CYCLE: usize = 1 + NUM_R1CS_INPUTS + NUM_PRODUCT_FACTORS; // 38
pub const NUM_EQ_CONSTRAINTS: usize = 19;
pub const NUM_PRODUCT_CONSTRAINTS: usize = 3;
pub const NUM_CONSTRAINTS_PER_CYCLE: usize = NUM_EQ_CONSTRAINTS + NUM_PRODUCT_CONSTRAINTS; // 22

/// An R1CS constraint as a symbolic expression.
///
/// The `expr` evaluates to zero when the constraint is satisfied.
/// Opening variable indices correspond to witness vector positions
/// (the `V_*` constants).
pub struct R1csConstraintExpr {
    pub name: &'static str,
    pub expr: Expr,
}

/// Shorthand for `b.opening(idx as u32)`.
macro_rules! v {
    ($b:expr, $idx:expr) => {
        $b.opening($idx as u32)
    };
}

/// Returns all 24 R1CS constraints as symbolic expressions.
///
/// Constraint ordering matches the sparse matrix layout in
/// `jolt_zkvm::r1cs::build_jolt_spartan_key`:
/// - Indices 0–18: eq-conditional (`guard · body = 0`)
/// - Indices 19–23: product (`left · right − output = 0`)
pub fn constraint_exprs() -> Vec<R1csConstraintExpr> {
    let mut cs = Vec::with_capacity(NUM_CONSTRAINTS_PER_CYCLE);

    // 0: guard = Load + Store
    //    body  = RamAddress − Rs1Value − Imm
    {
        let b = ExprBuilder::new();
        let guard = v!(b, V_FLAG_LOAD) + v!(b, V_FLAG_STORE);
        let body = v!(b, V_RAM_ADDRESS) - v!(b, V_RS1_VALUE) - v!(b, V_IMM);
        cs.push(R1csConstraintExpr {
            name: "RamAddrEqRs1PlusImmIfLoadStore",
            expr: b.build(guard * body),
        });
    }

    // 1: guard = 1 − Load − Store
    //    body  = RamAddress
    {
        let b = ExprBuilder::new();
        let guard = b.one() - v!(b, V_FLAG_LOAD) - v!(b, V_FLAG_STORE);
        let body = v!(b, V_RAM_ADDRESS);
        cs.push(R1csConstraintExpr {
            name: "RamAddrEqZeroIfNotLoadStore",
            expr: b.build(guard * body),
        });
    }

    // 2: guard = Load
    //    body  = RamReadValue − RamWriteValue
    {
        let b = ExprBuilder::new();
        let guard = v!(b, V_FLAG_LOAD);
        let body = v!(b, V_RAM_READ_VALUE) - v!(b, V_RAM_WRITE_VALUE);
        cs.push(R1csConstraintExpr {
            name: "RamReadEqRamWriteIfLoad",
            expr: b.build(guard * body),
        });
    }

    // 3: guard = Load
    //    body  = RamReadValue − RdWriteValue
    {
        let b = ExprBuilder::new();
        let guard = v!(b, V_FLAG_LOAD);
        let body = v!(b, V_RAM_READ_VALUE) - v!(b, V_RD_WRITE_VALUE);
        cs.push(R1csConstraintExpr {
            name: "RamReadEqRdWriteIfLoad",
            expr: b.build(guard * body),
        });
    }

    // 4: guard = Store
    //    body  = Rs2Value − RamWriteValue
    {
        let b = ExprBuilder::new();
        let guard = v!(b, V_FLAG_STORE);
        let body = v!(b, V_RS2_VALUE) - v!(b, V_RAM_WRITE_VALUE);
        cs.push(R1csConstraintExpr {
            name: "Rs2EqRamWriteIfStore",
            expr: b.build(guard * body),
        });
    }

    // 5: guard = Add + Sub + Mul
    //    body  = LeftLookupOperand
    {
        let b = ExprBuilder::new();
        let guard = v!(b, V_FLAG_ADD_OPERANDS)
            + v!(b, V_FLAG_SUBTRACT_OPERANDS)
            + v!(b, V_FLAG_MULTIPLY_OPERANDS);
        let body = v!(b, V_LEFT_LOOKUP_OPERAND);
        cs.push(R1csConstraintExpr {
            name: "LeftLookupZeroUnlessAddSubMul",
            expr: b.build(guard * body),
        });
    }

    // 6: guard = 1 − Add − Sub − Mul
    //    body  = LeftLookupOperand − LeftInstructionInput
    {
        let b = ExprBuilder::new();
        let guard = b.one()
            - v!(b, V_FLAG_ADD_OPERANDS)
            - v!(b, V_FLAG_SUBTRACT_OPERANDS)
            - v!(b, V_FLAG_MULTIPLY_OPERANDS);
        let body = v!(b, V_LEFT_LOOKUP_OPERAND) - v!(b, V_LEFT_INSTRUCTION_INPUT);
        cs.push(R1csConstraintExpr {
            name: "LeftLookupEqLeftInputOtherwise",
            expr: b.build(guard * body),
        });
    }

    // 7: guard = Add
    //    body  = RightLookupOperand − LeftInput − RightInput
    {
        let b = ExprBuilder::new();
        let guard = v!(b, V_FLAG_ADD_OPERANDS);
        let body = v!(b, V_RIGHT_LOOKUP_OPERAND)
            - v!(b, V_LEFT_INSTRUCTION_INPUT)
            - v!(b, V_RIGHT_INSTRUCTION_INPUT);
        cs.push(R1csConstraintExpr {
            name: "RightLookupAdd",
            expr: b.build(guard * body),
        });
    }

    // 8: guard = Sub
    //    body  = RightLookupOperand − LeftInput + RightInput − 2^64
    {
        let b = ExprBuilder::new();
        let guard = v!(b, V_FLAG_SUBTRACT_OPERANDS);
        let body = v!(b, V_RIGHT_LOOKUP_OPERAND) - v!(b, V_LEFT_INSTRUCTION_INPUT)
            + v!(b, V_RIGHT_INSTRUCTION_INPUT)
            - 0x1_0000_0000_0000_0000i128;
        cs.push(R1csConstraintExpr {
            name: "RightLookupSub",
            expr: b.build(guard * body),
        });
    }

    // 9: guard = Mul
    //    body  = RightLookupOperand − Product
    {
        let b = ExprBuilder::new();
        let guard = v!(b, V_FLAG_MULTIPLY_OPERANDS);
        let body = v!(b, V_RIGHT_LOOKUP_OPERAND) - v!(b, V_PRODUCT);
        cs.push(R1csConstraintExpr {
            name: "RightLookupEqProductIfMul",
            expr: b.build(guard * body),
        });
    }

    // 10: guard = 1 − Add − Sub − Mul − Advice
    //     body  = RightLookupOperand − RightInstructionInput
    {
        let b = ExprBuilder::new();
        let guard = b.one()
            - v!(b, V_FLAG_ADD_OPERANDS)
            - v!(b, V_FLAG_SUBTRACT_OPERANDS)
            - v!(b, V_FLAG_MULTIPLY_OPERANDS)
            - v!(b, V_FLAG_ADVICE);
        let body = v!(b, V_RIGHT_LOOKUP_OPERAND) - v!(b, V_RIGHT_INSTRUCTION_INPUT);
        cs.push(R1csConstraintExpr {
            name: "RightLookupEqRightInputOtherwise",
            expr: b.build(guard * body),
        });
    }

    // 11: guard = Assert
    //     body  = LookupOutput − 1
    {
        let b = ExprBuilder::new();
        let guard = v!(b, V_FLAG_ASSERT);
        let body = v!(b, V_LOOKUP_OUTPUT) - 1;
        cs.push(R1csConstraintExpr {
            name: "AssertLookupOne",
            expr: b.build(guard * body),
        });
    }

    // 12: guard = OpFlags(WriteLookupOutputToRD)
    //     body  = RdWriteValue − LookupOutput
    // Uses the raw circuit flag, not the product-derived WriteLookupOutputToRD variable.
    {
        let b = ExprBuilder::new();
        let guard = v!(b, V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD);
        let body = v!(b, V_RD_WRITE_VALUE) - v!(b, V_LOOKUP_OUTPUT);
        cs.push(R1csConstraintExpr {
            name: "RdWriteEqLookupIfWriteLookupToRd",
            expr: b.build(guard * body),
        });
    }

    // 13: guard = OpFlags(Jump)
    //     body  = RdWriteValue − UnexpandedPC − 4 + 2·IsCompressed
    // Uses the raw Jump flag. Trace rewriting ensures Jump implies rd != x0.
    {
        let b = ExprBuilder::new();
        let guard = v!(b, V_FLAG_JUMP);
        let body = v!(b, V_RD_WRITE_VALUE) - v!(b, V_UNEXPANDED_PC) - 4
            + 2i128 * v!(b, V_FLAG_IS_COMPRESSED);
        cs.push(R1csConstraintExpr {
            name: "RdWriteEqPCPlusConstIfWritePCtoRD",
            expr: b.build(guard * body),
        });
    }

    // 14: guard = ShouldJump
    //     body  = NextUnexpandedPC − LookupOutput
    {
        let b = ExprBuilder::new();
        let guard = v!(b, V_SHOULD_JUMP);
        let body = v!(b, V_NEXT_UNEXPANDED_PC) - v!(b, V_LOOKUP_OUTPUT);
        cs.push(R1csConstraintExpr {
            name: "NextUnexpPCEqLookupIfShouldJump",
            expr: b.build(guard * body),
        });
    }

    // 15: guard = ShouldBranch
    //     body  = NextUnexpandedPC − UnexpandedPC − Imm
    {
        let b = ExprBuilder::new();
        let guard = v!(b, V_SHOULD_BRANCH);
        let body = v!(b, V_NEXT_UNEXPANDED_PC) - v!(b, V_UNEXPANDED_PC) - v!(b, V_IMM);
        cs.push(R1csConstraintExpr {
            name: "NextUnexpPCEqPCPlusImmIfShouldBranch",
            expr: b.build(guard * body),
        });
    }

    // 16: guard = 1 − ShouldBranch − Jump
    //     body  = NextUnexpandedPC − UnexpandedPC − 4
    //             + 4·DoNotUpdateUnexpandedPC + 2·IsCompressed
    {
        let b = ExprBuilder::new();
        let guard = b.one() - v!(b, V_SHOULD_BRANCH) - v!(b, V_FLAG_JUMP);
        let body = v!(b, V_NEXT_UNEXPANDED_PC) - v!(b, V_UNEXPANDED_PC) - 4
            + 4i128 * v!(b, V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC)
            + 2i128 * v!(b, V_FLAG_IS_COMPRESSED);
        cs.push(R1csConstraintExpr {
            name: "NextUnexpPCUpdateOtherwise",
            expr: b.build(guard * body),
        });
    }

    // 17: guard = VirtualInstruction − IsLastInSequence
    //     body  = NextPC − PC − 1
    {
        let b = ExprBuilder::new();
        let guard = v!(b, V_FLAG_VIRTUAL_INSTRUCTION) - v!(b, V_FLAG_IS_LAST_IN_SEQUENCE);
        let body = v!(b, V_NEXT_PC) - v!(b, V_PC) - 1;
        cs.push(R1csConstraintExpr {
            name: "NextPCEqPCPlusOneIfInline",
            expr: b.build(guard * body),
        });
    }

    // 18: guard = NextIsVirtual − NextIsFirstInSequence
    //     body  = 1 − DoNotUpdateUnexpandedPC
    {
        let b = ExprBuilder::new();
        let guard = v!(b, V_NEXT_IS_VIRTUAL) - v!(b, V_NEXT_IS_FIRST_IN_SEQUENCE);
        let body = b.one() - v!(b, V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC);
        cs.push(R1csConstraintExpr {
            name: "MustStartSequenceFromBeginning",
            expr: b.build(guard * body),
        });
    }

    // 19: Product = LeftInput · RightInput
    {
        let b = ExprBuilder::new();
        let e =
            v!(b, V_LEFT_INSTRUCTION_INPUT) * v!(b, V_RIGHT_INSTRUCTION_INPUT) - v!(b, V_PRODUCT);
        cs.push(R1csConstraintExpr {
            name: "ProductEqLeftTimesRight",
            expr: b.build(e),
        });
    }

    // 20: ShouldBranch = LookupOutput · Branch
    {
        let b = ExprBuilder::new();
        let e = v!(b, V_LOOKUP_OUTPUT) * v!(b, V_BRANCH) - v!(b, V_SHOULD_BRANCH);
        cs.push(R1csConstraintExpr {
            name: "ShouldBranchEqLookupTimesBranch",
            expr: b.build(e),
        });
    }

    // 21: ShouldJump = Jump · (1 − NextIsNoop)
    {
        let b = ExprBuilder::new();
        let e = v!(b, V_FLAG_JUMP) * (b.one() - v!(b, V_NEXT_IS_NOOP)) - v!(b, V_SHOULD_JUMP);
        cs.push(R1csConstraintExpr {
            name: "ShouldJumpEqJumpTimesNotNoop",
            expr: b.build(e),
        });
    }

    debug_assert_eq!(cs.len(), NUM_CONSTRAINTS_PER_CYCLE);
    cs
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};

    fn f(val: u64) -> Fr {
        Fr::from_u64(val)
    }

    fn fi(val: i128) -> Fr {
        Fr::from_i128(val)
    }

    /// Build a witness vector (indices 0–40) from variable assignments.
    /// Index 0 (V_CONST) is unused by expressions but must be present
    /// for correct indexing.
    fn witness(assignments: &[(usize, Fr)]) -> Vec<Fr> {
        let mut w = vec![Fr::from_u64(0); NUM_VARS_PER_CYCLE];
        for &(idx, val) in assignments {
            w[idx] = val;
        }
        w
    }

    /// NOP: all zeros except PC update (NextUnexpandedPC = UnexpandedPC + 4).
    fn nop_witness() -> Vec<Fr> {
        witness(&[(V_UNEXPANDED_PC, f(0)), (V_NEXT_UNEXPANDED_PC, f(4))])
    }

    /// LOAD: Rs1=100, Imm=20, RamAddr=120, read/write=42.
    fn load_witness() -> Vec<Fr> {
        witness(&[
            (V_FLAG_LOAD, f(1)),
            (V_RS1_VALUE, f(100)),
            (V_IMM, fi(20)),
            (V_RAM_ADDRESS, f(120)),
            (V_RAM_READ_VALUE, f(42)),
            (V_RAM_WRITE_VALUE, f(42)),
            (V_RD_WRITE_VALUE, f(42)),
            (V_UNEXPANDED_PC, f(1000)),
            (V_NEXT_UNEXPANDED_PC, f(1004)),
            (V_PC, f(50)),
            (V_NEXT_PC, f(51)),
        ])
    }

    /// ADD: left=7, right=3, product=21, lookup adds → output=10.
    fn add_witness() -> Vec<Fr> {
        witness(&[
            (V_FLAG_ADD_OPERANDS, f(1)),
            (V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD, f(1)),
            (V_LEFT_INSTRUCTION_INPUT, f(7)),
            (V_RIGHT_INSTRUCTION_INPUT, f(3)),
            (V_PRODUCT, f(21)),
            (V_LEFT_LOOKUP_OPERAND, f(0)),
            (V_RIGHT_LOOKUP_OPERAND, f(10)),
            (V_LOOKUP_OUTPUT, f(10)),
            (V_RD_WRITE_VALUE, f(10)),
            (V_UNEXPANDED_PC, f(500)),
            (V_NEXT_UNEXPANDED_PC, f(504)),
            (V_PC, f(25)),
            (V_NEXT_PC, f(26)),
        ])
    }

    fn assert_all_constraints_satisfied(w: &[Fr]) {
        let constraints = constraint_exprs();
        for (i, c) in constraints.iter().enumerate() {
            let val: Fr = c.expr.evaluate(w, &[]);
            assert_eq!(val, Fr::from_u64(0), "constraint {i} ({}) violated", c.name,);
        }
    }

    #[test]
    fn constraint_count() {
        assert_eq!(constraint_exprs().len(), 22);
    }

    #[test]
    fn nop_satisfies_all() {
        assert_all_constraints_satisfied(&nop_witness());
    }

    #[test]
    fn load_satisfies_all() {
        assert_all_constraints_satisfied(&load_witness());
    }

    #[test]
    fn add_satisfies_all() {
        assert_all_constraints_satisfied(&add_witness());
    }

    #[test]
    fn nop_bad_pc_violates_constraint_16() {
        let mut w = nop_witness();
        w[V_NEXT_UNEXPANDED_PC] = f(999);
        let constraints = constraint_exprs();
        let val: Fr = constraints[16].expr.evaluate(&w, &[]);
        assert_ne!(val, Fr::from_u64(0));
    }

    #[test]
    fn load_bad_ram_addr_violates_constraint_0() {
        let mut w = load_witness();
        w[V_RAM_ADDRESS] = f(999);
        let constraints = constraint_exprs();
        let val: Fr = constraints[0].expr.evaluate(&w, &[]);
        assert_ne!(val, Fr::from_u64(0));
    }

    #[test]
    fn add_bad_product_violates_constraint_19() {
        let mut w = add_witness();
        w[V_PRODUCT] = f(999);
        let constraints = constraint_exprs();
        let val: Fr = constraints[19].expr.evaluate(&w, &[]);
        assert_ne!(val, Fr::from_u64(0));
    }

    /// Verify the Expr-based constraints agree with direct A·z × B·z = C·z
    /// evaluation for all three witness types.
    #[test]
    fn expr_matches_matrix_form() {
        use jolt_field::Field;

        type F = Fr;
        let one: F = F::from_u64(1);
        let neg_one: F = -one;

        // Reproduce the sparse matrix form inline and check agreement
        let witnesses = [nop_witness(), load_witness(), add_witness()];
        let constraints = constraint_exprs();

        // Helper: evaluate sparse dot product
        let dot =
            |entries: &[(usize, F)], w: &[F]| -> F { entries.iter().map(|&(i, c)| c * w[i]).sum() };

        for w in &witnesses {
            // Constraint 0: A=[Load,Store], B=[RamAddr,-Rs1,-Imm], C=[]
            let az = dot(&[(V_FLAG_LOAD, one), (V_FLAG_STORE, one)], w);
            let bz = dot(
                &[
                    (V_RAM_ADDRESS, one),
                    (V_RS1_VALUE, neg_one),
                    (V_IMM, neg_one),
                ],
                w,
            );
            let matrix_val = az * bz;
            let expr_val: F = constraints[0].expr.evaluate(w, &[]);
            assert_eq!(matrix_val, expr_val, "constraint 0 mismatch");

            // Constraint 16: most complex eq-conditional
            let az = dot(
                &[
                    (V_CONST, one),
                    (V_SHOULD_BRANCH, neg_one),
                    (V_FLAG_JUMP, neg_one),
                ],
                &{
                    let mut wc = w.clone();
                    wc[V_CONST] = one;
                    wc
                },
            );
            let bz = dot(
                &[
                    (V_NEXT_UNEXPANDED_PC, one),
                    (V_UNEXPANDED_PC, neg_one),
                    (V_CONST, F::from_i128(-4)),
                    (V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC, F::from_i128(4)),
                    (V_FLAG_IS_COMPRESSED, F::from_i128(2)),
                ],
                &{
                    let mut wc = w.clone();
                    wc[V_CONST] = one;
                    wc
                },
            );
            let matrix_val = az * bz;
            let expr_val: F = constraints[16].expr.evaluate(w, &[]);
            assert_eq!(matrix_val, expr_val, "constraint 16 mismatch");

            // Constraint 19: product
            let az = dot(&[(V_LEFT_INSTRUCTION_INPUT, one)], w);
            let bz = dot(&[(V_RIGHT_INSTRUCTION_INPUT, one)], w);
            let cz = dot(&[(V_PRODUCT, one)], w);
            let matrix_val = az * bz - cz;
            let expr_val: F = constraints[19].expr.evaluate(w, &[]);
            assert_eq!(matrix_val, expr_val, "constraint 19 mismatch");
        }
    }
}

#[cfg(all(test, feature = "z3"))]
mod z3_tests {
    use super::*;
    use crate::backends::z3::Z3Emitter;
    use z3::ast::Int;
    use z3::{SatResult, Solver};

    /// Bind a witness vector to a Z3Emitter (indices 1–40 as openings).
    fn bind_witness(emitter: &mut Z3Emitter, vals: &[(usize, i64)]) {
        for &(idx, val) in vals {
            if idx == V_CONST {
                continue;
            }
            emitter.bind_opening(idx as u32, Int::from_i64(val));
        }
    }

    /// Assert all 24 constraint expressions equal zero with the given witness.
    fn assert_z3_sat(name: &str, witness: &[(usize, i64)]) {
        let constraints = constraint_exprs();
        let solver = Solver::new();

        for (i, c) in constraints.iter().enumerate() {
            let mut emitter = Z3Emitter::new().with_opening_prefix(format!("c{i}"));
            bind_witness(&mut emitter, witness);
            let z3_expr = c.expr.to_circuit(&mut emitter);
            solver.assert(z3_expr.eq(Int::from_i64(0)));
        }

        assert_eq!(
            solver.check(),
            SatResult::Sat,
            "{name} witness should satisfy all constraints",
        );
    }

    #[test]
    fn z3_nop_satisfies() {
        assert_z3_sat("NOP", &[(V_UNEXPANDED_PC, 0), (V_NEXT_UNEXPANDED_PC, 4)]);
    }

    #[test]
    fn z3_load_satisfies() {
        assert_z3_sat(
            "LOAD",
            &[
                (V_FLAG_LOAD, 1),
                (V_RS1_VALUE, 100),
                (V_IMM, 20),
                (V_RAM_ADDRESS, 120),
                (V_RAM_READ_VALUE, 42),
                (V_RAM_WRITE_VALUE, 42),
                (V_RD_WRITE_VALUE, 42),
                (V_UNEXPANDED_PC, 1000),
                (V_NEXT_UNEXPANDED_PC, 1004),
                (V_PC, 50),
                (V_NEXT_PC, 51),
            ],
        );
    }

    #[test]
    fn z3_add_satisfies() {
        assert_z3_sat(
            "ADD",
            &[
                (V_FLAG_ADD_OPERANDS, 1),
                (V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD, 1),
                (V_LEFT_INSTRUCTION_INPUT, 7),
                (V_RIGHT_INSTRUCTION_INPUT, 3),
                (V_PRODUCT, 21),
                (V_LEFT_LOOKUP_OPERAND, 0),
                (V_RIGHT_LOOKUP_OPERAND, 10),
                (V_LOOKUP_OUTPUT, 10),
                (V_RD_WRITE_VALUE, 10),
                (V_UNEXPANDED_PC, 500),
                (V_NEXT_UNEXPANDED_PC, 504),
                (V_PC, 25),
                (V_NEXT_PC, 26),
            ],
        );
    }

    /// Product constraint (19) is UNSAT when Product ≠ Left × Right.
    #[test]
    fn z3_product_violation_unsat() {
        let constraints = constraint_exprs();
        let solver = Solver::new();

        let mut emitter = Z3Emitter::new();
        emitter.bind_opening(V_LEFT_INSTRUCTION_INPUT as u32, Int::from_i64(7));
        emitter.bind_opening(V_RIGHT_INSTRUCTION_INPUT as u32, Int::from_i64(3));
        emitter.bind_opening(V_PRODUCT as u32, Int::from_i64(999));
        let z3_expr = constraints[19].expr.to_circuit(&mut emitter);
        solver.assert(z3_expr.eq(Int::from_i64(0)));

        assert_eq!(solver.check(), SatResult::Unsat);
    }

    /// PC update (constraint 16) is UNSAT for NOP with wrong NextUnexpandedPC.
    #[test]
    fn z3_pc_update_violation_unsat() {
        let constraints = constraint_exprs();
        let solver = Solver::new();

        // NOP: guard = 1 - 0 - 0 = 1 (active), body must be 0
        // body = NextUnexpPC - UnexpPC - 4 = 999 - 0 - 4 = 995 ≠ 0
        let mut emitter = Z3Emitter::new();
        emitter.bind_opening(V_UNEXPANDED_PC as u32, Int::from_i64(0));
        emitter.bind_opening(V_NEXT_UNEXPANDED_PC as u32, Int::from_i64(999));
        emitter.bind_opening(V_SHOULD_BRANCH as u32, Int::from_i64(0));
        emitter.bind_opening(V_FLAG_JUMP as u32, Int::from_i64(0));
        emitter.bind_opening(V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC as u32, Int::from_i64(0));
        emitter.bind_opening(V_FLAG_IS_COMPRESSED as u32, Int::from_i64(0));
        let z3_expr = constraints[16].expr.to_circuit(&mut emitter);
        solver.assert(z3_expr.eq(Int::from_i64(0)));

        assert_eq!(solver.check(), SatResult::Unsat);
    }

    /// Symbolically verify: if Load=1 and all constraints hold,
    /// then RamAddress = Rs1Value + Imm (constraint 0 forces this).
    #[test]
    fn z3_load_forces_ram_addr() {
        let constraints = constraint_exprs();
        let solver = Solver::new();

        // Emit constraint 0 with symbolic variables (only bind Load=1)
        let mut emitter = Z3Emitter::new();
        emitter.bind_opening(V_FLAG_LOAD as u32, Int::from_i64(1));
        let z3_expr = constraints[0].expr.to_circuit(&mut emitter);

        // Constraint 0 = 0
        solver.assert(z3_expr.eq(Int::from_i64(0)));

        // Assert RamAddress ≠ Rs1 + Imm → should be UNSAT
        let ram = emitter.openings().get(&(V_RAM_ADDRESS as u32)).unwrap();
        let rs1 = emitter.openings().get(&(V_RS1_VALUE as u32)).unwrap();
        let imm = emitter.openings().get(&(V_IMM as u32)).unwrap();
        // Store flag is symbolic but constraint 0 guard = Load + Store ≥ 1
        let store = emitter.openings().get(&(V_FLAG_STORE as u32)).unwrap();
        solver.assert(store.eq(Int::from_i64(0)));
        solver.assert(ram.ne(rs1 + imm));

        assert_eq!(solver.check(), SatResult::Unsat);
    }

    // Parity audit: prove jolt-ir constraints are equivalent to old jolt-core.
    //
    // For each constraint, we build the old jolt-core definition manually
    // as a Z3 expression (using the same symbolic variables bound to the
    // Z3Emitter), then assert `old ≠ new` and verify UNSAT — proving
    // they're algebraically identical over all 8-bit variable assignments.
    //
    // Variable mapping: old JoltR1CSInputs indices → new V_* constants.
    // Old constraints 0–18: eq-conditional `A(z) * B(z) = 0`
    // Old product constraints 0–2 → new indices 19, 22, 23
    // New-only product constraints: 20 (WriteLookupOutputToRD), 21 (WritePCtoRD)

    /// Create 8-bit bounded symbolic variables for all V_* positions.
    fn make_bounded_vars(solver: &Solver) -> Vec<Int> {
        (0..NUM_VARS_PER_CYCLE)
            .map(|i| {
                let var = Int::new_const(format!("v{i}"));
                solver.assert(var.ge(Int::from_i64(0)));
                solver.assert(var.lt(Int::from_i64(256)));
                var
            })
            .collect()
    }

    /// Bind all symbolic variables (indices 1..41) to a Z3Emitter.
    fn bind_all_vars(emitter: &mut Z3Emitter, vars: &[Int]) {
        for (i, var) in vars.iter().enumerate().skip(1) {
            emitter.bind_opening(i as u32, var.clone());
        }
    }

    /// Reconstruct old jolt-core eq-conditional constraint `A(z) * B(z)`.
    ///
    /// Each old constraint was `condition * (left − right)` with variables
    /// indexed by `JoltR1CSInputs`, mapped here to V_* constants.
    #[allow(clippy::too_many_lines)]
    fn old_eq_conditional(idx: usize, vars: &[Int]) -> Int {
        let v = |i: usize| vars[i].clone();
        let c = |n: i64| Int::from_i64(n);
        let two_64 = || Int::from_i64(0x1_0000_0000) * Int::from_i64(0x1_0000_0000);

        match idx {
            // (Load + Store) * (RamAddr − Rs1 − Imm)
            0 => {
                (v(V_FLAG_LOAD) + v(V_FLAG_STORE)) * (v(V_RAM_ADDRESS) - v(V_RS1_VALUE) - v(V_IMM))
            }
            // (1 − Load − Store) * RamAddr
            1 => (c(1) - v(V_FLAG_LOAD) - v(V_FLAG_STORE)) * v(V_RAM_ADDRESS),
            // Load * (RamRead − RamWrite)
            2 => v(V_FLAG_LOAD) * (v(V_RAM_READ_VALUE) - v(V_RAM_WRITE_VALUE)),
            // Load * (RamRead − RdWrite)
            3 => v(V_FLAG_LOAD) * (v(V_RAM_READ_VALUE) - v(V_RD_WRITE_VALUE)),
            // Store * (Rs2 − RamWrite)
            4 => v(V_FLAG_STORE) * (v(V_RS2_VALUE) - v(V_RAM_WRITE_VALUE)),
            // (Add + Sub + Mul) * LeftLookup
            5 => {
                (v(V_FLAG_ADD_OPERANDS) + v(V_FLAG_SUBTRACT_OPERANDS) + v(V_FLAG_MULTIPLY_OPERANDS))
                    * v(V_LEFT_LOOKUP_OPERAND)
            }
            // (1 − Add − Sub − Mul) * (LeftLookup − LeftInput)
            6 => {
                (c(1)
                    - v(V_FLAG_ADD_OPERANDS)
                    - v(V_FLAG_SUBTRACT_OPERANDS)
                    - v(V_FLAG_MULTIPLY_OPERANDS))
                    * (v(V_LEFT_LOOKUP_OPERAND) - v(V_LEFT_INSTRUCTION_INPUT))
            }
            // Add * (RightLookup − LeftInput − RightInput)
            7 => {
                v(V_FLAG_ADD_OPERANDS)
                    * (v(V_RIGHT_LOOKUP_OPERAND)
                        - v(V_LEFT_INSTRUCTION_INPUT)
                        - v(V_RIGHT_INSTRUCTION_INPUT))
            }
            // Sub * (RightLookup − LeftInput + RightInput − 2^64)
            8 => {
                v(V_FLAG_SUBTRACT_OPERANDS)
                    * (v(V_RIGHT_LOOKUP_OPERAND) - v(V_LEFT_INSTRUCTION_INPUT)
                        + v(V_RIGHT_INSTRUCTION_INPUT)
                        - two_64())
            }
            // Mul * (RightLookup − Product)
            9 => v(V_FLAG_MULTIPLY_OPERANDS) * (v(V_RIGHT_LOOKUP_OPERAND) - v(V_PRODUCT)),
            // (1 − Add − Sub − Mul − Advice) * (RightLookup − RightInput)
            10 => {
                (c(1)
                    - v(V_FLAG_ADD_OPERANDS)
                    - v(V_FLAG_SUBTRACT_OPERANDS)
                    - v(V_FLAG_MULTIPLY_OPERANDS)
                    - v(V_FLAG_ADVICE))
                    * (v(V_RIGHT_LOOKUP_OPERAND) - v(V_RIGHT_INSTRUCTION_INPUT))
            }
            // Assert * (LookupOutput − 1)
            11 => v(V_FLAG_ASSERT) * (v(V_LOOKUP_OUTPUT) - c(1)),
            // WriteLookupFlag * (RdWrite − LookupOutput)
            12 => v(V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD) * (v(V_RD_WRITE_VALUE) - v(V_LOOKUP_OUTPUT)),
            // Jump * (RdWrite − UnexpPC − 4 + 2·IsCompressed)
            13 => {
                v(V_FLAG_JUMP)
                    * (v(V_RD_WRITE_VALUE) - v(V_UNEXPANDED_PC) - c(4)
                        + c(2) * v(V_FLAG_IS_COMPRESSED))
            }
            // ShouldJump * (NextUnexpPC − LookupOutput)
            14 => v(V_SHOULD_JUMP) * (v(V_NEXT_UNEXPANDED_PC) - v(V_LOOKUP_OUTPUT)),
            // ShouldBranch * (NextUnexpPC − UnexpPC − Imm)
            15 => v(V_SHOULD_BRANCH) * (v(V_NEXT_UNEXPANDED_PC) - v(V_UNEXPANDED_PC) - v(V_IMM)),
            // (1 − ShouldBranch − Jump) *
            //   (NextUnexpPC − UnexpPC − 4 + 4·DoNotUpdate + 2·IsCompressed)
            16 => {
                (c(1) - v(V_SHOULD_BRANCH) - v(V_FLAG_JUMP))
                    * (v(V_NEXT_UNEXPANDED_PC) - v(V_UNEXPANDED_PC) - c(4)
                        + c(4) * v(V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC)
                        + c(2) * v(V_FLAG_IS_COMPRESSED))
            }
            // (Virtual − IsLast) * (NextPC − PC − 1)
            17 => {
                (v(V_FLAG_VIRTUAL_INSTRUCTION) - v(V_FLAG_IS_LAST_IN_SEQUENCE))
                    * (v(V_NEXT_PC) - v(V_PC) - c(1))
            }
            // (NextIsVirtual − NextIsFirst) * (1 − DoNotUpdate)
            18 => {
                (v(V_NEXT_IS_VIRTUAL) - v(V_NEXT_IS_FIRST_IN_SEQUENCE))
                    * (c(1) - v(V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC))
            }
            _ => unreachable!(),
        }
    }

    /// Reconstruct old jolt-core product constraint `left * right − output`.
    fn old_product_constraint(idx: usize, vars: &[Int]) -> Int {
        let v = |i: usize| vars[i].clone();
        let c = |n: i64| Int::from_i64(n);

        match idx {
            // Product = LeftInput · RightInput
            0 => v(V_LEFT_INSTRUCTION_INPUT) * v(V_RIGHT_INSTRUCTION_INPUT) - v(V_PRODUCT),
            // ShouldBranch = LookupOutput · Branch
            1 => v(V_LOOKUP_OUTPUT) * v(V_BRANCH) - v(V_SHOULD_BRANCH),
            // ShouldJump = Jump · (1 − NextIsNoop)
            2 => v(V_FLAG_JUMP) * (c(1) - v(V_NEXT_IS_NOOP)) - v(V_SHOULD_JUMP),
            _ => unreachable!(),
        }
    }

    /// Prove all 19 eq-conditional constraints match old jolt-core.
    #[test]
    fn z3_parity_eq_conditional() {
        let constraints = constraint_exprs();

        for (idx, c) in constraints.iter().enumerate().take(NUM_EQ_CONSTRAINTS) {
            let solver = Solver::new();
            let vars = make_bounded_vars(&solver);

            let mut emitter = Z3Emitter::new();
            bind_all_vars(&mut emitter, &vars);
            let new_expr = c.expr.to_circuit(&mut emitter);
            let old_expr = old_eq_conditional(idx, &vars);

            solver.assert(old_expr.ne(new_expr));
            assert_eq!(
                solver.check(),
                SatResult::Unsat,
                "eq-conditional {idx} ({}) differs from old",
                c.name,
            );
        }
    }

    /// Prove all 3 product constraints match old jolt-core.
    ///
    /// Old → new index mapping:
    /// - 0 (Instruction)   → 19
    /// - 1 (ShouldBranch)  → 20
    /// - 2 (ShouldJump)    → 21
    #[test]
    fn z3_parity_product_constraints() {
        let constraints = constraint_exprs();
        let old_to_new = [(0, 19), (1, 20), (2, 21)];

        for (old_idx, new_idx) in old_to_new {
            let solver = Solver::new();
            let vars = make_bounded_vars(&solver);

            let mut emitter = Z3Emitter::new();
            bind_all_vars(&mut emitter, &vars);
            let new_expr = constraints[new_idx].expr.to_circuit(&mut emitter);
            let old_expr = old_product_constraint(old_idx, &vars);

            solver.assert(old_expr.ne(new_expr));
            assert_eq!(
                solver.check(),
                SatResult::Unsat,
                "product {old_idx} (new idx {new_idx}, {}) differs from old",
                constraints[new_idx].name,
            );
        }
    }
}
