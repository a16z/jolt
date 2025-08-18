use crate::{field::JoltField, zkvm::instruction::CircuitFlags};

use super::{
    inputs::JoltR1CSInputs,
    old_builder::{CombinedUniformBuilder, R1CSBuilder},
};

pub const PC_START_ADDRESS: i64 = 0x80000000;

pub trait R1CSConstraints<F: JoltField> {
    fn construct_constraints(padded_trace_length: usize) -> CombinedUniformBuilder<F> {
        let mut uniform_builder = R1CSBuilder::new();
        Self::uniform_constraints(&mut uniform_builder);

        CombinedUniformBuilder::construct(uniform_builder, padded_trace_length)
    }
    /// Constructs Jolt's uniform constraints.
    /// Uniform constraints are constraints that hold for each step of
    /// the execution trace.
    fn uniform_constraints(builder: &mut R1CSBuilder);
}

pub struct JoltRV32IMConstraints;
impl<F: JoltField> R1CSConstraints<F> for JoltRV32IMConstraints {
    fn uniform_constraints(cs: &mut R1CSBuilder) {
        // if LeftOperandIsRs1Value { assert!(LeftInstructionInput == Rs1Value) }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsRs1Value),
            JoltR1CSInputs::LeftInstructionInput,
            JoltR1CSInputs::Rs1Value,
        );

        // if LeftOperandIsPC { assert!(LeftInstructionInput == UnexpandedPC) }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsPC),
            JoltR1CSInputs::LeftInstructionInput,
            JoltR1CSInputs::UnexpandedPC,
        );

        // if !(LeftOperandIsRs1Value || LeftOperandIsPC)  {
        //     assert!(LeftInstructionInput == 0)
        // }
        // Note that LeftOperandIsRs1Value and LeftOperandIsPC are mutually exclusive flags
        cs.constrain_eq_conditional(
            1 - JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsRs1Value)
                - JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsPC),
            JoltR1CSInputs::LeftInstructionInput,
            0i128,
        );

        // if RightOperandIsRs2Value { assert!(RightInstructionInput == Rs2Value) }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsRs2Value),
            JoltR1CSInputs::RightInstructionInput,
            JoltR1CSInputs::Rs2Value,
        );

        // if RightOperandIsImm { assert!(RightInstructionInput == Imm) }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsImm),
            JoltR1CSInputs::RightInstructionInput,
            JoltR1CSInputs::Imm,
        );

        // if !(RightOperandIsRs2Value || RightOperandIsImm)  {
        //     assert!(RightInstructionInput == 0)
        // }
        // Note that RightOperandIsRs2Value and RightOperandIsImm are mutually exclusive flags
        cs.constrain_eq_conditional(
            1 - JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsRs2Value)
                - JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsImm),
            JoltR1CSInputs::RightInstructionInput,
            0i128,
        );

        // if Load || Store {
        //     assert!(RamAddress == Rs1Value + Imm)
        // } else {
        //     assert!(RamAddress == 0)
        // }
        let is_load_or_store = JoltR1CSInputs::OpFlags(CircuitFlags::Load)
            + JoltR1CSInputs::OpFlags(CircuitFlags::Store);
        cs.constrain_if_else(
            is_load_or_store,
            JoltR1CSInputs::Rs1Value + JoltR1CSInputs::Imm,
            0i128,
            JoltR1CSInputs::RamAddress,
        );

        // if Load {
        //     assert!(RamReadValue == RamWriteValue)
        // }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::Load),
            JoltR1CSInputs::RamReadValue,
            JoltR1CSInputs::RamWriteValue,
        );

        // if Load {
        //     assert!(RamReadValue == RdWriteValue)
        // }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::Load),
            JoltR1CSInputs::RamReadValue,
            JoltR1CSInputs::RdWriteValue,
        );

        // if Store {
        //     assert!(Rs2Value == RamWriteValue)
        // }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::Store),
            JoltR1CSInputs::Rs2Value,
            JoltR1CSInputs::RamWriteValue,
        );

        // if AddOperands || SubtractOperands || MultiplyOperands {
        //     // Lookup query is just RightLookupOperand
        //     assert!(LeftLookupOperand == 0)
        // } else {
        //     assert!(LeftLookupOperand == LeftInstructionInput)
        // }
        cs.constrain_if_else(
            JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands)
                + JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands)
                + JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands),
            0i128,
            JoltR1CSInputs::LeftInstructionInput,
            JoltR1CSInputs::LeftLookupOperand,
        );

        // If AddOperands {
        //     assert!(RightLookupOperand == LeftInstructionInput + RightInstructionInput)
        // }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands),
            JoltR1CSInputs::RightLookupOperand,
            JoltR1CSInputs::LeftInstructionInput + JoltR1CSInputs::RightInstructionInput,
        );

        // If SubtractOperands {
        //     assert!(RightLookupOperand == LeftInstructionInput - RightInstructionInput)
        // }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands),
            JoltR1CSInputs::RightLookupOperand,
            // Converts from unsigned to twos-complement representation
            JoltR1CSInputs::LeftInstructionInput - JoltR1CSInputs::RightInstructionInput
                + (0xffffffffffffffffi128 + 1),
        );

        // if MultiplyOperands {
        //     assert!(RightLookupOperand == Rs1Value * Rs2Value)
        // }
        cs.constrain_prod(
            JoltR1CSInputs::RightInstructionInput,
            JoltR1CSInputs::LeftInstructionInput,
            JoltR1CSInputs::Product,
        );
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands),
            JoltR1CSInputs::RightLookupOperand,
            JoltR1CSInputs::Product,
        );

        // if !(AddOperands || SubtractOperands || MultiplyOperands || Advice) {
        //     assert!(RightLookupOperand == RightInstructionInput)
        // }
        cs.constrain_eq_conditional(
            1 - JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands)
                - JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands)
                - JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands)
                // Arbitrary untrusted advice goes in right lookup operand
                - JoltR1CSInputs::OpFlags(CircuitFlags::Advice),
            JoltR1CSInputs::RightLookupOperand,
            JoltR1CSInputs::RightInstructionInput,
        );

        // if Assert {
        //     assert!(LookupOutput == 1)
        // }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::Assert),
            JoltR1CSInputs::LookupOutput,
            1i128,
        );

        // if Rd != 0 && WriteLookupOutputToRD {
        //     assert!(RdWriteValue == LookupOutput)
        // }
        cs.constrain_prod(
            JoltR1CSInputs::Rd,
            JoltR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToRD),
            JoltR1CSInputs::WriteLookupOutputToRD,
        );
        cs.constrain_eq_conditional(
            JoltR1CSInputs::WriteLookupOutputToRD,
            JoltR1CSInputs::RdWriteValue,
            JoltR1CSInputs::LookupOutput,
        );

        // if Rd != 0 && Jump {
        //     if !isCompressed {
        //          assert!(RdWriteValue == UnexpandedPC + 4)
        //     } else {
        //          assert!(RdWriteValue == UnexpandedPC + 2)
        //     }
        // }
        cs.constrain_prod(
            JoltR1CSInputs::Rd,
            JoltR1CSInputs::OpFlags(CircuitFlags::Jump),
            JoltR1CSInputs::WritePCtoRD,
        );
        cs.constrain_eq_conditional(
            JoltR1CSInputs::WritePCtoRD,
            JoltR1CSInputs::RdWriteValue,
            JoltR1CSInputs::UnexpandedPC + 4i128
                - 2 * JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed),
        );

        // if Jump && !NextIsNoop {
        //     assert!(NextUnexpandedPC == LookupOutput)
        // }
        cs.constrain_prod(
            JoltR1CSInputs::OpFlags(CircuitFlags::Jump),
            1 - JoltR1CSInputs::NextIsNoop,
            JoltR1CSInputs::ShouldJump,
        );
        cs.constrain_eq_conditional(
            JoltR1CSInputs::ShouldJump,
            JoltR1CSInputs::NextUnexpandedPC,
            JoltR1CSInputs::LookupOutput,
        );

        // if Branch && LookupOutput {
        //     assert!(NextUnexpandedPC == UnexpandedPC + Imm)
        // }
        cs.constrain_prod(
            JoltR1CSInputs::OpFlags(CircuitFlags::Branch),
            JoltR1CSInputs::LookupOutput,
            JoltR1CSInputs::ShouldBranch,
        );
        cs.constrain_eq_conditional(
            JoltR1CSInputs::ShouldBranch,
            JoltR1CSInputs::NextUnexpandedPC,
            JoltR1CSInputs::UnexpandedPC + JoltR1CSInputs::Imm,
        );

        // if !(ShouldBranch || Jump) {
        //     if DoNotUpdatePC {
        //         assert!(NextUnexpandedPC == UnexpandedPC)
        //     } else if isCompressed {
        //         assert!(NextUnexpandedPC == UnexpandedPC + 2)
        //     } else {
        //         assert!(NextUnexpandedPC == UnexpandedPC + 4)
        //     }
        // }
        // Note that ShouldBranch and Jump instructions are mutually exclusive
        cs.constrain_prod(
            JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed),
            JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC),
            JoltR1CSInputs::CompressedDoNotUpdateUnexpPC,
        );
        cs.constrain_eq_conditional(
            1 - JoltR1CSInputs::ShouldBranch - JoltR1CSInputs::OpFlags(CircuitFlags::Jump),
            JoltR1CSInputs::NextUnexpandedPC,
            JoltR1CSInputs::UnexpandedPC + 4i128
                - 4 * JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC)
                - 2 * JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed)
                + 2 * JoltR1CSInputs::CompressedDoNotUpdateUnexpPC,
        );

        // if Inline {
        //     assert!(NextPC == PC + 1)
        // }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::InlineSequenceInstruction),
            JoltR1CSInputs::NextPC,
            JoltR1CSInputs::PC + 1i128,
        );
    }
}

impl JoltRV32IMConstraints {
    /// Constructs uniform constraints using the compile-time constant constraint system
    pub fn construct_const_constraints<F: JoltField>(
        padded_trace_length: usize,
    ) -> super::old_builder::CombinedUniformBuilder<F> {
        use super::constraints::UNIFORM_R1CS;
        use super::old_builder::{CombinedUniformBuilder, R1CSBuilder};

        // Convert const constraints to dynamic constraints
        let dynamic_constraints = UNIFORM_R1CS
            .iter()
            .map(Self::convert_const_constraint_to_dynamic)
            .collect();

        let uniform_builder = R1CSBuilder {
            constraints: dynamic_constraints,
        };

        CombinedUniformBuilder::construct(uniform_builder, padded_trace_length)
    }

    /// Convert a single ConstraintConst to dynamic Constraint
    fn convert_const_constraint_to_dynamic(
        const_constraint: &super::constraints::ConstraintConst,
    ) -> super::old_builder::Constraint {
        use super::old_builder::Constraint;
        use super::old_ops::{Term, Variable, LC};

        let convert_const_lc_to_lc = |const_lc: &super::constraints::ConstLC| -> LC {
            let mut terms = Vec::new();

            // Add variable terms
            for i in 0..const_lc.num_terms() {
                if let Some(term) = const_lc.term(i) {
                    terms.push(Term(Variable::Input(term.input_index), term.coeff));
                }
            }

            // Add constant term if present
            if let Some(const_val) = const_lc.const_term() {
                terms.push(Term(Variable::Constant, const_val));
            }

            LC::new(terms)
        };

        Constraint {
            a: convert_const_lc_to_lc(&const_constraint.a),
            b: convert_const_lc_to_lc(&const_constraint.b),
            c: convert_const_lc_to_lc(&const_constraint.c),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::constraints::{ConstLC, ConstraintConst, NUM_R1CS_CONSTRAINTS, UNIFORM_R1CS};
    use super::super::old_ops::{Term, Variable, LC};
    use super::*;

    #[test]
    fn test_ground_truth_validation() {
        use ark_bn254::Fr as F;

        // Generate ground truth constraints using the original dynamic system
        let mut builder = R1CSBuilder::new();
        <JoltRV32IMConstraints as R1CSConstraints<F>>::uniform_constraints(&mut builder);
        let ground_truth_constraints = builder.get_constraints();

        // Verify we have the expected number of constraints
        assert_eq!(
            ground_truth_constraints.len(),
            NUM_R1CS_CONSTRAINTS,
            "Number of constraints mismatch! Expected {NUM_R1CS_CONSTRAINTS}, got {}",
            ground_truth_constraints.len()
        );

        // Compare each const constraint with its ground truth equivalent
        for (i, (const_constraint, ground_truth)) in UNIFORM_R1CS
            .iter()
            .zip(ground_truth_constraints.iter())
            .enumerate()
        {
            println!("Checking constraint {i}");

            // Convert const constraint to dynamic for comparison and compare
            let converted = const_constraint_to_dynamic_for_test(const_constraint);
            compare_constraints(&converted, ground_truth, i);
        }
    }

    /// Test-specific conversion function that always works regardless of feature flags
    fn const_constraint_to_dynamic_for_test(
        const_constraint: &ConstraintConst,
    ) -> super::super::old_builder::Constraint {
        use super::super::old_builder::Constraint;

        fn const_lc_to_dynamic_lc_for_test(const_lc: &ConstLC) -> LC {
            let mut terms = Vec::new();

            // Add variable terms
            for i in 0..const_lc.num_terms() {
                if let Some(term) = const_lc.term(i) {
                    terms.push(Term(Variable::Input(term.input_index), term.coeff));
                }
            }

            // Add constant term if present
            if let Some(const_val) = const_lc.const_term() {
                terms.push(Term(Variable::Constant, const_val));
            }

            LC::new(terms)
        }

        Constraint {
            a: const_lc_to_dynamic_lc_for_test(&const_constraint.a),
            b: const_lc_to_dynamic_lc_for_test(&const_constraint.b),
            c: const_lc_to_dynamic_lc_for_test(&const_constraint.c),
        }
    }

    // Helper function to compare two constraints for equality
    fn compare_constraints(
        converted: &super::super::old_builder::Constraint,
        ground_truth: &super::super::old_builder::Constraint,
        constraint_index: usize,
    ) {
        // Helper to sort terms for comparison (ground truth might have different ordering)
        // Also filter out zero coefficient terms since they don't affect the constraint
        fn sort_and_filter_terms(terms: &[Term]) -> Vec<Term> {
            let mut sorted: Vec<Term> = terms.iter()
                .filter(|term| term.1 != 0)  // Filter out zero coefficients
                .copied()
                .collect();
            sorted.sort_by_key(|term| match term.0 {
                Variable::Input(idx) => (0, idx, term.1),
                Variable::Constant => (1, 0, term.1),
            });
            sorted
        }

        let converted_a = sort_and_filter_terms(converted.a.terms());
        let ground_truth_a = sort_and_filter_terms(ground_truth.a.terms());
        let converted_b = sort_and_filter_terms(converted.b.terms());
        let ground_truth_b = sort_and_filter_terms(ground_truth.b.terms());
        let converted_c = sort_and_filter_terms(converted.c.terms());
        let ground_truth_c = sort_and_filter_terms(ground_truth.c.terms());

        assert_eq!(
            converted_a, ground_truth_a,
            "Constraint {constraint_index} A terms mismatch!\nConverted: {converted_a:?}\nGround truth: {ground_truth_a:?}"
        );

        assert_eq!(
            converted_b, ground_truth_b,
            "Constraint {constraint_index} B terms mismatch!\nConverted: {converted_b:?}\nGround truth: {ground_truth_b:?}"
        );

        assert_eq!(
            converted_c, ground_truth_c,
            "Constraint {constraint_index} C terms mismatch!\nConverted: {converted_c:?}\nGround truth: {ground_truth_c:?}"
        );
    }

    #[test]
    fn from_index_to_index() {
        use super::super::inputs::{JoltR1CSInputs, ALL_R1CS_INPUTS};

        for i in 0..JoltR1CSInputs::num_inputs() {
            assert_eq!(i, JoltR1CSInputs::from_index(i).to_index());
        }
        for var in ALL_R1CS_INPUTS {
            assert_eq!(
                var,
                JoltR1CSInputs::from_index(JoltR1CSInputs::to_index(&var))
            );
        }
    }
}
