use crate::jolt::{
    execution_trace::JoltONNXR1CSInputs,
    r1cs::builder::{CombinedUniformBuilder, R1CSBuilder},
};
use jolt_core::field::JoltField;
use onnx_tracer::{constants::MAX_TENSOR_SIZE, trace_types::CircuitFlags};

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

pub struct JoltONNXConstraints;
impl<F: JoltField> R1CSConstraints<F> for JoltONNXConstraints {
    fn uniform_constraints(cs: &mut R1CSBuilder) {
        for i in 0..MAX_TENSOR_SIZE {
            // if AddOperands || SubtractOperands || MultiplyOperands {
            //     // Lookup query is just RightLookupOperand
            //     assert!(LeftLookupOperand == 0)
            // } else {
            //     assert!(LeftLookupOperand == LeftInstructionInput)
            // }
            cs.constrain_if_else(
                JoltONNXR1CSInputs::OpFlags(CircuitFlags::AddOperands)
                    + JoltONNXR1CSInputs::OpFlags(CircuitFlags::SubtractOperands)
                    + JoltONNXR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands),
                0,
                JoltONNXR1CSInputs::LeftInstructionInput(i),
                JoltONNXR1CSInputs::LeftLookupOperand(i),
            );

            // If AddOperands {
            //     assert!(RightLookupOperand == LeftInstructionInput + RightInstructionInput)
            // }
            cs.constrain_eq_conditional(
                JoltONNXR1CSInputs::OpFlags(CircuitFlags::AddOperands),
                JoltONNXR1CSInputs::RightLookupOperand(i),
                JoltONNXR1CSInputs::LeftInstructionInput(i)
                    + JoltONNXR1CSInputs::RightInstructionInput(i),
            );

            // If SubtractOperands {
            //     assert!(RightLookupOperand == LeftInstructionInput - RightInstructionInput)
            // }
            cs.constrain_eq_conditional(
                JoltONNXR1CSInputs::OpFlags(CircuitFlags::SubtractOperands),
                JoltONNXR1CSInputs::RightLookupOperand(i),
                // Converts from unsigned to twos-complement representation
                JoltONNXR1CSInputs::LeftInstructionInput(i)
                    - JoltONNXR1CSInputs::RightInstructionInput(i)
                    + (0xffffffffi64 + 1),
            );

            // if MultiplyOperands {
            //     assert!(RightLookupOperand == Rs1Value * Rs2Value)
            // }
            cs.constrain_prod(
                JoltONNXR1CSInputs::RightInstructionInput(i),
                JoltONNXR1CSInputs::LeftInstructionInput(i),
                JoltONNXR1CSInputs::Product(i),
            );
            cs.constrain_eq_conditional(
                JoltONNXR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands),
                JoltONNXR1CSInputs::RightLookupOperand(i),
                JoltONNXR1CSInputs::Product(i),
            );

            // if Rd != 0 && WriteLookupOutputToRD {
            //     assert!(RdWriteValue == LookupOutput)
            // }
            cs.constrain_prod(
                JoltONNXR1CSInputs::Rd(i),
                JoltONNXR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToTD),
                JoltONNXR1CSInputs::WriteLookupOutputToTD(i),
            );
            cs.constrain_eq_conditional(
                JoltONNXR1CSInputs::WriteLookupOutputToTD(i),
                JoltONNXR1CSInputs::RdWriteValue(i),
                JoltONNXR1CSInputs::LookupOutput(i),
            );
        }
    }
}
