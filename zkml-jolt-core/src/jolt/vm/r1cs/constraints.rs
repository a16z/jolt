use jolt_core::field::JoltField;
use onnx_tracer::trace_types::CircuitFlags;

use crate::jolt::vm::r1cs::{
    builder::{CombinedUniformBuilder, R1CSBuilder},
    inputs::JoltONNXR1CSInputs,
};

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
        // // if AddOperands || SubtractOperands || MultiplyOperands {
        // //     // Lookup query is just RightLookupOperand
        // //     assert!(LeftLookupOperand == 0)
        // // } else {
        // //     assert!(LeftLookupOperand == LeftInstructionInput)
        // // }
        // cs.constrain_if_else(
        //     JoltONNXR1CSInputs::OpFlags(CircuitFlags::AddOperands)
        //         + JoltONNXR1CSInputs::OpFlags(CircuitFlags::SubtractOperands)
        //         + JoltONNXR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands),
        //     0,
        //     JoltONNXR1CSInputs::LeftInstructionInput,
        //     JoltONNXR1CSInputs::LeftLookupOperand,
        // );

        // If AddOperands {
        //     assert!(RightLookupOperand == LeftInstructionInput + RightInstructionInput)
        // }
        cs.constrain_eq_conditional(
            JoltONNXR1CSInputs::OpFlags(CircuitFlags::AddOperands),
            JoltONNXR1CSInputs::RightLookupOperand,
            JoltONNXR1CSInputs::LeftInstructionInput + JoltONNXR1CSInputs::RightInstructionInput,
        );

        // // If SubtractOperands {
        // //     assert!(RightLookupOperand == LeftInstructionInput - RightInstructionInput)
        // // }
        // cs.constrain_eq_conditional(
        //     JoltONNXR1CSInputs::OpFlags(CircuitFlags::SubtractOperands),
        //     JoltONNXR1CSInputs::RightLookupOperand,
        //     // Converts from unsigned to twos-complement representation
        //     JoltONNXR1CSInputs::LeftInstructionInput - JoltONNXR1CSInputs::RightInstructionInput
        //         + (0xffffffffi64 + 1),
        // );

        // if MultiplyOperands {
        //     assert!(RightLookupOperand == Rs1Value * Rs2Value)
        // }
        cs.constrain_prod(
            JoltONNXR1CSInputs::RightInstructionInput,
            JoltONNXR1CSInputs::LeftInstructionInput,
            JoltONNXR1CSInputs::Product,
        );
        cs.constrain_eq_conditional(
            JoltONNXR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands),
            JoltONNXR1CSInputs::RightLookupOperand,
            JoltONNXR1CSInputs::Product,
        );

        // if Rd != 0 && WriteLookupOutputToRD {
        //     assert!(RdWriteValue == LookupOutput)
        // }
        cs.constrain_prod(
            JoltONNXR1CSInputs::Rd,
            JoltONNXR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToRD),
            JoltONNXR1CSInputs::WriteLookupOutputToRD,
        );
        cs.constrain_eq_conditional(
            JoltONNXR1CSInputs::WriteLookupOutputToRD,
            JoltONNXR1CSInputs::RdWriteValue,
            JoltONNXR1CSInputs::LookupOutput,
        );
    }
}
