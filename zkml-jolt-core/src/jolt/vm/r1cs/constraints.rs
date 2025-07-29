use jolt_core::{
    field::JoltField,
    jolt::instruction::CircuitFlags,
    r1cs::{builder::R1CSBuilder, constraints::R1CSConstraints},
};

use crate::jolt::vm::r1cs::inputs::JoltONNXR1CSInputs;

pub struct JoltONNXConstraints;
impl<F: JoltField> R1CSConstraints<F> for JoltONNXConstraints {
    fn uniform_constraints(cs: &mut R1CSBuilder) {
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
            JoltONNXR1CSInputs::LeftInstructionInput,
            JoltONNXR1CSInputs::LeftLookupOperand,
        );
    }
}
