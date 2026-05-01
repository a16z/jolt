use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual ASSERT_EQ: returns 1 if operands are equal, 0 otherwise.
    AssertEq,
    circuit flags: [Assert],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
