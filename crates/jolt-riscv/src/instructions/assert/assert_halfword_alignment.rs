use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual ASSERT_HALFWORD_ALIGNMENT: checks whether `rs1 + imm` is 2-byte aligned.
    AssertHalfwordAlignment,
    circuit flags: [AddOperands, Assert],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
