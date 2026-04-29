use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual ASSERT_WORD_ALIGNMENT: checks whether `rs1 + imm` is 4-byte aligned.
    AssertWordAlignment,
    circuit flags: [AddOperands, Assert],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
