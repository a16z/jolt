use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual ASSERT_MULU_NO_OVERFLOW: checks unsigned multiply doesn't overflow.
    /// Returns 1 if the upper XLEN bits of `x * y` are all zero.
    AssertMulUNoOverflow,
    circuit flags: [MultiplyOperands, Assert],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
