use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual ASSERT_VALID_UNSIGNED_REMAINDER: validates unsigned remainder.
    /// Returns 1 if divisor is 0 or remainder < divisor.
    AssertValidUnsignedRemainder,
    circuit flags: [Assert],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
