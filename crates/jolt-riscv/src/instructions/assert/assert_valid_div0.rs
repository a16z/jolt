use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual ASSERT_VALID_DIV0: validates `(divisor, quotient)` for division-by-zero handling.
    /// Returns 1 if the divisor is nonzero, or if the divisor is 0 and the quotient is MAX.
    AssertValidDiv0,
    circuit flags: [Assert],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
