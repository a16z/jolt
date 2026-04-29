use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual ASSERT_LTE: returns 1 if `x <= y` (unsigned), 0 otherwise.
    AssertLte,
    circuit flags: [Assert],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
