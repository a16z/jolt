use crate::jolt_instruction;

jolt_instruction!(
    /// RV64M DIVUW: 32-bit unsigned division, sign-extended to 64 bits.
    /// Returns `u64::MAX` on division by zero.
    DivUW,
    circuit flags: [],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
