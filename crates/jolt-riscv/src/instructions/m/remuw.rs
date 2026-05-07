use crate::jolt_instruction;

jolt_instruction!(
    /// RV64M REMUW: 32-bit unsigned remainder, sign-extended to 64 bits.
    /// Returns `x` (truncated to 32 bits, sign-extended) on division by zero.
    RemUW,
    circuit flags: [],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
