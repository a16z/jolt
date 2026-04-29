use crate::jolt_instruction;

jolt_instruction!(
    /// RV64M MULW: 32-bit multiply, sign-extended to 64 bits.
    MulW,
    circuit flags: [MultiplyOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
