use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual MULIW: multiply by immediate, then sign-extend the low word.
    MulIW,
    circuit flags: [MultiplyOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
