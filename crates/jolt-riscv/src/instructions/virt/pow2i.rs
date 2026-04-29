use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual POW2I: computes `2^imm` with immediate exponent.
    Pow2I,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [RightOperandIsImm]
);
