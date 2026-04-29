use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual POW2IW: computes `2^(imm mod 32)` for 32-bit immediate mode.
    Pow2IW,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [RightOperandIsImm]
);
