use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual POW2W: computes `2^(rs1 mod 32)` for 32-bit mode.
    Pow2W,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value]
);
