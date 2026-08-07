use crate::jolt_instruction;

jolt_instruction!(
    /// Computes the byte-lane multiplier selected by `rs1 + imm`.
    VirtualPow2Lane,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
