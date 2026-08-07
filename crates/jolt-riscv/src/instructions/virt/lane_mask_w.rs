use crate::jolt_instruction;

jolt_instruction!(
    /// Builds the word-lane mask selected by `rs1 + imm`.
    VirtualLaneMaskW,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
