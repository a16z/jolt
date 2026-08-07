use crate::jolt_instruction;

jolt_instruction!(
    /// Builds the byte-lane mask selected by `rs1 + imm`.
    VirtualLaneMaskB,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
