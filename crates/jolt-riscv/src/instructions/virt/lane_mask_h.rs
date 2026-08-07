use crate::jolt_instruction;

jolt_instruction!(
    /// Builds the halfword-lane mask selected by `rs1 + imm`.
    VirtualLaneMaskH,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
