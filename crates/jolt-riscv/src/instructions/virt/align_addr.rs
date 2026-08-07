use crate::jolt_instruction;

jolt_instruction!(
    /// Aligns `rs1 + imm` down to its containing doubleword address.
    VirtualAlignAddr,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
