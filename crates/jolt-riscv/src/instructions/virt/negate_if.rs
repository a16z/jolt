use crate::jolt_instruction;

jolt_instruction!(
    /// Negates `rs2` when `rs1` is negative, otherwise returns `rs2`.
    VirtualNegateIf,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
