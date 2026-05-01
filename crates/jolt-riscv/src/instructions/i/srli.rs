use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I SRLI: shift right logical by immediate.
    SrlI,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
