use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual SRAI: arithmetic right shift using a bitmask immediate.
    VirtualSrai,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
